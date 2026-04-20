# scorers/cft_scorer.py
#
# Method C (CFT-style) Scorer — All-in-one
#
# Two-phase CFT-style token scoring that produces a binary critical mask
# per token for each sample in GSM8K train.
#
# Phase 1-A: Greedy filter 500 correct samples + counterfactual perturbation → ground-truth masks
# Phase 1-B: Brief fine-tune on those masks → M_cft
# Phase 2:   P_cft - P_base probability gap → top 15% critical for all remaining samples
# Merge:     Phase 1 + Phase 2 → expand_mask → cft_masks_final.jsonl
#
# Usage:
#   python cft_scorer.py --data gsm8k_train.jsonl
#   python cft_scorer.py --data gsm8k_train.jsonl --skip_phase1  # if M_cft already exists

import json
import os
import re
import argparse
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any

from scoring.context_expansion import expand_mask

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
TARGET_SAMPLES   = 500
TOPK             = 3
MAX_NEW_TOKENS   = 512
SUFFIX_EXTRA     = 20
CRITICAL_RATIO   = 0.15
FINETUNE_STEPS   = 75
LEARNING_RATE    = 2e-5
BATCH_SIZE       = 2
GRAD_ACCUM       = 4
MAX_LENGTH       = 512

FORMAT_STRINGS = {"Question:", "Answer:", "\n", " "}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

_ANSWER_RE = re.compile(r"####\s*([\d,\.\-]+)")


def extract_answer(text: str) -> Optional[float]:
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "").strip())
        except ValueError:
            return None
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def save_masks(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            out = {
                "id":            r["id"],
                "question":      r["question"],
                "response":      r["response"],
                "token_ids":     r["token_ids"],
                "critical_mask": r["critical_mask"],
            }
            for k, v in r.items():
                if k not in out:
                    out[k] = v
            f.write(json.dumps(out) + "\n")


def load_masks(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def is_format_token(tok: str) -> bool:
    return tok.strip() in FORMAT_STRINGS or tok in FORMAT_STRINGS


def load_model(model_name_or_dir: str, device: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_dir, torch_dtype=dtype
    ).to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1-A  Greedy filter + counterfactual perturbation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
) -> Tuple[str, List[int]]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp_ids = out[0, prompt_len:].tolist()
    return tokenizer.decode(resp_ids, skip_special_tokens=True), resp_ids


@torch.no_grad()
def get_topk_at_each_position(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    prompt_len: int,
    device: str,
) -> torch.Tensor:
    logits      = model(full_ids.to(device)).logits[0]
    resp_logits = logits[prompt_len - 1 : -1]
    return resp_logits.topk(TOPK, dim=-1).indices.cpu()


@torch.no_grad()
def run_counterfactual_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: List[int],
    resp_ids: List[int],
    alt_rank: int,
    topk_ids: torch.Tensor,
    gold_answer: float,
    device: str,
) -> List[int]:
    resp_len = len(resp_ids)
    pad_id   = tokenizer.pad_token_id or tokenizer.eos_token_id

    batch_inputs, suffix_lens = [], []
    for i in range(resp_len):
        alt_id = topk_ids[i, alt_rank].item()
        batch_inputs.append(prompt_ids + resp_ids[:i] + [alt_id])
        suffix_lens.append(resp_len - i - 1)

    max_len = max(len(x) for x in batch_inputs)
    padded  = [([pad_id] * (max_len - len(x))) + x for x in batch_inputs]
    attn    = [[0] * (max_len - len(x)) + [1] * len(x) for x in batch_inputs]

    out = model.generate(
        input_ids=torch.tensor(padded, dtype=torch.long).to(device),
        attention_mask=torch.tensor(attn, dtype=torch.long).to(device),
        max_new_tokens=max(suffix_lens) + SUFFIX_EXTRA,
        do_sample=False,
        temperature=1.0,
        pad_token_id=pad_id,
    )

    results = []
    for i, gen in enumerate(out.tolist()):
        new_toks = gen[len(padded[i]):]
        gen_text = tokenizer.decode(new_toks, skip_special_tokens=True)
        pred     = extract_answer(gen_text)
        results.append(1 if (pred is not None and pred == gold_answer) else 0)
    return results


def annotate_one_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    response: str,
    resp_ids: List[int],
    gold_answer: float,
    device: str,
) -> Tuple[List[int], List[int]]:
    prompt     = build_prompt(question)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    full_ids   = torch.tensor([prompt_ids + resp_ids], dtype=torch.long)

    topk_ids = get_topk_at_each_position(model, full_ids, len(prompt_ids), device)

    F_alt2 = run_counterfactual_batch(
        model, tokenizer, prompt_ids, resp_ids, 1, topk_ids, gold_answer, device)
    F_alt3 = run_counterfactual_batch(
        model, tokenizer, prompt_ids, resp_ids, 2, topk_ids, gold_answer, device)

    resp_tokens = [tokenizer.decode([tid]) for tid in resp_ids]
    resp_mask   = []
    for i, tok in enumerate(resp_tokens):
        if is_format_token(tok):
            resp_mask.append(1)
        else:
            resp_mask.append(1 if (F_alt2[i] == 0 and F_alt3[i] == 0) else 0)

    return prompt_ids + resp_ids, [1] * len(prompt_ids) + resp_mask


def run_phase1_annotate(
    data_path: str,
    output_path: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    target_n: int = TARGET_SAMPLES,
) -> List[Dict]:
    print(f"\n{'='*60}")
    print("[Phase 1-A] Greedy filter + counterfactual annotation")
    print(f"{'='*60}")

    with open(data_path) as f:
        all_samples = [json.loads(l) for l in f]

    annotations = []
    print(f"Scanning {len(all_samples)} samples for {target_n} correctly-solved ones...")

    for item in tqdm(all_samples):
        if len(annotations) >= target_n:
            break

        question    = item["question"]
        gold_answer = extract_answer(item["answer"])
        if gold_answer is None:
            continue

        response, resp_ids = greedy_generate(model, tokenizer, build_prompt(question), device)
        if extract_answer(response) != gold_answer:
            continue

        try:
            all_ids, mask = annotate_one_sample(
                model, tokenizer, question, response, resp_ids, gold_answer, device)
        except Exception as e:
            print(f"  Annotation error: {e}")
            continue

        annotations.append({
            "id":            len(annotations),
            "question":      question,
            "response":      response,
            "token_ids":     all_ids,
            "critical_mask": mask,
        })

    print(f"Collected {len(annotations)} annotated samples.")
    save_masks(annotations, output_path)
    print(f"Saved → {output_path}")
    return annotations


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1-B  Brief fine-tune → M_cft
# ══════════════════════════════════════════════════════════════════════════════

class CriticalMaskDataset(Dataset):
    def __init__(self, annotations: List[Dict], max_length: int = MAX_LENGTH):
        self.samples    = annotations
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item   = self.samples[idx]
        ids    = item["token_ids"][:self.max_length]
        mask   = item["critical_mask"][:self.max_length]
        labels = [tid if m == 1 else -100 for tid, m in zip(ids, mask)]
        return {
            "input_ids":      torch.tensor(ids,    dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: List[Dict], pad_id: int) -> Dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    def pad(t, val): return torch.cat([t, torch.full((max_len - t.shape[0],), val, dtype=torch.long)])
    return {
        "input_ids":      torch.stack([pad(x["input_ids"],      pad_id) for x in batch]),
        "attention_mask": torch.stack([pad(x["attention_mask"], 0)      for x in batch]),
        "labels":         torch.stack([pad(x["labels"],        -100)    for x in batch]),
    }


def run_phase1_finetune(
    annotations: List[Dict],
    output_dir: str,
    base_model_name: str,
    device: str,
    dtype: torch.dtype,
    steps: int = FINETUNE_STEPS,
) -> str:
    print(f"\n{'='*60}")
    print("[Phase 1-B] Brief fine-tune → M_cft")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype
    ).to(device).train()

    loader = DataLoader(
        CriticalMaskDataset(annotations),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        drop_last=False,
    )
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn    = nn.CrossEntropyLoss(ignore_index=-100)
    global_step, accum_loss = 0, 0.0
    optimizer.zero_grad()

    print(f"Fine-tuning for {steps} steps (bs={BATCH_SIZE}, accum={GRAD_ACCUM}, lr={LEARNING_RATE})...")
    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break
            ids    = batch["input_ids"].to(device)
            attn   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                logits = model(input_ids=ids, attention_mask=attn).logits.float()
                loss   = loss_fn(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                ) / GRAD_ACCUM

            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if global_step % 10 == 0:
                    print(f"  step {global_step:4d}/{steps}  loss={accum_loss:.4f}")
                accum_loss = 0.0
            global_step += 1

    optimizer.step()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"M_cft saved → {output_dir}")
    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2  Probability-gap scoring
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_token_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    logits    = model(input_ids.to(device)).logits[0].float()
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    targets   = input_ids[0, 1:]
    return log_probs[torch.arange(log_probs.shape[0]), targets].exp().cpu()


def run_phase2_score(
    data_path: str,
    output_path: str,
    base_model: AutoModelForCausalLM,
    cft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    phase1_ids: set,
    critical_ratio: float = CRITICAL_RATIO,
) -> List[Dict]:
    print(f"\n{'='*60}")
    print("[Phase 2] Probability-gap scoring (all samples)")
    print(f"{'='*60}")

    with open(data_path) as f:
        all_samples = [json.loads(l) for l in f]

    cache: List[Optional[Dict]] = []
    all_resp_scores: List[float] = []

    print(f"Pass 1: scoring {len(all_samples)} samples...")
    for idx, item in enumerate(tqdm(all_samples)):
        if idx in phase1_ids:
            cache.append(None)
            continue

        question = item["question"]
        prompt   = build_prompt(question)

        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = base_model.generate(
                **enc.to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        p_len    = enc["input_ids"].shape[1]
        resp_ids = out[0, p_len:].tolist()
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)

        full_text = prompt + " " + response
        enc2      = tokenizer(full_text, return_tensors="pt",
                              max_length=MAX_LENGTH, truncation=True)
        input_ids = enc2["input_ids"]
        seq_len   = input_ids.shape[1]

        prompt_enc = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        try:
            probs_cft  = get_token_probs(cft_model,  input_ids, device)
            probs_base = get_token_probs(base_model, input_ids, device)
            gap_scores = (probs_cft - probs_base).numpy()
        except Exception as e:
            print(f"  Scoring error at sample {idx}: {e}")
            cache.append(None)
            continue

        token_ids  = input_ids[0].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        for i in range(seq_len):
            if i >= prompt_len and not is_format_token(token_strs[i]):
                score_idx = i - 1
                if score_idx < len(gap_scores):
                    all_resp_scores.append(float(gap_scores[score_idx]))

        cache.append({
            "sample_id":  idx,
            "question":   question,
            "response":   response,
            "token_ids":  token_ids,
            "token_strs": token_strs,
            "prompt_len": prompt_len,
            "gap_scores": gap_scores,
        })

    if all_resp_scores:
        arr       = np.array(all_resp_scores, dtype=np.float32)
        threshold = float(np.percentile(arr, 100 * (1 - critical_ratio)))
        print(f"\nThreshold (top {critical_ratio*100:.0f}%): {threshold:.6f}  "
              f"[range {arr.min():.4f} .. {arr.max():.4f}]")
    else:
        threshold = 0.0
        print("Warning: no scores collected. Threshold set to 0.")

    print("Pass 2: building critical masks...")
    records = []
    for result in tqdm(cache):
        if result is None:
            continue
        token_ids  = result["token_ids"]
        token_strs = result["token_strs"]
        gap_scores = result["gap_scores"]
        p_len      = result["prompt_len"]
        seq_len    = len(token_ids)

        mask = []
        for i in range(seq_len):
            if i < p_len or is_format_token(token_strs[i]):
                mask.append(1)
            else:
                score_idx = i - 1
                score = float(gap_scores[score_idx]) if score_idx < len(gap_scores) else -999.0
                mask.append(1 if score >= threshold else 0)

        records.append({
            "id":            result["sample_id"],
            "question":      result["question"],
            "response":      result["response"],
            "token_ids":     token_ids,
            "critical_mask": mask,
        })

    save_masks(records, output_path)
    total    = sum(len(r["critical_mask"]) for r in records)
    critical = sum(sum(r["critical_mask"]) for r in records)
    print(f"Saved {len(records)} samples → {output_path}")
    print(f"  Total tokens: {total:,}  |  Critical: {critical:,} ({critical/max(total,1)*100:.1f}%)")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Merge + expand
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_expand_masks(
    phase1_records: List[Dict],
    phase2_records: List[Dict],
    output_path: str,
    tokenizer: AutoTokenizer,
    context_mode: str = "window",  # "window" | "span"
    window: int = 2,
) -> None:
    """
    Merge Phase 1 + Phase 2 records, then apply context expansion once
    to the combined critical_mask of every sample.
    """
    print(f"\n{'='*60}")
    print(f"[Merge] Combining Phase 1 + Phase 2  (context_mode={context_mode!r}, window={window})")
    print(f"{'='*60}")

    p1_ids   = {r["id"] for r in phase1_records}
    p2_dedup = [r for r in phase2_records if r["id"] not in p1_ids]
    merged   = sorted(phase1_records + p2_dedup, key=lambda r: r["id"])

    expanded_records = []
    for r in tqdm(merged, desc="Expanding masks"):
        token_ids  = r["token_ids"]
        raw_mask   = r["critical_mask"]

        token_strings = [
            tokenizer.decode([tid], skip_special_tokens=False)
            for tid in token_ids
        ]
        anchor_mask  = torch.tensor(raw_mask, dtype=torch.bool)
        final_mask   = expand_mask(
            anchor_mask,
            token_strings,
            mode=context_mode,
            window=window,
        )
        expanded_records.append({**r, "critical_mask": final_mask.tolist()})

    save_masks(expanded_records, output_path)
    total    = sum(len(r["critical_mask"]) for r in expanded_records)
    critical = sum(sum(r["critical_mask"]) for r in expanded_records)
    print(f"Merged + expanded {len(expanded_records)} samples → {output_path}")
    print(f"  Total tokens: {total:,}  |  Critical: {critical:,} ({critical/max(total,1)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_dataset_cft(
    data_path:      str,
    output_path:    str,
    base_model_name: str   = BASE_MODEL_NAME,
    phase1_n:       int    = TARGET_SAMPLES,
    finetune_steps: int    = FINETUNE_STEPS,
    critical_ratio: float  = CRITICAL_RATIO,
    context_mode:   str    = "window",   # "window" | "span"
    window:         int    = 2,
    skip_phase1:    bool   = False,
    workdir:        str    = None,
):
    """
    Full CFT-style scoring pipeline. Drop-in equivalent of
    preprocess_dataset_* in other scorers.

    Parameters
    ----------
    data_path       : GSM8K train JSONL path.
    output_path     : Where to write the final masked JSONL.
    base_model_name : Base model for Phase 1 annotation + Phase 2 scoring.
    phase1_n        : Number of correctly-solved samples to annotate in Phase 1.
    finetune_steps  : Gradient steps for Phase 1-B fine-tune.
    critical_ratio  : Top fraction selected as critical in Phase 2.
    context_mode    : "window" or "span" — passed to expand_mask at merge.
    window          : Half-width for window mode.
    skip_phase1     : If True and intermediate files exist, skip Phase 1.
    workdir         : Directory for intermediate files. Defaults to the
                      directory containing output_path.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(workdir, exist_ok=True)

    p1_path = os.path.join(workdir, "phase1_annotations.jsonl")
    cft_dir = os.path.join(workdir, "M_cft")
    p2_path = os.path.join(workdir, "phase2_masks.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if skip_phase1 and os.path.exists(p1_path) and os.path.exists(os.path.join(cft_dir, "config.json")):
        print("\n[Phase 1] Skipping — loading existing outputs...")
        phase1_records = load_masks(p1_path)
    else:
        print(f"\nLoading base model: {base_model_name}")
        tokenizer  = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = load_model(base_model_name, device, dtype)

        phase1_records = run_phase1_annotate(
            data_path, p1_path, base_model, tokenizer, device, phase1_n)

        run_phase1_finetune(
            phase1_records, cft_dir, base_model_name, device, dtype, finetune_steps)

        del base_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"\nLoading models for Phase 2...")
    tokenizer  = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = load_model(base_model_name, device, dtype)
    cft_model  = load_model(cft_dir, device, dtype)

    phase1_ids     = {r["id"] for r in phase1_records}
    phase2_records = run_phase2_score(
        data_path, p2_path,
        base_model, cft_model, tokenizer, device,
        phase1_ids=phase1_ids,
        critical_ratio=critical_ratio,
    )

    # ── Merge + expand ────────────────────────────────────────────────────────
    merge_and_expand_masks(
        phase1_records, phase2_records, output_path,
        tokenizer=tokenizer,
        context_mode=context_mode,
        window=window,
    )

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CFT-style scorer (all-in-one)")
    parser.add_argument("--data",           required=True,      help="GSM8K train JSONL")
    parser.add_argument("--workdir",        default=".",        help="Output directory")
    parser.add_argument("--base_model",     default=BASE_MODEL_NAME)
    parser.add_argument("--phase1_n",       type=int,   default=TARGET_SAMPLES)
    parser.add_argument("--finetune_steps", type=int,   default=FINETUNE_STEPS)
    parser.add_argument("--critical_ratio", type=float, default=CRITICAL_RATIO)
    parser.add_argument("--context_mode",   default="window", choices=["window", "span"],
                        help="Context expansion mode applied at merge step")
    parser.add_argument("--window",         type=int,   default=2,
                        help="Half-width for window mode (ignored for span)")
    parser.add_argument("--skip_phase1",    action="store_true",
                        help="Skip Phase 1 if phase1_annotations.jsonl + M_cft already exist")
    args = parser.parse_args()

    final_path = os.path.join(args.workdir, "cft_masks_final.jsonl")

    preprocess_dataset_cft(
        data_path=args.data,
        output_path=final_path,
        base_model_name=args.base_model,
        phase1_n=args.phase1_n,
        finetune_steps=args.finetune_steps,
        critical_ratio=args.critical_ratio,
        context_mode=args.context_mode,
        window=args.window,
        skip_phase1=args.skip_phase1,
        workdir=args.workdir,
    )

    print(f"\n✓ Done!  Final mask file: {final_path}")


if __name__ == "__main__":
    main()