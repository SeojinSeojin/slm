# scoring/entropy_gap_scorer.py
# Approach A: Entropy Gap Scoring
#
# Select tokens where general model is uncertain but math model is confident.
#
# Score: delta_H(x_i) = H_general(x_i) - H_math(x_i)
#   -> high delta_H = general LM struggles, math LM finds it easy
#   -> these are "math-critical" tokens
#
# Models:
#   General: Qwen2.5-1.5B
#   Math:    Qwen2.5-Math-7B (stronger math signal, same tokenizer family)

import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

GENERAL_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SYMBOLIC_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy: H(x_i) = -sum_v p(v|x<i) * log p(v|x<i)"""
    probs     = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def get_model_entropy(
    model:     AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device:    str = "cuda",
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    outputs   = model(input_ids)
    logits    = outputs.logits[0].float()
    entropy   = compute_token_entropy(logits[:-1])
    return entropy.cpu()


class EntropyGapScorer:
    """
    Approach A: select tokens with high H_general - H_math.
    Expand with context window to preserve local reasoning context.
    """

    def __init__(
        self,
        general_model_name:  str = GENERAL_MODEL_NAME,
        symbolic_model_name: str = SYMBOLIC_MODEL_NAME,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (
            self.device == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16

        print(f"Device: {self.device}")
        print(f"Loading general model: {general_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(general_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.general_model = AutoModelForCausalLM.from_pretrained(
            general_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        print(f"Loading math model: {symbolic_model_name}")
        self.symbolic_model = AutoModelForCausalLM.from_pretrained(
            symbolic_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        self.sym_vocab_size = self.symbolic_model.config.vocab_size
        gen_vocab = self.general_model.config.vocab_size
        if self.sym_vocab_size != gen_vocab:
            print(f"  Vocab mismatch: general={gen_vocab}, math={self.sym_vocab_size} — will clamp")

        print("Entropy Gap scorer ready.")

    @torch.no_grad()
    def score(
        self,
        text:           str,
        select_ratio:   float = 0.25,
        context_window: int   = 2,
        max_length:     int   = 512,
    ) -> dict:
        """
        Select top select_ratio% by entropy gap, expand with context window.
        select_ratio=0.25 + context_window=2 -> ~75% final ratio.
        """
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        token_ids = enc["input_ids"]

        entropy_g = get_model_entropy(self.general_model, token_ids, self.device)

        token_ids_sym = token_ids.clone().clamp(max=self.sym_vocab_size - 1)
        entropy_s = get_model_entropy(self.symbolic_model, token_ids_sym, self.device)

        min_len   = min(len(entropy_g), len(entropy_s))
        entropy_g = entropy_g[:min_len]
        entropy_s = entropy_s[:min_len]

        gap = (entropy_g - entropy_s).numpy()

        token_ids_list = token_ids[0, :min_len].tolist()

        # Top-k% selection
        n_select  = max(1, int(min_len * select_ratio))
        threshold = np.sort(gap)[-n_select]
        base_mask = gap >= threshold

        # Expand with context window
        expanded = base_mask.copy()
        for i in range(min_len):
            if base_mask[i]:
                lo = max(0, i - context_window)
                hi = min(min_len, i + context_window + 1)
                expanded[lo:hi] = True

        mask = expanded.tolist()

        return {
            "token_ids":        token_ids_list,
            "mask":             mask,
            "entropy_general":  entropy_g.tolist(),
            "entropy_symbolic": entropy_s.tolist(),
            "entropy_gap":      gap.tolist(),
        }


def preprocess_dataset_entropy_gap(
    data_path:           str,
    output_path:         str,
    general_model_name:  str   = GENERAL_MODEL_NAME,
    symbolic_model_name: str   = SYMBOLIC_MODEL_NAME,
    select_ratio:        float = 0.25,
    context_window:      int   = 2,
    max_length:          int   = 512,
):
    """Apply entropy gap scoring to entire JSONL dataset."""
    scorer = EntropyGapScorer(general_model_name, symbolic_model_name)

    with open(data_path, "r") as f:
        lines = f.readlines()

    print(f"Processing {len(lines):,} documents with entropy gap scoring...")
    print("  (2 forward passes per document)")

    results         = []
    total_tokens    = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        try:
            scored = scorer.score(
                item["text"],
                select_ratio=select_ratio,
                context_window=context_window,
                max_length=max_length,
            )
            results.append({
                "text":      item["text"],
                "token_ids": scored["token_ids"],
                "mask":      scored["mask"],
                "method":    "entropy_gap",
            })
            total_tokens    += len(scored["token_ids"])
            selected_tokens += sum(scored["mask"])
        except Exception as e:
            print(f"  Skipping (error): {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / max(total_tokens, 1) * 100
    print(f"\nDone! Saved: {output_path}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Selected tokens: {selected_tokens:,} ({ratio:.1f}%)")


if __name__ == "__main__":
    scorer = EntropyGapScorer()
    text = "Question: John takes a pill every 6 hours. How many pills does he take a week?\nAnswer: He takes 24/6=<<24/6=4>>4 pills a day\nSo he takes 4*7=<<4*7=28>>28 pills a week\n#### 28"
    tokenizer = AutoTokenizer.from_pretrained(GENERAL_MODEL_NAME)

    for ratio in [0.20, 0.25, 0.30]:
        r = scorer.score(text, select_ratio=ratio, context_window=2)
        tokens = [tokenizer.decode([i]) for i in r["token_ids"]]
        sel = sum(r["mask"])
        total = len(r["mask"])
        rendered = "".join(f"[{t}]" if m else t for t, m in zip(tokens, r["mask"]))
        print(f"\nselect_ratio={ratio}: {sel}/{total}={sel/total*100:.0f}%")
        print(rendered)
