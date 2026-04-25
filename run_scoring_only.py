#!/usr/bin/env python3
# run_scoring_only.py
# Run token scoring (masking) only — no training, no evaluation.
# Usage: python3 run_scoring_only.py

import os, sys, json, gc, torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SAVE_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(SAVE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "gsm8k_train.jsonl")

# ============================================================
# CONFIG — toggle as needed
# ============================================================
RUN_RHO1   = False
RUN_OURS_B = False
RUN_OURS_A = True
RUN_OURS_C = False

RHO1_SELECT_RATIO   = 0.10
OURS_A_SELECT_RATIO = 0.12
OURS_C_GAP_RATIO    = 0.40
OURS_C_ABS_RATIO    = 0.40
MAX_LENGTH          = 512

# context expansion variants to run for each scorer
# (mode, window) pairs — applied to all scorers that support it
CONTEXT_VARIANTS = [
    ("window", 1),
    ("window", 3),
    ("span",   1),
]
# ============================================================


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  Free VRAM: {free:.1f} GB")


def need_scoring(path):
    if os.path.exists(path):
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"  Already exists ({n} lines): {os.path.basename(path)} — skipping")
        return False
    return True


def variant_suffix(mode, window):
    """e.g. 'span_w3' or 'window_w1'"""
    return f"{mode}_w{window}"


# ============================================================
# Download GSM8K if needed
# ============================================================
if not os.path.exists(RAW_PATH):
    print("Downloading GSM8K train set...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    with open(RAW_PATH, "w") as f:
        for item in ds:
            f.write(json.dumps({"question": item['question'], "answer": item['answer']}) + "\n")
            # text = f"Question: {item['question'].strip()}\nAnswer: {item['answer'].strip()}"
            # f.write(json.dumps({"text": text}) + "\n")
    print(f"  Saved {len(ds)} examples")
else:
    with open(RAW_PATH) as f:
        n = sum(1 for _ in f)
    print(f"GSM8K already downloaded ({n} examples)")


# ============================================================
# RHO-1 scoring
# — base: no expansion (original behaviour)
# — variants: window/span × w1/w3
# ============================================================
if RUN_RHO1:
    from symbolic_slm.scoring.rho1_scorer import preprocess_dataset_rho1

    # base (no expansion)
    rho1_base_path = os.path.join(DATA_DIR, "gsm8k_rho1_base.jsonl")
    if need_scoring(rho1_base_path):
        print("\n[RHO-1] base (no expansion)...")
        clear_memory()
        preprocess_dataset_rho1(
            data_path=RAW_PATH,
            output_path=rho1_base_path,
            select_ratio=RHO1_SELECT_RATIO,
            context_mode=None,
            max_length=MAX_LENGTH,
        )

    # expansion variants
    for mode, w in CONTEXT_VARIANTS:
        path = os.path.join(DATA_DIR, f"gsm8k_rho1_{variant_suffix(mode, w)}.jsonl")
        if need_scoring(path):
            print(f"\n[RHO-1] {variant_suffix(mode, w)}...")
            clear_memory()
            preprocess_dataset_rho1(
                data_path=RAW_PATH,
                output_path=path,
                select_ratio=RHO1_SELECT_RATIO,
                context_mode=mode,
                window=w,
                max_length=MAX_LENGTH,
            )

    clear_memory()
    print("[RHO-1] Done.")


# ============================================================
# Ours-B scoring (symbolic rule-based)
# — variants: window/span × w1/w3
# ============================================================
if RUN_OURS_B:
    from symbolic_slm.scoring.symbolic_scorer import preprocess_dataset_symbolic

    for mode, w in CONTEXT_VARIANTS:
        path = os.path.join(DATA_DIR, f"gsm8k_symbolic_{variant_suffix(mode, w)}.jsonl")
        if need_scoring(path):
            print(f"\n[Ours-B] {variant_suffix(mode, w)}...")
            clear_memory()
            preprocess_dataset_symbolic(
                data_path=RAW_PATH,
                output_path=path,
                context_mode=mode,
                window=w,
                max_length=MAX_LENGTH,
            )

    clear_memory()
    print("[Ours-B] Done.")


# ============================================================
# Ours-A scoring (entropy gap)
# — variants: window/span × w1/w3
# ============================================================
if RUN_OURS_A:
    from symbolic_slm.scoring.entropy_gap_scorer import preprocess_dataset_entropy_gap

    for mode, w in CONTEXT_VARIANTS:
        path = os.path.join(DATA_DIR, f"gsm8k_entropy_gap_{variant_suffix(mode, w)}.jsonl")
        if need_scoring(path):
            print(f"\n[Ours-A] {variant_suffix(mode, w)}...")
            clear_memory()
            preprocess_dataset_entropy_gap(
                data_path=RAW_PATH,
                output_path=path,
                select_ratio=OURS_A_SELECT_RATIO,
                context_mode=mode,
                window=w,
                max_length=MAX_LENGTH,
            )

    clear_memory()
    print("[Ours-A] Done.")


# ============================================================
# Ours-C scoring (dual-threshold entropy gap)
# — variants: window/span × w1/w3
# ============================================================
if RUN_OURS_C:
    from symbolic_slm.scoring.cft_scorer import preprocess_dataset_cft_variants

    variants = [
        (mode, w, os.path.join(DATA_DIR, f"gsm8k_cft_{variant_suffix(mode, w)}.jsonl"))
        for mode, w in CONTEXT_VARIANTS
        if need_scoring(os.path.join(DATA_DIR, f"gsm8k_cft_{variant_suffix(mode, w)}.jsonl"))
    ]
    if variants:
        preprocess_dataset_cft_variants(
            data_path=RAW_PATH,
            variants=variants,
            workdir=os.path.join(DATA_DIR, "cft_workdir"),
        )


# ============================================================
# Sanity check
# ============================================================
print("\n" + "=" * 65)
print("Sanity check (first example, selected/total tokens):")

all_outputs = []
if RUN_RHO1:
    all_outputs.append(("RHO-1 base", os.path.join(DATA_DIR, "gsm8k_rho1_base.jsonl")))
    for mode, w in CONTEXT_VARIANTS:
        tag = variant_suffix(mode, w)
        all_outputs.append((f"RHO-1 {tag}",  os.path.join(DATA_DIR, f"gsm8k_rho1_{tag}.jsonl")))
if RUN_OURS_B:
    for mode, w in CONTEXT_VARIANTS:
        tag = variant_suffix(mode, w)
        all_outputs.append((f"Ours-B {tag}", os.path.join(DATA_DIR, f"gsm8k_symbolic_{tag}.jsonl")))
if RUN_OURS_A:
    for mode, w in CONTEXT_VARIANTS:
        tag = variant_suffix(mode, w)
        all_outputs.append((f"Ours-A {tag}", os.path.join(DATA_DIR, f"gsm8k_entropy_gap_{tag}.jsonl")))
if RUN_OURS_C:
    for mode, w in CONTEXT_VARIANTS:
        tag = variant_suffix(mode, w)
        all_outputs.append((f"Ours-C {tag}", os.path.join(DATA_DIR, f"gsm8k_cft_{tag}.jsonl")))

for label, path in all_outputs:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        item = json.loads(f.readline())
    tokens   = len(item["token_ids"])
    selected = sum(item["mask"])
    print(f"  {label:22s}: {selected}/{tokens} = {selected/tokens*100:.1f}%")

print("=" * 65)
print("Scoring complete. Ready for training.")