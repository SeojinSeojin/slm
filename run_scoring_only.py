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
RUN_RHO1   = True
RUN_OURS_B = True
RUN_OURS_A = True
RUN_OURS_C = True

RHO1_SELECT_RATIO   = 0.77
OURS_A_SELECT_RATIO = 0.25   # + context_window=3 -> ~75%
OURS_C_GAP_RATIO    = 0.40
OURS_C_ABS_RATIO    = 0.40
MAX_LENGTH          = 512
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


# ============================================================
# Download GSM8K if needed
# ============================================================
if not os.path.exists(RAW_PATH):
    print("Downloading GSM8K train set...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    with open(RAW_PATH, "w") as f:
        for item in ds:
            text = f"Question: {item['question'].strip()}\nAnswer: {item['answer'].strip()}"
            f.write(json.dumps({"text": text}) + "\n")
    print(f"  Saved {len(ds)} examples")
else:
    with open(RAW_PATH) as f:
        n = sum(1 for _ in f)
    print(f"GSM8K already downloaded ({n} examples)")


# ============================================================
# RHO-1 scoring (Qwen2.5-0.5B reference, same tokenizer)
# ============================================================
rho1_path = os.path.join(DATA_DIR, "gsm8k_scored_rho1.jsonl")
if RUN_RHO1 and need_scoring(rho1_path):
    print("\n[RHO-1] Scoring with Qwen2.5-0.5B reference...")
    clear_memory()
    from symbolic_slm.scoring.rho1_scorer import preprocess_dataset_rho1
    preprocess_dataset_rho1(
        data_path=RAW_PATH, output_path=rho1_path,
        select_ratio=RHO1_SELECT_RATIO, max_length=MAX_LENGTH,
    )
    clear_memory()
    print("[RHO-1] Done.")


# ============================================================
# Ours-B scoring (symbolic rule-based)
# ============================================================
b_path = os.path.join(DATA_DIR, "gsm8k_scored_symbolic.jsonl")
if RUN_OURS_B and need_scoring(b_path):
    print("\n[Ours-B] Symbolic rule-based scoring...")
    clear_memory()
    from symbolic_slm.scoring.symbolic_scorer import preprocess_dataset_symbolic
    preprocess_dataset_symbolic(
        data_path=RAW_PATH, output_path=b_path,
        max_length=MAX_LENGTH,
    )
    clear_memory()
    print("[Ours-B] Done.")


# ============================================================
# Ours-A scoring (entropy gap, Qwen2.5-Math-7B)
# ============================================================
a_path = os.path.join(DATA_DIR, "gsm8k_scored_entropy_gap.jsonl")
if RUN_OURS_A and need_scoring(a_path):
    print("\n[Ours-A] Entropy gap scoring (Qwen2.5-Math-7B)...")
    clear_memory()
    from symbolic_slm.scoring.entropy_gap_scorer import preprocess_dataset_entropy_gap
    preprocess_dataset_entropy_gap(
        data_path=RAW_PATH, output_path=a_path,
        select_ratio=OURS_A_SELECT_RATIO,
        context_window=3,
        max_length=MAX_LENGTH,
    )
    clear_memory()
    print("[Ours-A] Done.")


# ============================================================
# Ours-C scoring (dual-threshold)
# ============================================================
c_path = os.path.join(DATA_DIR, "gsm8k_scored_math_confidence.jsonl")
if RUN_OURS_C and need_scoring(c_path):
    print("\n[Ours-C] Dual-threshold scoring...")
    clear_memory()
    from symbolic_slm.scoring.math_confidence_scorer import preprocess_dataset_math_confidence
    preprocess_dataset_math_confidence(
        data_path=RAW_PATH, output_path=c_path,
        gap_ratio=OURS_C_GAP_RATIO, abs_ratio=OURS_C_ABS_RATIO,
        context_window=3, max_length=MAX_LENGTH,
    )
    clear_memory()
    print("[Ours-C] Done.")


# ============================================================
# Sanity check
# ============================================================
print("\n" + "=" * 55)
print("Sanity check (first example):")
for label, path in [
    ("RHO-1",  rho1_path),
    ("Ours-B", b_path),
    ("Ours-A", a_path),
    ("Ours-C", c_path),
]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        item = json.loads(f.readline())
    tokens   = len(item["token_ids"])
    selected = sum(item["mask"])
    print(f"  {label:8s}: {selected}/{tokens} = {selected/tokens*100:.1f}%")
print("=" * 55)
print("Scoring complete. Ready for training.")
