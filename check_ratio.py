#!/usr/bin/env python3
# check_ratio.py
# Check token selection ratio on a small sample without full scoring.
# Usage: python3 check_ratio.py [n_samples]

import os, sys, json, gc
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 20

SAVE_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(SAVE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "gsm8k_train.jsonl")

print(f"Checking token selection ratio on {N_SAMPLES} samples...")
print("=" * 55)

# Load sample
with open(RAW_PATH) as f:
    lines = [f.readline() for _ in range(N_SAMPLES)]
texts = [json.loads(l)["text"] for l in lines]


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_scored_jsonl(path, label):
    """Check ratio from already-scored jsonl."""
    if not os.path.exists(path):
        return
    total, selected, n = 0, 0, 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            total    += len(item["mask"])
            selected += sum(item["mask"])
            n += 1
    print(f"  {label}: {selected}/{total} = {selected/total*100:.1f}%  ({n} docs)")


# ============================================================
# 1. Check existing scored files first
# ============================================================
print("\n[Existing scored files]")
for label, fname in [
    ("CT",     "gsm8k_scored_ct.jsonl"),
    ("RHO-1",  "gsm8k_scored_rho1.jsonl"),
    ("Ours-B", "gsm8k_scored_symbolic.jsonl"),
    ("Ours-A", "gsm8k_scored_entropy_gap.jsonl"),
    ("Ours-C", "gsm8k_scored_math_confidence.jsonl"),
]:
    path = os.path.join(DATA_DIR, fname)
    check_scored_jsonl(path, label)


# ============================================================
# 2. Live scoring on N_SAMPLES for new methods
# ============================================================
print(f"\n[Live scoring on {N_SAMPLES} samples]")

# RHO-1 (Qwen2.5-0.5B)
rho1_path = os.path.join(DATA_DIR, "gsm8k_scored_rho1.jsonl")
if not os.path.exists(rho1_path):
    print("\n  Scoring RHO-1 sample...")
    from symbolic_slm.scoring.rho1_scorer import RHO1Scorer
    scorer = RHO1Scorer()
    total, selected = 0, 0
    for text in texts:
        r = scorer.score(text, select_ratio=0.60)
        total    += len(r["mask"])
        selected += sum(r["mask"])
    del scorer; clear()
    print(f"  RHO-1 (new): {selected}/{total} = {selected/total*100:.1f}%")

# Ours-C
c_path = os.path.join(DATA_DIR, "gsm8k_scored_math_confidence.jsonl")
if not os.path.exists(c_path):
    print("\n  Scoring Ours-C sample...")
    from symbolic_slm.scoring.math_confidence_scorer import MathConfidenceScorer
    scorer = MathConfidenceScorer()
    total, selected = 0, 0
    for text in texts:
        r = scorer.score(text, gap_ratio=0.40, abs_ratio=0.50, context_window=2)
        total    += len(r["mask"])
        selected += sum(r["mask"])
    del scorer; clear()
    print(f"  Ours-C:      {selected}/{total} = {selected/total*100:.1f}%")

print("\n" + "=" * 55)
print("Done.")
