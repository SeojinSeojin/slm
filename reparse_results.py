#!/usr/bin/env python3
# reparse_results.py
# Re-extract answers from saved per_problem JSONs without re-running inference.
# Useful when normalize_answer or extract functions are updated.
#
# Usage: python3 reparse_results.py

import os
import json
import re
import re

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DETAIL_DIR  = os.path.join(BASE_DIR, "results", "per_problem")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "eval_results_reparsed.json")

EXPERIMENTS = ["Qwen2.5-base", "CT", "RHO-1", "Ours-B", "Ours-A"]

# ============================================================
# Import extractors and normalizer from evaluate.py
# ============================================================
import sys
sys.path.insert(0, BASE_DIR)
from symbolic_slm.eval.evaluate import (
    extract_answer_gsm8k,
    extract_answer_math,
    extract_answer_svamp,
    normalize_answer,
)


# ============================================================
# Re-extract per benchmark
# ============================================================

def reparse_gsm8k(problems):
    correct, total = 0, 0
    for p in problems:
        pred = extract_answer_gsm8k(p["generated"])
        gold = p["gold"]
        is_correct = normalize_answer(pred) == normalize_answer(gold)
        p["pred"]    = pred
        p["correct"] = is_correct
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else 0.0
    print(f"    GSM8K:  {correct}/{total} = {acc:.1f}%")
    return acc, correct, total


def reparse_math(problems):
    # Reload dataset to fix truncated gold values
    try:
        from datasets import load_dataset
        dataset = load_dataset("lighteval/MATH-Hard", split="test")
        solutions = [item.get("solution", "") for item in dataset]
        has_dataset = len(solutions) >= len(problems)
        print(f"    Re-extracting gold from dataset ({len(solutions)} solutions)")
    except Exception as e:
        print(f"    Could not reload dataset: {e}")
        has_dataset = False
        solutions = []

    correct, total = 0, 0
    for i, p in enumerate(problems):
        generated = p["generated"]

        # Re-extract gold from original solution if available
        if has_dataset and i < len(solutions):
            gold = extract_answer_math(solutions[i])
            if gold:
                p["gold"] = gold

        pred = extract_answer_math(generated)
        if not pred:
            m = re.search(r"[Tt]he answer is\s*\$?([^\n$]+)\$?", generated)
            if m:
                pred = m.group(1).strip().rstrip(".")

        gold = p["gold"]
        is_correct = normalize_answer(pred) == normalize_answer(gold)
        p["pred"]    = pred
        p["correct"] = is_correct
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else 0.0
    print(f"    MATH:   {correct}/{total} = {acc:.1f}%")
    return acc, correct, total


def reparse_asdiv(problems):
    correct, total = 0, 0
    for p in problems:
        pred = extract_answer_gsm8k(p["generated"])
        gold = p["gold"]
        is_correct = normalize_answer(pred) == normalize_answer(gold)
        p["pred"]    = pred
        p["correct"] = is_correct
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else 0.0
    print(f"    ASDiv:  {correct}/{total} = {acc:.1f}%")
    return acc, correct, total


def reparse_svamp(problems):
    correct, total = 0, 0
    for p in problems:
        pred = extract_answer_svamp(p["generated"])
        gold = p["gold"]
        is_correct = normalize_answer(pred) == normalize_answer(gold)
        p["pred"]    = pred
        p["correct"] = is_correct
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else 0.0
    print(f"    SVAMP:  {correct}/{total} = {acc:.1f}%")
    return acc, correct, total


BENCH_REPARSE = {
    "gsm8k": reparse_gsm8k,
    "math":  reparse_math,
    "asdiv": reparse_asdiv,
    "svamp": reparse_svamp,
}

# ============================================================
# Run
# ============================================================

all_results = []

for exp in EXPERIMENTS:
    detail_path = os.path.join(DETAIL_DIR, f"{exp}.json")
    if not os.path.exists(detail_path):
        print(f"  ⚠️  없음: {exp}")
        continue

    with open(detail_path) as f:
        details = json.load(f)

    print(f"\n[{exp}]")
    result = {"experiment": exp, "model_path": exp}
    acc_keys = []

    for bench, reparse_fn in BENCH_REPARSE.items():
        if bench not in details or not details[bench]:
            continue
        acc, correct, total = reparse_fn(details[bench])
        result[f"{bench}_accuracy"] = acc
        result[f"{bench}_correct"]  = correct
        result[f"{bench}_total"]    = total
        acc_keys.append(f"{bench}_accuracy")

    if acc_keys:
        result["avg_accuracy"] = sum(result[k] for k in acc_keys) / len(acc_keys)

    # Save updated per_problem
    with open(detail_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    all_results.append(result)

# Save summary
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"  Results saved: {RESULTS_PATH}")
print(f"{'='*60}")
print(f"\n  {'Model':<20} {'GSM8K':>7} {'MATH':>7} {'ASDiv':>7} {'SVAMP':>7} {'Avg':>7}")
print(f"  {'-'*57}")
for r in all_results:
    g = r.get("gsm8k_accuracy", 0)
    m = r.get("math_accuracy",  0)
    a = r.get("asdiv_accuracy", 0)
    s = r.get("svamp_accuracy", 0)
    avg = r.get("avg_accuracy", 0)
    print(f"  {r['experiment']:<20} {g:>6.1f}% {m:>6.1f}% {a:>6.1f}% {s:>6.1f}% {avg:>6.1f}%")
