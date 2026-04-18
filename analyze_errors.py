#!/usr/bin/env python3
# analyze_errors.py
# Per-problem error analysis across experiments
# Usage: python3 analyze_errors.py

import json
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DETAIL_DIR  = os.path.join(BASE_DIR, "results", "per_problem")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "error_analysis.txt")

EXPERIMENTS = ["Qwen2.5-base", "CT", "RHO-1", "Ours-B", "Ours-A"]
BENCHMARKS  = ["gsm8k", "math", "asdiv", "svamp"]

# ============================================================
# Load all per-problem results
# ============================================================

data = {}
for exp in EXPERIMENTS:
    path = os.path.join(DETAIL_DIR, f"{exp}.json")
    if not os.path.exists(path):
        print(f"  skipping: {path}")
        continue
    with open(path) as f:
        data[exp] = json.load(f)
    print(f"  loaded: {exp}")

available_exps = list(data.keys())
print(f"\nAvailable: {available_exps}\n")


# ============================================================
# Helpers
# ============================================================

def get_question(prob):
    """Different benchmarks use different field names."""
    return prob.get("question", prob.get("problem", prob.get("body", "")))


def analyze_benchmark(bench, n_show=5):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {bench.upper()}")
    lines.append("=" * 70)

    exps_with_bench = [e for e in available_exps if bench in data[e] and len(data[e][bench]) > 0]
    if not exps_with_bench:
        lines.append("  No data.\n")
        return "\n".join(lines)

    n_problems = len(data[exps_with_bench[0]][bench])
    correct = {exp: [p["correct"] for p in data[exp][bench]] for exp in exps_with_bench}

    # Accuracy
    lines.append("\n[Accuracy]")
    for exp in exps_with_bench:
        acc = sum(correct[exp]) / len(correct[exp]) * 100
        lines.append(f"  {exp:<20}: {acc:.1f}%  ({sum(correct[exp])}/{len(correct[exp])})")

    # Case 1: base correct, ALL others wrong
    if "Qwen2.5-base" in exps_with_bench:
        others = [e for e in exps_with_bench if e != "Qwen2.5-base"]
        cases = [
            i for i in range(n_problems)
            if correct["Qwen2.5-base"][i]
            and all(not correct[e][i] for e in others)
        ]
        lines.append(f"\n[Base correct, all others wrong]: {len(cases)}")
        for i in cases[:n_show]:
            prob = data["Qwen2.5-base"][bench][i]
            lines.append(f"\n  Problem #{i}")
            lines.append(f"  Q: {get_question(prob)[:120]}")
            lines.append(f"  Gold: {prob['gold']}")
            lines.append(f"  Base pred: {prob['pred']}")
            for e in others:
                p = data[e][bench][i]
                lines.append(f"  {e} pred: {p['pred']}")
            lines.append(f"  Base output: {prob['generated'][:200]}")

    # Case 2: RHO-1 wrong, Ours-B correct (key comparison)
    if "RHO-1" in exps_with_bench and "Ours-B" in exps_with_bench:
        cases = [
            i for i in range(n_problems)
            if not correct["RHO-1"][i] and correct["Ours-B"][i]
        ]
        lines.append(f"\n[RHO-1 wrong, Ours-B correct]: {len(cases)}  <- key comparison")
        for i in cases[:n_show]:
            prob_b = data["Ours-B"][bench][i]
            prob_r = data["RHO-1"][bench][i]
            lines.append(f"\n  Problem #{i}")
            lines.append(f"  Q: {get_question(prob_b)[:120]}")
            lines.append(f"  Gold: {prob_b['gold']}")
            lines.append(f"  RHO-1 pred:  {prob_r['pred']}")
            lines.append(f"  Ours-B pred: {prob_b['pred']}")
            lines.append(f"  RHO-1 output:  {prob_r['generated'][:150]}")
            lines.append(f"  Ours-B output: {prob_b['generated'][:150]}")

    # Case 3: base wrong, Ours-B or Ours-A correct
    for ours in ["Ours-B", "Ours-A"]:
        if ours not in exps_with_bench or "Qwen2.5-base" not in exps_with_bench:
            continue
        cases = [
            i for i in range(n_problems)
            if not correct["Qwen2.5-base"][i] and correct[ours][i]
        ]
        lines.append(f"\n[Base wrong, {ours} correct]: {len(cases)}")
        for i in cases[:n_show]:
            prob = data[ours][bench][i]
            base_prob = data["Qwen2.5-base"][bench][i]
            lines.append(f"\n  Problem #{i}")
            lines.append(f"  Q: {get_question(prob)[:120]}")
            lines.append(f"  Gold: {prob['gold']}")
            lines.append(f"  Base pred: {base_prob['pred']}")
            lines.append(f"  {ours} pred: {prob['pred']}")
            lines.append(f"  {ours} output: {prob['generated'][:200]}")

    # Case 4: all wrong
    all_wrong = [i for i in range(n_problems) if all(not correct[e][i] for e in exps_with_bench)]
    lines.append(f"\n[All wrong (hardest)]: {len(all_wrong)}")

    # Case 5: all correct
    all_correct_idx = [i for i in range(n_problems) if all(correct[e][i] for e in exps_with_bench)]
    lines.append(f"[All correct (easiest)]: {len(all_correct_idx)}")

    # Error type: no answer vs wrong answer
    lines.append(f"\n[Error type breakdown]")
    lines.append(f"  {'Model':<20} {'No answer':>10} {'Wrong ans':>10} {'Total wrong':>12}")
    lines.append(f"  {'-'*55}")
    for exp in exps_with_bench:
        problems = data[exp][bench]
        wrong    = [p for p in problems if not p["correct"]]
        no_ans   = sum(1 for p in wrong if not p["pred"])
        wrong_ans = len(wrong) - no_ans
        lines.append(f"  {exp:<20} {no_ans:>10} {wrong_ans:>10} {len(wrong):>12}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Run
# ============================================================

output = ["=" * 70, "  Per-Problem Error Analysis", "=" * 70, ""]

for bench in BENCHMARKS:
    output.append(analyze_benchmark(bench, n_show=5))

result_text = "\n".join(output)
print(result_text)

with open(OUTPUT_PATH, "w") as f:
    f.write(result_text)
print(f"\nSaved: {OUTPUT_PATH}")
