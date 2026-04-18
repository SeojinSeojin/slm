#!/usr/bin/env python3
# export_to_csv.py
# Convert loss JSON files and eval results to CSV for Google Sheets
# Usage: python3 export_to_csv.py

import os
import json
import csv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv_export")
os.makedirs(CSV_DIR, exist_ok=True)

EXPERIMENTS = {
    "ct":                 "CT",
    "rho1":               "RHO-1",
    "ours_b_symbolic":    "Ours-B",
    "ours_a_entropy_gap": "Ours-A",
}

# ============================================================
# 1. Loss curves → one CSV (step, CT, RHO-1, Ours-B, Ours-A)
# ============================================================

def export_loss():
    all_loss = {}  # exp_name → {step: loss}
    all_steps = set()

    for key, name in EXPERIMENTS.items():
        path = os.path.join(MODELS_DIR, f"{key}_loss.json")
        if not os.path.exists(path):
            print(f"  Loss not found: {key}")
            continue
        with open(path) as f:
            history = json.load(f)
        all_loss[name] = {h["step"]: h["loss"] for h in history}
        all_steps.update(all_loss[name].keys())
        print(f"  Loaded: {key}_loss.json ({len(history)} steps)")

    if not all_loss:
        print("  No loss files found.")
        return

    steps = sorted(all_steps)
    names = list(all_loss.keys())

    out_path = os.path.join(CSV_DIR, "loss_curves.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step"] + names)
        for step in steps:
            row = [step] + [all_loss[name].get(step, "") for name in names]
            writer.writerow(row)

    print(f"  Saved: loss_curves.csv ({len(steps)} steps x {len(names)} experiments)")


# ============================================================
# 2. Eval results → CSV
# ============================================================

def export_eval():
    # Find best available results file
    for fname in ["eval_results_reparsed.json", "eval_results_full.json", "eval_results.json"]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            print(f"  Loading: {fname}")
            with open(path) as f:
                results = json.load(f)
            break
    else:
        print("  No eval results found.")
        return

    benchmarks = ["gsm8k", "math", "asdiv", "svamp"]
    bench_labels = {"gsm8k": "GSM8K", "math": "MATH", "asdiv": "ASDiv", "svamp": "SVAMP"}

    # Filter benchmarks that have data
    active = [b for b in benchmarks if any(r.get(f"{b}_accuracy", 0) > 0 for r in results)]

    out_path = os.path.join(CSV_DIR, "eval_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"] + [bench_labels[b] for b in active] + ["Avg"]
        writer.writerow(header)
        for r in results:
            vals = [r.get(f"{b}_accuracy", 0.0) for b in active]
            avg  = sum(vals) / len(vals) if vals else 0.0
            row  = [r["experiment"]] + [f"{v:.1f}" for v in vals] + [f"{avg:.1f}"]
            writer.writerow(row)

    print(f"  Saved: eval_results.csv ({len(results)} models x {len(active)} benchmarks)")


# ============================================================
# Run
# ============================================================

print("Exporting to CSV...\n")

print("[1/2] Loss curves")
export_loss()

print("\n[2/2] Eval results")
export_eval()

print(f"\nDone! Files saved to: {CSV_DIR}")
print("  loss_curves.csv  → upload to Google Sheets for loss graph")
print("  eval_results.csv → upload to Google Sheets for bar chart")
