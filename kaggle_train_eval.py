#!/usr/bin/env python3
# kaggle_train_eval.py
# Training + Evaluation only (scoring already done on GTX)
# Upload scored jsonl files as Kaggle dataset before running

import os, sys, json, gc, shutil, subprocess
import torch

# ============================================================
# Install dependencies
# ============================================================
subprocess.run([
    "pip", "install", "-q",
    "transformers", "datasets", "tqdm", "accelerate"
], check=True)

# ============================================================
# Paths
# ============================================================
BASE_DIR  = "/kaggle/working/slm_vcl"
DATA_DIR  = os.path.join(BASE_DIR, "results", "data")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
SAVE_DIR  = os.path.join(BASE_DIR, "results")

# Scored JSONL files uploaded as Kaggle dataset
# Change this path to match your dataset name
DATASET_DIR = "/kaggle/input/slm-scored"

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Copy scored files from dataset to working dir
for fname in [
    "gsm8k_scored_rho1.jsonl",
    "gsm8k_scored_symbolic.jsonl",
    "gsm8k_scored_entropy_gap.jsonl",
    "gsm8k_scored_math_confidence.jsonl",
]:
    src = os.path.join(DATASET_DIR, fname)
    dst = os.path.join(DATA_DIR, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Copied: {fname}")
    elif os.path.exists(dst):
        print(f"Already exists: {fname}")
    else:
        print(f"NOT FOUND: {src}")

# Clone or copy slm_vcl code
# Option 1: upload as dataset
SLM_CODE_DIR = "/kaggle/input/slm-code/slm_vcl"
if os.path.exists(SLM_CODE_DIR):
    if not os.path.exists(BASE_DIR):
        shutil.copytree(SLM_CODE_DIR, BASE_DIR)
        print("Code copied from dataset")
else:
    print("WARNING: Code dataset not found. Upload slm_vcl as dataset.")

sys.path.insert(0, BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "num_samples":       7473,
    "max_length":        512,
    "base_model":        "Qwen/Qwen2.5-1.5B",

    "max_steps":         8000,
    "batch_size":        1,
    "grad_accum_steps":  32,
    "learning_rate":     5e-5,
    "warmup_steps":      25,
    "log_interval":      10,

    "gsm8k_samples":     200,
    "math_samples":      200,
    "asdiv_samples":     200,
    "svamp_samples":     200,

    "run_rho1":   True,
    "run_ours_b": True,
    "run_ours_a": True,
    "run_ours_c": True,
}

results_path = os.path.join(SAVE_DIR, "eval_results.json")


# ============================================================
# Helpers
# ============================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  Free VRAM: {free:.1f} GB")


def already_evaluated(name):
    if not os.path.exists(results_path):
        return False
    with open(results_path) as f:
        results = json.load(f)
    return any(r["experiment"] == name for r in results)


def need_scoring(path):
    return not os.path.exists(path)


def train_eval_delete(exp_key, scored_path, exp_name, ct_mode=False):
    if already_evaluated(exp_name):
        print(f"  Already evaluated (skip): {exp_name}")
        return

    if need_scoring(scored_path):
        print(f"  ERROR: scored file not found: {scored_path}")
        return

    from symbolic_slm.train.slm_trainer import train_slm
    from symbolic_slm.eval.evaluate import run_evaluation

    model_path = os.path.join(MODEL_DIR, f"{exp_key}_final")

    clear_memory()
    print(f"\n{'='*50}")
    print(f"  Training: {exp_name}")
    print(f"{'='*50}")

    train_slm(
        base_model_name  = CONFIG["base_model"],
        scored_data_path = scored_path,
        output_dir       = model_path,
        max_steps        = CONFIG["max_steps"],
        batch_size       = CONFIG["batch_size"],
        grad_accum_steps = CONFIG["grad_accum_steps"],
        learning_rate    = CONFIG["learning_rate"],
        warmup_steps     = CONFIG["warmup_steps"],
        max_length       = CONFIG["max_length"],
        log_interval     = CONFIG["log_interval"],
        ct_mode          = ct_mode,
        loss_save_path   = os.path.join(MODEL_DIR, f"{exp_key}_loss.json"),
    )

    clear_memory()
    run_evaluation(
        model_path, exp_name, results_path,
        gsm8k_samples = CONFIG["gsm8k_samples"],
        math_samples  = CONFIG["math_samples"],
        asdiv_samples = CONFIG["asdiv_samples"],
        svamp_samples = CONFIG["svamp_samples"],
    )

    clear_memory()
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"  Deleted model: {model_path}")


# ============================================================
# Run
# ============================================================
from symbolic_slm.eval.evaluate import run_evaluation

rho1_path = os.path.join(DATA_DIR, "gsm8k_scored_rho1.jsonl")
b_path    = os.path.join(DATA_DIR, "gsm8k_scored_symbolic.jsonl")
a_path    = os.path.join(DATA_DIR, "gsm8k_scored_entropy_gap.jsonl")
c_path    = os.path.join(DATA_DIR, "gsm8k_scored_math_confidence.jsonl")

if CONFIG["run_rho1"]:
    train_eval_delete("rho1", rho1_path, "RHO-1")

if CONFIG["run_ours_b"]:
    train_eval_delete("ours_b_symbolic", b_path, "Ours-B")

if CONFIG["run_ours_a"]:
    train_eval_delete("ours_a_entropy_gap", a_path, "Ours-A")

if CONFIG["run_ours_c"]:
    train_eval_delete("ours_c_math_confidence", c_path, "Ours-C")

# ============================================================
# Print results
# ============================================================
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    print(f"\n{'='*60}")
    print(f"  {'Model':<20} {'GSM8K':>7} {'MATH':>7} {'ASDiv':>7} {'SVAMP':>7}")
    print(f"  {'-'*52}")
    for r in results:
        g = r.get("gsm8k_accuracy", 0)
        m = r.get("math_accuracy",  0)
        a = r.get("asdiv_accuracy", 0)
        s = r.get("svamp_accuracy", 0)
        print(f"  {r['experiment']:<20} {g:>6.1f}% {m:>6.1f}% {a:>6.1f}% {s:>6.1f}%")
    print(f"{'='*60}")

# Copy results to output
import shutil
out = "/kaggle/working/results"
if os.path.exists(SAVE_DIR):
    shutil.copytree(SAVE_DIR, out, dirs_exist_ok=True)
    print(f"\nResults saved to {out}")
