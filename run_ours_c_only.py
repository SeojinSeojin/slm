#!/usr/bin/env python3
import os, sys, json, gc, shutil, torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SAVE_DIR  = os.path.join(BASE_DIR, "results")
DATA_DIR  = os.path.join(SAVE_DIR, "data")
MODEL_DIR = os.path.join(SAVE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG = {
    "base_model":        "Qwen/Qwen2.5-1.5B",
    "max_steps":         8000,
    "batch_size":        1,
    "grad_accum_steps":  32,
    "learning_rate":     5e-5,
    "warmup_steps":      25,
    "max_length":        512,
    "log_interval":      10,
    "gsm8k_samples":     200,
    "math_samples":      200,
    "asdiv_samples":     200,
    "svamp_samples":     200,
}

results_path = os.path.join(SAVE_DIR, "eval_results.json")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def already_evaluated(name):
    if not os.path.exists(results_path):
        return False
    with open(results_path) as f:
        return any(r["experiment"] == name for r in json.load(f))

from symbolic_slm.train.slm_trainer import SLMTrainer, SLMDataset
from symbolic_slm.eval.evaluate import run_evaluation

scored_path = os.path.join(DATA_DIR, "gsm8k_scored_math_confidence.jsonl")
exp_key     = "ours_c_math_confidence"
exp_name    = "Ours-C"
model_path  = os.path.join(MODEL_DIR, f"{exp_key}_final")

if already_evaluated(exp_name):
    print(f"Already evaluated: {exp_name}")
else:
    print(f"Training: {exp_name}")
    clear_memory()
    dataset = SLMDataset(scored_path, CONFIG["max_length"])
    trainer = SLMTrainer(
        model_name       = CONFIG["base_model"],
        save_dir         = MODEL_DIR,
        learning_rate    = CONFIG["learning_rate"],
        batch_size       = CONFIG["batch_size"],
        grad_accum_steps = CONFIG["grad_accum_steps"],
        max_steps        = CONFIG["max_steps"],
        warmup_steps     = CONFIG["warmup_steps"],
        max_length       = CONFIG["max_length"],
        log_interval     = CONFIG["log_interval"],
    )
    trainer.train(dataset, exp_key)
    del trainer, dataset
    clear_memory()

    run_evaluation(
        model_path, exp_name, results_path,
        gsm8k_samples=CONFIG["gsm8k_samples"],
        math_samples=CONFIG["math_samples"],
        asdiv_samples=CONFIG["asdiv_samples"],
        svamp_samples=CONFIG["svamp_samples"],
    )
    clear_memory()
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    print(f"Done: {exp_name}")
