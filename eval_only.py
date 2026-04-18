#!/usr/bin/env python3
# eval_only.py
# Run evaluation on already-saved models (no training needed)
# Usage: python3 eval_only.py

import os, sys, json, gc, torch, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from symbolic_slm.eval.evaluate import run_evaluation

# ============================================================
# Config — adjust paths and samples as needed
# ============================================================
MODELS_DIR   = os.path.join(BASE_DIR, "results", "models")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "eval_results_full.json")

BASE_MODEL   = "Qwen/Qwen2.5-3B"   # base model name

# ============================================================
# ★ 여기서 켜고 끄기 ★
# ============================================================
RUN_BASE   = False
RUN_CT     = False
RUN_RHO1   = True
RUN_OURS_B = True
RUN_OURS_A = True
# ============================================================

EXPERIMENTS  = [
    ("ct_step250",                 "CT",     RUN_CT),
    ("rho1_step250",               "RHO-1",  RUN_RHO1),
    ("ours_b_symbolic_step250",    "Ours-B", RUN_OURS_B),
    ("ours_a_entropy_gap_step250", "Ours-A", RUN_OURS_A),
]

GSM8K_SAMPLES  = 200
MATH_SAMPLES   = 200
ASDIV_SAMPLES  = 200
SVAMP_SAMPLES  = 200

# ============================================================
# ★ 벤치마크 켜고 끄기 ★
# ============================================================
RUN_GSM8K  = True
RUN_MATH   = True
RUN_ASDIV  = True
RUN_SVAMP  = True
# ============================================================

# ============================================================
# Check available models
# ============================================================
print("=" * 60)
print("  Eval-only run on saved models")
print("=" * 60)
print(f"\nModels dir: {MODELS_DIR}")
for folder, name in EXPERIMENTS:
    path = os.path.join(MODELS_DIR, folder)
    has_model = os.path.exists(path) and any(
        "safetensors" in f for f in os.listdir(path)
    )
    print(f"  {'✅' if has_model else '❌'} {name} ({folder})")

print()

# ============================================================
# Evaluate base model
# ============================================================
if RUN_BASE:
    print("Evaluating base model...")
    run_evaluation(
        BASE_MODEL, "Qwen2.5-base", RESULTS_PATH,
        gsm8k_samples=GSM8K_SAMPLES  if RUN_GSM8K  else 0,
        math_samples=MATH_SAMPLES    if RUN_MATH   else 0,
        asdiv_samples=ASDIV_SAMPLES  if RUN_ASDIV  else 0,
        svamp_samples=SVAMP_SAMPLES  if RUN_SVAMP  else 0,
        run_asdiv=RUN_ASDIV,
        run_svamp=RUN_SVAMP,
    )
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Skipping base model (RUN_BASE=False)")

# ============================================================
# Evaluate saved models
# ============================================================
for folder, name, run_flag in EXPERIMENTS:
    if not run_flag:
        print(f"Skipping {name} (flag=False)")
        continue

    model_path = os.path.join(MODELS_DIR, folder)

    if not os.path.exists(model_path):
        print(f"\n❌ {name}: 모델 없음 ({model_path})")
        continue
    if not any("safetensors" in f for f in os.listdir(model_path)):
        print(f"\n❌ {name}: safetensors 없음")
        continue

    print(f"\nEvaluating: {name}")
    try:
        run_evaluation(
            model_path, name, RESULTS_PATH,
            gsm8k_samples=GSM8K_SAMPLES  if RUN_GSM8K  else 0,
            math_samples=MATH_SAMPLES    if RUN_MATH   else 0,
            asdiv_samples=ASDIV_SAMPLES  if RUN_ASDIV  else 0,
            svamp_samples=SVAMP_SAMPLES  if RUN_SVAMP  else 0,
            run_asdiv=RUN_ASDIV,
            run_svamp=RUN_SVAMP,
        )
    except Exception as e:
        print(f"❌ {name} evaluation failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()

print(f"\n🎉 Done! Results: {RESULTS_PATH}")
print(f"   Per-problem: {os.path.dirname(RESULTS_PATH)}/per_problem/")
