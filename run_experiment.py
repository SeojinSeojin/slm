#!/usr/bin/env python3
# run_experiment.py
# RTX 2080 Ti 11GB
# 실행: python3 run_experiment.py

import os, sys, json, gc, torch, shutil

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SAVE_DIR  = os.path.join(BASE_DIR, "results")
DATA_DIR  = os.path.join(SAVE_DIR, "data")
MODEL_DIR = os.path.join(SAVE_DIR, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "max_length":        384,
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

    "run_base": False,
    "run_ct":   False,
}

# scored 파일 → (experiment_name, display_name) 매핑
# results/data/ 에 있는 jsonl을 모두 실험 대상으로 등록
EXPERIMENTS = [
    # (scored_path,                                               exp_name,                        display_name,             ct_mode)
    # ("gsm8k_scored_ct.jsonl",                                    "ct",                            "CT",                     True),
    # ("gsm8k_rho1_base.jsonl",                                    "rho1_base",                     "RHO-1 base",             False),
    ("gsm8k_rho1_window_w1.jsonl",                               "rho1_window_w1",                "RHO-1 window_w1",        False),
    ("gsm8k_rho1_window_w3.jsonl",                               "rho1_window_w3",                "RHO-1 window_w3",        False),
    # ("gsm8k_rho1_span_w1.jsonl",                                 "rho1_span_w1",                  "RHO-1 span_w1",          False),
    # ("gsm8k_symbolic_window_w1.jsonl",                           "symbolic_window_w1",            "Ours-B window_w1",       False),
    # ("gsm8k_symbolic_window_w3.jsonl",                           "symbolic_window_w3",            "Ours-B window_w3",       False),
    # ("gsm8k_symbolic_span_w1.jsonl",                             "symbolic_span_w1",              "Ours-B span_w1",         False),
    # ("gsm8k_entropy_gap_window_w1.jsonl",                        "entropy_gap_window_w1",         "Ours-A window_w1",       False),
    # ("gsm8k_entropy_gap_window_w3.jsonl",                        "entropy_gap_window_w3",         "Ours-A window_w3",       False),
    # ("gsm8k_entropy_gap_span_w1.jsonl",                          "entropy_gap_span_w1",           "Ours-A span_w1",         False),
]

print("=" * 60)
print("  Symbolic-Aware Token Selection (VCL RTX 2080 Ti)")
print("=" * 60)
print(f"  base_model:      {CONFIG['base_model']}")
print(f"  optimizer_steps: {CONFIG['max_steps'] // CONFIG['grad_accum_steps']}")
print(f"  batch={CONFIG['batch_size']}  grad_accum={CONFIG['grad_accum_steps']}"
      f"  eff_batch={CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
print(f"  experiments:     {len(EXPERIMENTS)}")
print("=" * 60)

results_path = os.path.join(SAVE_DIR, "eval_results.json")


# ============================================================
# Helpers
# ============================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  VRAM 여유: {free:.1f} GB")


def already_evaluated(display_name):
    if not os.path.exists(results_path):
        return False
    with open(results_path) as f:
        existing = json.load(f)
    return any(r["experiment"] == display_name for r in existing)


def make_trainer():
    from symbolic_slm.train.slm_trainer import SLMTrainer
    return SLMTrainer(
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


def train_eval_delete(exp_name, scored_path, display_name, ct_mode=False):
    """학습 → 평가 → 모델 삭제. 실패해도 다음 실험 계속."""
    from symbolic_slm.train.slm_trainer import SLMDataset
    from symbolic_slm.eval.evaluate import run_evaluation

    if already_evaluated(display_name):
        print(f"  평가됨 (스킵): {display_name}")
        model_path = os.path.join(MODEL_DIR, f"{exp_name}_final")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        return

    if not os.path.exists(scored_path):
        print(f"  ⚠️  scored 파일 없음 (스킵): {os.path.basename(scored_path)}")
        return

    model_path = os.path.join(MODEL_DIR, f"{exp_name}_final")
    model_ok   = (
        os.path.exists(model_path) and
        any("safetensors" in f for f in os.listdir(model_path))
    )

    if not model_ok:
        print(f"\n{'='*60}")
        print(f"[{display_name}] 학습")
        print(f"{'='*60}")
        clear_memory()
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        try:
            dataset = SLMDataset(scored_path, CONFIG["max_length"], ct_mode=ct_mode)
            trainer = make_trainer()
            trainer.train(dataset, exp_name)
            del trainer, dataset
            clear_memory()
        except Exception as e:
            print(f"\n❌ [{display_name}] 학습 실패: {e}")
            print(f"   → 다음 실험으로 넘어갑니다.")
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            clear_memory()
            return
    else:
        print(f"\n  모델 있음 (학습 스킵): {display_name}")

    try:
        clear_memory()
        run_evaluation(
            model_path, display_name, results_path,
            gsm8k_samples = CONFIG["gsm8k_samples"],
            math_samples  = CONFIG["math_samples"],
            asdiv_samples = CONFIG["asdiv_samples"],
            svamp_samples = CONFIG["svamp_samples"],
        )
    except Exception as e:
        print(f"\n❌ [{display_name}] 평가 실패: {e}")

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"  모델 삭제: {display_name}")
    clear_memory()


# ============================================================
# Step 1: Base 평가 (선택)
# ============================================================
if CONFIG["run_base"] and not already_evaluated("Qwen2.5-base"):
    from symbolic_slm.eval.evaluate import run_evaluation
    clear_memory()
    run_evaluation(
        CONFIG["base_model"], "Qwen2.5-base", results_path,
        gsm8k_samples = CONFIG["gsm8k_samples"],
        math_samples  = CONFIG["math_samples"],
        asdiv_samples = CONFIG["asdiv_samples"],
        svamp_samples = CONFIG["svamp_samples"],
    )


# ============================================================
# Step 2: 모든 실험 순차 실행
# ============================================================
print(f"\n총 {len(EXPERIMENTS)}개 실험 시작\n")

for filename, exp_name, display_name, ct_mode in EXPERIMENTS:
    # CT 실험은 CONFIG로 on/off
    if filename == "gsm8k_scored_ct.jsonl" and not CONFIG["run_ct"]:
        print(f"  스킵 (run_ct=False): {display_name}")
        continue

    scored_path = os.path.join(DATA_DIR, filename)
    train_eval_delete(exp_name, scored_path, display_name, ct_mode=ct_mode)


# ============================================================
# 결과 요약
# ============================================================
print(f"\n{'='*60}")
print("실험 완료! 결과 요약:")
print(f"{'='*60}")

if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    # display_name 기준으로 정렬해서 출력
    name_order = {display: i for i, (_, _, display, _) in enumerate(EXPERIMENTS)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["experiment"], 999))
    for r in results_sorted:
        gsm = r.get("gsm8k", {}).get("accuracy", "N/A")
        print(f"  {r['experiment']:30s}  GSM8K: {gsm}")

print(f"\n결과 파일: {results_path}")