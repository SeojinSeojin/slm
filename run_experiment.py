# run_experiment.py (VCL 버전)
# RTX 2080 Ti 11GB
# 실행: python3 run_experiment.py

import os, sys, json, gc, torch, shutil

# 이 파일 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)   # symbolic_slm 패키지 찾는 경로

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SAVE_DIR  = os.path.join(BASE_DIR, "results")
DATA_DIR  = os.path.join(SAVE_DIR, "data")
MODEL_DIR = os.path.join(SAVE_DIR, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# CONFIG
# RTX 2080 Ti 11GB:
#   모델 weights:  ~3GB
#   optimizer:     ~6GB  ← AdamW가 weights의 2배
#   activations:   ~1.5GB
#   합계:          ~10.5GB → batch=1만 안전
#   grad_accum=32 → eff_batch=32 유지
# ============================================================
CONFIG = {
    "num_samples":       7473,
    "max_length":        512,
    "base_model":        "Qwen/Qwen2.5-1.5B",

    "max_steps":         8000,
    "batch_size":        1,       # 2080 Ti 11GB — OOM 방지
    "grad_accum_steps":  32,      # eff_batch=32 유지
    "learning_rate":     5e-5,
    "warmup_steps":      25,      # optimizer_steps 250의 10%

    "log_interval":      10,
    "select_ratio_rho1": 0.77,

    "gsm8k_samples":     200,
    "math_samples":      200,
    "asdiv_samples":     200,
    "svamp_samples":     200,

    "run_ct":     False,
    "run_rho1":   True,
    "run_ours_b": True,
    "run_ours_a": True,
    "run_ours_c": False,
    "run_base":   False,
}

print("=" * 60)
print("  Symbolic-Aware Token Selection (VCL RTX 2080 Ti)")
print("=" * 60)
print(f"  base_model: {CONFIG['base_model']}")
print(f"  max_length={CONFIG['max_length']}"
      f" | optimizer_steps={CONFIG['max_steps']//CONFIG['grad_accum_steps']}")
print(f"  batch={CONFIG['batch_size']}"
      f" | grad_accum={CONFIG['grad_accum_steps']}"
      f" | eff_batch={CONFIG['batch_size']*CONFIG['grad_accum_steps']}")
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


def need_scoring(path):
    if os.path.exists(path):
        print(f"  스코어링 있음 (스킵): {os.path.basename(path)}")
        return False
    return True


def already_evaluated(name):
    if not os.path.exists(results_path):
        return False
    with open(results_path) as f:
        existing = json.load(f)
    return any(r["experiment"] == name for r in existing)


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


def train_eval_delete(experiment_name, scored_path, display_name, ct_mode=False):
    """학습 → 평가 → 모델 삭제. 실패해도 다음 실험 계속 진행."""
    from symbolic_slm.train.slm_trainer import SLMDataset
    from symbolic_slm.eval.evaluate import run_evaluation

    model_path = os.path.join(MODEL_DIR, f"{experiment_name}_final")

    if already_evaluated(display_name):
        print(f"  평가됨 (스킵): {display_name}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        return

    model_ok = (
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
            trainer.train(dataset, experiment_name)
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
            gsm8k_samples=CONFIG["gsm8k_samples"],
            math_samples=CONFIG["math_samples"],
            asdiv_samples=CONFIG["asdiv_samples"],
            svamp_samples=CONFIG["svamp_samples"],
        )
    except Exception as e:
        print(f"\n❌ [{display_name}] 평가 실패: {e}")

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"  모델 삭제: {display_name}")
    clear_memory()


# ============================================================
# Step 1: 데이터
# ============================================================

raw_path = os.path.join(DATA_DIR, "gsm8k_train.jsonl")

if os.path.exists(raw_path):
    with open(raw_path) as f:
        n = sum(1 for _ in f)
    if n >= 7000:
        print(f"\n[Step 1] GSM8K train 있음: {n:,}개 (스킵)")
    else:
        print(f"\n[Step 1] 손상 ({n}개) → 재다운로드")
        os.remove(raw_path)

if not os.path.exists(raw_path):
    print("\n[Step 1] GSM8K train 다운로드...")
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    with open(raw_path, "w") as f:
        for item in dataset:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"  ✅ {len(dataset):,}개 저장")


# ============================================================
# Step 2: 스코어링
# ============================================================

print("\n" + "=" * 60)
print("[스코어링]")
print("=" * 60)

ct_scored_path = os.path.join(DATA_DIR, "gsm8k_scored_ct.jsonl")
if need_scoring(ct_scored_path):
    print("  CT tokenize...")
    from transformers import AutoTokenizer
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(raw_path) as f:
        lines = f.readlines()
    with open(ct_scored_path, "w") as out:
        for line in tqdm(lines, desc="CT"):
            item = json.loads(line.strip())
            enc  = tokenizer(item["text"], max_length=CONFIG["max_length"], truncation=True)
            ids  = enc["input_ids"]
            if len(ids) < 8:
                continue
            out.write(json.dumps({
                "text": item["text"], "token_ids": ids, "mask": [True] * len(ids),
            }) + "\n")
    del tokenizer
    print("  ✅ CT 완료")

b_scored_path = os.path.join(DATA_DIR, "gsm8k_scored_symbolic.jsonl")
if CONFIG["run_ours_b"] and need_scoring(b_scored_path):
    clear_memory()
    from symbolic_slm.scoring.symbolic_scorer import preprocess_dataset_symbolic
    preprocess_dataset_symbolic(
        data_path=raw_path, output_path=b_scored_path,
        max_length=CONFIG["max_length"],
    )
    clear_memory()

rho1_scored_path = os.path.join(DATA_DIR, "gsm8k_scored_rho1.jsonl")
if CONFIG["run_rho1"] and need_scoring(rho1_scored_path):
    clear_memory()
    from symbolic_slm.scoring.rho1_scorer import preprocess_dataset_rho1
    preprocess_dataset_rho1(
        data_path=raw_path, output_path=rho1_scored_path,
        select_ratio=CONFIG["select_ratio_rho1"],
        max_length=CONFIG["max_length"],
    )
    clear_memory()

a_scored_path = os.path.join(DATA_DIR, "gsm8k_scored_entropy_gap.jsonl")
if CONFIG["run_ours_a"] and need_scoring(a_scored_path):
    clear_memory()
    from symbolic_slm.scoring.entropy_gap_scorer import preprocess_dataset_entropy_gap
    preprocess_dataset_entropy_gap(
        data_path=raw_path, output_path=a_scored_path,
        select_ratio=0.25, max_length=CONFIG["max_length"],
    )
    clear_memory()

c_scored_path = os.path.join(DATA_DIR, "gsm8k_scored_math_confidence.jsonl")
if CONFIG["run_ours_c"] and need_scoring(c_scored_path):
    clear_memory()
    from symbolic_slm.scoring.math_confidence_scorer import preprocess_dataset_math_confidence
    preprocess_dataset_math_confidence(
        data_path=raw_path, output_path=c_scored_path,
        gap_ratio=0.40, abs_ratio=0.40, context_window=2,
        max_length=CONFIG["max_length"],
    )
    clear_memory()

print("\n✅ 스코어링 완료. 학습 시작.")


# ============================================================
# Step 3+4: 학습 → 평가 → 삭제
# ============================================================

from symbolic_slm.eval.evaluate import run_evaluation

# Base 평가
if CONFIG["run_base"] and not already_evaluated("Qwen2.5-base"):
    clear_memory()
    run_evaluation(
        CONFIG["base_model"], "Qwen2.5-base", results_path,
        gsm8k_samples=CONFIG["gsm8k_samples"],
        math_samples=CONFIG["math_samples"],
        asdiv_samples=CONFIG["asdiv_samples"],
        svamp_samples=CONFIG["svamp_samples"],
    )

# CT
if CONFIG["run_ct"]:
    ct_model = os.path.join(MODEL_DIR, "ct_final")
    if already_evaluated("CT"):
        print("  평가됨 (스킵): CT")
        if os.path.exists(ct_model):
            shutil.rmtree(ct_model)
    else:
        train_eval_delete("ct", ct_scored_path, "CT", ct_mode=True)

# RHO-1
if CONFIG["run_rho1"]:
    train_eval_delete("rho1", rho1_scored_path, "RHO-1")

# Ours-B
if CONFIG["run_ours_b"]:
    train_eval_delete("ours_b_symbolic", b_scored_path, "Ours-B")

# Ours-A
if CONFIG["run_ours_a"]:
    train_eval_delete("ours_a_entropy_gap", a_scored_path, "Ours-A")

# Ours-C
if CONFIG["run_ours_c"]:
    train_eval_delete("ours_c_math_confidence", c_scored_path, "Ours-C")

print(f"\n실험 완료!")
print(f"결과: {results_path}")
