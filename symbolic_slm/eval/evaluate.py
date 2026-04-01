# eval/evaluate.py
# GSM8K + MATH 평가
# 수정 이력:
#   - import os 누락 추가
#   - torch_dtype → dtype
#   - MATH도 few-shot CoT로 변경 (논문과 동일 조건)
#   - temperature=0.0 + do_sample=False 중복 warning 제거

import re
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# Few-shot 프롬프트
# ============================================================

GSM8K_FEWSHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: {question}
Answer:"""

# 논문과 동일하게 MATH도 few-shot CoT
MATH_FEWSHOT = """Problem: What is the value of $x$ if $2x + 3 = 11$?
Solution: Subtracting 3 from both sides: $2x = 8$. Dividing by 2: $x = 4$. The answer is $\\boxed{4}$.

Problem: A rectangle has length 8 and width 5. What is its area?
Solution: Area = length times width = $8 \\times 5 = 40$. The answer is $\\boxed{40}$.

Problem: If $f(x) = x^2 + 2x + 1$, what is $f(3)$?
Solution: $f(3) = 9 + 6 + 1 = 16$. The answer is $\\boxed{16}$.

Problem: {problem}
Solution:"""


# ============================================================
# 정답 추출
# ============================================================

def extract_answer_gsm8k(text: str) -> str:
    match = re.search(r"[Tt]he answer is\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def extract_answer_math(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def normalize_answer(ans: str) -> str:
    return ans.replace(",", "").replace(" ", "").lower().strip()


# ============================================================
# 평가 함수
# ============================================================

@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    n_samples:      int = 100,
    max_new_tokens: int = 256,
    device:         str = "cuda",
) -> dict:
    """GSM8K few-shot CoT 평가 (논문과 동일)"""

    print(f"\n📊 GSM8K 평가 중 ({n_samples}개)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(dataset):
        dataset = dataset.select(list(range(n_samples)))

    correct = 0
    total   = 0

    for item in tqdm(dataset):
        question    = item["question"]
        gold_answer = item["answer"].split("####")[-1].strip().replace(",", "")
        prompt      = GSM8K_FEWSHOT.format(question=question)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy (temperature 파라미터 제거 → warning 없음)
            pad_token_id=tokenizer.eos_token_id,
        )

        generated      = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        pred_answer    = extract_answer_gsm8k(generated_text)

        if normalize_answer(pred_answer) == normalize_answer(gold_answer):
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"  GSM8K: {correct}/{total} = {accuracy:.1f}%")
    return {"gsm8k_accuracy": accuracy, "gsm8k_correct": correct, "gsm8k_total": total}


@torch.no_grad()
def evaluate_math(
    model,
    tokenizer,
    n_samples:      int = 50,
    max_new_tokens: int = 256,
    device:         str = "cuda",
) -> dict:
    """MATH few-shot CoT 평가 (논문과 동일 조건)"""

    print(f"\n📊 MATH 평가 중 ({n_samples}개)...")

    # 현재 HuggingFace Hub에서 사용 가능한 MATH 데이터셋
    dataset = None
    candidates = [
        ("EleutherAI/hendrycks_math", "all",  "test"),
        ("lighteval/MATH-Hard",        None,   "test"),
        ("openai/gsm8k",               "main", "test"),  # 최후 fallback (GSM8K로 대체)
    ]
    for path, config, split in candidates:
        try:
            dataset = load_dataset(path, config, split=split) if config \
                else load_dataset(path, split=split)
            print(f"  데이터셋 로드: {path}")
            break
        except Exception:
            continue

    if dataset is None:
        print("  ⚠️  MATH 데이터셋을 찾을 수 없습니다. 스킵.")
        return {"math_accuracy": 0.0, "math_correct": 0, "math_total": 0}

    if n_samples < len(dataset):
        dataset = dataset.select(list(range(n_samples)))

    correct = 0
    total   = 0

    for item in tqdm(dataset):
        problem  = item.get("problem",  item.get("question", ""))
        solution = item.get("solution", item.get("answer",   ""))

        gold_answer = extract_answer_math(solution)
        if not gold_answer:
            continue

        # .format() 대신 replace 사용
        # 이유: MATH 문제에 LaTeX {숫자} 패턴이 있어서 format()이 오작동
        prompt = MATH_FEWSHOT.replace("{problem}", problem)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated      = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        pred_answer    = extract_answer_math(generated_text)

        if normalize_answer(pred_answer) == normalize_answer(gold_answer):
            correct += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"  MATH: {correct}/{total} = {accuracy:.1f}%")
    return {"math_accuracy": accuracy, "math_correct": correct, "math_total": total}


# ============================================================
# 전체 평가 실행
# ============================================================

def run_evaluation(
    model_path:      str,
    experiment_name: str,
    results_path:    str = "./eval_results.json",
    gsm8k_samples:   int = 100,
    math_samples:    int = 50,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 평가: {experiment_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,   # ← 수정: torch_dtype → dtype
        device_map="auto",
    ).eval()

    gsm8k_result = evaluate_gsm8k(model, tokenizer, n_samples=gsm8k_samples, device=device)
    math_result  = evaluate_math(model,  tokenizer, n_samples=math_samples,  device=device)

    avg_acc = (gsm8k_result["gsm8k_accuracy"] + math_result["math_accuracy"]) / 2

    result = {
        "experiment":  experiment_name,
        "model_path":  model_path,
        "avg_accuracy": avg_acc,
        **gsm8k_result,
        **math_result,
    }

    print(f"\n{'='*50}")
    print(f"  {experiment_name}")
    print(f"  GSM8K: {gsm8k_result['gsm8k_accuracy']:.1f}%")
    print(f"  MATH:  {math_result['math_accuracy']:.1f}%")
    print(f"  Avg:   {avg_acc:.1f}%")
    print(f"{'='*50}")

    # 저장 (기존 결과 유지)
    existing = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing.append(result)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"💾 결과 저장: {results_path}")
    return result
