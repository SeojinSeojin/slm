# eval/evaluate.py
# GSM8K + MATH + ASDiv + SVAMP evaluation

import re
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# Few-shot prompts
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

MATH_FEWSHOT = """Question: What is the value of $x$ if $2x + 3 = 11$?
Answer: Subtract 3 from both sides: $2x = 8$. Divide by 2: $x = 4$. The answer is 4.

Question: A rectangle has length 8 and width 5. What is its area?
Answer: Area = length times width = $8 \\times 5 = 40$. The answer is 40.

Question: If $f(x) = x^2 + 2x + 1$, what is $f(3)$?
Answer: $f(3) = 3^2 + 2(3) + 1 = 9 + 6 + 1 = 16$. The answer is 16.

Question: {problem}
Answer:"""

ASDIV_FEWSHOT = """Question: Seven red apples and two green apples are in the basket. How many apples are in the basket?
Answer: 7 + 2 = 9. The answer is 9.

Question: Ellen has six more balls than Marin. Marin has nine balls. How many balls does Ellen have?
Answer: 9 + 6 = 15. The answer is 15.

Question: There are 18 students in a class and 6 sit in each row. How many rows are there?
Answer: 18 / 6 = 3. The answer is 3.

Question: A box has 4 rows of 5 chocolates each. How many chocolates are there in total?
Answer: 4 x 5 = 20. The answer is 20.

Question: {question}
Answer:"""

SVAMP_FEWSHOT = """Question: There are 15 trees in the grove and workers planted 6 more. How many trees are there now?
Answer: 15 + 6 = 21. The answer is 21.

Question: Sara had 10 apples and ate 3. How many apples does she have left?
Answer: 10 - 3 = 7. The answer is 7.

Question: A bag has 4 groups of 5 candies each. How many candies are there in total?
Answer: 4 x 5 = 20. The answer is 20.

Question: 24 students are divided equally into 6 groups. How many students are in each group?
Answer: 24 / 6 = 4. The answer is 4.

Question: {question}
Answer:"""


# ============================================================
# Answer extraction
# ============================================================

def extract_answer_gsm8k(text: str) -> str:
    # "The answer is $X" or "The answer is X" — $ 있어도 처리
    match = re.search(r"[Tt]he answer is\s*\$?\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # #### 패턴
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # 위 두 패턴 모두 실패하면 빈 문자열 반환
    # 기존엔 마지막 숫자를 fallback으로 뽑았는데,
    # 모델이 헛소리를 길게 생성하면 엉뚱한 숫자가 뽑히는 문제가 있었음
    return ""


def extract_answer_math(text: str) -> str:
    """Extract answer from MATH-style output. Handles nested braces in \\boxed{}."""
    text = re.sub(r"\\[dt]frac", r"\\frac", text)

    idx = text.find("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()

    m = re.search(r"[Tt]he answer is\s*\$?([^\n$]+)\$?", text)
    if m:
        return m.group(1).strip().rstrip(".")

    fracs = list(re.finditer(r"\\frac\{[^{}]+\}\{[^{}]+\}", text))
    if fracs:
        return fracs[-1].group(0).strip()

    # FIX: 기존과 동일하게 마지막 숫자 fallback 유지 (MATH는 숫자 외 표현이 많아서 필요)
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def extract_answer_svamp(text: str) -> str:
    # #### 패턴
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # "The answer is X"
    match = re.search(r"[Tt]he answer is\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # FIX: 기존 코드에 return 뒤 dead code(try 블록)가 있었음 — 제거
    # base model이 few-shot 예시를 반복하는 경향이 있어서 첫 번째 숫자 사용
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    return numbers[0] if numbers else ""


def normalize_answer(s: str) -> str:
    if not s:
        return ""

    s = s.strip()
    s = s.replace("$", "").strip()

    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        s = m.group(1).strip()

    s = s.replace("\\cup", "U").replace("\\cap", "N")
    s = s.replace("∪", "U").replace("∩", "N")
    s = s.replace("\\infty", "inf").replace("∞", "inf")
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\\[dt]?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s+(\S+)", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(" ", "")

    frac_match = re.match(r"^\(?(-?\d+\.?\d*)\)?/\(?(-?\d+\.?\d*)\)?$", s)
    if frac_match:
        try:
            val = float(frac_match.group(1)) / float(frac_match.group(2))
            return str(int(val)) if val == int(val) else str(round(val, 6))
        except Exception:
            pass

    try:
        val = float(s.replace(",", ""))
        return str(int(val)) if val == int(val) else str(round(val, 6))
    except Exception:
        pass

    if "U" in s or "N" in s or "(" in s or "[" in s:
        s = s.replace("−", "-")
        s = s.replace("infty", "inf").replace("infinity", "inf")
        return s.lower()

    return s.lower()


normalize_number = normalize_answer


# ============================================================
# Generation helper
# ============================================================

def generate_answer(model, tokenizer, prompt, max_new_tokens, device):
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
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================
# GSM8K
# ============================================================

@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    n_samples:      int = 200,
    max_new_tokens: int = 256,
    device:         str = "cuda",
) -> dict:
    print(f"\n📊 GSM8K evaluation ({n_samples} samples)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(dataset):
        dataset = dataset.select(list(range(n_samples)))

    correct, total = 0, 0
    per_problem = []

    for item in tqdm(dataset):
        question   = item["question"]
        gold       = item["answer"].split("####")[-1].strip().replace(",", "")
        prompt     = GSM8K_FEWSHOT.format(question=question)
        generated  = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        pred       = extract_answer_gsm8k(generated)
        is_correct = normalize_answer(pred) == normalize_answer(gold)

        if is_correct:
            correct += 1
        total += 1

        per_problem.append({
            "question":  question,
            "gold":      gold,
            "pred":      pred,
            "correct":   is_correct,
            "generated": generated,
        })

    accuracy = correct / total * 100
    print(f"  GSM8K: {correct}/{total} = {accuracy:.1f}%")
    return {
        "gsm8k_accuracy":     accuracy,
        "gsm8k_correct":      correct,
        "gsm8k_total":        total,
        "gsm8k_per_problem":  per_problem,
    }


# ============================================================
# MATH-Hard
# ============================================================

@torch.no_grad()
def evaluate_math(
    model,
    tokenizer,
    n_samples:      int = 200,
    max_new_tokens: int = 512,  # FIX: 256 → 512 (LaTeX 수식 생성에 공간 더 필요)
    device:         str = "cuda",
) -> dict:
    print(f"\n📊 MATH-Hard evaluation ({n_samples} samples)...")
    dataset = None
    for path, config, split in [
        ("lighteval/MATH-Hard",       None,  "test"),
        ("EleutherAI/hendrycks_math", "all", "test"),
    ]:
        try:
            dataset = load_dataset(path, config, split=split) if config \
                else load_dataset(path, split=split)
            print(f"  Loaded: {path}")
            break
        except Exception:
            continue

    if dataset is None:
        print("  MATH dataset not found, skipping.")
        return {"math_accuracy": 0.0, "math_correct": 0, "math_total": 0, "math_per_problem": []}

    if n_samples < len(dataset):
        dataset = dataset.select(list(range(n_samples)))

    correct, total = 0, 0
    per_problem = []

    for item in tqdm(dataset):
        problem   = item.get("problem", item.get("question", ""))
        solution  = item.get("solution", item.get("answer", ""))
        gold      = extract_answer_math(solution)
        if not gold:
            continue

        prompt    = MATH_FEWSHOT.replace("{problem}", problem)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        pred      = extract_answer_gsm8k(generated)
        if not pred:
            pred = extract_answer_math(generated)
        is_correct = normalize_answer(pred) == normalize_answer(gold)

        if is_correct:
            correct += 1
        total += 1

        per_problem.append({
            "problem":   problem,
            "gold":      gold,
            "pred":      pred,
            "correct":   is_correct,
            "generated": generated,
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"  MATH: {correct}/{total} = {accuracy:.1f}%")
    return {
        "math_accuracy":    accuracy,
        "math_correct":     correct,
        "math_total":       total,
        "math_per_problem": per_problem,
    }


# ============================================================
# ASDiv
# ============================================================

@torch.no_grad()
def evaluate_asdiv(
    model,
    tokenizer,
    n_samples:      int = 200,
    max_new_tokens: int = 256,
    device:         str = "cuda",
) -> dict:
    print(f"\n📊 ASDiv evaluation ({n_samples} samples)...")
    try:
        dataset = load_dataset("EleutherAI/asdiv", split="validation")
    except Exception as e:
        print(f"  ASDiv load failed: {e}")
        return {"asdiv_accuracy": 0.0, "asdiv_correct": 0, "asdiv_total": 0, "asdiv_per_problem": []}

    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    correct, total = 0, 0
    per_problem = []

    for item in tqdm(dataset):
        question   = item["body"] + " " + item["question"]
        raw_answer = str(item["answer"]).split("(")[0].strip().replace(",", "")
        gold       = normalize_number(raw_answer)

        prompt     = ASDIV_FEWSHOT.format(question=question)
        generated  = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        pred       = normalize_number(extract_answer_gsm8k(generated))
        is_correct = normalize_answer(pred) == normalize_answer(gold)

        if is_correct:
            correct += 1
        total += 1

        per_problem.append({
            "question":  question,
            "gold":      gold,
            "pred":      pred,
            "correct":   is_correct,
            "generated": generated,
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"  ASDiv: {correct}/{total} = {accuracy:.1f}%")
    return {
        "asdiv_accuracy":    accuracy,
        "asdiv_correct":     correct,
        "asdiv_total":       total,
        "asdiv_per_problem": per_problem,
    }


# ============================================================
# SVAMP
# ============================================================

@torch.no_grad()
def evaluate_svamp(
    model,
    tokenizer,
    n_samples:      int = 200,
    max_new_tokens: int = 128,
    device:         str = "cuda",
) -> dict:
    print(f"\n📊 SVAMP evaluation ({n_samples} samples)...")
    try:
        dataset = load_dataset("ChilleD/SVAMP", split="train")
    except Exception as e:
        print(f"  SVAMP load failed: {e}")
        return {"svamp_accuracy": 0.0, "svamp_correct": 0, "svamp_total": 0, "svamp_per_problem": []}

    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    correct, total = 0, 0
    per_problem = []

    for item in tqdm(dataset):
        question   = item["Body"] + " " + item["Question"]
        gold       = normalize_number(str(item["Answer"]).replace(",", "").strip())
        prompt     = SVAMP_FEWSHOT.format(question=question)
        generated  = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        pred       = normalize_number(extract_answer_svamp(generated))
        is_correct = normalize_answer(pred) == normalize_answer(gold)

        if is_correct:
            correct += 1
        total += 1

        per_problem.append({
            "question":  question,
            "gold":      gold,
            "pred":      pred,
            "correct":   is_correct,
            "generated": generated,
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"  SVAMP: {correct}/{total} = {accuracy:.1f}%")
    return {
        "svamp_accuracy":    accuracy,
        "svamp_correct":     correct,
        "svamp_total":       total,
        "svamp_per_problem": per_problem,
    }


# ============================================================
# run_evaluation
# ============================================================

def run_evaluation(
    model_path:      str,
    experiment_name: str,
    results_path:    str  = "./eval_results.json",
    gsm8k_samples:   int  = 200,
    math_samples:    int  = 200,
    asdiv_samples:   int  = 200,
    svamp_samples:   int  = 200,
    run_asdiv:       bool = True,
    run_svamp:       bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 Evaluating: {experiment_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FIX: dtype= → torch_dtype= (오타로 인해 float32로 로딩되던 문제)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    gsm8k_result = evaluate_gsm8k(model, tokenizer, n_samples=gsm8k_samples, device=device) \
        if gsm8k_samples > 0 else {"gsm8k_accuracy": 0.0, "gsm8k_correct": 0, "gsm8k_total": 0, "gsm8k_per_problem": []}
    math_result  = evaluate_math(model,  tokenizer, n_samples=math_samples,  device=device) \
        if math_samples  > 0 else {"math_accuracy":  0.0, "math_correct":  0, "math_total":  0, "math_per_problem":  []}

    result = {
        "experiment": experiment_name,
        "model_path": model_path,
        **{k: v for k, v in gsm8k_result.items() if "per_problem" not in k},
        **{k: v for k, v in math_result.items()  if "per_problem" not in k},
    }

    if run_asdiv:
        asdiv_result = evaluate_asdiv(model, tokenizer, n_samples=asdiv_samples, device=device)
        result.update({k: v for k, v in asdiv_result.items() if "per_problem" not in k})

    if run_svamp:
        svamp_result = evaluate_svamp(model, tokenizer, n_samples=svamp_samples, device=device)
        result.update({k: v for k, v in svamp_result.items() if "per_problem" not in k})

    acc_keys = [k for k in result if k.endswith("_accuracy")]
    result["avg_accuracy"] = sum(result[k] for k in acc_keys) / len(acc_keys)

    print(f"\n{'='*50}")
    print(f"  {experiment_name}")
    for k in acc_keys:
        bench = k.replace("_accuracy", "").upper()
        print(f"  {bench}: {result[k]:.1f}%")
    print(f"  Avg: {result['avg_accuracy']:.1f}%")
    print(f"{'='*50}")

    detail_dir = os.path.join(os.path.dirname(results_path), "per_problem")
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, f"{experiment_name}.json")

    if os.path.exists(detail_path):
        with open(detail_path) as f:
            details = json.load(f)
    else:
        details = {"experiment": experiment_name}

    if gsm8k_samples > 0:
        details["gsm8k"] = gsm8k_result.get("gsm8k_per_problem", [])
    if math_samples > 0:
        details["math"]  = math_result.get("math_per_problem", [])
    if run_asdiv:
        details["asdiv"] = asdiv_result.get("asdiv_per_problem", [])
    if run_svamp:
        details["svamp"] = svamp_result.get("svamp_per_problem", [])

    with open(detail_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"  Per-problem details: {detail_path}")

    existing = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)

    existing_entry = next((r for r in existing if r["experiment"] == experiment_name), {})
    existing_entry.update(result)
    acc_keys = [k for k in existing_entry if k.endswith("_accuracy")]
    existing_entry["avg_accuracy"] = sum(existing_entry[k] for k in acc_keys) / len(acc_keys)

    existing = [r for r in existing if r["experiment"] != experiment_name]
    existing.append(existing_entry)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Summary saved: {results_path}")
    return result