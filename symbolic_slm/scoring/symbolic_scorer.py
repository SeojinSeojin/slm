# scoring/symbolic_scorer.py
# Approach B: Deterministic Symbolic Scoring
# 수학적으로 중요한 토큰을 rule-based로 선별
# → 빠름, 추가 모델 불필요, 먼저 실험하기 좋음

import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================
# 수학적으로 중요한 패턴 정의
# ============================================================

# 논리/수학 연산자 (키워드)
LOGICAL_KEYWORDS = {
    # 수량/변화 동사
    "loses", "gains", "increases", "decreases", "doubles", "halves",
    "multiplies", "divides", "adds", "subtracts", "equals", "totals",
    # 비교
    "greater", "less", "more", "fewer", "larger", "smaller",
    "maximum", "minimum", "most", "least",
    # 논리
    "if", "then", "therefore", "thus", "hence", "since", "because",
    "given", "suppose", "assume", "let",
    # 수학 용어
    "sum", "difference", "product", "quotient", "remainder",
    "ratio", "proportion", "percent", "rate", "average", "mean",
    "total", "each", "per", "every",
}

# 수식 패턴 (regex)
MATH_PATTERNS = [
    r'\d+\.?\d*',                          # 숫자 (97, 3.14)
    r'[+\-*/=<>≤≥≠±]',                   # 연산자
    r'\\frac\{',                           # LaTeX fraction
    r'\\sum|\\prod|\\int',                 # LaTeX 합/적분
    r'\\sqrt|\\log|\\ln|\\exp',           # LaTeX 함수
    r'\\[a-zA-Z]+',                        # LaTeX 명령어 일반
    r'\$[^$]+\$',                          # inline math $...$
    r'\$\$[^$]+\$\$',                      # display math $$...$$
    r'[a-zA-Z]_\{?\d+\}?',               # 변수 subscript (x_1, a_{ij})
    r'\([^)]*=\s*[^)]*\)',                # 수식 포함 괄호
    # Variable transitions (proposal 명시 항목)
    # "variable transitions" = 변수가 다른 값으로 전환되는 표현
    r'[a-zA-Z]\s*=\s*[a-zA-Z0-9]',       # 변수 대입 (x = 3, a = b)
    r'[a-zA-Z]\s*:=\s*',                  # 정의식 (x := ...)
    r'therefore\s+[a-zA-Z]',              # "therefore x = ..."
    r'let\s+[a-zA-Z]\s*=',               # "let x = ..."
    r'where\s+[a-zA-Z]\s*=',             # "where x = ..."
    r'→|⟹|⟺|\s*=>\s*',               # 전이 화살표
]

COMPILED_PATTERNS = [re.compile(p) for p in MATH_PATTERNS]


def is_symbolic_token(token_str: str) -> bool:
    """
    토큰 하나가 수학적으로 중요한지 판단.
    True면 학습에 포함, False면 제외.
    """
    t = token_str.strip().lower()

    # 빈 토큰, 공백 토큰 제외
    if not t or t in {"", " ", "\n", "\t"}:
        return False

    # 1) 논리/수학 키워드
    if t in LOGICAL_KEYWORDS:
        return True

    # 2) 수식 패턴 매칭
    for pattern in COMPILED_PATTERNS:
        if pattern.search(token_str):
            return True

    # 3) 숫자 포함 토큰
    if any(c.isdigit() for c in token_str):
        return True

    # 4) 특수 수학 기호
    math_symbols = set("+=−×÷≤≥≠∈∉⊂⊃∪∩∀∃∧∨¬→↔∞∑∏∫√π")
    if any(c in math_symbols for c in token_str):
        return True

    return False


def is_answer_separator(token_str: str) -> bool:
    """Force-include #### tokens — answer separator must always be learned."""
    return token_str.strip().startswith("####") or token_str.strip() == "##"


def score_tokens_symbolic(
    token_ids:    list,
    tokenizer:    AutoTokenizer,
    select_ratio: float = 0.40,  # window 확장 후 실제 비율이 높아지므로 참고값
    context_window: int = 3,     # symbolic 토큰 앞뒤 N개 토큰도 포함
) -> torch.Tensor:
    """
    Symbolic 토큰 + 주변 context window 포함.

    이전 방식(8.3%)의 문제:
    → 숫자/기호만 고립되어 학습 → 맥락 없어 모델 붕괴

    수정:
    → symbolic 토큰 발견 시 앞뒤 context_window개 토큰도 함께 선택
    → 모델이 "97 kg", "loses 3 kg per month" 같은 표현을 통째로 학습
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    seq_len  = len(token_ids)
    symbolic = np.zeros(seq_len, dtype=bool)

    # 1) symbolic 토큰 먼저 표시
    for i, tid in enumerate(token_ids):
        token_str    = tokenizer.decode([tid], skip_special_tokens=False)
        symbolic[i]  = is_symbolic_token(token_str)

    # 2) 각 symbolic 토큰 주변 context_window 확장
    mask = np.zeros(seq_len, dtype=bool)
    for i in range(seq_len):
        if symbolic[i]:
            lo = max(0, i - context_window)
            hi = min(seq_len, i + context_window + 1)
            mask[lo:hi] = True

    return torch.tensor(mask, dtype=torch.bool)


# ============================================================
# 데이터셋 전체에 스코어 적용 (오프라인 전처리)
# ============================================================

def preprocess_dataset_symbolic(
    data_path: str,
    output_path: str,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
    select_ratio: float = 0.40,
    max_length: int = 512,
):
    """
    JSONL 데이터셋을 읽어서 각 문서마다 symbolic mask를 계산하고 저장.
    학습 시 이 mask를 그대로 사용해서 속도 최적화.
    """
    print(f"📂 데이터 로드: {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    with open(data_path, "r") as f:
        lines = f.readlines()

    print(f"🔄 {len(lines):,}개 문서 처리 중...")
    total_tokens = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        text = item["text"]

        # 토크나이즈
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = enc["input_ids"][0]

        # Symbolic scoring
        mask = score_tokens_symbolic(token_ids, tokenizer, select_ratio)

        total_tokens += len(token_ids)
        selected_tokens += mask.sum().item()

        results.append({
            "text": text,
            "token_ids": token_ids.tolist(),
            "mask": mask.tolist(),   # True = 학습, False = 무시
        })

    # 저장
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / total_tokens * 100
    print(f"\n✅ 완료! 저장: {output_path}")
    print(f"   전체 토큰: {total_tokens:,}")
    print(f"   선택 토큰: {selected_tokens:,} ({ratio:.1f}%)")

    return output_path


# ============================================================
# 테스트 실행
# ============================================================
if __name__ == "__main__":
    # 빠른 테스트
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )

    test_text = "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?"
    enc = tokenizer(test_text, return_tensors="pt")
    token_ids = enc["input_ids"][0]

    mask = score_tokens_symbolic(token_ids, tokenizer, select_ratio=0.10)

    print("=== Symbolic Scorer 테스트 ===")
    print(f"입력: {test_text}")
    print(f"\n선택된 토큰:")
    for i, (tid, m) in enumerate(zip(token_ids.tolist(), mask.tolist())):
        token_str = tokenizer.decode([tid])
        if m:
            print(f"  [{i:3d}] '{token_str}'  ✅")

    print(f"\n선택 비율: {mask.sum().item()}/{len(mask)} = {mask.float().mean():.1%}")
