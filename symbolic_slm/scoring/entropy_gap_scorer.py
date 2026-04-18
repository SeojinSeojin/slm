# scoring/entropy_gap_scorer.py
# Approach A: Information-Theoretic Entropy Gap Scoring
# General Model vs Symbolic Model의 entropy 차이로 "Critical Token" 식별
#
# 핵심 아이디어:
#   - General Model → 수학 토큰 보면 entropy 높음 (불확실)
#   - Symbolic Model → 수학 토큰 보면 entropy 낮음 (확실)
#   - Entropy Gap = H_general - H_symbolic 이 크면 → Critical Token

import torch
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# ============================================================
# 설정
# ============================================================

# General Reference: 일반 언어모델 (수학 특화 X)
GENERAL_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SYMBOLIC_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


# ============================================================
# Entropy 계산 함수
# ============================================================

def compute_token_entropy(
    logits: torch.Tensor,   # [seq_len, vocab_size]
) -> torch.Tensor:
    """
    각 position에서 다음 토큰의 분포 entropy 계산.
    H(x_i) = -sum_v p(v|x<i) * log p(v|x<i)

    Returns:
        entropy: [seq_len] float tensor
    """
    # logits → probabilities
    probs = F.softmax(logits, dim=-1)    # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # entropy = -sum(p * log p)
    entropy = -(probs * log_probs).sum(dim=-1)   # [seq_len]
    return entropy


@torch.no_grad()
def get_model_entropy(
    model:     AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device:    str = "cuda",
) -> torch.Tensor:
    """
    각 position의 entropy 반환.
    float32로 cast 후 계산 (bfloat16 cross_entropy 미지원 방지).
    """
    input_ids = input_ids.to(device)
    outputs   = model(input_ids)
    logits    = outputs.logits[0].float()   # ← float32로 cast

    entropy = compute_token_entropy(logits[:-1])
    return entropy.cpu()


# ============================================================
# Entropy Gap Scorer
# ============================================================

class EntropyGapScorer:
    """
    두 모델의 entropy gap으로 critical token을 식별.

    핵심: RHO-1과 동일하게 general model tokenizer 기준으로 tokenize,
    symbolic model도 같은 token_ids를 입력받음.
    → position i의 entropy_g와 entropy_s가 같은 토큰 x_i에 대응됨.
    """

    def __init__(
        self,
        general_model_name:  str = GENERAL_MODEL_NAME,
        symbolic_model_name: str = SYMBOLIC_MODEL_NAME,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (
            self.device == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16

        print(f"🖥️  디바이스: {self.device}")

        print(f"📥 General Model 로딩: {general_model_name}")
        # general tokenizer만 사용 (두 모델 공용)
        self.tokenizer = AutoTokenizer.from_pretrained(general_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.general_model = AutoModelForCausalLM.from_pretrained(
            general_model_name,
            dtype=dtype,   # ← 수정: torch_dtype → dtype
        ).to(self.device).eval()

        print(f"📥 Symbolic Model 로딩: {symbolic_model_name}")
        self.symbolic_model = AutoModelForCausalLM.from_pretrained(
            symbolic_model_name,
            dtype=dtype,   # ← 수정: torch_dtype → dtype
        ).to(self.device).eval()

        # symbolic model vocab_size 확인 (clamp 용도)
        sym_vocab   = self.symbolic_model.config.vocab_size
        gen_vocab   = self.general_model.config.vocab_size
        if sym_vocab != gen_vocab:
            print(f"  ⚠️  vocab 불일치: general={gen_vocab}, symbolic={sym_vocab}")
            print(f"      token_ids를 symbolic vocab 범위로 clamp합니다.")
        self.sym_vocab_size = sym_vocab

        print("✅ 두 모델 로딩 완료")

    @torch.no_grad()
    def score(
        self,
        text:           str,
        select_ratio:   float = 0.10,
        context_window: int   = 3,
        max_length:     int   = 512,
    ) -> dict:
        """
        Entropy Gap = H_general(x_i) - H_symbolic(x_i)
        같은 tokenizer, 같은 token_ids → 같은 position i끼리 비교 보장.
        """
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        token_ids = enc["input_ids"]

        entropy_g = get_model_entropy(self.general_model, token_ids, self.device)

        token_ids_sym = token_ids.clone().clamp(max=self.sym_vocab_size - 1)
        entropy_s = get_model_entropy(self.symbolic_model, token_ids_sym, self.device)

        min_len   = min(len(entropy_g), len(entropy_s))
        entropy_g = entropy_g[:min_len]
        entropy_s = entropy_s[:min_len]

        # Entropy Gap
        gap = (entropy_g - entropy_s).numpy()

        # Top-k% 선택
        token_ids_list = token_ids[0, :min_len].tolist()
        n_select  = max(1, int(min_len * select_ratio))
        threshold = np.sort(gap)[-n_select]
        base_mask = gap >= threshold

        # Expand with context window
        expanded = base_mask.copy()
        for i in range(min_len):
            if base_mask[i]:
                lo = max(0, i - context_window)
                hi = min(min_len, i + context_window + 1)
                expanded[lo:hi] = True

        mask = expanded.tolist()

        return {
            "token_ids":       token_ids_list,
            "mask":            mask,
            "entropy_general": entropy_g.tolist(),
            "entropy_symbolic":entropy_s.tolist(),
            "entropy_gap":     gap.tolist(),
        }


# ============================================================
# 데이터셋 전체 처리
# ============================================================

def preprocess_dataset_entropy_gap(
    data_path: str,
    output_path: str,
    general_model_name: str = GENERAL_MODEL_NAME,
    symbolic_model_name: str = SYMBOLIC_MODEL_NAME,
    select_ratio: float = 0.10,
    context_window: int = 3,
    max_length: int = 2048,
    batch_size: int = 1,
):
    """
    JSONL 데이터를 entropy gap으로 스코어링해서 저장.
    """
    scorer = EntropyGapScorer(general_model_name, symbolic_model_name)

    with open(data_path, "r") as f:
        lines = f.readlines()

    print(f"🔄 {len(lines):,}개 문서 entropy gap 스코어링 중...")
    print("⚠️  이 작업은 시간이 걸립니다 (문서당 2번 forward pass)")

    results = []
    total_tokens = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        text = item["text"]

        try:
            scored = scorer.score(text, select_ratio=select_ratio,
                                  context_window=context_window,
                                  max_length=max_length)
            results.append({
                "text": text,
                "token_ids": scored["token_ids"],
                "mask": scored["mask"],
                "method": "entropy_gap",
            })
            total_tokens += len(scored["token_ids"])
            selected_tokens += sum(scored["mask"])
        except Exception as e:
            print(f"⚠️  스킵 (오류): {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / max(total_tokens, 1) * 100
    print(f"\n✅ 완료! 저장: {output_path}")
    print(f"   전체 토큰: {total_tokens:,}")
    print(f"   선택 토큰: {selected_tokens:,} ({ratio:.1f}%)")


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    # 단일 예시 테스트 (모델 로딩 있으므로 시간 걸림)
    scorer = EntropyGapScorer()

    test_text = "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?"
    result = scorer.score(test_text, select_ratio=0.10)

    tokenizer = AutoTokenizer.from_pretrained(GENERAL_MODEL_NAME)

    print("=== Entropy Gap Scorer 테스트 ===")
    print(f"입력: {test_text}\n")
    print("선택된 토큰 (Critical Tokens):")
    for i, (tid, m, g_ent, s_ent, gap) in enumerate(zip(
        result["token_ids"], result["mask"],
        result["entropy_general"], result["entropy_symbolic"],
        result["entropy_gap"],
    )):
        tok = tokenizer.decode([tid])
        if m:
            print(f"  [{i:3d}] '{tok:15s}'  gap={gap:.3f}  (H_gen={g_ent:.3f}, H_sym={s_ent:.3f})  ✅")

    total = len(result["mask"])
    selected = sum(result["mask"])
    print(f"\n선택 비율: {selected}/{total} = {selected/total:.1%}")
