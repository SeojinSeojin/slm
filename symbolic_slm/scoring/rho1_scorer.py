# scoring/rho1_scorer.py
# RHO-1 방식: Reference Model의 excess loss 기준 토큰 선별
#
# 핵심 수식 (논문 Eq.3):
#   L_delta(x_i) = L_theta(x_i) - L_RM(x_i)
#   → training model loss - reference model loss
#   → 이 값이 높은 토큰 = "아직 배울 게 많은 토큰" → 선택
#
# 개선사항:
#   1. Reference model을 TinyLlama로 교체
#      → Qwen2.5-Math와 Qwen2.5-base는 너무 유사해서 excess loss 차이 거의 없음
#      → TinyLlama는 완전히 다른 모델이라 차별화 뚜렷
#   2. 포맷 토큰 강제 포함
#      → "Question:", "Answer:" 같은 Q&A 구조 토큰은 무조건 mask=True
#      → 이 토큰들이 마스킹되면 생성 패턴 학습 불가

import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

TRAINING_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
REFERENCE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # 개선: 충분히 다른 모델

# Q&A 포맷 토큰 — 무조건 포함 (마스킹하면 생성 패턴 학습 불가)
FORMAT_KEYWORDS = {
    "question", "answer", "solution", "problem",
    "Q", "A", ":", "\n",
}


def is_format_token(token_str: str) -> bool:
    """Q&A 구조 토큰이면 True — 무조건 학습에 포함."""
    t = token_str.strip().lower()
    return t in FORMAT_KEYWORDS or token_str.strip() in {":", "\n", "##"}


@torch.no_grad()
def get_token_losses(
    model:     AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device:    str,
) -> torch.Tensor:
    """
    각 position의 token-level CE loss 반환.
    cross_entropy는 float32 필요 → logits를 float32로 cast 후 계산.
    """
    input_ids    = input_ids.to(device)
    outputs      = model(input_ids)
    logits       = outputs.logits[0].float()   # ← bfloat16/float16 → float32

    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, reduction="none"
    )
    return per_token_loss.cpu()


class RHO1Scorer:
    """
    RHO-1 방식 토큰 스코어링.

    논문 Eq.3: L_Δ(x_i) = L_θ(x_i) - L_RM(x_i)
    → 같은 토큰 x_i에 대해 두 모델의 loss 차이를 계산해야 함.

    핵심: 두 모델의 vocab이 달라도 training model tokenizer 기준으로
    tokenize하고, reference model도 그 token_ids를 그대로 입력받음.
    → 같은 position i의 loss끼리 빼는 게 보장됨.
    """

    def __init__(
        self,
        training_model_name:  str = TRAINING_MODEL_NAME,
        reference_model_name: str = REFERENCE_MODEL_NAME,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (
            self.device == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16

        print(f"📥 Training Model 로딩: {training_model_name}")
        # training model tokenizer만 사용 (두 모델 공용)
        self.tokenizer = AutoTokenizer.from_pretrained(training_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.training_model = AutoModelForCausalLM.from_pretrained(
            training_model_name, dtype=dtype,
        ).to(self.device).eval()

        print(f"📥 Reference Model 로딩: {reference_model_name}")
        # reference model도 training tokenizer의 token_ids를 입력받음
        # vocab 크기가 다를 수 있어서 reference model config의 vocab_size 확인
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name, dtype=dtype,
        ).to(self.device).eval()

        # reference model vocab_size 확인
        ref_vocab = self.reference_model.config.vocab_size
        train_vocab = self.training_model.config.vocab_size
        if ref_vocab != train_vocab:
            print(f"  ⚠️  vocab 크기 불일치: training={train_vocab}, ref={ref_vocab}")
            print(f"      token_ids를 ref vocab 범위로 clamp해서 사용합니다.")
        self.ref_vocab_size = ref_vocab

        print("✅ RHO-1 Scorer 준비 완료")

    def score(
        self,
        text:         str,
        select_ratio: float = 0.77,
        max_length:   int   = 512,
    ) -> dict:
        """
        논문 Eq.3: excess loss = L_training(x_i) - L_reference(x_i)

        개선:
        1. 포맷 토큰("Question:", "Answer:" 등)은 무조건 mask=True
           → 이 토큰 마스킹 시 Q&A 패턴 학습 불가
        2. 나머지 토큰은 excess loss 상위 k%로 선택
        """
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        token_ids = enc["input_ids"]

        loss_t = get_token_losses(self.training_model, token_ids, self.device)

        token_ids_ref = token_ids.clone()
        token_ids_ref = token_ids_ref.clamp(max=self.ref_vocab_size - 1)
        loss_r = get_token_losses(self.reference_model, token_ids_ref, self.device)

        min_len = min(len(loss_t), len(loss_r))
        loss_t  = loss_t[:min_len].numpy()
        loss_r  = loss_r[:min_len].numpy()

        excess = loss_t - loss_r

        # 포맷 토큰 판별
        token_ids_list = token_ids[0, :min_len].tolist()
        format_mask = np.array([
            is_format_token(self.tokenizer.decode([tid]))
            for tid in token_ids_list
        ], dtype=bool)

        # excess loss 기준 상위 k% 선택 (포맷 토큰 제외하고 계산)
        n_select  = max(1, int(min_len * select_ratio))
        threshold = np.sort(excess)[-n_select]
        excess_mask = excess >= threshold

        # 최종 mask: 포맷 토큰 OR excess loss 상위 k%
        mask = (format_mask | excess_mask).tolist()

        return {
            "token_ids":   token_ids_list,
            "mask":        mask,
            "excess_loss": excess.tolist(),
        }


def preprocess_dataset_rho1(
    data_path:            str,
    output_path:          str,
    training_model_name:  str   = TRAINING_MODEL_NAME,
    reference_model_name: str   = REFERENCE_MODEL_NAME,
    select_ratio:         float = 0.77,
    max_length:           int   = 512,
):
    """JSONL 전체에 RHO-1 스코어링 적용."""

    # Colab T4: 두 모델 동시 로딩 시 VRAM 부족 가능
    # → 메모리 체크
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  현재 여유 VRAM: {free_gb:.1f} GB")
        if free_gb < 6.0:
            print("  ⚠️  VRAM 부족 가능. 런타임 재시작 후 실행 권장.")

    scorer = RHO1Scorer(training_model_name, reference_model_name)

    with open(data_path) as f:
        lines = f.readlines()

    print(f"🔄 {len(lines):,}개 문서 RHO-1 스코어링 중...")
    results         = []
    total_tokens    = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        try:
            scored = scorer.score(
                item["text"], select_ratio=select_ratio, max_length=max_length
            )
            results.append({
                "text":      item["text"],
                "token_ids": scored["token_ids"],
                "mask":      scored["mask"],
                "method":    "rho1",
            })
            total_tokens    += len(scored["token_ids"])
            selected_tokens += sum(scored["mask"])
        except Exception as e:
            print(f"  ⚠️  스킵: {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / max(total_tokens, 1) * 100
    print(f"\n✅ 완료! 저장: {output_path}")
    print(f"   전체 토큰: {total_tokens:,}")
    print(f"   선택 토큰: {selected_tokens:,} ({ratio:.1f}%)")


if __name__ == "__main__":
    # 단일 문서 테스트
    scorer = RHO1Scorer()
    text   = "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?"
    result = scorer.score(text, select_ratio=0.77)

    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_NAME)
    print("\n=== RHO-1 선택 토큰 ===")
    for i, (tid, m, ex) in enumerate(zip(
        result["token_ids"], result["mask"], result["excess_loss"]
    )):
        tok = tokenizer.decode([tid])
        if m:
            print(f"  [{i:3d}] '{tok:15s}'  excess={ex:.3f}  ✅")

    total = len(result["mask"])
    sel   = sum(result["mask"])
    print(f"\n선택: {sel}/{total} = {sel/total:.1%}")
