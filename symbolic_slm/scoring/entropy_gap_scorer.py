# scoring/entropy_gap_scorer.py

import torch
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

GENERAL_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SYMBOLIC_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs     = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def get_model_entropy(model, input_ids, device="cuda") -> torch.Tensor:
    input_ids = input_ids.to(device)
    outputs   = model(input_ids)
    logits    = outputs.logits[0].float()
    entropy   = compute_token_entropy(logits[:-1])
    return entropy.cpu()


# ============================================================
# ▼▼▼ 신규: margin 확장 헬퍼 ▼▼▼
# ============================================================

def expand_mask_with_margin(
    selected_indices: np.ndarray,
    seq_len: int,
    context_window: int = 3,
) -> np.ndarray:
    """
    선택된 인덱스 각각의 앞뒤 context_window개 토큰도 True로 설정.
    symbolic_scorer의 context_window 방식과 동일한 로직.
    """
    mask = np.zeros(seq_len, dtype=bool)
    for i in selected_indices:
        lo = max(0, i - context_window)
        hi = min(seq_len, i + context_window + 1)
        mask[lo:hi] = True
    return mask


# ============================================================
# Entropy Gap Scorer
# ============================================================

class EntropyGapScorer:

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
        self.tokenizer = AutoTokenizer.from_pretrained(general_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.general_model = AutoModelForCausalLM.from_pretrained(
            general_model_name, dtype=dtype,
        ).to(self.device).eval()

        print(f"📥 Symbolic Model 로딩: {symbolic_model_name}")
        self.symbolic_model = AutoModelForCausalLM.from_pretrained(
            symbolic_model_name, dtype=dtype,
        ).to(self.device).eval()

        sym_vocab = self.symbolic_model.config.vocab_size
        gen_vocab = self.general_model.config.vocab_size
        if sym_vocab != gen_vocab:
            print(f"  ⚠️  vocab 불일치: general={gen_vocab}, symbolic={sym_vocab}")
            print(f"      token_ids를 symbolic vocab 범위로 clamp합니다.")
        self.sym_vocab_size = sym_vocab

        print("✅ 두 모델 로딩 완료")

    @torch.no_grad()
    def score(
        self,
        text:           str,
        select_ratio:   float = 0.25,   # margin 확장 후 실제 ~75% → 수동 튜닝
        context_window: int   = 3,      # symbolic_scorer와 동일한 margin 크기
        max_length:     int   = 512,
    ) -> dict:
        """
        1) Entropy Gap 상위 select_ratio% 토큰을 seed로 선택  (argsort 기반, 동점 버그 없음)
        2) 각 seed 토큰의 앞뒤 context_window개 토큰도 함께 선택 (margin 확장)

        select_ratio 튜닝 가이드:
            context_window=3 기준, 텍스트 밀도에 따라 다르지만
            select_ratio=0.40 → 실제 ~65%
            select_ratio=0.50 → 실제 ~75%  ← 기본값 (목표 범위)
            select_ratio=0.55 → 실제 ~80%
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

        gap = (entropy_g - entropy_s).numpy()

        # ▼ 버그 수정: argsort로 정확히 n개만 seed 선택
        n_seed           = max(1, int(min_len * select_ratio))
        top_indices      = np.argsort(gap)[::-1][:n_seed]   # 내림차순 상위 n개

        # ▼ 신규: seed 주변 margin 확장
        mask = expand_mask_with_margin(top_indices, min_len, context_window)

        return {
            "token_ids":        token_ids[0, :min_len].tolist(),
            "mask":             mask.tolist(),
            "entropy_general":  entropy_g.tolist(),
            "entropy_symbolic": entropy_s.tolist(),
            "entropy_gap":      gap.tolist(),
        }


# ============================================================
# 데이터셋 전체 처리
# ============================================================

def preprocess_dataset_entropy_gap(
    data_path:           str,
    output_path:         str,
    general_model_name:  str   = GENERAL_MODEL_NAME,
    symbolic_model_name: str   = SYMBOLIC_MODEL_NAME,
    select_ratio:        float = 0.25,   # ← margin 감안한 값
    context_window:      int   = 3,
    max_length:          int   = 2048,
):
    scorer = EntropyGapScorer(general_model_name, symbolic_model_name)

    with open(data_path, "r") as f:
        lines = f.readlines()

    print(f"🔄 {len(lines):,}개 문서 entropy gap 스코어링 중...")
    print(f"   select_ratio={select_ratio}, context_window=±{context_window}")

    results        = []
    total_tokens   = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        text = item["text"]

        try:
            scored = scorer.score(
                text,
                select_ratio=select_ratio,
                context_window=context_window,
                max_length=max_length,
            )
            results.append({
                "text":      text,
                "token_ids": scored["token_ids"],
                "mask":      scored["mask"],
                "method":    "entropy_gap",
            })
            total_tokens    += len(scored["token_ids"])
            selected_tokens += sum(scored["mask"])
        except Exception as e:
            print(f"⚠️  스킵 (오류): {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / max(total_tokens, 1) * 100
    print(f"\n✅ 완료! 저장: {output_path}")
    print(f"   전체 토큰:   {total_tokens:,}")
    print(f"   선택 토큰:   {selected_tokens:,} ({ratio:.1f}%)")
    print(f"   목표 범위:   70~80%  {'✅' if 70 <= ratio <= 80 else '⚠️  select_ratio 재조정 필요'}")


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    scorer = EntropyGapScorer()

    test_text = "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?"
    result    = scorer.score(test_text, select_ratio=0.25, context_window=3)

    tokenizer = AutoTokenizer.from_pretrained(GENERAL_MODEL_NAME)
    tokens    = tokenizer.convert_ids_to_tokens(result["token_ids"])

    print("=== Entropy Gap Scorer 테스트 ===")
    print(f"입력: {test_text}\n")
    print(f"{'idx':>4}  {'token':15}  {'gap':>7}  {'H_gen':>7}  {'H_sym':>7}  sel")
    print("-" * 55)
    for i, (tok, m, g_ent, s_ent, gap) in enumerate(zip(
        tokens,
        result["mask"],
        result["entropy_general"],
        result["entropy_symbolic"],
        result["entropy_gap"],
    )):
        marker = "✅" if m else "  "
        print(f"[{i:3d}]  {tok:15}  {gap:7.3f}  {g_ent:7.3f}  {s_ent:7.3f}  {marker}")

    total    = len(result["mask"])
    selected = sum(result["mask"])
    print(f"\n선택 비율: {selected}/{total} = {selected/total:.1%}")