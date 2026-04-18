# scoring/math_confidence_scorer.py
# Approach C: Dual-Threshold Entropy Gap Scoring
#
# Core idea:
#   Select tokens that satisfy BOTH conditions:
#   1. H_general - H_math > gap_threshold  (gap in top gap_ratio%)
#      → math model has clear advantage over general model
#   2. H_math < abs_threshold              (math model actually confident)
#      → filters out tokens both models find hard (narrative filler)
#
# This avoids the failure mode of Ours-A, where narrative filler
# (high H_general, high H_math) passes the gap condition alone.

import torch
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

TRAINING_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MATH_MODEL_NAME     = "Qwen/Qwen2.5-Math-1.5B"


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-position entropy of the next-token distribution.
    H(x_i) = -sum_v p(v|x<i) * log p(v|x<i)
    """
    probs     = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy   = -(probs * log_probs).sum(dim=-1)
    return entropy


@torch.no_grad()
def get_model_entropy(
    model:     AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device:    str = "cuda",
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    outputs   = model(input_ids)
    logits    = outputs.logits[0].float()   # cast to float32
    entropy   = compute_token_entropy(logits[:-1])
    return entropy.cpu()


class MathConfidenceScorer:
    """
    Ours-C: Dual-threshold entropy gap scoring.

    Unlike Ours-A which selects tokens where H_general - H_math is large,
    Ours-C adds a second condition: H_math must actually be low.

    This filters out narrative filler where BOTH models are uncertain,
    keeping only tokens where the math model has a genuine advantage.
    """

    def __init__(
        self,
        general_model_name: str = TRAINING_MODEL_NAME,
        math_model_name:    str = MATH_MODEL_NAME,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (
            self.device == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16

        print(f"Device: {self.device}")

        print(f"Loading tokenizer + general model: {general_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(general_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.general_model = AutoModelForCausalLM.from_pretrained(
            general_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        print(f"Loading math model: {math_model_name}")
        self.math_model = AutoModelForCausalLM.from_pretrained(
            math_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        self.math_vocab_size = self.math_model.config.vocab_size
        gen_vocab = self.general_model.config.vocab_size
        if self.math_vocab_size != gen_vocab:
            print(f"  Vocab mismatch: general={gen_vocab}, math={self.math_vocab_size}")
            print(f"  Will clamp token_ids to math vocab range.")

        print("Ours-C (Dual-threshold) scorer ready.")

    @torch.no_grad()
    def score(
        self,
        text:            str,
        gap_ratio:       float = 0.40,
        abs_ratio:       float = 0.50,
        context_window:  int   = 3,
        max_length:      int   = 512,
    ) -> dict:
        """
        Dual-threshold entropy gap scoring.

        Select tokens that satisfy BOTH:
          1. H_general - H_math > gap_threshold   (gap in top gap_ratio%)
             → math model has clear advantage over general model
          2. H_math < abs_threshold               (math model actually confident)
             → filters out tokens both models find hard (narrative filler)

        This avoids:
          - Narrative filler: H_general high, H_math also high → fails condition 2
          - Common words:    H_general low,  H_math low        → fails condition 1
          - Target tokens:   H_general high, H_math low        → passes both

        gap_ratio=0.40, abs_ratio=0.50, context_window=3 → ~70% total
        """
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        token_ids = enc["input_ids"]
        n = token_ids.shape[1]

        # General model entropy (no clamping needed — same tokenizer)
        entropy_g = get_model_entropy(self.general_model, token_ids, self.device).numpy()

        # Math model entropy (clamp to math vocab range)
        token_ids_math = token_ids.clone().clamp(max=self.math_vocab_size - 1)
        entropy_m = get_model_entropy(self.math_model, token_ids_math, self.device).numpy()

        min_len = min(len(entropy_g), len(entropy_m))
        entropy_g = entropy_g[:min_len]
        entropy_m = entropy_m[:min_len]

        gap = entropy_g - entropy_m  # positive = math model more confident

        # Condition 1: gap in top gap_ratio%
        n_gap     = max(1, int(min_len * gap_ratio))
        gap_thresh = np.sort(gap)[-n_gap]
        cond1     = gap >= gap_thresh

        # Condition 2: H_math in bottom abs_ratio% (math model actually confident)
        n_abs      = max(1, int(min_len * abs_ratio))
        abs_thresh = np.sort(entropy_m)[n_abs - 1]
        cond2      = entropy_m <= abs_thresh

        # Both conditions must hold
        base_mask = cond1 & cond2

        # Expand with context window
        expanded = base_mask.copy()
        for i in range(min_len):
            if base_mask[i]:
                lo = max(0, i - context_window)
                hi = min(min_len, i + context_window + 1)
                expanded[lo:hi] = True

        return {
            "token_ids":     token_ids[0, :min_len].tolist(),
            "mask":          expanded.tolist(),
            "entropy_gap":   gap.tolist(),
            "entropy_math":  entropy_m.tolist(),
            "entropy_general": entropy_g.tolist(),
        }


def preprocess_dataset_math_confidence(
    data_path:          str,
    output_path:        str,
    general_model_name: str   = TRAINING_MODEL_NAME,
    math_model_name:    str   = MATH_MODEL_NAME,
    gap_ratio:          float = 0.40,
    abs_ratio:          float = 0.50,
    context_window:     int   = 3,
    max_length:         int   = 512,
):
    """Apply Ours-C dual-threshold scoring to entire JSONL dataset."""

    scorer = MathConfidenceScorer(general_model_name, math_model_name)

    with open(data_path) as f:
        lines = f.readlines()

    print(f"Processing {len(lines):,} documents with Ours-C (dual-threshold) scoring...")

    results         = []
    total_tokens    = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        try:
            scored = scorer.score(
                item["text"],
                gap_ratio=gap_ratio,
                abs_ratio=abs_ratio,
                context_window=context_window,
                max_length=max_length,
            )
            results.append({
                "text":      item["text"],
                "token_ids": scored["token_ids"],
                "mask":      scored["mask"],
                "method":    "math_confidence",
            })
            total_tokens    += len(scored["token_ids"])
            selected_tokens += sum(scored["mask"])
        except Exception as e:
            print(f"  Skipping (error): {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / max(total_tokens, 1) * 100
    print(f"\nDone! Saved: {output_path}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Selected tokens: {selected_tokens:,} ({ratio:.1f}%)")


if __name__ == "__main__":
    scorer = MathConfidenceScorer()

    test_text = "Question: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?\nAnswer: There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\n#### 35"

    from transformers import AutoTokenizer as _Tok
    tokenizer = _Tok.from_pretrained(TRAINING_MODEL_NAME)

    for gap_r, abs_r in [(0.40, 0.50), (0.30, 0.40), (0.50, 0.60)]:
        result = scorer.score(test_text, gap_ratio=gap_r, abs_ratio=abs_r, context_window=3)
        tokens = [tokenizer.decode([i]) for i in result["token_ids"]]
        total = len(result["mask"])
        sel   = sum(result["mask"])
        rendered = "".join(f"[{t}]" if m else t for t, m in zip(tokens, result["mask"]))
        print(f"\n=== gap_ratio={gap_r} abs_ratio={abs_r}: {sel}/{total}={sel/total*100:.0f}% ===")
        print(rendered)
