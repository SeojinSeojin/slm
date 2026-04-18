# scoring/rho1_scorer.py
# RHO-1: Select tokens with high excess loss (Lin et al., NeurIPS 2024, Eq.3)
#
# Score: L_delta(x_i) = L_theta(x_i) - L_RM(x_i)
#   -> tokens where training model loss >> reference model loss
#   -> "tokens the model has not yet learned"
#
# Implementation notes:
#   - Reference model: Qwen2.5-Math-7B (same tokenizer, math-specialized)
#   - Closer to original paper intent: RM trained on high-quality math data
#   - Applied to fine-tuning (original paper: pretraining)
#   - Offline static scoring (original paper: dynamic per-batch)
#   - No format token forced inclusion

import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAINING_MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
REFERENCE_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"  # Same tokenizer, math-specialized


@torch.no_grad()
def get_token_losses(
    model:     AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device:    str,
) -> torch.Tensor:
    """
    Return per-position token-level CE loss.
    Cast logits to float32 before cross_entropy.
    """
    input_ids    = input_ids.to(device)
    outputs      = model(input_ids)
    logits       = outputs.logits[0].float()

    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, reduction="none"
    )
    return per_token_loss.cpu()


class RHO1Scorer:
    """
    RHO-1 token scoring: select tokens by excess loss.
    Eq.3: L_delta(x_i) = L_theta(x_i) - L_RM(x_i)
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

        print(f"Loading training model: {training_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(training_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.training_model = AutoModelForCausalLM.from_pretrained(
            training_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        print(f"Loading reference model: {reference_model_name}")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name, torch_dtype=dtype,
        ).to(self.device).eval()

        ref_vocab   = self.reference_model.config.vocab_size
        train_vocab = self.training_model.config.vocab_size
        if ref_vocab != train_vocab:
            print(f"  Vocab mismatch: training={train_vocab}, ref={ref_vocab} — will clamp")
        self.ref_vocab_size = ref_vocab

        print("RHO-1 scorer ready.")

    def score(
        self,
        text:         str,
        select_ratio: float = 0.77,
        max_length:   int   = 512,
    ) -> dict:
        """Eq.3: excess loss = L_training(x_i) - L_reference(x_i). Select top select_ratio%."""
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        token_ids = enc["input_ids"]

        loss_t = get_token_losses(self.training_model, token_ids, self.device)

        token_ids_ref = token_ids.clone().clamp(max=self.ref_vocab_size - 1)
        loss_r = get_token_losses(self.reference_model, token_ids_ref, self.device)

        min_len = min(len(loss_t), len(loss_r))
        loss_t  = loss_t[:min_len].numpy()
        loss_r  = loss_r[:min_len].numpy()

        excess = loss_t - loss_r

        token_ids_list = token_ids[0, :min_len].tolist()
        n_select  = max(1, int(min_len * select_ratio))
        threshold = np.sort(excess)[-n_select]
        mask      = (excess >= threshold).tolist()

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
    """Apply RHO-1 scoring to entire JSONL dataset."""
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  Free VRAM: {free_gb:.1f} GB")

    scorer = RHO1Scorer(training_model_name, reference_model_name)

    with open(data_path) as f:
        lines = f.readlines()

    print(f"Processing {len(lines):,} documents with RHO-1 scoring...")
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
    scorer = RHO1Scorer()
    text   = "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?"
    result = scorer.score(text)
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_NAME)
    tokens = [tokenizer.decode([i]) for i in result["token_ids"]]
    rendered = "".join(f"[{t}]" if m else t for t, m in zip(tokens, result["mask"]))
    total = len(result["mask"])
    sel   = sum(result["mask"])
    print(f"\n{sel}/{total} = {sel/total:.1%}")
    print(rendered)   