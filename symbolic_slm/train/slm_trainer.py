# train/slm_trainer.py

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import bitsandbytes as bnb

# ============================================================
# Dataset
# ============================================================

class SLMDataset(Dataset):
    """
    CT mode           : mask 무시, 전체 토큰 loss
    SLM mode          : mask=True 토큰에만 loss
    answer_weight mode: 전체 토큰 loss + #### 이후 answer 토큰에 가중치
    """

    def __init__(
        self,
        data_path:     str,
        max_length:    int   = 384,
        ct_mode:       bool  = False,
        answer_weight: float = 1.0,   # > 1.0 → weight answer tokens higher
    ):
        self.max_length    = max_length
        self.ct_mode       = ct_mode
        self.answer_weight = answer_weight
        self.data          = []

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                ids  = item["token_ids"][:max_length]
                mask = item.get("mask", [True] * len(ids))[:max_length]
                if len(ids) < 8:
                    continue
                if ct_mode:
                    mask = [True] * len(ids)

                # Build per-token weight using text to find #### boundary
                if answer_weight != 1.0:
                    text     = item.get("text", "")
                    boundary = _find_answer_boundary(ids, text)
                    weights  = [answer_weight if i >= boundary else 1.0
                                for i in range(len(ids))][:max_length]
                else:
                    weights = [1.0] * len(ids)

                self.data.append({
                    "input_ids": ids,
                    "mask":      mask,
                    "weights":   weights,
                })

        mode_str = "CT (전체)" if ct_mode else (
            f"Answer-weighted (w={answer_weight})" if answer_weight != 1.0 else "SLM (선별)"
        )
        selected = sum(sum(d["mask"]) for d in self.data)
        total    = sum(len(d["input_ids"]) for d in self.data)
        print(f"✅ {len(self.data):,}개 문서 [{mode_str}]"
              f" | 선택 토큰: {selected:,}/{total:,} ({selected/max(total,1):.1%})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _find_answer_boundary(token_ids: list, text: str) -> int:
    """
    Find the token index where the answer region starts (after ####).
    Uses character-level position of '####' in text, then maps to token index
    via approximate character-to-token ratio.
    Falls back to last 15% of tokens if #### not found.
    """
    idx = text.find("####")
    if idx == -1:
        return int(len(token_ids) * 0.85)

    ratio    = idx / max(len(text), 1)
    boundary = int(len(token_ids) * ratio)
    return max(0, min(boundary, len(token_ids) - 1))


# ============================================================
# Collate
# ============================================================

def collate_fn(batch, pad_token_id: int, max_length: int):
    max_len = min(max(len(b["input_ids"]) for b in batch), max_length)

    input_ids_list      = []
    attention_mask_list = []
    loss_mask_list      = []
    weights_list        = []

    for b in batch:
        ids     = b["input_ids"][:max_len]
        mask    = b["mask"][:max_len]
        weights = b.get("weights", [1.0] * len(ids))[:max_len]
        length  = len(ids)
        pad     = max_len - length

        input_ids_list.append(     ids     + [pad_token_id] * pad)
        attention_mask_list.append([1]     * length + [0]   * pad)
        loss_mask_list.append(     mask    + [False] * pad)
        weights_list.append(       weights + [1.0]   * pad)

    return {
        "input_ids":      torch.tensor(input_ids_list,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "loss_mask":      torch.tensor(loss_mask_list,      dtype=torch.bool),
        "weights":        torch.tensor(weights_list,        dtype=torch.float),
    }


# ============================================================
# Loss
# ============================================================

def compute_loss(
    logits:    torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    weights:   torch.Tensor = None,
) -> torch.Tensor:
    """
    CE loss on mask=True tokens, optionally weighted per token.
    weights: float tensor same shape as input_ids (1.0 = normal, >1.0 = emphasized).
    """
    shift_logits = logits[:,   :-1, :].contiguous()
    shift_labels = input_ids[:, 1: ].contiguous()
    shift_mask   = loss_mask[:, 1: ].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask   = shift_mask.view(-1)

    selected_logits = flat_logits[flat_mask]
    selected_labels = flat_labels[flat_mask]

    if selected_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    if weights is not None:
        shift_weights    = weights[:, 1:].contiguous()
        flat_weights     = shift_weights.view(-1)
        selected_weights = flat_weights[flat_mask]
        per_token_loss   = nn.functional.cross_entropy(
            selected_logits, selected_labels, reduction="none"
        )
        return (per_token_loss * selected_weights).mean()

    return nn.functional.cross_entropy(selected_logits, selected_labels)


# ============================================================
# Trainer
# ============================================================

class SLMTrainer:
    def __init__(
        self,
        model_name:       str   = "Qwen/Qwen2.5-1.5B",
        save_dir:         str   = "./checkpoints",
        learning_rate:    float = 5e-5,
        batch_size:       int   = 16,
        grad_accum_steps: int   = 4,
        max_steps:        int   = 8000,
        warmup_steps:     int   = 200,
        max_length:       int   = 384,
        log_interval:     int   = 20,
    ):
        self.save_dir         = save_dir
        self.batch_size       = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps        = max_steps
        self.warmup_steps     = warmup_steps
        self.max_length       = max_length
        self.log_interval     = log_interval

        os.makedirs(save_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype      = torch.bfloat16
            self.use_scaler = False
            print(f"🖥️  {self.device} | bfloat16")
        elif self.device == "cuda":
            self.dtype      = torch.float16
            self.use_scaler = True
            print(f"🖥️  {self.device} | float16 + GradScaler")
        else:
            self.dtype      = torch.float32
            self.use_scaler = False
            print(f"🖥️  cpu | float32")

        print(f"📥 모델 로딩: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.model.gradient_checkpointing_enable()
        self.model.train()

        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        print("✅ 준비 완료")

    def train(self, dataset: SLMDataset, experiment_name: str) -> str:
        # 실험 시작 전 캐시 비우기
        torch.cuda.empty_cache()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(
                b,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.max_length,
            ),
            num_workers=0,
            pin_memory=False,
        )

        total_optimizer_steps = self.max_steps // self.grad_accum_steps
        effective_log         = max(1, min(self.log_interval, total_optimizer_steps // 10))
        effective_warmup      = min(self.warmup_steps, max(1, total_optimizer_steps // 10))

        if effective_warmup != self.warmup_steps:
            print(f"  warmup {self.warmup_steps} → {effective_warmup} (total={total_optimizer_steps})")

        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=effective_warmup,
            num_training_steps=total_optimizer_steps,
        )

        print(f"\n🚀 [{experiment_name}]"
              f" opt_steps={total_optimizer_steps}"
              f" | batch={self.batch_size}"
              f" | accum={self.grad_accum_steps}"
              f" | eff_batch={self.batch_size * self.grad_accum_steps}"
              f" | max_len={self.max_length}")

        global_step    = 0
        optimizer_step = 0
        running_loss   = 0.0
        loss_history   = []
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(total=total_optimizer_steps, desc=f"[{experiment_name}]",
                    unit="step", dynamic_ncols=True)

        while optimizer_step < total_optimizer_steps:
            for batch in dataloader:
                if optimizer_step >= total_optimizer_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                weights   = batch["weights"].to(self.device) if "weights" in batch else None

                with torch.autocast(device_type=self.device, dtype=self.dtype,
                                    enabled=(self.device == "cuda")):
                    # outputs 전체 대신 logits만 참조 → 나머지 hidden states 즉시 해제
                    logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                    loss   = compute_loss(logits, input_ids, loss_mask, weights)
                    loss   = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward() if self.use_scaler else loss.backward()

                running_loss += loss.item() * self.grad_accum_steps
                global_step  += 1

                if global_step % self.grad_accum_steps == 0:
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    if self.use_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)  # gradient tensor 완전 해제
                    optimizer_step += 1

                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{loss.item() * self.grad_accum_steps:.3f}",
                        "lr":   f"{lr:.1e}",
                    })
                    pbar.update(1)

                    if optimizer_step % effective_log == 0:
                        avg_loss = running_loss / effective_log
                        print(f"\n  [{experiment_name}] {optimizer_step}/{total_optimizer_steps}"
                              f" | loss={avg_loss:.4f} | lr={lr:.2e}")
                        loss_history.append({"step": optimizer_step, "loss": avg_loss})
                        running_loss = 0.0

            # 에폭 끝마다 단편화된 캐시 정리
            torch.cuda.empty_cache()

        pbar.close()

        history_path = os.path.join(self.save_dir, f"{experiment_name}_loss.json")
        with open(history_path, "w") as f:
            json.dump(loss_history, f, indent=2)
        print(f"  📊 {history_path}")

        self._save_checkpoint(experiment_name, optimizer_step, final=True)
        print(f"✅ [{experiment_name}] 완료\n")
        return history_path

    def _save_checkpoint(self, name: str, step: int, final: bool = False):
        suffix = "final" if final else f"step{step}"
        path   = os.path.join(self.save_dir, f"{name}_{suffix}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 {path}")