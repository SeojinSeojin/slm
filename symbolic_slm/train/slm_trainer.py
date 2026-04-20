# train/slm_trainer.py

import os
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


# ============================================================
# Dataset
# ============================================================

class SLMDataset(Dataset):
    """
    CT mode  : mask 무시, 전체 토큰 loss
    SLM mode : mask=True 토큰에만 loss
    """

    def __init__(self, data_path: str, max_length: int = 384, ct_mode: bool = False):
        self.max_length = max_length
        self.ct_mode    = ct_mode
        self.data       = []

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                ids  = item["token_ids"][:max_length]
                mask = item.get("mask", [True] * len(ids))[:max_length]
                if len(ids) < 8:
                    continue
                if ct_mode:
                    mask = [True] * len(ids)
                self.data.append({"input_ids": ids, "mask": mask})

        mode_str = "CT (전체)" if ct_mode else "SLM (선별)"
        selected = sum(sum(d["mask"]) for d in self.data)
        total    = sum(len(d["input_ids"]) for d in self.data)
        print(f"✅ {len(self.data):,}개 문서 [{mode_str}]"
              f" | 선택 토큰: {selected:,}/{total:,} ({selected/max(total,1):.1%})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# Collate
# ============================================================

def collate_fn(batch, pad_token_id: int, max_length: int):
    max_len = min(max(len(b["input_ids"]) for b in batch), max_length)

    input_ids_list      = []
    attention_mask_list = []
    loss_mask_list      = []

    for b in batch:
        ids    = b["input_ids"][:max_len]
        mask   = b["mask"][:max_len]
        length = len(ids)
        pad    = max_len - length

        input_ids_list.append(     ids  + [pad_token_id] * pad)
        attention_mask_list.append([1]  * length + [0]   * pad)
        loss_mask_list.append(     mask + [False] * pad)

    return {
        "input_ids":      torch.tensor(input_ids_list,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "loss_mask":      torch.tensor(loss_mask_list,      dtype=torch.bool),
    }


# ============================================================
# Loss
# ============================================================

def compute_loss(
    logits:    torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    mask=True 토큰에만 CE loss.
    OOM 방지: 선택 토큰만 추출 후 CE 계산.
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
        max_length:       int   = 512,
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
            torch_dtype=self.dtype,  # fixed: was `dtype=`
        ).to(self.device)

        self.model.gradient_checkpointing_enable()
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        print("✅ 준비 완료")

    def train(self, dataset: SLMDataset, experiment_name: str) -> str:

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
            pin_memory=(self.device == "cuda"),
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
        self.optimizer.zero_grad()

        pbar = tqdm(total=total_optimizer_steps, desc=f"[{experiment_name}]",
                    unit="step", dynamic_ncols=True)

        while optimizer_step < total_optimizer_steps:
            for batch in dataloader:
                if optimizer_step >= total_optimizer_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype,
                                    enabled=(self.device == "cuda")):
                    outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                    loss    = compute_loss(outputs.logits, input_ids, loss_mask)
                    loss    = loss / self.grad_accum_steps

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
                    self.optimizer.zero_grad()
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

        pbar.close()

        history_path = os.path.join(self.save_dir, f"{experiment_name}_loss.json")
        with open(history_path, "w") as f:
            json.dump(loss_history, f, indent=2)
        print(f"  📊 {history_path}")

        # Save final checkpoint once (removed duplicate mid-loop save)
        self._save_checkpoint(experiment_name, optimizer_step, final=True)
        print(f"✅ [{experiment_name}] 완료\n")
        return history_path

    def _save_checkpoint(self, name: str, step: int, final: bool = False):
        suffix = "final" if final else f"step{step}"
        path   = os.path.join(self.save_dir, f"{name}_{suffix}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 {path}")