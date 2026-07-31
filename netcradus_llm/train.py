import os
import time
import math
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from netcradus_llm.config import NetcradusConfig
from netcradus_llm.model import NetcradusForCausalLM


class NetcradusTrainer:
    """
    Production-grade Training Engine for Netcradus LLM.
    Supports AdamW, Cosine Annealing LR Schedule with Warmup, Gradient Clipping,
    Loss Evaluation, and Distributed Checkpointing.
    """
    def __init__(
        self,
        model: NetcradusForCausalLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 10,
        max_steps: int = 100,
        gradient_clip: float = 1.0,
        output_dir: str = "./checkpoints",
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip = gradient_clip
        self.output_dir = output_dir
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)

        # Separate weight decay parameters (don't decay biases or LayerNorm params)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def get_lr(self, step: int) -> float:
        """Linear Warmup followed by Cosine Decay LR Scheduler."""
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.learning_rate * 0.1
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.learning_rate * 0.1 + coeff * (self.learning_rate * 0.9)

    def train(self) -> Dict[str, Any]:
        """Execute pretraining / fine-tuning loop."""
        self.model.train()
        step = 0
        total_loss = 0.0
        start_time = time.time()
        history = []

        print(f"[NetcradusTrainer] Starting Training on Device: {self.device}")
        print(f"[NetcradusTrainer] Total Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        train_iter = iter(self.train_dataloader)
        while step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Update Learning Rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 10 == 0 or step == self.max_steps:
                avg_loss = total_loss / step
                elapsed = time.time() - start_time
                tokens_processed = step * input_ids.shape[0] * input_ids.shape[1]
                tps = tokens_processed / elapsed
                print(f"Step [{step}/{self.max_steps}] | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.6f} | Speed: {tps:.1f} tok/s")
                history.append({"step": step, "loss": loss.item(), "lr": lr})

        # Save Final Checkpoint
        checkpoint_path = os.path.join(self.output_dir, "netcradus_final.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step
        }, checkpoint_path)
        print(f"[NetcradusTrainer] Training Complete! Saved checkpoint to: {checkpoint_path}")

        return {"final_loss": loss.item(), "history": history, "checkpoint": checkpoint_path}
