from dataclasses import dataclass
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import get_batch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable cross entropy over final dimension."""
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = -target_logits + logsumexp
    return nll.mean()


@dataclass
class TrainingConfig:
    batch_size: int = 16
    seq_len: int = 128
    max_steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    log_jsonl: bool = True


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_ids: np.ndarray,
        val_ids: np.ndarray | None,
        config: TrainingConfig,
        run_metadata: dict | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.config = config
        self.step = 0
        self.run_metadata = run_metadata or {}
        self.best_val_loss: float | None = None
        self.best_step: int | None = None
        self.best_checkpoint_path: str | None = None

    def _checkpoint_payload(self, val_loss: float | None = None, train_loss: float | None = None) -> dict:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "meta": self.run_metadata,
        }
        if val_loss is not None:
            payload["val_loss"] = val_loss
        if train_loss is not None:
            payload["train_loss"] = train_loss
        return payload

    def _append_training_log(self, row: dict) -> None:
        if not self.config.log_jsonl:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "training_log.jsonl"
        row = {**row, "ts": datetime.now(timezone.utc).isoformat()}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            source = self.val_ids if self.val_ids is not None else self.train_ids
            for _ in range(self.config.eval_batches):
                xb, yb = get_batch(source, self.config.batch_size, self.config.seq_len, self.config.device)
                logits = self.model(xb)
                loss = cross_entropy_loss(logits, yb)
                losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses)) if losses else float("nan")

    def save_checkpoint(self, val_loss: float | None = None, train_loss: float | None = None) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step_{self.step}.pt"
        torch.save(self._checkpoint_payload(val_loss=val_loss, train_loss=train_loss), ckpt_path)
        return ckpt_path

    def save_best_checkpoint(self, val_loss: float, train_loss: float) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"
        payload = self._checkpoint_payload(val_loss=val_loss, train_loss=train_loss)
        payload["train_loss_at_best"] = train_loss
        torch.save(payload, best_path)
        self.best_val_loss = val_loss
        self.best_step = self.step
        self.best_checkpoint_path = str(best_path)
        return best_path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = int(ckpt["step"])
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])

    def train(self) -> list[dict]:
        logs = []
        self.model.train()
        pbar = tqdm(
            total=self.config.max_steps,
            initial=self.step,
            desc="Training",
            unit="step",
        )
        while self.step < self.config.max_steps:
            next_step = self.step + 1
            lr = self._lr_for_step(next_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            xb, yb = get_batch(self.train_ids, self.config.batch_size, self.config.seq_len, self.config.device)
            logits = self.model(xb)
            loss = cross_entropy_loss(logits, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            self.step += 1
            pbar.update(1)
            pbar.set_postfix(
                train_loss=f"{float(loss.item()):.4f}",
                lr=f"{lr:.2e}",
                refresh=False,
            )

            if self.step % self.config.eval_interval == 0 or self.step == 1:
                val_loss = self.evaluate()
                train_loss_f = float(loss.item())
                log_row = {
                    "step": self.step,
                    "train_loss": train_loss_f,
                    "val_loss": val_loss,
                    "lr": lr,
                }
                if self.val_ids is not None and (
                    self.best_val_loss is None or val_loss < self.best_val_loss
                ):
                    best_path = self.save_best_checkpoint(val_loss, train_loss_f)
                    log_row["new_best"] = True
                    log_row["best_path"] = str(best_path)
                    print(
                        f"best: val_loss={val_loss:.4f} -> saved {best_path}",
                        flush=True,
                    )
                logs.append(log_row)
                self._append_training_log(log_row)
                self.save_checkpoint(val_loss=val_loss, train_loss=train_loss_f)

        pbar.close()
        return logs

    def _lr_for_step(self, step: int) -> float:
        if step <= self.config.warmup_steps:
            return self.config.lr * (step / max(self.config.warmup_steps, 1))

        if self.config.max_steps <= self.config.warmup_steps:
            return self.config.min_lr

        progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.lr - self.config.min_lr) * cosine
