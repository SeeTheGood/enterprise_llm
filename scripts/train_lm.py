#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import torch

from llm.data import load_token_ids
from llm.model import TransformerConfig, TransformerLM
from llm.optim import AdamW
from llm.training import Trainer, TrainingConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-ids", required=True)
    p.add_argument("--val-ids")
    p.add_argument(
        "--tokenizer-json",
        help="Optional path to tokenizer.json used for this run (stored in checkpoint meta for generate/eval)",
    )
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--mlp-type", choices=["relu", "swiglu"], default="relu")
    p.add_argument("--use-rope", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--resume-checkpoint", help="Path to checkpoint to continue training from")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1, dest="weight_decay")
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (embeddings + block residuals)")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    args = p.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    train_ids = load_token_ids(args.train_ids, mmap=True)
    val_ids = load_token_ids(args.val_ids, mmap=True) if args.val_ids else None

    model_cfg = TransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
        use_rope=args.use_rope,
    )
    model = TransformerLM(model_cfg).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        max_steps=args.steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
    )

    run_metadata = {
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "max_seq_len": model_cfg.max_seq_len,
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
            "norm_type": model_cfg.norm_type,
            "mlp_type": model_cfg.mlp_type,
            "use_rope": model_cfg.use_rope,
        },
        "train_ids_path": args.train_ids,
        "val_ids_path": args.val_ids,
    }
    if args.tokenizer_json:
        run_metadata["tokenizer_json_path"] = str(Path(args.tokenizer_json).expanduser().resolve())
    trainer = Trainer(model, optim, train_ids, val_ids, train_cfg, run_metadata=run_metadata)
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
        print(f"Resumed from {args.resume_checkpoint} at step={trainer.step} on device={device}")
        if trainer.step >= args.steps:
            raise ValueError(
                f"--steps ({args.steps}) must be greater than resumed step ({trainer.step}) "
                "to continue training."
            )
    logs = trainer.train()
    for row in logs:
        print(row)
    last_ckpt = f"{train_cfg.checkpoint_dir}/step_{trainer.step}.pt"
    print(f"DONE: training complete at step={trainer.step}")
    print(f"DONE: last checkpoint expected at {last_ckpt}")
    if trainer.best_checkpoint_path is not None:
        print(
            f"DONE: best val checkpoint: {trainer.best_checkpoint_path} "
            f"(val_loss={trainer.best_val_loss:.4f} at step={trainer.best_step})"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
