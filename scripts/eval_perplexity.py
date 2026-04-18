#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse

import numpy as np
import torch

from llm.checkpoint_utils import assert_checkpoint_compatible, config_from_checkpoint_meta
from llm.model import TransformerConfig, TransformerLM
from llm.training import cross_entropy_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--token-ids", required=True)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Required if checkpoint has no embedded model_config; ignored when meta is present.",
    )
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--mlp-type", choices=["relu", "swiglu"], default="relu")
    p.add_argument("--use-rope", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    p.add_argument("--eval-batches", type=int, default=100, help="Number of random eval batches")
    p.add_argument("--batch-size", type=int, default=8)
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = config_from_checkpoint_meta(ckpt)
    if cfg is None:
        if args.vocab_size is None:
            raise ValueError(
                "--vocab-size is required when the checkpoint has no embedded model_config."
            )
        cfg = TransformerConfig(
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
    if ckpt.get("meta", {}).get("model_config"):
        assert_checkpoint_compatible(
            cfg,
            ckpt,
            hint="Use generation/eval args that exactly match training config for this checkpoint.",
        )
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seq_len = cfg.max_seq_len
    ids = np.load(args.token_ids)
    if len(ids) <= seq_len + 1:
        raise ValueError(
            f"Validation token array must be larger than max_seq_len+1. "
            f"Got len(ids)={len(ids)}, max_seq_len={seq_len}"
        )

    losses = []
    with torch.no_grad():
        max_start = len(ids) - seq_len - 1
        for _ in range(args.eval_batches):
            starts = np.random.randint(0, max_start, size=args.batch_size)
            x_np = np.stack([ids[s : s + seq_len] for s in starts])
            y_np = np.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
            x = torch.tensor(x_np, dtype=torch.long, device=device)
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            logits = model(x)
            losses.append(cross_entropy_loss(logits, y).item())

    loss = float(np.mean(losses))
    ppl = float(torch.exp(torch.tensor(loss)).item())
    print(
        {
            "loss": loss,
            "perplexity": ppl,
            "eval_batches": args.eval_batches,
            "batch_size": args.batch_size,
        }
    )


if __name__ == "__main__":
    main()
