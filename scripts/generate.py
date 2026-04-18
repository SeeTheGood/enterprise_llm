#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import torch

from llm.checkpoint_utils import assert_checkpoint_compatible, config_from_checkpoint_meta
from llm.model import TransformerConfig, TransformerLM
from llm.tokenizer import BPETokenizer


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer JSON (same file used when training this checkpoint), e.g. data/tokenizer.json",
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint, e.g. checkpoints/best.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
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
    args = p.parse_args()

    # Catch copy-paste of example paths from docs (not real files on disk).
    _tok = args.tokenizer.replace("\\", "/").lower()
    if "actual/path" in _tok or "path/to/your" in _tok:
        raise SystemExit(
            "Invalid --tokenizer: that value looks like a documentation placeholder, not a real file.\n"
            "Use a path that exists on your machine, for example:\n"
            "  --tokenizer data/tokenizer.json\n"
            "Train or copy the tokenizer JSON that matches your checkpoint (same vocab as training)."
        )

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    tok_path = Path(args.tokenizer).expanduser()
    if not tok_path.is_file():
        raise SystemExit(
            f"Tokenizer file not found: {tok_path.resolve()}\n"
            "Use a real path to your BPE JSON (for example data/tokenizer.json), "
            "not a placeholder from the docs."
        )
    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path.resolve()}")

    tok = BPETokenizer.load(str(tok_path))
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = config_from_checkpoint_meta(ckpt)
    if cfg is None:
        cfg = TransformerConfig(
            vocab_size=tok.vocab_size,
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
        assert_checkpoint_compatible(cfg, ckpt)
    if tok.vocab_size > cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tok.vocab_size} exceeds model vocab_size={cfg.vocab_size}. "
            "The tokenizer must not assign ids outside the checkpoint embedding table."
        )
    if tok.vocab_size < cfg.vocab_size:
        _log(
            f"Note: tokenizer vocab_size={tok.vocab_size} < model {cfg.vocab_size} "
            f"(OK when train ids only use 0…{tok.vocab_size - 1})."
        )
    _log("Building model and loading weights…")
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    input_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    eos_id = tok.vocab.get(b"<|endoftext|>")
    _log(f"Generating up to {args.max_new_tokens} new tokens…")
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_id,
    )
    text = tok.decode(out[0].tolist())
    _log("Done.")
    print(text)


if __name__ == "__main__":
    main()
