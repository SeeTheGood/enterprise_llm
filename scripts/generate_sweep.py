#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import torch

from llm.checkpoint_utils import assert_checkpoint_compatible, config_from_checkpoint_meta
from llm.model import TransformerConfig, TransformerLM
from llm.tokenizer import BPETokenizer


PRESETS = {
    "strict": {"temperature": 0.4, "top_k": 10, "top_p": 0.75, "repetition_penalty": 1.25},
    "balanced": {"temperature": 0.55, "top_k": 20, "top_p": 0.85, "repetition_penalty": 1.15},
    "creative": {"temperature": 0.8, "top_k": 40, "top_p": 0.92, "repetition_penalty": 1.08},
    "wild": {"temperature": 1.0, "top_k": 80, "top_p": 0.97, "repetition_penalty": 1.02},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer JSON (same as training), e.g. data/tokenizer.json",
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--mlp-type", choices=["relu", "swiglu"], default="relu")
    p.add_argument("--use-rope", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    p.add_argument(
        "--presets",
        default="strict,balanced,creative,wild",
        help="Comma-separated list from: strict,balanced,creative,wild",
    )
    args = p.parse_args()

    _tok = args.tokenizer.replace("\\", "/").lower()
    if "actual/path" in _tok or "path/to/your" in _tok:
        raise SystemExit(
            "Invalid --tokenizer: documentation placeholder, not a real path. "
            "Use something that exists, e.g. data/tokenizer.json"
        )

    names = [n.strip() for n in args.presets.split(",") if n.strip()]
    for n in names:
        if n not in PRESETS:
            raise ValueError(f"Unknown preset '{n}'. Choose from: {','.join(PRESETS)}")

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
            "Use a real path to your BPE JSON (e.g. data/tokenizer.json)."
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
            f"Tokenizer vocab_size={tok.vocab_size} exceeds model vocab_size={cfg.vocab_size}."
        )
    if tok.vocab_size < cfg.vocab_size:
        _log(
            f"Note: tokenizer vocab_size={tok.vocab_size} < model {cfg.vocab_size} "
            f"(OK if training ids only used token ids 0…{tok.vocab_size - 1})."
        )
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    base_input = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    eos_id = tok.vocab.get(b"<|endoftext|>")

    for name in names:
        s = PRESETS[name]
        out = model.generate(
            input_ids=base_input.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=s["temperature"],
            top_k=s["top_k"],
            top_p=s["top_p"],
            repetition_penalty=s["repetition_penalty"],
            eos_token_id=eos_id,
        )
        text = tok.decode(out[0].tolist())
        print("\n" + "=" * 24)
        print(f"PRESET: {name}")
        print(
            f"temperature={s['temperature']} top_k={s['top_k']} "
            f"top_p={s['top_p']} repetition_penalty={s['repetition_penalty']}"
        )
        print("-" * 24)
        print(text)


if __name__ == "__main__":
    main()
