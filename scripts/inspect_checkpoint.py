#!/usr/bin/env python3
"""Print meta fields from a training checkpoint (model config, data paths, tokenizer path if saved)."""
import _repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path, help="Path to .pt file")
    args = p.parse_args()
    path = args.checkpoint.expanduser()
    if not path.is_file():
        raise SystemExit(f"Not found: {path.resolve()}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = ckpt.get("meta") or {}
    out = {
        "checkpoint": str(path.resolve()),
        "meta": meta,
        "model_config": meta.get("model_config"),
        "train_ids_path": meta.get("train_ids_path"),
        "val_ids_path": meta.get("val_ids_path"),
        "tokenizer_json_path": meta.get("tokenizer_json_path"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
