#!/usr/bin/env python3
"""
Remove derived LM artifacts so you can redo tokenizer → tokenize → train from a clean slate.

By default keeps plaintext under data/ (train.txt, val.txt, corpus.txt). Use --also-text to remove those too.
"""
import _repo_bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files (default is dry-run only)",
    )
    p.add_argument(
        "--also-text",
        action="store_true",
        help="Also delete data/train.txt, data/val.txt, data/corpus.txt (destructive)",
    )
    args = p.parse_args()

    data = root / "data"
    ckpt_dir = root / "checkpoints"

    to_remove: list[Path] = []

    for pattern in ("tokenizer.json", "train_ids.npy", "val_ids.npy"):
        f = data / pattern
        if f.is_file():
            to_remove.append(f)

    if args.also_text:
        for name in ("train.txt", "val.txt", "corpus.txt"):
            f = data / name
            if f.is_file():
                to_remove.append(f)

    if ckpt_dir.is_dir():
        for f in sorted(ckpt_dir.rglob("*")):
            if f.is_file():
                to_remove.append(f)

    if not to_remove:
        print("Nothing to remove (expected artifacts already absent).", file=sys.stderr)
        return

    print("Paths to remove:", file=sys.stderr)
    for f in to_remove:
        print(f"  {f}", file=sys.stderr)

    if not args.yes:
        print(
            "\nDry-run only. Re-run with --yes to delete these files.",
            file=sys.stderr,
        )
        return

    for f in to_remove:
        f.unlink()
        print(f"Removed {f}")


if __name__ == "__main__":
    main()
