#!/usr/bin/env python3
"""Check that a tokenizer reproduces token ids in a .npy file for the given plaintext corpus."""
import _repo_bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

from llm.tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Encode --text with --tokenizer and compare to --token-ids. "
            "Use this after retraining a tokenizer to confirm it matches the run that built train_ids.npy."
        )
    )
    p.add_argument("--tokenizer", required=True, help="tokenizer.json to test")
    p.add_argument("--text", required=True, help="Same plaintext file used to build token-ids (e.g. data/train.txt)")
    p.add_argument("--token-ids", required=True, dest="token_ids", help="Reference .npy (e.g. data/train_ids.npy)")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="If >0, only compare the first N token positions (faster for huge files)",
    )
    args = p.parse_args()

    tok_path = Path(args.tokenizer).expanduser()
    if not tok_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tok_path.resolve()}")

    text_path = Path(args.text).expanduser()
    if not text_path.is_file():
        raise SystemExit(f"Text not found: {text_path.resolve()}")

    ids_path = Path(args.token_ids).expanduser()
    if not ids_path.is_file():
        raise SystemExit(f"token-ids not found: {ids_path.resolve()}")

    tok = BPETokenizer.load(str(tok_path))
    print(f"Tokenizer vocab_size={tok.vocab_size}", file=sys.stderr)

    text = text_path.read_text(encoding="utf-8")
    ref = np.load(str(ids_path), mmap_mode="r")
    n_ref = len(ref)
    print(f"Reference array length={n_ref}", file=sys.stderr)

    print("Encoding corpus (may take a while)…", file=sys.stderr)
    enc = tok.encode(text)
    n_enc = len(enc)

    if n_enc != n_ref:
        print(
            f"MISMATCH: encoded length {n_enc} != reference length {n_ref}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.max_tokens > 0:
        n = min(args.max_tokens, n_enc)
    else:
        n = n_enc

    mismatches = 0
    first_bad = None
    for i in range(n):
        if enc[i] != int(ref[i]):
            mismatches += 1
            if first_bad is None:
                first_bad = (i, enc[i], int(ref[i]))
    if mismatches:
        print(
            f"MISMATCH: {mismatches} differing positions in first {n} tokens. "
            f"First at index {first_bad[0]}: got id {first_bad[1]}, ref {first_bad[2]}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"OK: first {n} token ids match reference (full length {n_enc}).")


if __name__ == "__main__":
    main()
