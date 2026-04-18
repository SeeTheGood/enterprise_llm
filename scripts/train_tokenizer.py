#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from llm.tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to plaintext corpus")
    p.add_argument("--output", required=True, help="Path to tokenizer json")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = p.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer(special_tokens=args.special_token)
    tok.train(text, vocab_size=args.vocab_size)
    tok.save(args.output)
    print(f"Saved tokenizer with vocab_size={tok.vocab_size} to {args.output}")


if __name__ == "__main__":
    main()
