#!/usr/bin/env python3
import _repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from llm.data import save_token_ids
from llm.tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="Output .npy path")
    args = p.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    text = Path(args.input).read_text(encoding="utf-8")
    ids = tok.encode(text)
    save_token_ids(args.output, ids)
    print(f"Saved {len(ids)} token ids to {args.output}")


if __name__ == "__main__":
    main()
