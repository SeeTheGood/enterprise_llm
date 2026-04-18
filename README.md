# enterprise_llm

LLM-style Transformer LM implementation from scratch.

## What is implemented

- Byte-level BPE tokenizer with GPT-2 regex pre-tokenization
- Transformer LM (embeddings, causal attention, MLP blocks, generation)
- Stable cross-entropy and AdamW optimizer
- Training loop with evaluation and checkpoint save/load
- CLI scripts for tokenizer train, corpus tokenization, LM training, perplexity eval, and generation
- Unit + smoke-ish tests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
```

## End-to-end commands

```bash
python scripts/train_tokenizer.py --input data/train.txt --output data/tokenizer.json --vocab-size 4096 --special-token "<|endoftext|>"
python scripts/tokenize_corpus.py --tokenizer data/tokenizer.json --input data/train.txt --output data/train_ids.npy
python scripts/tokenize_corpus.py --tokenizer data/tokenizer.json --input data/val.txt --output data/val_ids.npy
python scripts/train_lm.py --train-ids data/train_ids.npy --val-ids data/val_ids.npy --vocab-size 4096 --max-seq-len 128 --steps 500 --device cpu
python scripts/eval_perplexity.py --checkpoint checkpoints/step_500.pt --token-ids data/val_ids.npy --vocab-size 4096 --max-seq-len 128
python scripts/generate.py --tokenizer data/tokenizer.json --checkpoint checkpoints/step_500.pt --prompt "Once upon a time" --max-new-tokens 100
```

## Notes

- Checkpoints are saved in `checkpoints/` as `step_*.pt`.
- Current implementation prioritizes clarity and assignment alignment over maximum throughput.
