import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find chunk boundaries without splitting a special token."""
    assert isinstance(split_special_token, bytes), "Must represent special token as bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(file_size // desired_num_chunks, 1)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        initial_position = boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(boundaries))


@dataclass
class BPEModelState:
    vocab: dict[str, int]
    id_to_token: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]


class BPETokenizer:
    """Byte-level BPE tokenizer with regex pre-tokenization."""

    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []
        self.vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.id_to_token: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []

        next_id = 256
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            self.vocab[b] = next_id
            self.id_to_token[next_id] = b
            next_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _compile_special_regex(self) -> re.Pattern | None:
        if not self.special_tokens:
            return None
        escaped = [re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)]
        return re.compile("(" + "|".join(escaped) + ")")

    def _iter_non_special_segments(self, text: str) -> Iterable[tuple[bool, str]]:
        special_re = self._compile_special_regex()
        if special_re is None:
            yield False, text
            return
        start = 0
        for m in special_re.finditer(text):
            if m.start() > start:
                yield False, text[start : m.start()]
            yield True, m.group(0)
            start = m.end()
        if start < len(text):
            yield False, text[start:]

    def pretokenize_counts(self, text: str) -> Counter[bytes]:
        counts: Counter[bytes] = Counter()
        for is_special, chunk in self._iter_non_special_segments(text):
            if is_special:
                counts[chunk.encode("utf-8")] += 1
                continue
            for m in re.finditer(PAT, chunk):
                piece = m.group(0)
                if piece:
                    counts[piece.encode("utf-8")] += 1
        return counts

    def _word_to_symbols(self, word: bytes) -> list[bytes]:
        return [bytes([b]) for b in word]

    def _merge_pair_in_symbols(self, symbols: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def train(self, text: str, vocab_size: int, verbose: bool = True, log_every: int = 50) -> None:
        if vocab_size < len(self.vocab):
            raise ValueError("Target vocab_size is smaller than current vocabulary")

        word_counts = self.pretokenize_counts(text)
        tokenized_words: dict[bytes, list[bytes]] = {
            w: self._word_to_symbols(w) for w in word_counts.keys() if w not in self.vocab
        }
        start_vocab_size = len(self.vocab)
        target_new_merges = max(vocab_size - start_vocab_size, 0)
        if verbose:
            print(
                f"[bpe] starting training: pre_tokens={len(word_counts)}, "
                f"start_vocab={start_vocab_size}, target_vocab={vocab_size}",
                flush=True,
            )

        pbar = None
        if verbose and target_new_merges > 0:
            pbar = tqdm(total=target_new_merges, desc="BPE merges", unit="merge")

        while len(self.vocab) < vocab_size:
            pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            for word, symbols in tokenized_words.items():
                if len(symbols) < 2:
                    continue
                freq = word_counts[word]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            max_count = max(pair_counts.values())
            best_pairs = [pair for pair, cnt in pair_counts.items() if cnt == max_count]
            best_pair = max(best_pairs)
            merged_token = best_pair[0] + best_pair[1]

            if merged_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged_token] = new_id
                self.id_to_token[new_id] = merged_token
                self.merges.append(best_pair)
                if pbar is not None:
                    pbar.update(1)
                if verbose:
                    completed_merges = len(self.vocab) - start_vocab_size
                    if completed_merges == 1 or completed_merges % log_every == 0:
                        print(
                            f"[bpe] merge {completed_merges}/{target_new_merges} "
                            f"vocab={len(self.vocab)} max_pair_count={max_count}",
                            flush=True,
                        )

            for word, symbols in tokenized_words.items():
                tokenized_words[word] = self._merge_pair_in_symbols(symbols, best_pair)
        if pbar is not None:
            pbar.close()
        if verbose:
            print(f"[bpe] done: final_vocab={len(self.vocab)}", flush=True)

    def _apply_merges(self, symbols: list[bytes]) -> list[bytes]:
        for pair in self.merges:
            symbols = self._merge_pair_in_symbols(symbols, pair)
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for is_special, chunk in self._iter_non_special_segments(text):
            if is_special:
                ids.append(self.vocab[chunk.encode("utf-8")])
                continue
            for m in re.finditer(PAT, chunk):
                piece = m.group(0)
                if not piece:
                    continue
                symbols = [bytes([b]) for b in piece.encode("utf-8")]
                symbols = self._apply_merges(symbols)
                ids.extend(self.vocab[s] for s in symbols)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.id_to_token[i] for i in token_ids]
        return b"".join(tokens).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        state = {
            "vocab": {k.hex(): v for k, v in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        p = Path(path).expanduser()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tokenizer file not found: {p.resolve()}\n"
                "Use a real path to your BPE JSON (e.g. data/tokenizer.json in the repo root), "
                "not a placeholder like /actual/path/to/… from the docs."
            ) from e
        tok = cls(special_tokens=raw.get("special_tokens", []))
        tok.vocab = {bytes.fromhex(k): v for k, v in raw["vocab"].items()}
        tok.id_to_token = {v: k for k, v in tok.vocab.items()}
        tok.merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw["merges"]]
        return tok
