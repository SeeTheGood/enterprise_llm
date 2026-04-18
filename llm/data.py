from pathlib import Path

import numpy as np
import torch


def save_token_ids(path: str | Path, token_ids: list[int]) -> None:
    arr = np.asarray(token_ids, dtype=np.int32)
    np.save(path, arr)


def load_token_ids(path: str | Path, mmap: bool = True) -> np.ndarray:
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def get_batch(
    token_ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= seq_len:
        raise ValueError("Token array must be larger than seq_len")

    starts = np.random.randint(0, len(token_ids) - seq_len - 1, size=batch_size)
    offsets = np.arange(seq_len, dtype=np.int64)
    x_idx = starts[:, None] + offsets[None, :]
    y_idx = x_idx + 1
    x = token_ids[x_idx]
    y = token_ids[y_idx]

    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t
