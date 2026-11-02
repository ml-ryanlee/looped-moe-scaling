"""Token data: a memory-mapped uint16 array of GPT-2 token ids, read as a fixed random
permutation of non-overlapping context-length chunks."""
import numpy as np
import torch


def chunk_order(num_tokens: int, context_length: int, seed: int = 42) -> np.ndarray:
    """Fixed random order of chunk indices; step s reads chunks [s*B, (s+1)*B).

    Each chunk also needs the token after it (the last target), hence num_tokens - 1.
    """
    return np.random.RandomState(seed=seed).permutation((num_tokens - 1) // context_length)


def get_batch(tokens: np.ndarray, order: np.ndarray, step: int, batch_size: int,
              context_length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = [], []
    for i in range(batch_size):
        start = int(order[step * batch_size + i]) * context_length
        window = np.asarray(tokens[start:start + context_length + 1], dtype=np.int64)
        inputs.append(window[:-1])
        targets.append(window[1:])
    x = torch.from_numpy(np.stack(inputs)).to(device)
    y = torch.from_numpy(np.stack(targets)).to(device)
    return x, y
