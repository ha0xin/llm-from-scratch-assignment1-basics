import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    num_possible = len(dataset) - context_length
    assert num_possible > 0, "Dataset too small for context length"

    starts = np.random.randint(0, num_possible, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts])

    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t
