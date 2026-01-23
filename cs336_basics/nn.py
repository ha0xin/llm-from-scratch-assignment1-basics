from __future__ import annotations

import torch
from torch import nn


class Linear(nn.Module):
    """Linear layer without bias: y = W x.

    Notes:
        - Weight is stored as shape (out_features, in_features) to match the handout.
        - Initialization uses truncated normal per assignment spec.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype,
        )
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transform.

        Args:
            x: Tensor with last dim = in_features.
        """
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # weight.T: (in_features, out_features)
        # output: (..., out_features)
        return torch.matmul(x, self.weight.T)


class Embedding(nn.Module):
    """Embedding layer that looks up token ids in a weight matrix."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype,
        )
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings for token ids.

        Args:
            token_ids: Long tensor of shape (batch_size, sequence_length)
        """
        return self.weight[token_ids]
