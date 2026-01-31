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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (no bias)."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        weight = torch.empty(
            (d_model,),
            device=device,
            dtype=dtype,
        )
        nn.init.ones_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the last dimension.

        Args:
            x: Tensor with last dim = d_model.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return torch.rsqrt(rms + self.eps) * x * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer (no bias)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        w1 = torch.empty((d_ff, d_model), device=device, dtype=dtype)
        w2 = torch.empty((d_model, d_ff), device=device, dtype=dtype)
        w3 = torch.empty((d_ff, d_model), device=device, dtype=dtype)

        sigma_w1 = (2.0 / (d_model + d_ff)) ** 0.5
        sigma_w2 = (2.0 / (d_ff + d_model)) ** 0.5
        sigma_w3 = (2.0 / (d_model + d_ff)) ** 0.5

        nn.init.trunc_normal_(w1, mean=0.0, std=sigma_w1, a=-3 * sigma_w1, b=3 * sigma_w1)
        nn.init.trunc_normal_(w2, mean=0.0, std=sigma_w2, a=-3 * sigma_w2, b=3 * sigma_w2)
        nn.init.trunc_normal_(w3, mean=0.0, std=sigma_w3, a=-3 * sigma_w3, b=3 * sigma_w3)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU to the last dimension.

        Args:
            x: Tensor with last dim = d_model.
        """
        x_w1 = torch.matmul(x, self.w1.T)
        x_w3 = torch.matmul(x, self.w3.T)
        gate = x_w1 * torch.sigmoid(x_w1)
        return torch.matmul(gate * x_w3, self.w2.T)
