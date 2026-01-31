import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def apply_rope(x: Tensor, theta: float, token_positions: Tensor) -> Tensor:
    d_k = x.shape[-1]
    assert d_k % 2 == 0, "RoPE requires even head dimension"

    device = x.device
    token_positions = token_positions.to(device=device)

    inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=x.dtype) / d_k)
    pos = token_positions.to(dtype=x.dtype).unsqueeze(-1)
    angles = pos * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.stack([out_even, out_odd], dim=-1)
    out = out.reshape(x.shape)
    return out


def multihead_self_attention(
    x: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    num_heads: int,
    use_rope: bool = False,
    theta: float | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = torch.matmul(x, q_proj_weight.T)
    k = torch.matmul(x, k_proj_weight.T)
    v = torch.matmul(x, v_proj_weight.T)

    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    if use_rope:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if token_positions.dim() == 1:
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.unsqueeze(1)
        q = apply_rope(q, theta, token_positions)
        k = apply_rope(k, theta, token_positions)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    out = torch.matmul(out, o_proj_weight.T)
    return out
