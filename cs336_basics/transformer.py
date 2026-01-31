import torch
from torch import Tensor

from .attention import multihead_self_attention
from .nn import RMSNorm, SwiGLU


def transformer_block(
    x: Tensor,
    weights: dict[str, Tensor],
    num_heads: int,
    theta: float,
) -> Tensor:
    d_model = x.shape[-1]
    ln1 = RMSNorm(d_model, eps=1e-5, device=x.device, dtype=x.dtype)
    ln2 = RMSNorm(d_model, eps=1e-5, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        ln1.weight.copy_(weights["ln1.weight"])
        ln2.weight.copy_(weights["ln2.weight"])

    h = ln1(x)
    attn_out = multihead_self_attention(
        h,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        num_heads=num_heads,
        use_rope=True,
        theta=theta,
        token_positions=None,
    )
    x = x + attn_out

    h2 = ln2(x)
    ffn = SwiGLU(d_model, weights["ffn.w1.weight"].shape[0], device=x.device, dtype=x.dtype)
    with torch.no_grad():
        ffn.w1.copy_(weights["ffn.w1.weight"])
        ffn.w2.copy_(weights["ffn.w2.weight"])
        ffn.w3.copy_(weights["ffn.w3.weight"])
    ffn_out = ffn(h2)
    x = x + ffn_out
    return x


def transformer_lm(
    in_indices: Tensor,
    weights: dict[str, Tensor],
    num_layers: int,
    num_heads: int,
    theta: float,
) -> Tensor:
    device = in_indices.device
    token_emb = weights["token_embeddings.weight"].to(device=device)
    x = token_emb[in_indices]

    for layer in range(num_layers):
        prefix = f"layers.{layer}."
        block_weights = {
            "attn.q_proj.weight": weights[prefix + "attn.q_proj.weight"],
            "attn.k_proj.weight": weights[prefix + "attn.k_proj.weight"],
            "attn.v_proj.weight": weights[prefix + "attn.v_proj.weight"],
            "attn.output_proj.weight": weights[prefix + "attn.output_proj.weight"],
            "ln1.weight": weights[prefix + "ln1.weight"],
            "ffn.w1.weight": weights[prefix + "ffn.w1.weight"],
            "ffn.w2.weight": weights[prefix + "ffn.w2.weight"],
            "ffn.w3.weight": weights[prefix + "ffn.w3.weight"],
            "ln2.weight": weights[prefix + "ln2.weight"],
        }
        x = transformer_block(x, block_weights, num_heads=num_heads, theta=theta)

    d_model = x.shape[-1]
    ln_final = RMSNorm(d_model, eps=1e-5, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        ln_final.weight.copy_(weights["ln_final.weight"])
    x = ln_final(x)

    lm_head = weights["lm_head.weight"].to(device=device)
    logits = torch.matmul(x, lm_head.T)
    return logits
