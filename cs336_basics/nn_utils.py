import torch
from torch import Tensor
from collections.abc import Iterable


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # inputs: (batch, vocab)
    logsumexp = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p is not None and p.grad is not None]
    if not grads:
        return
    device = grads[0].device
    total_norm = torch.zeros((), device=device)
    for g in grads:
        total_norm = total_norm + g.detach().pow(2).sum()
    total_norm = torch.sqrt(total_norm)
    eps = 1e-6
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
