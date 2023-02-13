import math
import torch
import torch.nn as nn


def gelu(x):
    # Implementation of Gaussian Error Linear Units(GELU)
    # (https://arxiv.org/abs/1606.08415)
    # 0.5 * x * (1 + tanh[sqrt(2/π)(x + 0.044715x^3)])
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, ndim, *, dim=-1, eps=1e-5, bias=True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        u = torch.mean(x, keepdim=True, dim=self.dim)
        s = torch.mean(torch.pow(x-u, 2), keepdim=True, dim=self.dim)
        x = (x-u) / torch.sqrt(s + self.eps)
        if self.bias is not None:
            return x * self.weight + self.bias
        return x * self.weight

