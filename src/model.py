import math
from typing import List, Union
import torch
import torch.nn as nn


def gelu(x: torch.Tensor):
    """
    Implementation of Gaussian Error Linear Units(GELU)
    (https://arxiv.org/abs/1606.08415)
    0.5 * x * (1 + tanh[sqrt(2/π)(x + 0.044715x^3)])
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    """
    Applies layer normalization over each individual example in a batch
    as described in paper: https://arxiv.org/abs/1607.06450

    Args:
        in_ch(int): input shape from an expected input of size
        eps(float): a value used in denominator for numerical
                    stability.
                    Default: 1e-5
        affine(bool): when True, learnable per element parameters 
                      are initialized to one (for weight) and zero
                      (for bias) respectively.
                      Default: True 
    """
    def __init__(self, in_ch: Union[int, List], *, 
                 eps: float=1e-5, affine: bool=True) -> None:
        super().__init__()
        if isinstance(in_ch, int):
            shape = (in_ch, )
        else:
            shape = (in_ch[-1], )
        shape = torch.Size(shape)
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*shape))
            self.bias = nn.Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the weight parameter to 1 and bias
        parameter to 0.
        """
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        u = torch.mean(x, keepdim=True, dim=-1)
        s = torch.mean(torch.pow(x-u, 2), keepdim=True, dim=-1)
        x = (x-u) / torch.sqrt(s + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x
