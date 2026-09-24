# -*- coding: utf-8 -*-
"""Core building blocks shared by the RAGFAS heads.

* :class:`RandomNodeDropout` — "DropNode" (paper Sec. 3.2/4.1): zeroes entire
  node feature vectors with probability ``p`` (0.3).  Unlike ordinary dropout
  it does not rescale activations, and it is toggled per mini-batch by a
  Bernoulli gate (legacy behaviour preserved).
* :class:`GradientReversal` — GRL (paper Sec. 3.4): identity in the forward
  pass, ``-lambda * grad`` in the backward pass with the progressive schedule
  ``lambda = 2 / (1 + exp(-alpha * i / N)) - 1`` (alpha = 10, i = iteration,
  N = maximum iterations).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RandomNodeDropout(nn.Module):
    """Zero out randomly chosen *nodes* (all feature channels of a node)."""

    def __init__(self, drop_prob: float = 0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not self.training or not enabled:
            return x
        num_rows = x.size(0)
        mask = torch.rand(num_rows, device=x.device) > self.drop_prob
        mask = mask.float().unsqueeze(1).expand_as(x)
        return x * mask


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReversal(nn.Module):
    """GRL with the progressive warm-up schedule of the paper (Sec. 3.4)."""

    def __init__(self, max_iter: int = 4000, alpha: float = 10.0, max_coeff: float = 1.0):
        super().__init__()
        self.iter_num = 0
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_coeff = max_coeff

    def schedule(self) -> float:
        """Current schedule value ``lambda = 2 / (1 + exp(-alpha * i / N)) - 1``."""
        return 2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.iter_num += 1
        coeff = self.schedule() * self.max_coeff
        return GradientReversalFunction.apply(x, coeff)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, max_iter={self.max_iter}, max_coeff={self.max_coeff}"


__all__ = ["RandomNodeDropout", "GradientReversalFunction", "GradientReversal"]