# -*- coding: utf-8 -*-
"""Classifier head (paper Sec. 3.3).

Each tapped graph embedding ``Z_i`` is node-wise global-average-pooled, and
``{H_1, ..., H_4}`` are stacked into ``H_input in R^{k' x 4}`` with
``k' = 128``.  The ``k'`` feature channels are treated as ``T = 128`` tokens
whose 4-dimensional feature aggregates the four levels; each token is
linearly projected to an embedding size of 8 and a multi-head transformer
encoder captures cross-token dependencies (Eq. 3).  The flattened output is
mapped to a single logit (trained with binary cross-entropy, Eq. 4; sigmoid
is applied at inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MultiLevelTokenClassifier(nn.Module):
    """Token-transformer classifier over the four level embeddings."""

    def __init__(self, tokens: int = 128, levels: int = 4, embed_dim: int = 8,
                 num_layers: int = 5, num_heads: int = 8, out_features: int = 1):
        super().__init__()
        self.project = nn.Linear(levels, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(tokens * embed_dim, out_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h``: ``[B, tokens, levels]`` stacked level embeddings (Sec. 3.3)."""
        x = self.project(h)                 # [B, T=128, embed=8]
        x = self.encoder(x)                 # sequence = 128 tokens, batch = B
        x = x.reshape(x.size(0), -1)
        return self.head(x)                 # [B, 1] raw logit


__all__ = ["MultiLevelTokenClassifier"]