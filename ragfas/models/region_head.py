# -*- coding: utf-8 -*-
"""Region Prediction Network — RPN (paper Sec. 3.5).

A self-supervised auxiliary head classifies every node into its anatomical
region (seven classes over the 468 MediaPipe landmarks).  The head is a pair
of graph convolutions over ``Z1`` followed by a transformer; per the paper it
is "composed of ChebConv layers and a transformer", while the legacy released
checkpoints use GraphSAGE layers (configurable via ``conv``; parameter names
differ, so released checkpoints require ``conv="sage"``).

Due to GPU memory limits only ``num_samples`` (=64) of the 468 nodes are
randomly sampled per batch when training this head (paper Sec. 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import ChebConv, GraphSAGE


@dataclass
class RegionHeadConfig:
    embed_dim: int = 64
    num_layers: int = 10
    num_heads: int = 8
    num_classes: int = 7
    num_samples: int = 64
    conv: str = "cheb"   # "cheb" (paper text) | "sage" (legacy released checkpoints)


class RegionTransformerHead(nn.Module):
    """Transformer head of the RPN: embedding 64, 10 layers, 8 heads."""

    def __init__(self, sequence_len: int, embed_dim: int, num_layers: int,
                 num_heads: int, out_features: int):
        super().__init__()
        self.project = nn.Linear(1, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(sequence_len * embed_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[num_samples, sequence_len, 1]`` selected node features."""
        x = self.project(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


class RegionPredictionNetwork(nn.Module):
    """RPN (paper Sec. 3.5): conv layers on ``Z1`` + transformer -> region logits."""

    def __init__(self, hidden_channels: int, config: RegionHeadConfig = RegionHeadConfig()):
        super().__init__()
        torch.manual_seed(12345)  # deterministic init, as in the original code
        self.conv_type = config.conv
        if config.conv == "sage":
            # Legacy checkpoints: conv1_node_class / conv2_node_class GraphSAGE.
            self.conv1_node_class = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
            self.conv2_node_class = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
        else:
            self.cheb1 = ChebConv(hidden_channels, hidden_channels, 2)
            self.cheb2 = ChebConv(hidden_channels, hidden_channels, 2)
        self.head = RegionTransformerHead(
            sequence_len=hidden_channels,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            out_features=config.num_classes,
        )

    def forward(self, z1: torch.Tensor, edge_index: torch.Tensor,
                node_idx: torch.Tensor) -> torch.Tensor:
        """``z1``: node features of the whole batch; ``node_idx``: sampled nodes.

        Returns region logits ``[num_samples, 7]`` for the sampled nodes.
        """
        if self.conv_type == "sage":
            h = self.conv1_node_class(z1, edge_index)
            h = h.relu()
            h = self.conv2_node_class(h, edge_index)
        else:
            h = self.cheb1(z1, edge_index)
            h = h.relu()
            h = self.cheb2(h, edge_index)
            h = h.relu()

        h = h[node_idx]                       # [num_samples, hidden]
        h = torch.unsqueeze(h, dim=2)         # [num_samples, hidden, 1]
        return self.head(h)


__all__ = ["RegionPredictionNetwork", "RegionHeadConfig", "RegionTransformerHead"]