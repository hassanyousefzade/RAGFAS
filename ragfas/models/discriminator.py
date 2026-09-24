# -*- coding: utf-8 -*-
"""Domain Discriminator Graph — DG (paper Sec. 3.4, Eq. 6-7).

Three interchangeable strategies (Strategy pattern) selected by
``DiscriminatorConfig.pooling``:

* ``"node"`` (default, paper-faithful): DG predicts the domain of *every
  node* from ``Z1`` and ``A`` (Eq. 6) — GraphSAGE layers followed by a
  per-node dense layer and softmax; the focal adversarial loss (Eq. 7) is
  averaged over all nodes of the batch.
* ``"graph"``: the behaviour of the original research script — GraphSAGE
  layers over node features, then a graph-level mean-pool before the dense
  layer (produces one domain prediction per graph).  Parameter names are
  identical to the node variant, so released checkpoints load in both modes.
* ``"global_mlp"``: the "global" ablation of paper Table 3 — features are
  mean-pooled first, then passed through a GRL + plain MLP discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import global_mean_pool

from ragfas.models.layers import GradientReversal, RandomNodeDropout


@dataclass
class DiscriminatorConfig:
    pooling: str = "node"      # "node" (paper) | "graph" (legacy ckpts) | "global_mlp" (Table 3)
    num_sage_layers: int = 4
    dropnode_prob: float = 0.3


class DomainDiscriminator(nn.Module):
    """Node-level domain discriminator with a GRL on the generator path."""

    def __init__(self, hidden_channels: int, num_domains: int, max_iter: int = 4000,
                 alpha: float = 10.0, max_coeff: float = 1.0,
                 config: DiscriminatorConfig = DiscriminatorConfig()):
        super().__init__()
        torch.manual_seed(12345)  # deterministic init, as in the original code
        self.pooling = config.pooling
        self.grl = GradientReversal(max_iter=max_iter, alpha=alpha, max_coeff=max_coeff)

        if self.pooling == "global_mlp":
            # Table 3 "global" row: pool first, then GRL + MLP (no GraphSAGE).
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, num_domains),
            )
            return

        self.conv1 = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
        self.conv2 = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
        self.conv3 = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
        self.conv4 = GraphSAGE(hidden_channels, hidden_channels, 2, dropout=0.5)
        self.drop_1 = RandomNodeDropout(config.dropnode_prob)
        self.drop_2 = RandomNodeDropout(config.dropnode_prob)
        self.fc2 = nn.Linear(hidden_channels, num_domains)

    def forward(self, z1: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                drop_enabled: bool = False) -> torch.Tensor:
        """Returns softmax domain probabilities.

        * node mode       -> ``[N_total, S]`` (one prediction per node)
        * graph mode      -> ``[B, S]``
        * global_mlp mode -> ``[B, S]``
        """
        if self.pooling == "global_mlp":
            x = global_mean_pool(z1, batch)
            x = self.grl(x)
            x = self.mlp(x)
            return x.softmax(dim=-1)

        x = self.grl(z1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.drop_1(x, drop_enabled)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.drop_2(x, drop_enabled)

        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()

        if self.pooling == "node":
            x = self.fc2(x)                     # per-node dense layer (paper Eq. 6)
            return x.softmax(dim=-1)            # [N_total, S]

        # "graph" mode: legacy released-checkpoint behaviour.
        x = global_mean_pool(x, batch)
        x = self.fc2(x)
        return x.softmax(dim=-1)                # [B, S]


__all__ = ["DomainDiscriminator", "DiscriminatorConfig"]