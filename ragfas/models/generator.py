# -*- coding: utf-8 -*-
"""Graph Feature Generator — GFG (paper Sec. 3.2).

A stack of Chebyshev graph convolutions (ChebConv, polynomial order K=2) so
that every block aggregates exactly the 2-hop neighbourhood at a cost linear
in the number of edges.  Each Cheb block applies ChebConv -> BatchNorm -> ReLU
-> DropNode.  To mitigate over-smoothing, the outputs of blocks 2 and 4 are
concatenated before block 5.  Four representations are tapped (Eq. 2):

    Z1 after block 2 (also consumed by the two node-level objectives),
    Z2 after block 3,
    Z3 after block 5,
    Z4 after the final (block 8) convolution + graph-level pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import BatchNorm

from ragfas.models.layers import RandomNodeDropout


@dataclass
class DropNodeConfig:
    prob: float = 0.3
    # Paper: DropNode is zero in the first two blocks (it harms the region
    # classifier).  Set True to reproduce the legacy released checkpoints.
    first_two_blocks: bool = False
    # Per-batch Bernoulli gate: DropNode of the whole network is toggled with
    # this probability at every training iteration (legacy behaviour).
    enable_prob: float = 0.5


class GraphFeatureGenerator(nn.Module):
    """Eight ChebConv(K=2) blocks with BatchNorm/ReLU/DropNode and skip concat.

    Returns the tapped node/level features used by the classifier and the two
    node-level objectives (``z1`` is the "discriminator" feature of the paper).
    """

    def __init__(self, in_channels: int, hidden_channels: int, cheb_order: int = 2,
                 dropnode: DropNodeConfig = DropNodeConfig()):
        super().__init__()
        torch.manual_seed(12345)  # deterministic init, as in the original code

        self.conv1 = ChebConv(in_channels, hidden_channels, cheb_order)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, 2)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, 2)
        self.conv4 = ChebConv(hidden_channels, hidden_channels, 2)
        self.conv5 = ChebConv(hidden_channels * 2, hidden_channels, 2)  # 2x from the skip concat
        self.conv6 = ChebConv(hidden_channels, hidden_channels, 2)
        self.conv7 = ChebConv(hidden_channels, hidden_channels, 2)
        self.conv8 = ChebConv(hidden_channels, hidden_channels, 2)

        self.drop_1 = RandomNodeDropout(dropnode.prob)
        self.drop_2 = RandomNodeDropout(dropnode.prob)
        self.drop_3 = RandomNodeDropout(dropnode.prob)
        self.drop_4 = RandomNodeDropout(dropnode.prob)
        self.drop_5 = RandomNodeDropout(dropnode.prob)
        self.drop_6 = RandomNodeDropout(dropnode.prob)

        self.b1 = BatchNorm(hidden_channels)
        self.b2 = BatchNorm(hidden_channels)
        self.b3 = BatchNorm(hidden_channels)
        self.b4 = BatchNorm(hidden_channels)
        self.b5 = BatchNorm(hidden_channels)
        self.b6 = BatchNorm(hidden_channels)
        self.b7 = BatchNorm(hidden_channels)
        self._first_two_blocks = dropnode.first_two_blocks

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                drop_enabled: bool = False) -> Dict[str, torch.Tensor]:
        # -- blocks 1-2 (DropNode disabled per paper; see DropNodeConfig) ------
        x = self.conv1(x, edge_index)
        x = self.b1(x)
        x = x.relu()
        if self._first_two_blocks:
            x = self.drop_1(x, drop_enabled)

        x = self.conv2(x, edge_index)
        x = self.b2(x)
        x = x.relu()
        if self._first_two_blocks:
            x = self.drop_2(x, drop_enabled)

        z1 = x                              # fine-grained node features (Sec. 3.2)
        h1 = global_mean_pool(x, batch)     # level-1 graph embedding

        # -- blocks 3-5, skip concat of block 2 before block 5 ----------------
        x = self.conv3(x, edge_index)
        x = self.b3(x)
        x = x.relu()
        x = self.drop_3(x, drop_enabled)
        h3 = global_mean_pool(x, batch)     # level-2 graph embedding

        x = self.conv4(x, edge_index)
        x = self.b4(x)
        x = x.relu()
        x = self.drop_4(x, drop_enabled)

        x = torch.cat((x, z1), dim=1)       # over-smoothing mitigation (Sec. 3.2)

        x = self.conv5(x, edge_index)
        x = self.b5(x)
        x = x.relu()
        x = self.drop_5(x, drop_enabled)
        h5 = global_mean_pool(x, batch)     # level-3 graph embedding

        # -- blocks 6-8 (the paper's "additional ChebConv and dropout") -------
        x = self.conv6(x, edge_index)
        x = self.b6(x)
        x = x.relu()
        x = self.drop_6(x, drop_enabled)

        x = self.conv7(x, edge_index)
        x = self.b7(x)
        x = x.relu()

        x = self.conv8(x, edge_index)
        h_final = global_mean_pool(x, batch)

        return {"z1": z1, "h1": h1, "h3": h3, "h5": h5, "h_final": h_final}


__all__ = ["GraphFeatureGenerator"]