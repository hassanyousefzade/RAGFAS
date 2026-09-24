# -*- coding: utf-8 -*-
"""RAGFAS: the complete model (paper Fig. 2).

    FGRE (precomputed D121-LowFE node features)
        -> GFG  (Graph Feature Generator, Sec. 3.2)
            ├── Classifier            (Sec. 3.3, spoof logit)
            ├── Node-level Discriminator over GRL   (Sec. 3.4, domain probs)
            └── Region Prediction Network           (Sec. 3.5, region logits)

The forward signature mirrors the original research code so that both the
released checkpoints and the training/evaluation loops remain 1:1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragfas.models.classifier import MultiLevelTokenClassifier
from ragfas.models.discriminator import DiscriminatorConfig, DomainDiscriminator
from ragfas.models.generator import DropNodeConfig, GraphFeatureGenerator
from ragfas.models.region_head import RegionHeadConfig, RegionPredictionNetwork


class RAGFAS(nn.Module):
    """Graph generator + classifier + domain discriminator + region head."""

    def __init__(self, in_channels: int, num_domains: int, hidden_channels: int = 128,
                 cheb_order: int = 2, dropnode: DropNodeConfig = DropNodeConfig(),
                 classifier_cfg: dict = None, region_cfg: RegionHeadConfig = None,
                 discriminator_cfg: DiscriminatorConfig = None,
                 grl_alpha: float = 10.0, grl_max_iter: int = 4000,
                 grl_max_coeff: float = 1.0, classifier_dropout: float = 0.5):
        super().__init__()
        classifier_cfg = classifier_cfg or {}
        region_cfg = region_cfg or RegionHeadConfig()
        disc_cfg = discriminator_cfg or DiscriminatorConfig()

        self.generator = GraphFeatureGenerator(
            in_channels, hidden_channels, cheb_order, dropnode)
        self.classifier = MultiLevelTokenClassifier(
            tokens=classifier_cfg.get("tokens", 128),
            levels=classifier_cfg.get("levels", 4),
            embed_dim=classifier_cfg.get("embed_dim", 8),
            num_layers=classifier_cfg.get("num_layers", 5),
            num_heads=classifier_cfg.get("num_heads", 8),
        )
        self.discriminator = DomainDiscriminator(
            hidden_channels, num_domains,
            max_iter=grl_max_iter, alpha=grl_alpha, max_coeff=grl_max_coeff,
            config=disc_cfg,
        )
        self.region_head = RegionPredictionNetwork(hidden_channels, region_cfg)
        self.classifier_dropout = classifier_dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                node_dropout_enabled: bool = False,
                region_sample_idx: Optional[torch.Tensor] = None):
        """One forward pass of the full multi-task model.

        Args:
            x, edge_index, batch: batched landmark graph (PyG batch).
            node_dropout_enabled: per-batch Bernoulli gate for DropNode.
            region_sample_idx: 64 node indices for the RPN branch.

        Returns:
            logit ``[B, 1]``, domain probs (``[N, S]`` node mode / ``[B, S]``
            graph modes), region logits ``[num_samples, 7]``.
        """
        taps = self.generator(x, edge_index, batch, node_dropout_enabled)

        # Stack the four level embeddings: {Z4-final, Z1, Z3(block5), Z2(block3)}
        # exactly as in the original code (paper Eq. 2/3).
        stacked = torch.stack(
            [taps["h_final"], taps["h1"], taps["h5"], taps["h3"]], dim=2)  # [B, k', 4]
        stacked = F.dropout(stacked, p=self.classifier_dropout, training=self.training)
        logit = self.classifier(stacked)                                   # Eq. 3

        domain_probs = self.discriminator(
            taps["z1"], edge_index, batch, node_dropout_enabled)           # Eq. 6

        region_logits = None
        if region_sample_idx is not None:
            region_logits = self.region_head(taps["z1"], edge_index, region_sample_idx)

        return logit, domain_probs, region_logits


__all__ = ["RAGFAS"]