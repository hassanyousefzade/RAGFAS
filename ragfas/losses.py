# -*- coding: utf-8 -*-
"""Loss functions (paper Sec. 3.3-3.5).

* :class:`FocalLoss` — the adversarial domain-classification loss ``L_Adv``
  (Eq. 7), applied on *softmax probabilities* exactly like the original
  implementation:  ``-mean((1 - p_y)^gamma * log(p_y))``.
* :class:`RAGFASCriterion` — the integrated objective
  ``L_Total = L_Cls + L_Adv + L_Self`` (unweighted sum, Sec. 3.5):
  binary cross-entropy (Eq. 4) + focal domain loss (Eq. 7) + region
  cross-entropy (Eq. 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss over softmax probabilities (as used by the original code).

    Args:
        inputs: ``[N, C]`` softmax probabilities.
        targets: ``[N]`` integer class labels.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1, 1).long()
        pt = inputs.gather(1, targets)
        pt = pt.clamp(min=1e-7, max=1.0)
        loss = -1.0 * (1.0 - pt) ** self.gamma * torch.log(pt)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class LossWeights:
    cls: float = 1.0
    adv: float = 1.0
    self_supervised: float = 1.0


def expand_domain_targets(domain_probs: torch.Tensor, graph_domain: torch.Tensor,
                          batch: torch.Tensor) -> torch.Tensor:
    """Align domain targets with the discriminator output granularity.

    * graph mode: ``domain_probs`` is ``[B, S]`` -> per-graph ids are returned.
    * node mode:  ``domain_probs`` is ``[N_total, S]`` -> the per-graph ids are
      broadcast to every node with the PyG ``batch`` vector, implementing the
      per-node sum of Eq. 7.
    """
    if domain_probs.size(0) == graph_domain.size(0):
        return graph_domain
    return graph_domain[batch]


class RAGFASCriterion(nn.Module):
    """Integrated multi-task objective ``L_Total = L_Cls + L_Adv + L_Self``."""

    def __init__(self, loss_weights: LossWeights = None, focal_gamma: float = 2.0):
        super().__init__()
        self.cls_criterion = nn.BCEWithLogitsLoss()          # Eq. 4
        self.adv_criterion = FocalLoss(gamma=focal_gamma)    # Eq. 7
        self.region_criterion = nn.CrossEntropyLoss()        # Eq. 8
        self.weights = loss_weights or LossWeights()

    def forward(self, logit: torch.Tensor, y: torch.Tensor,
                domain_probs: torch.Tensor, domain_targets: torch.Tensor,
                region_logits: torch.Tensor, region_targets: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_cls = self.cls_criterion(logit.squeeze(), y.float())
        loss_adv = self.adv_criterion(domain_probs, domain_targets)
        loss_self = self.region_criterion(region_logits.squeeze(), region_targets.float())

        total = (self.weights.cls * loss_cls
                 + self.weights.adv * loss_adv
                 + self.weights.self_supervised * loss_self)
        stats = {"cls": loss_cls.item(), "adv": loss_adv.item(),
                 "self": loss_self.item(), "total": total.item()}
        return total, stats


__all__ = ["FocalLoss", "LossWeights", "expand_domain_targets", "RAGFASCriterion"]