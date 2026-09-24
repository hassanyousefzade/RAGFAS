# -*- coding: utf-8 -*-
"""Training engine (Template Method over the training loop).

The Trainer owns everything optimisation-related (DropNode gate, node
subsampling, the three losses, logging) while the caller drives epoch-level
logic (evaluation, model selection, checkpointing) in ``train.py``.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ragfas.data.dataset import sample_node_indices
from ragfas.losses import RAGFASCriterion, expand_domain_targets
from ragfas.utils.logging import show_tqdm


class Trainer:
    """Runs the training loop of one RAGFAS model."""

    def __init__(self, model: torch.nn.Module, criterion: RAGFASCriterion, optimizer,
                 scheduler, device: torch.device, dropnode_enable_prob: float,
                 log_every: int, logger: logging.Logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.enable_prob = dropnode_enable_prob
        self.log_every = log_every
        self.logger = logger

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        """One epoch; returns per-sample averaged loss statistics."""
        self.model.train()
        sums = {"total": 0.0, "cls": 0.0, "adv": 0.0, "self": 0.0}
        seen = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1} (Train)", unit="batch",
                        disable=not show_tqdm())
        for it, graph in enumerate(progress):
            graph = graph.to(self.device)
            drop_enabled = bool(np.random.binomial(1, self.enable_prob))
            index_arr = sample_node_indices(graph.x.shape[0], self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logit, domain_probs, region_logits = self.model(
                graph.x, graph.edge_index, graph.batch, drop_enabled, index_arr)

            domain_targets = expand_domain_targets(
                domain_probs, graph.domain, graph.batch)
            loss, loss_stats = self.criterion(
                logit, graph.y, domain_probs, domain_targets,
                region_logits, graph.y_node[index_arr])
            loss.backward()
            self.optimizer.step()

            bs = graph.num_graphs
            seen += bs
            for key in sums:
                sums[key] += loss_stats[key] * bs

            if (it + 1) % self.log_every == 0:
                self.logger.info(
                    "  [E%d] iter %d/%d | running avg: cls=%.6f adv=%.6f "
                    "self=%.6f total=%.6f",
                    epoch + 1, it + 1, len(loader),
                    sums["cls"] / seen, sums["adv"] / seen,
                    sums["self"] / seen, sums["total"] / seen)

        stats = {k: v / max(seen, 1) for k, v in sums.items()}
        self.logger.info(
            "[EPOCH %d] TRAIN LOSSES: cls=%.6f | adv=%.6f | self=%.6f | total=%.6f",
            epoch + 1, stats["cls"], stats["adv"], stats["self"], stats["total"])
        return stats


__all__ = ["Trainer"]