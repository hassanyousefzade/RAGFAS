# -*- coding: utf-8 -*-
"""Evaluation engine: frame- and person-level metrics (paper Sec. 4.1).

For every graph the sigmoid of the spoof logit is written into a score file
(``<score> <label>`` with label 1 = live); a second file averages the frame
scores of every person/video folder to produce the person-level metrics.  The
model-selection metric follows the original implementation (best test
person-level AUC) and is configurable.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ragfas.data.dataset import sample_node_indices
from ragfas.metrics import performances_val
from ragfas.utils.logging import show_tqdm


class Evaluator:
    """Writes score files and computes ACC/HTER/AUC at both granularities."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 scores_dir: str, logger: logging.Logger, save_scores: bool = True):
        self.model = model
        self.device = device
        self.scores_dir = scores_dir
        self.logger = logger
        self.save_scores = save_scores

    @torch.no_grad()
    def evaluate(self, loader, tag: str, epoch: int = -1,
                 per_domain: bool = False) -> Dict[str, float]:
        """Evaluate ``loader``; optionally break the frame metrics down per domain."""
        self.model.eval()
        scores_list, person_dict = [], {}
        domain_frames: Dict[str, list] = {}

        progress = tqdm(loader, desc=f"Epoch {epoch + 1} ({tag})", unit="batch",
                        disable=not show_tqdm())
        for graph in progress:
            index_arr = sample_node_indices(graph.x.shape[0], self.device)
            graph = graph.to(self.device)
            logit, _, _ = self.model(graph.x, graph.edge_index, graph.batch,
                                     False, index_arr)
            logit = torch.sigmoid(logit)

            logit_cpu = logit.reshape(-1).cpu().numpy()
            label_cpu = graph.y.float().reshape(-1).cpu().numpy()
            person_ids = graph.person_id
            domain_names = getattr(graph, "domain_name", [""] * graph.num_graphs)

            for score, label, pid, dom in zip(logit_cpu, label_cpu, person_ids, domain_names):
                scores_list.append(f"{score} {label}\n")
                if pid not in person_dict:
                    person_dict[pid] = {"scores": [], "label": label}
                person_dict[pid]["scores"].append(score)
                if per_domain and dom:
                    domain_frames.setdefault(dom, []).append((score, label))

        if self.save_scores:
            score_dir = os.path.join(self.scores_dir, f"{tag}_epoch_{epoch + 1}")
            os.makedirs(score_dir, exist_ok=True)
        else:
            score_dir = tempfile.mkdtemp(prefix="ragfas_eval_")

        frame_file = os.path.join(score_dir, "score.txt")
        with open(frame_file, "w") as f:
            f.writelines(scores_list)
        person_file = os.path.join(score_dir, "score_person.txt")
        with open(person_file, "w") as f:
            f.writelines(f"{np.mean(v['scores'])} {v['label']}\n"
                         for v in person_dict.values())

        m = performances_val(frame_file)
        p = performances_val(person_file)
        self.logger.info("[EPOCH %d][%s][Frame ] ACC=%.4f HTER=%.4f AUC=%.4f ERR=%.4f TPR=%.4f",
                         epoch + 1, tag, m[0], m[3], m[4], m[5], m[6])
        self.logger.info("[EPOCH %d][%s][Person] ACC=%.4f HTER=%.4f AUC=%.4f ERR=%.4f TPR=%.4f",
                         epoch + 1, tag, p[0], p[3], p[4], p[5], p[6])
        self.logger.info("  scores: %s", frame_file)
        self.logger.info("  person: %s", person_file)

        if per_domain:
            self._log_per_domain(domain_frames, tag, epoch)

        return {"frame_acc": m[0], "frame_hter": m[3], "frame_auc": m[4],
                "person_acc": p[0], "person_hter": p[3], "person_auc": p[4]}

    def _log_per_domain(self, domain_frames: Dict[str, list], tag: str, epoch: int) -> None:
        for dom, items in domain_frames.items():
            labels = {lb for _, lb in items}
            if len(labels) < 2:
                self.logger.warning("[EPOCH %d][%s][domain=%s] single class present; "
                                    "metrics skipped", epoch + 1, tag, dom)
                continue
            dom_dir = os.path.join(self.scores_dir, f"{tag}_epoch_{epoch + 1}_domain_{dom}")
            os.makedirs(dom_dir, exist_ok=True)
            dom_file = os.path.join(dom_dir, "score.txt")
            with open(dom_file, "w") as f:
                f.writelines(f"{s} {lb}\n" for s, lb in items)
            d = performances_val(dom_file)
            self.logger.info("[EPOCH %d][%s][domain=%s][Frame] ACC=%.4f HTER=%.4f "
                             "AUC=%.4f ERR=%.4f TPR=%.4f",
                             epoch + 1, tag, dom, d[0], d[3], d[4], d[5], d[6])


__all__ = ["Evaluator"]