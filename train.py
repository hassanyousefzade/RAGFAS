#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train RAGFAS on one of the paper's protocols.

Examples::

    # Limited-source protocol MI -> C (paper Table 2)
    python train.py --config configs/protocols/MI_to_C.yaml

    # Leave-one-out protocol OMI -> C (paper Table 1) with overrides
    python train.py --config configs/protocols/OMI_to_C.yaml \
        --set data.train_list=/data/lists/graph_train_O_M_I.txt \
        --set data.test_list=/data/lists/graph_test_C.txt \
        --set train.epochs=1000

    # Match the original research scripts exactly (released checkpoints)
    python train.py --config configs/protocols/MI_to_C.yaml --legacy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader

from ragfas.config import RAGFASConfig, config_dict, load_config
from ragfas.data.dataset import GraphListDataset
from ragfas.engine.evaluator import Evaluator
from ragfas.engine.trainer import Trainer
from ragfas.losses import LossWeights, RAGFASCriterion
from ragfas.models import RAGFAS
from ragfas.utils.logging import setup_logger
from ragfas.utils.paths import check_dir
from ragfas.utils.seed import get_device, set_seed

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SELECTION_METRICS = ("person_auc", "frame_auc", "frame_hter")


def infer_num_domains(state_dict: dict) -> int:
    """Read the discriminator output size from checkpoint weights."""
    for key in ("discriminator.fc2.weight", "discriminator.mlp.8.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    raise KeyError("discriminator output layer not found in the checkpoint")


def resolve_paths(cfg: RAGFASConfig) -> RAGFASConfig:
    """Fill in repo-relative defaults (landmark colors asset)."""
    if cfg.data.landmark_colors is None:
        cfg.data.landmark_colors = os.path.join(REPO_ROOT, "assets", "landmark_colors.npy")
    return cfg


def build_model(cfg: RAGFASConfig, in_channels: int, num_domains: int) -> RAGFAS:
    """Factory: assemble the RAGFAS model from the typed config."""
    m = cfg.model
    return RAGFAS(
        in_channels=in_channels,
        num_domains=num_domains,
        hidden_channels=m.hidden_channels,
        cheb_order=m.cheb_order,
        dropnode=m.dropnode,
        classifier_cfg={"tokens": m.classifier.tokens, "levels": m.classifier.levels,
                        "embed_dim": m.classifier.embed_dim,
                        "num_layers": m.classifier.num_layers,
                        "num_heads": m.classifier.num_heads},
        region_cfg=m.region_head,
        discriminator_cfg=m.discriminator,
        grl_alpha=m.grl.alpha,
        grl_max_iter=m.grl.max_iter,
        grl_max_coeff=m.grl.max_coeff,
    )


def append_metrics_csv(csv_path: str, row: dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAGFAS (ICASSP 2027).")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Config overrides, e.g. train.epochs=1000")
    parser.add_argument("--legacy", action="store_true",
                        help="Match the original research scripts exactly "
                             "(required for loading the released checkpoints).")
    args = parser.parse_args()

    cfg: RAGFASConfig = resolve_paths(load_config(args.config, args.set, args.legacy))
    if cfg.train.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"unknown selection metric: {cfg.train.selection_metric}")

    protocol = cfg.protocol
    out_dir = check_dir(cfg.output.dir)
    logger, log_file = setup_logger(f"ragfas.train.{protocol}",
                                    os.path.join(out_dir, "logs"),
                                    tag=f"train_{protocol}")

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    logger.info("=" * 72)
    logger.info("RAGFAS training run: %s", protocol)
    logger.info("log file : %s", log_file)
    logger.info("config   : %s", config_dict(cfg))
    logger.info("torch %s | device: %s", torch.__version__, device)

    train_dataset = GraphListDataset(cfg.data.train_list, train_mode=True,
                                     landmark_colors=cfg.data.landmark_colors)
    test_dataset = GraphListDataset(cfg.data.test_list, train_mode=False)
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Empty dataset - check the list files.")
        sys.exit(1)

    # num_domains auto-detected from the train list (e.g. MI -> 2, OCI -> 3).
    num_domains = cfg.model.num_domains or len(train_dataset.domain_counts)
    in_channels = cfg.model.in_channels or train_dataset.num_node_features
    logger.info("num_domains=%d domains=%s | node features=%d",
                num_domains, sorted(train_dataset.domain_counts), in_channels)

    loader_kwargs = dict(num_workers=cfg.data.num_workers,
                         pin_memory=cfg.data.pin_memory,
                         persistent_workers=cfg.data.num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                              shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size,
                             shuffle=False, **loader_kwargs)

    model = build_model(cfg, in_channels, num_domains).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable parameters: %.2fM", n_params / 1e6)
    logger.info("model:\n%s", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_step,
                                                gamma=cfg.train.lr_gamma)
    criterion = RAGFASCriterion(
        LossWeights(cls=cfg.train.w_cls, adv=cfg.train.w_adv,
                    self_supervised=cfg.train.w_self),
        focal_gamma=cfg.train.focal_gamma)

    trainer = Trainer(model, criterion, optimizer, scheduler, device,
                      dropnode_enable_prob=cfg.model.dropnode.enable_prob,
                      log_every=cfg.train.log_every, logger=logger)
    evaluator = Evaluator(model, device, os.path.join(out_dir, "Scores"), logger,
                          save_scores=cfg.output.save_scores)

    ckpt_dir = check_dir(os.path.join(out_dir, "checkpoints"))
    best_path = os.path.join(ckpt_dir, f"{protocol}_best.pth")
    last_path = os.path.join(ckpt_dir, f"{protocol}_last.pth")

    best_metric = -1.0
    if cfg.train.resume and os.path.exists(best_path):
        logger.info("Resuming from %s", best_path)
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))
        resume_metrics = evaluator.evaluate(test_loader, "test", -1)
        best_metric = resume_metrics[cfg.train.selection_metric]
        logger.info("Continuing with %s=%.4f", cfg.train.selection_metric, best_metric)
    else:
        logger.info("Training from scratch (resume=False or no checkpoint found).")

    csv_path = os.path.join(out_dir, "logs",
                            f"metrics_{protocol}_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logger.info("metrics csv: %s", csv_path)

    for epoch in range(cfg.train.epochs):
        train_stats = trainer.train_epoch(train_loader, epoch)
        metrics = evaluator.evaluate(test_loader, "test", epoch)
        lr = optimizer.param_groups[0]["lr"]

        append_metrics_csv(csv_path, {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_stats.items()},
            "lr": lr,
            **{f"test_{k}": v for k, v in metrics.items()},
        })

        current = metrics[cfg.train.selection_metric]
        if current > best_metric:
            logger.info("improve: %s %.4f -> %.4f | saving %s",
                        cfg.train.selection_metric, best_metric, current, best_path)
            torch.save(model.state_dict(), best_path)
            best_metric = current

        torch.save(model.state_dict(), last_path)
        logger.info("[EPOCH %d] done. lr=%.2e | best %s=%.4f",
                    epoch + 1, lr, cfg.train.selection_metric, best_metric)
        scheduler.step()

    logger.info("Training finished. best %s=%.4f", cfg.train.selection_metric, best_metric)


if __name__ == "__main__":
    main()