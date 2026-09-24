#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference-only evaluation of a trained RAGFAS checkpoint.

Strictly loads every parameter from the checkpoint (strict=True) and reports
frame- and person-level ACC/HTER/AUC; when the domain name is recoverable
from the graph paths, per-domain frame metrics are logged too.  The number of
domains and the input feature size are inferred from the checkpoint and the
dataset, so evaluation matches training automatically.

Examples::

    python evaluate.py --config configs/protocols/MI_to_C.yaml \
        --checkpoint runs/MI_to_C/checkpoints/MI_to_C_best.pth

    python evaluate.py --config configs/protocols/MI_to_C.yaml \
        --checkpoint runs/MI_to_C/checkpoints/MI_to_C_last.pth \
        --split train            # evaluate the training domains as well
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch_geometric.loader import DataLoader

from ragfas.config import RAGFASConfig, load_config
from ragfas.data.dataset import GraphListDataset
from ragfas.engine.evaluator import Evaluator
from ragfas.utils.logging import setup_logger
from ragfas.utils.seed import get_device, set_seed
from train import build_model, infer_num_domains, resolve_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RAGFAS checkpoint.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Model weights (.pth)")
    parser.add_argument("--split", default="test", choices=["test", "train", "both"])
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides")
    parser.add_argument("--legacy", action="store_true",
                        help="Match the original research scripts exactly "
                             "(required for the released checkpoints).")
    args = parser.parse_args()

    cfg: RAGFASConfig = resolve_paths(load_config(args.config, args.set, args.legacy))
    protocol = cfg.protocol
    out_dir = os.path.join(cfg.output.dir, "inference")
    logger, log_file = setup_logger(f"ragfas.eval.{protocol}",
                                    os.path.join(out_dir, "logs"),
                                    tag=f"evaluate_{protocol}")

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    logger.info("=" * 72)
    logger.info("RAGFAS inference-only evaluation: %s", protocol)
    logger.info("log file: %s", log_file)
    logger.info("checkpoint: %s", args.checkpoint)
    if not os.path.exists(args.checkpoint):
        logger.error("checkpoint not found")
        sys.exit(1)

    loaders = {}
    if args.split in ("test", "both"):
        loaders["test"] = GraphListDataset(cfg.data.test_list, train_mode=False)
    if args.split in ("train", "both"):
        loaders["train"] = GraphListDataset(cfg.data.train_list, train_mode=False)
    for split, dataset in loaders.items():
        if len(dataset) == 0:
            logger.error("%s dataset is empty", split)
            sys.exit(1)

    ref = next(iter(loaders.values()))
    in_channels = cfg.model.in_channels or ref.num_node_features

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_domains = infer_num_domains(state_dict)
    logger.info("in_channels=%d | num_domains=%d (from checkpoint)", in_channels, num_domains)

    model = build_model(cfg, in_channels, num_domains).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("checkpoint loaded (strict).")
    except RuntimeError as e:
        logger.error("checkpoint load failed: %s", e)
        logger.error("If this checkpoint comes from the original research scripts, "
                     "re-run with --legacy.")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("parameters: %.2fM", n_params / 1e6)

    evaluator = Evaluator(model, device, os.path.join(out_dir, "Scores"), logger)
    for split, dataset in loaders.items():
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers,
                            pin_memory=cfg.data.pin_memory)
        m = evaluator.evaluate(loader, split, epoch=-1, per_domain=True)
        logger.info("SUMMARY [%s]: Frame ACC=%.4f HTER=%.4f AUC=%.4f | "
                    "Person ACC=%.4f HTER=%.4f AUC=%.4f",
                    split, m["frame_acc"], m["frame_hter"], m["frame_auc"],
                    m["person_acc"], m["person_hter"], m["person_auc"])

    logger.info("Done.")


if __name__ == "__main__":
    main()