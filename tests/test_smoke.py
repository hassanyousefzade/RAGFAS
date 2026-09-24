# -*- coding: utf-8 -*-
"""Smoke test: model shapes, checkpoint round-trip and one train/eval step.

Run on the training machine (needs torch + torch_geometric)::

    python tests/test_smoke.py

It builds two small random landmark graphs (64 nodes instead of 468 to keep
the test fast), runs a full RAGFAS forward in every discriminator mode and
executes a single optimisation step + evaluation for each mode.  It also
verifies that a state-dict saved in "node" mode loads in "graph" mode (the
two variants only differ in the forward path, not in the parameters).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ragfas.config import RAGFASConfig, load_config
from ragfas.data.dataset import sample_node_indices
from ragfas.engine.evaluator import Evaluator
from ragfas.engine.trainer import Trainer
from ragfas.losses import LossWeights, RAGFASCriterion
from ragfas.models import RAGFAS
from train import build_model

IN_CHANNELS, N_NODES, N_GRAPHS, N_DOMAINS = 640, 64, 4, 2


def make_graph(seed: int) -> Data:
    rng = np.random.RandomState(seed)
    x = torch.randn(N_NODES, IN_CHANNELS)
    # ring graph
    src = torch.arange(N_NODES)
    dst = (src + 1) % N_NODES
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    region = torch.eye(7)[torch.from_numpy(rng.randint(0, 7, N_NODES))]
    return Data(x=x, edge_index=edge_index,
                y=torch.tensor(rng.randint(0, 2)),
                domain=int(rng.randint(0, N_DOMAINS)),
                y_node=region,
                person_id=f"person_{seed}",
                domain_name="fake_domain")


def run_mode(cfg: RAGFASConfig, tag: str) -> dict:
    print(f"--- mode {tag}: pooling={cfg.model.discriminator.pooling}, "
          f"first_two_blocks={cfg.model.dropnode.first_two_blocks}, "
          f"region conv={cfg.model.region_head.conv}")
    model = build_model(cfg, IN_CHANNELS, N_DOMAINS)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    params: {n_params / 1e6:.2f}M")

    loader = DataLoader([make_graph(i) for i in range(N_GRAPHS)], batch_size=2)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = RAGFASCriterion(LossWeights(), focal_gamma=2.0)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("smoke")

    class CfgStub:  # minimal attrs used by Trainer
        pass

    trainer = Trainer(model, criterion, optimizer, None, device,
                      dropnode_enable_prob=cfg.model.dropnode.enable_prob,
                      log_every=1, logger=logger)
    trainer.train_epoch(loader, 0)
    print("    train step OK")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ev = Evaluator(model, device, tmp, logger, save_scores=True)
        metrics = ev.evaluate(loader, "test", 0)
    print(f"    eval OK: frame AUC={metrics['frame_auc']:.3f} "
          f"person AUC={metrics['person_auc']:.3f}")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) forward shapes for every discriminator strategy
    batch = torch.zeros(N_NODES, dtype=torch.long)   # batch of one graph
    for pooling in ("node", "graph", "global_mlp"):
        cfg = RAGFASConfig()
        cfg.model.discriminator.pooling = pooling
        model = build_model(cfg, IN_CHANNELS, N_DOMAINS)
        data = make_graph(0)
        index = sample_node_indices(N_NODES)
        logit, domain, region = model(data.x, data.edge_index, batch, True, index)
        expected = (N_NODES, N_DOMAINS) if pooling == "node" else (1, N_DOMAINS)
        assert logit.shape == (1, 1), logit.shape
        assert tuple(domain.shape) == expected, (pooling, domain.shape)
        assert region.shape == (index.numel(), 7), region.shape
        print(f"forward OK [{pooling}]: logit{tuple(logit.shape)} "
              f"domain{tuple(domain.shape)} region{tuple(region.shape)}")

    # 2) one train + eval step in paper mode and legacy mode
    paper_cfg = load_config(os.path.join("configs", "default.yaml"))
    paper_cfg.model.discriminator.pooling = "node"
    paper_state = run_mode(paper_cfg, "paper")

    legacy_cfg = load_config(os.path.join("configs", "default.yaml"))
    legacy_cfg.apply_legacy()
    legacy_state = run_mode(legacy_cfg, "legacy")

    # 3) node <-> graph checkpoint compatibility (identical parameter names)
    model = build_model(legacy_cfg, IN_CHANNELS, N_DOMAINS)
    model.load_state_dict(paper_state, strict=True)
    print("checkpoint compatibility OK: node-mode weights load into graph mode")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()