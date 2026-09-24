# -*- coding: utf-8 -*-
"""Typed configuration system (Factory pattern).

Every entry point loads a YAML file into :class:`RAGFASConfig` dataclasses,
optionally applies ``--set key=value`` overrides and a ``--legacy`` switch
that restores the exact behaviour of the original research scripts (needed to
load the released checkpoints bit-exactly):

* ``model.discriminator.pooling = "graph"``  (pooled before the dense layer)
* ``model.dropnode.first_two_blocks = true`` (DropNode also in blocks 1-2)
* ``model.region_head.conv = "sage"``        (GraphSAGE inside the RPN)
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml

# The model-side dataclasses are the single source of truth; the config
# re-exports them so YAML keys map 1:1 onto the model constructors.
from ragfas.models.discriminator import DiscriminatorConfig
from ragfas.models.generator import DropNodeConfig
from ragfas.models.region_head import RegionHeadConfig


@dataclass
class ClassifierConfig:
    tokens: int = 128
    levels: int = 4
    embed_dim: int = 8
    num_layers: int = 5
    num_heads: int = 8


@dataclass
class GRLConfig:
    alpha: float = 10.0
    max_iter: int = 4000
    max_coeff: float = 1.0


@dataclass
class ModelConfig:
    hidden_channels: int = 128
    cheb_order: int = 2
    in_channels: Optional[int] = None      # None -> taken from the dataset (640)
    num_domains: Optional[int] = None      # None -> auto-detected from the train list
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    region_head: RegionHeadConfig = field(default_factory=RegionHeadConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    dropnode: DropNodeConfig = field(default_factory=DropNodeConfig)
    grl: GRLConfig = field(default_factory=GRLConfig)


@dataclass
class DataConfig:
    train_list: str = ""
    test_list: str = ""
    landmark_colors: Optional[str] = None  # None -> <repo>/assets/landmark_colors.npy
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    seed: int = 1378
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1.0e-4
    lr_step: int = 500
    lr_gamma: float = 0.5
    w_cls: float = 1.0
    w_adv: float = 1.0
    w_self: float = 1.0
    focal_gamma: float = 2.0
    resume: bool = True
    selection_metric: str = "person_auc"   # person_auc | frame_auc | frame_hter
    log_every: int = 200
    device: str = "auto"


@dataclass
class OutputConfig:
    dir: str = "runs/run"
    save_scores: bool = True


@dataclass
class RAGFASConfig:
    protocol: str = "custom"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def apply_legacy(self) -> None:
        """Restore the exact behaviour of the original research scripts."""
        self.model.discriminator.pooling = "graph"
        self.model.dropnode.first_two_blocks = True
        self.model.region_head.conv = "sage"


def _build(cls, data: Dict[str, Any]):
    """Recursively build a dataclass tree from a (possibly partial) dict."""
    if not is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data or data[f.name] is None:
            continue
        value = data[f.name]
        field_type = hints[f.name]
        if is_dataclass(field_type):
            kwargs[f.name] = _build(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _parse_value(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_config(path: str, overrides: Optional[List[str]] = None,
                legacy: bool = False) -> RAGFASConfig:
    """Load a YAML config, apply ``key.subkey=value`` overrides and --legacy."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    top = {f.name for f in fields(RAGFASConfig)}
    cfg = _build(RAGFASConfig, {k: v for k, v in raw.items() if k in top})

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got: {ov}")
        key, value = ov.split("=", 1)
        node, leaf = key.rsplit(".", 1)
        target = cfg
        for part in node.split("."):
            target = getattr(target, part)
        setattr(target, leaf, _parse_value(value))

    if legacy:
        cfg.apply_legacy()
    return cfg


def config_dict(cfg: RAGFASConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


__all__ = [
    "RAGFASConfig", "DataConfig", "ModelConfig", "TrainConfig", "OutputConfig",
    "DropNodeConfig", "ClassifierConfig", "RegionHeadConfig", "DiscriminatorConfig",
    "GRLConfig", "load_config", "config_dict",
]