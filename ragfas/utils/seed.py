# -*- coding: utf-8 -*-
"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CUDA included) for reproducible runs.

    Note: the model modules additionally re-seed ``torch`` inside their
    ``__init__`` (seed 12345) exactly like the original implementation, so
    weight initialisation is deterministic regardless of the global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


__all__ = ["set_seed", "get_device"]