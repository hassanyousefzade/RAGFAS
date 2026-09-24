# -*- coding: utf-8 -*-
"""Path helpers shared by the data pipeline and the training loop.

The original research code parsed dataset names and person/video ids out of the
absolute path of every preprocessed graph file, e.g.::

    /media/.../oulu/preposess/oulu_dev_live_1_1_21_1/cropped_0005.jpg.pt
                            ^^^^^ domain      ^^^^^^^^^^^^^^^^^^^^^ person/video

The helpers below keep that behaviour (including tolerance for windows-style
separators inside the list files) while making it independent of the depth of
the directory tree.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, List

# Name of the dataset folder (as it appears in the path) -> domain label.
# The values are the ones used to train all released checkpoints; new source
# domains can be appended without touching the model code.
DOMAIN_MAP = {
    "MSU_MFSD": 0,
    "Replay_attack_dataset": 1,
    "oulu": 2,
    # CASIA never appears as a *target* domain but is a source in OCI->M.
    "casia": 3,
    "casia-ds": 3,
}


def split_path(path: str) -> list:
    """Split a (possibly windows-style) path into its components."""
    p = str(path).strip().strip('"').replace("\\", "/")
    return osp.normpath(p).split("/")


def find_domain_name(path: str, domain_map=None) -> str:
    """Return the first path component that is a known dataset folder."""
    domain_map = DOMAIN_MAP if domain_map is None else domain_map
    for part in split_path(path):
        if part in domain_map:
            return part
    raise ValueError(f"unknown domain in path: {path}")


def check_dir(d: str) -> str:
    """Create ``d`` (recursively) if it does not exist and return it."""
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d


def person_id_from_path(path: str) -> str:
    """Person/video folder name: the component right before the file name."""
    return split_path(path)[-2]


__all__ = [
    "DOMAIN_MAP",
    "split_path",
    "find_domain_name",
    "check_dir",
    "person_id_from_path",
]