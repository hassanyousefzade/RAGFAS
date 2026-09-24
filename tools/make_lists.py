#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the train/test graph list files of a protocol.

Walks ``<data-root>/<dataset-folder>/preposess/*/*.pt``, groups the graphs by
dataset (domain) and writes one plain-text list per split exactly in the
format the training code expects (one absolute ``.pt`` path per line).

Dataset folders are matched by name (case-insensitive contains):

    O -> "oulu",  C -> "casia",  M -> "msu",  I -> "replay"

Source datasets use their whole graph set for training; for the target
dataset only the *test* partitions are used (paths containing "test", which
is how CASIA-FASD and the other benchmarks name their evaluation split -
adjust ``--test-pattern`` if your folder layout differs).

Example (limited-source MI -> C)::

    python tools/make_lists.py --data-root /data \
        --protocol MI_to_C --out-dir /data/lists
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

from ragfas.data.protocols import DATASET_KEYS, PROTOCOLS


def collect(data_root: str) -> Dict[str, List[str]]:
    """Return {dataset key: [graph paths]} for every dataset folder found."""
    by_key: Dict[str, List[str]] = {k: [] for k in DATASET_KEYS}
    pattern = os.path.join(data_root, "**", "preposess", "*", "*.pt")
    for path in sorted(glob.glob(pattern, recursive=True)):
        folder = os.path.normpath(path).split(os.sep)
        pre_idx = folder.index("preposess")
        dataset_folder = folder[pre_idx - 1].lower()
        for key, needle in DATASET_KEYS.items():
            if needle.lower() in dataset_folder:
                by_key[key].append(path)
                break
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protocol train/test lists.")
    parser.add_argument("--data-root", required=True,
                        help="Root that contains <dataset>/preposess/*/*.pt")
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-pattern", default="test",
                        help="Substring marking the target's test partitions")
    args = parser.parse_args()

    sources, target = PROTOCOLS[args.protocol]
    by_key = collect(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)

    train_name = f"graph_train_{''.join(sources)}.txt"
    test_name = f"graph_test_{target}.txt"
    train_path = os.path.join(args.out_dir, train_name)
    test_path = os.path.join(args.out_dir, test_name)

    train_lines = [p for key in sources for p in by_key[key]]
    test_lines = [p for p in by_key[target] if args.test_pattern in os.path.basename(
        os.path.dirname(p))]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    counts = {key: len(by_key[key]) for key in DATASET_KEYS if by_key[key]}
    print(f"protocol {args.protocol}: sources={sources} target={target}")
    print(f"  graphs per dataset: {counts}")
    print(f"  train list: {train_path} ({len(train_lines)} graphs)")
    print(f"  test  list: {test_path} ({len(test_lines)} graphs)")


if __name__ == "__main__":
    main()