#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2 of the FGRE pipeline (paper Sec. 3.1): build one landmark graph per
cropped face and store it as ``<frame>.jpg.pt``.

For every ``cropped_XXXX.jpg`` (with its ``cropped_XXXX.npy`` landmarks):

1. the 40x40 patch of every one of the 468 landmarks is extracted from the
   zero-padded image (``kernel_size = 20``),
2. all patches are encoded with the frozen D121-LowFE extractor
   (DenseNet-121, first two blocks, GAP + concat -> 640 features per node),
3. the edges follow the canonical FACEMESH_TESSELATION topology
   (undirected, fixed for all samples),
4. the label comes from the video/folder name keywords
   (``live``/``real`` -> 1, ``spoof``/``attack`` -> 0),
5. ``Data(x=[468, 640], edge_index=[2, E], y=label)`` is saved as
   ``<frame>.jpg.pt`` next to the images, and the paths are appended to the
   optional ``--list-out`` file.

Example::

    python tools/build_graphs.py --input /data/oulu/preposess \
        --output-list /data/lists/graph_train_oulu.txt
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragfas.data.topology import facemesh_edge_index   # noqa: E402
from ragfas.models.lowfe import DenseNet121LowFE      # noqa: E402

KERNEL_SIZE = 20   # 40x40 patch around each landmark (paper Sec. 4.1)
FACE_SIZE = 256
BATCH = 64         # patches encoded per DenseNet forward


def label_from_path(path: str) -> int:
    """Label from the video/folder name: live/real -> 1, spoof/attack -> 0."""
    name = os.path.basename(os.path.dirname(path)).split("_")
    if "live" in name or "real" in name:
        return 1
    if "spoof" in name or "attack" in name:
        return 0
    raise ValueError(f"cannot infer the label from: {path}")


def extract_patches(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Zero-pad the image by KERNEL_SIZE and cut a 40x40 patch per landmark."""
    landmarks = np.clip(landmarks, 0, FACE_SIZE) + np.array([KERNEL_SIZE, KERNEL_SIZE])
    padded = np.pad(img, ((KERNEL_SIZE, KERNEL_SIZE), (KERNEL_SIZE, KERNEL_SIZE), (0, 0)),
                    mode="constant", constant_values=0)
    landmarks = np.round(landmarks).astype(int)

    patches = []
    for x, y in landmarks:
        patches.append(padded[y - KERNEL_SIZE:y + KERNEL_SIZE,
                              x - KERNEL_SIZE:x + KERNEL_SIZE])
    return np.array(patches, dtype=np.float32)   # [468, 40, 40, 3]


def build_graph(lowfe: DenseNet121LowFE, edge_index: torch.Tensor,
                path: str, device: torch.device) -> Data:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    info = np.load(os.path.splitext(path)[0] + ".npy", allow_pickle=True).item()
    landmarks = info["landmark"]

    patches = extract_patches(img, landmarks)          # [468, 40, 40, 3]
    features = encode_patches(lowfe, patches, device)  # [468, 640]
    return Data(x=features, edge_index=edge_index, y=torch.tensor(label_from_path(path)))


def encode_patches(lowfe: DenseNet121LowFE, patches: np.ndarray, device) -> torch.Tensor:
    """Encode all patches (one batched D121-LowFE pass per chunk)."""
    chunks = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH):
            np_batch = patches[i:i + BATCH].transpose(0, 3, 1, 2)  # NHWC -> NCHW
            t = torch.from_numpy(np.ascontiguousarray(np_batch)).to(device)
            chunks.append(lowfe(t).cpu())
    return torch.cat(chunks, dim=0).float().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGFAS stage 2: build landmark graphs.")
    parser.add_argument("--input", required=True,
                        help="Folder tree containing cropped_*.jpg + cropped_*.npy")
    parser.add_argument("--list-out", default=None,
                        help="Write the path of every produced .pt into this list file")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Use random DenseNet weights (debug only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    lowfe = DenseNet121LowFE(pretrained=not args.no_pretrained).to(device).eval()
    edge_index = facemesh_edge_index()

    paths = sorted(glob.glob(os.path.join(args.input, "**", "cropped_*.jpg"), recursive=True))
    print(f"{len(paths)} cropped faces found in {args.input}")

    list_file = open(args.list_out, "a", encoding="utf-8") if args.list_out else None
    done = 0
    for path in tqdm(paths, desc="Building graphs"):
        out_path = path + ".pt"
        if os.path.exists(out_path):
            done += 1
            if list_file:
                list_file.write(out_path + "\n")
            continue
        try:
            data = build_graph(lowfe, edge_index, path, device)
            torch.save(data, out_path)
            done += 1
            if list_file:
                list_file.write(out_path + "\n")
        except Exception as e:  # noqa: BLE001 - keep batch going, report at the end
            print(f"FAILED {path}: {e}")
    if list_file:
        list_file.close()
    print(f"done: {done}/{len(paths)} graphs saved"
          + (f" | list: {args.list_out}" if args.list_out else ""))


if __name__ == "__main__":
    main()