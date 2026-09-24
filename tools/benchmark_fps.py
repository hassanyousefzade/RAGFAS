#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end inference-speed benchmark of RAGFAS (paper Sec. 4.5).

Times the full single-image pipeline reported in the paper (198 FPS on a
Tesla T4, batch size 1), including MediaPipe landmark detection:

    image -> MediaPipe FaceMesh -> 256x256 crop -> 468 patches of 40x40
          -> ONE batched D121-LowFE forward on the [468, 3, 40, 40] tensor
          -> RAGFAS graph forward (generator + classifier + heads)
          -> sigmoid spoof score

Usage::

    python tools/benchmark_fps.py --config configs/protocols/MI_to_C.yaml \
        --checkpoint runs/MI_to_C/checkpoints/MI_to_C_best.pth \
        --image sample.jpg [--n-iter 200]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragfas.config import load_config              # noqa: E402
from ragfas.data.dataset import sample_node_indices  # noqa: E402
from ragfas.data.topology import facemesh_edge_index  # noqa: E402
from ragfas.models import DenseNet121LowFE         # noqa: E402
from train import build_model, infer_num_domains, resolve_paths  # noqa: E402

FACE_SIZE = 256
KERNEL = 20      # 40x40 patch per landmark


def detect_landmarks(face_mesh, image_rgb: np.ndarray) -> np.ndarray:
    """468 landmark coordinates on the 256x256 crop."""
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        raise RuntimeError("no face detected in the benchmark image")
    lm = results.multi_face_landmarks[0].landmark
    return np.array([(p.x, p.y) for p in lm]) * FACE_SIZE


def to_patch_tensor(image_rgb: np.ndarray, landmarks: np.ndarray) -> torch.Tensor:
    """468 zero-padded 40x40 patches stacked into one [468, 3, 40, 40] tensor."""
    pts = np.clip(landmarks, 0, FACE_SIZE) + KERNEL
    padded = np.pad(image_rgb, ((KERNEL, KERNEL), (KERNEL, KERNEL), (0, 0)),
                    mode="constant", constant_values=0)
    pts = np.round(pts).astype(int)
    patches = np.stack([padded[y - KERNEL:y + KERNEL, x - KERNEL:x + KERNEL]
                        for x, y in pts]).astype(np.float32)
    return torch.from_numpy(patches.transpose(0, 3, 1, 2).copy())


def run(image_rgb: np.ndarray, face_mesh, lowfe, model, edge_index,
        device: torch.device) -> float:
    """One end-to-end pass; returns the spoof score (sigmoid of the logit)."""
    landmarks = detect_landmarks(face_mesh, image_rgb)
    patches = to_patch_tensor(image_rgb, landmarks).to(device)

    with torch.no_grad():
        features = lowfe(patches)                       # [468, 640], one batched pass
        batch = torch.zeros(features.size(0), dtype=torch.long, device=device)
        index_arr = sample_node_indices(features.size(0), device)
        logit, _, _ = model(features, edge_index, batch, False, index_arr)
        score = torch.sigmoid(logit).item()
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGFAS end-to-end FPS benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="any photo containing a face")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides")
    args = parser.parse_args()

    import mediapipe as mp

    cfg = resolve_paths(load_config(args.config, args.set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {device} ({name})")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_domains = infer_num_domains(state)
    model = build_model(cfg, in_channels=640, num_domains=num_domains).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    lowfe = DenseNet121LowFE(pretrained=True).to(device).eval()
    edge_index = facemesh_edge_index().to(device)

    img = cv2.imread(args.image)
    assert img is not None, f"cannot read {args.image}"
    image_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (FACE_SIZE, FACE_SIZE))

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                         max_num_faces=1) as face_mesh:
        for _ in range(args.n_warmup):
            run(image_rgb, face_mesh, lowfe, model, edge_index, device)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.n_iter):
            score = run(image_rgb, face_mesh, lowfe, model, edge_index, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    fps = args.n_iter / elapsed
    print(f"sample score: {score:.4f}")
    print(f"end-to-end (MediaPipe included): {fps:.1f} FPS "
          f"({1000.0 * elapsed / args.n_iter:.2f} ms/image, {args.n_iter} iterations)")


if __name__ == "__main__":
    main()