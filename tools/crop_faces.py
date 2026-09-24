#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1 of the FGRE pipeline (paper Sec. 3.1): face alignment, cropping and
MediaPipe landmark extraction.

For every input frame/video the tool

1. detects the face with MediaPipe FaceMesh,
2. rotates the image so that both eyes are horizontal (roll alignment),
3. crops the landmark bounding box plus 20 px of padding and resizes it to
   256x256,
4. rescales the 468 landmarks to the cropped face and stores them next to the
   cropped image as ``cropped_XXXX.npy`` (``{"landmark": array}``).

Frames whose landmark detection fails are dropped (the paper reports the
failure rates in Sec. 4.6; the pipeline degrades gracefully).

Examples::

    python tools/crop_faces.py --input /data/oulu/preposess \
        --output /data/oulu/preposess --stage frames --sample-ratio 5
    python tools/crop_faces.py --input /data/oulu/preposess \
        --output /data/oulu/preposess --stage crop
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

PAD = 20
FACE_SIZE = 256


def align_face(rgb_img: np.ndarray, face_mesh) -> Optional[np.ndarray]:
    """Roll-align a face so that the eyes are horizontal (as in the paper)."""
    for _ in range(5):
        results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    height, width, _ = rgb_img.shape
    pts = [(int(l.x * width), int(l.y * height)) for l in landmarks]

    left_eye = np.mean(pts[33:36] + pts[133:136], axis=0)
    right_eye = np.mean(pts[362:365] + pts[263:266], axis=0)
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(rgb_img, matrix, (width, height),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_extract_landmarks(rgb_image: np.ndarray, face_mesh
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Crop the landmark bbox (+PAD px), resize to 256x256 and rescale landmarks."""
    original_h, original_w = rgb_image.shape[:2]
    for _ in range(5):
        results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    pts = [(l.x * original_w, l.y * original_h) for l in landmarks]

    x_min = int(min(p[0] for p in pts))
    x_max = int(max(p[0] for p in pts))
    y_min = int(min(p[1] for p in pts))
    y_max = int(max(p[1] for p in pts))

    x_min = max(x_min - PAD, 0)
    x_max = min(x_max + PAD, original_w)
    y_min = max(y_min - PAD, 0)
    y_max = min(y_max + PAD, original_h)

    cropped = rgb_image[y_min:y_max, x_min:x_max]
    ch, cw = cropped.shape[:2]
    resized = cv2.resize(cropped, (FACE_SIZE, FACE_SIZE))
    scaled = [(x - x_min, y - y_min) for x, y in pts]
    rescaled = np.array([(x * FACE_SIZE / cw, y * FACE_SIZE / ch) for x, y in scaled])
    return resized, rescaled


def process_image(path: str, save_dir: str, face_mesh) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped, landmarks = crop_extract_landmarks(img, face_mesh)

    stem = os.path.basename(path).split(".")[0].replace("org_", "cropped_")
    if cropped is None:
        print(f"no face detected, skipped: {path}")
        return False
    plt.imsave(os.path.join(save_dir, f"{stem}.jpg"), cropped)
    np.save(os.path.join(save_dir, f"{stem}.npy"),
            {"landmark": landmarks}, allow_pickle=True)
    return True


def extract_frames(video_path: str, save_dir: str, sample_ratio: int, max_frames: int) -> None:
    """Dump sampled frames of a video as ``org_XXXX.jpg``."""
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success or frame_id > max_frames:
            break
        if frame_id % sample_ratio == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imsave(os.path.join(save_dir, f"org_{frame_id:04d}.jpg"), frame)
        frame_id += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGFAS stage 1: align/crop/landmarks.")
    parser.add_argument("--input", required=True,
                        help="Folder of frames (stage=crop) or videos (stage=frames)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["crop", "frames"], default="crop")
    parser.add_argument("--sample-ratio", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    import mediapipe as mp

    os.makedirs(args.output, exist_ok=True)

    if args.stage == "frames":
        for video in sorted(glob.glob(os.path.join(args.input, "**", "*.mp4"),
                                      recursive=True)):
            out = os.path.join(args.output, os.path.splitext(os.path.basename(video))[0])
            os.makedirs(out, exist_ok=True)
            extract_frames(video, out, args.sample_ratio, args.max_frames)
            print(f"frames extracted: {video} -> {out}")
        return

    pattern = os.path.join(args.input, "**", "*.jpg" if args.recursive else "*.jpg")
    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [p for p in paths if not os.path.basename(p).startswith("cropped_")]
    print(f"{len(paths)} frames to process")

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2) as mesh:
        ok = 0
        for path in paths:
            save_dir = os.path.dirname(path)
            ok += int(process_image(path, save_dir, mesh))
    print(f"done: {ok}/{len(paths)} cropped faces saved")


if __name__ == "__main__":
    main()