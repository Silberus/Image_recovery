from __future__ import annotations

import cv2
import numpy as np

from .core import sharpness


def clipped_huber_mean(stack: np.ndarray, delta: float = 1.5, iterations: int = 5) -> np.ndarray:
    x = stack.astype(np.float32)
    estimate = np.median(x, axis=0)
    scale = np.median(np.abs(x - estimate), axis=0) * 1.4826 + 1.0
    for _ in range(iterations):
        residual = (x - estimate) / scale
        weights = np.minimum(1.0, float(delta) / (np.abs(residual) + 1e-6))
        estimate = np.sum(weights * x, axis=0) / np.maximum(np.sum(weights, axis=0), 1e-6)
    return np.clip(estimate, 0, 255)


def lucky_tile_mosaic(images: list[np.ndarray], tile: int, overlap: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float]]]:
    h, w = images[0].shape[:2]
    accum = np.zeros_like(images[0], dtype=np.float32)
    weight = np.zeros((h, w), np.float32)
    donor_map = np.zeros((h, w), np.uint16)
    donors = []
    step = max(4, tile - overlap)
    window_1d = np.hanning(tile) if tile > 4 else np.ones(tile)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(h, y + tile), min(w, x + tile)
            local_window = cv2.resize(window, (x2 - x, y2 - y))
            scores = [sharpness(im[y:y2, x:x2]) for im in images]
            donor = int(np.argmax(scores))
            patch = images[donor][y:y2, x:x2].astype(np.float32)
            accum[y:y2, x:x2] += patch * local_window[..., None]
            weight[y:y2, x:x2] += local_window
            donor_map[y:y2, x:x2] = donor + 1
            donors.append({"x": x, "y": y, "width": x2 - x, "height": y2 - y, "donor": donor, "sharpness": float(scores[donor])})
    result = accum / np.maximum(weight[..., None], 1e-6)
    return np.clip(result, 0, 255), donor_map, donors


def split_temporal_blocks(count: int, block: int) -> tuple[list[int], list[int]]:
    build, holdout = [], []
    for i in range(count):
        (build if (i // max(2, block)) % 2 == 0 else holdout).append(i)
    if not holdout:
        build = list(range(0, count, 2))
        holdout = list(range(1, count, 2))
    return build, holdout


def fuse(images: list[np.ndarray], config: dict) -> tuple[dict[str, np.ndarray], dict]:
    if len(images) < 2:
        raise ValueError("At least two registered observations are required")
    fcfg = config.get("fusion", {})
    build_idx, hold_idx = split_temporal_blocks(len(images), int(fcfg.get("validation_block_frames", 6)))
    build = [images[i] for i in build_idx]
    stack = np.stack(build).astype(np.float32)
    sharp_scores = [sharpness(i) for i in build]
    best = build[int(np.argmax(sharp_scores))]
    median = np.median(stack, axis=0)
    huber = clipped_huber_mean(stack, float(fcfg.get("huber_delta", 1.5)), int(fcfg.get("huber_iterations", 5)))
    mosaic, donor_map, donor_rows = lucky_tile_mosaic(build, int(fcfg.get("tile_size", 96)), int(fcfg.get("tile_overlap", 24)))
    mad = np.median(np.abs(stack - np.median(stack, axis=0)), axis=0).mean(axis=2)
    support = np.sum(np.mean(np.abs(stack - median), axis=3) < float(fcfg.get("support_threshold", 18.0)), axis=0)
    outputs = {
        "best_observed": best,
        "temporal_median": median,
        "huber_mean": huber,
        "observed_tile_mosaic": mosaic,
        "temporal_mad": mad,
        "support_count": support,
        "donor_map": donor_map,
    }
    meta = {"build_indices": build_idx, "holdout_indices": hold_idx, "donor_rows": donor_rows, "best_build_index": int(np.argmax(sharp_scores))}
    return outputs, meta
