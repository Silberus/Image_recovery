#!/usr/bin/env python3
"""Fail-closed analysis of tiny static or changing HMI fields.

The input is the ``aligned_observations`` directory produced by
``native_register_superresolution.py``.  The script never performs OCR and
never manufactures a glyph.  It preserves each observation, detects temporal
state changes, fuses only compatible states, and emits topology/support views
for a human audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def _imwrite(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalize(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (2.0, 98.0))
    if hi - lo < 1.0:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _huber(stack: np.ndarray, iterations: int = 5, delta: float = 1.5) -> np.ndarray:
    estimate = np.median(stack, axis=0)
    for _ in range(iterations):
        residual = stack - estimate[None, ...]
        scale = np.median(np.abs(residual), axis=0) * 1.4826 + 1e-3
        weight = np.minimum(1.0, delta * scale[None, ...] / (np.abs(residual) + 1e-6))
        estimate = np.sum(weight * stack, axis=0) / np.maximum(np.sum(weight, axis=0), 1e-6)
    return estimate


def _text_band(gray_stack: np.ndarray) -> tuple[int, int]:
    median = np.median(gray_stack, axis=0).astype(np.uint8)
    normalized = _normalize(median)
    # Text produces repeated vertical transitions (x-gradient). A window border
    # can be very dark but nearly constant across x and must not win the band
    # detector merely because it contains more black pixels.
    x_gradient = np.abs(cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3))
    profile = np.mean(x_gradient, axis=1)
    profile = cv2.GaussianBlur(profile[:, None], (1, 0), 2.0).ravel()
    threshold = max(float(np.percentile(profile, 70)), float(np.max(profile) * 0.28))
    rows = np.flatnonzero(profile >= threshold)
    if rows.size == 0:
        return 0, median.shape[0]
    # Retain the strongest contiguous group; distant groups commonly belong to
    # borders or neighboring UI controls.
    groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
    group = max(groups, key=lambda item: float(np.sum(profile[item])))
    return max(0, int(group[0]) - 8), min(median.shape[0], int(group[-1]) + 9)


def _segments(
    gray_stack: np.ndarray, y0: int, y1: int, max_changes: int
) -> tuple[list[tuple[int, int]], list[float], dict[str, Any]]:
    normalized = np.asarray([_normalize(frame) for frame in gray_stack], dtype=np.float32)
    blurred = np.asarray([cv2.GaussianBlur(frame[y0:y1], (0, 0), 1.2) for frame in normalized])
    diffs = [0.0]
    for index in range(1, len(blurred)):
        diffs.append(float(np.mean(np.abs(blurred[index] - blurred[index - 1]))))
    audit: dict[str, Any] = {"mode": "persistent_change_point", "max_changes": max_changes, "candidates": []}
    if max_changes <= 0 or len(blurred) < 8:
        audit["accepted_cut"] = None
        return [(0, len(gray_stack))], diffs, audit

    min_side = max(4, len(blurred) // 8)
    candidates: list[dict[str, float | int]] = []
    for cut in range(min_side, len(blurred) - min_side + 1):
        left, right = blurred[:cut], blurred[cut:]
        left_median, right_median = np.median(left, axis=0), np.median(right, axis=0)
        between = np.abs(left_median - right_median)
        left_noise = np.median(np.abs(left - left_median[None, ...]), axis=0)
        right_noise = np.median(np.abs(right - right_median[None, ...]), axis=0)
        noise = left_noise + right_noise + 1.0
        excess = np.maximum(between - 2.0 * noise, 0.0)
        column_energy = np.mean(excess, axis=0)
        ordered = np.sort(column_energy)[::-1]
        top = max(1, len(ordered) // 5)
        concentration = float(np.sum(ordered[:top]) / max(float(np.sum(ordered)), 1e-6))
        candidate = {
            "cut": cut,
            "score": float(np.mean(excess) / max(float(np.mean(noise)), 1e-6)),
            "significant_fraction": float(np.mean(excess > 3.0)),
            "top_20pct_column_concentration": concentration,
        }
        candidates.append(candidate)
    best = max(candidates, key=lambda row: float(row["score"]))
    audit["candidates"] = candidates
    # A state change must persist on both sides and outperform the within-state
    # blur/noise.  It must also be spatially localized; global differences are
    # more likely focus, perspective, or compression changes.
    accepted = (
        float(best["score"]) >= 0.55
        and 0.002 <= float(best["significant_fraction"]) <= 0.30
        and float(best["top_20pct_column_concentration"]) >= 0.45
    )
    audit["accepted_cut"] = int(best["cut"]) if accepted else None
    audit["acceptance_thresholds"] = {
        "score_min": 0.55,
        "significant_fraction_min": 0.002,
        "significant_fraction_max": 0.30,
        "top_20pct_column_concentration_min": 0.45,
    }
    if not accepted:
        return [(0, len(gray_stack))], diffs, audit
    cut = int(best["cut"])
    return [(0, cut), (cut, len(gray_stack))], diffs, audit


def _topology(stack: np.ndarray, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.asarray([_normalize(frame) for frame in stack], dtype=np.uint8)
    median = np.median(normalized, axis=0).astype(np.uint8)
    masks = np.zeros_like(normalized, dtype=bool)
    for index, frame in enumerate(normalized):
        band = frame[y0:y1]
        threshold = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        masks[index, y0:y1] = band < threshold
    support = np.mean(masks, axis=0)
    stable = np.zeros_like(median)
    stable[support >= 0.80] = 255
    uncertain = np.zeros_like(median)
    uncertain[(support >= 0.20) & (support < 0.80)] = 255
    return median, stable, uncertain


def _render_profile(values: np.ndarray, width: int, height: int = 160) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    vmax = max(float(np.max(values)), 1e-6)
    points = []
    for x in range(width):
        index = min(len(values) - 1, int(x * len(values) / width))
        y = height - 12 - int(values[index] / vmax * (height - 28))
        points.append((x, y))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _atlas(images: list[tuple[str, np.ndarray]], scale: int = 3) -> np.ndarray:
    tiles: list[np.ndarray] = []
    max_width = max(image.shape[1] for _, image in images) * scale
    for label, image in images:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        enlarged = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        tile = np.full((enlarged.shape[0] + 28, max_width, 3), 245, dtype=np.uint8)
        cv2.putText(tile, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        tile[28 : 28 + enlarged.shape[0], : enlarged.shape[1]] = enlarged
        tiles.append(tile)
    return np.vstack(tiles)


def analyze(input_dir: Path, output_dir: Path, max_changes: int) -> dict[str, Any]:
    ledger = input_dir / "observations.csv"
    rows = list(csv.DictReader(ledger.open("r", encoding="utf-8-sig", newline="")))
    if len(rows) < 2:
        raise RuntimeError("At least two aligned observations are required")
    images = [_imread(input_dir / row["file"], cv2.IMREAD_GRAYSCALE) for row in rows]
    if len({image.shape for image in images}) != 1:
        raise RuntimeError("Aligned observations have inconsistent dimensions")
    stack = np.asarray(images, dtype=np.float32)
    y0, y1 = _text_band(stack)
    segments, frame_diffs, change_audit = _segments(stack, y0, y1, max_changes)
    temporal_mad = np.median(np.abs(stack - np.median(stack, axis=0, keepdims=True)), axis=0)
    x_change = np.mean(temporal_mad[y0:y1], axis=0)
    normalized_median, stable, uncertain = _topology(stack, y0, y1)

    output_dir.mkdir(parents=True, exist_ok=True)
    mad_view = np.clip(temporal_mad / max(float(np.percentile(temporal_mad, 99)), 1e-6) * 255, 0, 255).astype(np.uint8)
    _imwrite(output_dir / "01_all_frame_median.png", normalized_median)
    _imwrite(output_dir / "02_temporal_mad.png", mad_view)
    _imwrite(output_dir / "03_stable_ink_support_ge_80pct.png", stable)
    _imwrite(output_dir / "04_uncertain_ink_support_20_80pct.png", uncertain)
    _imwrite(output_dir / "05_horizontal_change_profile.png", _render_profile(x_change, images[0].shape[1] * 3))

    atlas_items: list[tuple[str, np.ndarray]] = []
    segment_rows: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(segments, start=1):
        fused = np.clip(_huber(stack[start:end]), 0, 255).astype(np.uint8)
        normalized = _normalize(fused)
        name = f"segment_{number:02d}_frames_{start:02d}_{end - 1:02d}.png"
        _imwrite(output_dir / name, normalized)
        atlas_items.append((f"state {number}: observations {start}-{end - 1}", normalized))
        segment_rows.append(
            {
                "segment": number,
                "start_observation": start,
                "end_observation": end - 1,
                "start_time_seconds": float(rows[start]["time_seconds"]),
                "end_time_seconds": float(rows[end - 1]["time_seconds"]),
                "observations": end - start,
                "file": name,
            }
        )
    _imwrite(output_dir / "06_temporal_state_atlas.png", _atlas(atlas_items))

    with (output_dir / "temporal_states.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(segment_rows)

    changed = len(segments) > 1
    manifest: dict[str, Any] = {
        "schema": "evidence-media-restoration/temporal-field-topology/0.1",
        "input_ledger": str(ledger),
        "input_ledger_sha256": _sha256(ledger),
        "observations": len(rows),
        "image_shape": list(images[0].shape),
        "detected_text_band_y": [y0, y1],
        "temporal_change_audit": change_audit,
        "frame_to_frame_residuals": frame_diffs,
        "temporal_states": segment_rows,
        "field_class": "CHANGING" if changed else "STATIC_WITHIN_INTERVAL",
        "policy": {
            "ocr_used": False,
            "generative_model_used": False,
            "reading_status": "UNRESOLVED until source-observation topology audit",
            "classification": "DERIVED_DETERMINISTIC",
            "fusion_rule": "Only observations inside the same detected temporal state are fused",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="aligned_observations directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-changes",
        type=int,
        default=1,
        help="Maximum semantic state changes allowed in this interval; use 0 for a known-static field",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.input_dir.resolve(), args.output_dir.resolve(), max(0, args.max_changes)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
