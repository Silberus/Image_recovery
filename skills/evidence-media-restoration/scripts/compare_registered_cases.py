#!/usr/bin/env python
"""Compare and fuse two registered manifestations from one witness cluster."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evidence_media.core import residual_diagnostics, save_image, sha256_file, ssim_global, write_csv, write_json
from evidence_media.registration import align_to_reference
from evidence_media.restoration import deconvolution_sweep


GROUPS = [
    ("Hot Oil Zone 1", [0, 1, 2, 3]),
    ("Hot Oil Zone 2", [4, 5, 6, 7]),
    ("Aging Zone 3", [8, 9, 10, 11]),
    ("Cool Down Zone 4", [12, 14, 16, 18]),
    ("Cool Down Zone 4 - 1 Hour Average", [13, 15, 17, 19]),
]


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def labelled(image: np.ndarray, label: str, scale: int = 10) -> np.ndarray:
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((view.shape[0] + 30, view.shape[1], 3), 242, np.uint8)
    canvas[: view.shape[0]] = view
    cv2.putText(canvas, label, (4, view.shape[0] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
    return canvas


def row(cells: list[np.ndarray]) -> np.ndarray:
    h = max(c.shape[0] for c in cells)
    normalized = []
    for c in cells:
        x = np.full((h, c.shape[1], 3), 242, np.uint8)
        x[: c.shape[0]] = c
        normalized.append(x)
    return np.hstack(normalized)


def column(rows: list[np.ndarray], title: str) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    header = np.full((42, w, 3), 240, np.uint8)
    cv2.putText(header, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2, cv2.LINE_AA)
    output = [header]
    for r in rows:
        x = np.full((r.shape[0], w, 3), 242, np.uint8)
        x[:, : r.shape[1]] = r
        output.append(x)
    return np.vstack(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_a", type=Path)
    parser.add_argument("case_b", type=Path)
    parser.add_argument("boxes_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a = read(args.case_a / "03_huber_mean.png")
    b_raw = read(args.case_b / "03_huber_mean.png")
    b, alignment = align_to_reference(a, b_raw, "affine", 300, 1e-8)
    consensus = np.clip((a.astype(np.float32) + b.astype(np.float32)) / 2, 0, 255)
    difference = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    save_image(args.output / "01_encode_a_huber.png", a)
    save_image(args.output / "02_encode_b_huber_aligned.png", b)
    save_image(args.output / "03_cross_encode_consensus.png", consensus)
    save_image(args.output / "04_cross_encode_difference.png", cv2.applyColorMap(cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_MAGMA))
    boxes = [{k: int(float(v)) if k in {"register", "x", "y", "width", "height"} else v for k, v in item.items()} for item in csv.DictReader(args.boxes_csv.open(encoding="utf-8-sig"))]
    metrics, panels = [], {}
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        ca, cb, cc = a[y : y + h, x : x + w], b[y : y + h, x : x + w], consensus[y : y + h, x : x + w]
        diff = np.mean(np.abs(ca.astype(np.float32) - cb.astype(np.float32)), axis=2)
        deconv, sweep, selected = deconvolution_sweep(cc, {"deconvolution": {"enabled": True, "psf_size": 7, "gaussian_sigmas": [0.45, 0.65, 0.85, 1.1], "motion_lengths": [], "wiener_balances": [0.002, 0.006, 0.018], "rl_iterations": [3, 6, 9], "noise_sigma_target": float(np.median(diff)), "keep_top": 3}})
        model = deconv[selected] if selected else cc
        heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
        panel = row([labelled(ca, "encode_A"), labelled(cb, "encode_B"), labelled(cc, "dependent_consensus"), labelled(model, "selected_model"), labelled(heat, "A-B_difference")])
        panels[box["register"]] = panel
        metrics.append({
            "register": box["register"],
            "cross_encode_mae": float(diff.mean()),
            "cross_encode_p90": float(np.percentile(diff, 90)),
            "global_ssim": ssim_global(ca, cb),
            "selected_model": selected,
            "selected_objective": sweep[0]["objective"] if sweep else None,
            "classification": "DEPENDENT_MANIFESTATIONS",
            "reading": "UNRESOLVED",
        })
    sheets = []
    for title, indices in GROUPS:
        sheet = column([panels[i] for i in indices], title)
        sheets.append(sheet)
    save_image(args.output / "00_cross_encode_register_atlas.png", column(sheets, "AGING TOWER - dependent encode comparison"))
    write_csv(args.output / "cross_encode_register_metrics.csv", metrics)
    manifest = {
        "schema": "evidence-media-restoration/dependent-manifestation-comparison-0.2",
        "case_a_manifest_sha256": sha256_file(args.case_a / "manifest.json"),
        "case_b_manifest_sha256": sha256_file(args.case_b / "manifest.json"),
        "alignment": alignment,
        "whole_monitor_diagnostics": {"ssim": ssim_global(a, b), **residual_diagnostics(a, b)},
        "policy": {"independent_witnesses": 1, "consensus_class": "DERIVED_DETERMINISTIC_DEPENDENT", "ocr_used": False, "generative_model_used": False},
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
