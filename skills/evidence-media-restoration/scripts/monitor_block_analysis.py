#!/usr/bin/env python
"""Block-wise audit of small HMI registers from a registered monitor case.

The script deliberately produces images and stability diagnostics, not OCR.
It preserves a strict distinction between observed, fused and PSF-dependent views.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evidence_media.core import save_image, sha256_file, sharpness, write_csv, write_json
from evidence_media.fusion import clipped_huber_mean, split_temporal_blocks
from evidence_media.registration import align_to_reference
from evidence_media.restoration import deconvolution_sweep


DEFAULT_GROUPS = [
    ("hot_oil_zone_1", "Hot Oil Zone 1", [0, 1, 2, 3]),
    ("hot_oil_zone_2", "Hot Oil Zone 2", [4, 5, 6, 7]),
    ("aging_zone_3", "Aging Zone 3", [8, 9, 10, 11]),
    ("cool_down_zone_4", "Cool Down Zone 4", [12, 14, 16, 18]),
    ("zone_4_hour_average", "Cool Down Zone 4 - 1 Hour Average", [13, 15, 17, 19]),
]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def load_case(case: Path) -> tuple[list[np.ndarray], list[dict]]:
    rows = list(csv.DictReader((case / "source_observations.csv").open(encoding="utf-8-sig")))
    images = [read_image(case / row["file"]) for row in rows]
    return images, rows


def load_boxes(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    return [{k: int(float(v)) if k in {"register", "x", "y", "width", "height"} else v for k, v in row.items()} for row in rows]


def crop_box(image: np.ndarray, box: dict, pad: int) -> np.ndarray:
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    return image[y0:y1, x0:x1]


def core_crop(image: np.ndarray, pad: int, width: int, height: int) -> np.ndarray:
    return image[pad : pad + height, pad : pad + width]


def normalize_map(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)


def labelled_cell(image: np.ndarray, label: str, scale: int = 10) -> np.ndarray:
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cell = np.full((resized.shape[0] + 32, resized.shape[1], 3), 242, np.uint8)
    cell[: resized.shape[0]] = resized if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(cell, label, (4, resized.shape[0] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return cell


def join_row(cells: list[np.ndarray]) -> np.ndarray:
    height = max(c.shape[0] for c in cells)
    padded = []
    for cell in cells:
        canvas = np.full((height, cell.shape[1], 3), 242, np.uint8)
        canvas[: cell.shape[0]] = cell
        padded.append(canvas)
    return np.hstack(padded)


def join_column(rows: list[np.ndarray], title: str) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    header = np.full((46, width, 3), 238, np.uint8)
    cv2.putText(header, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (15, 15, 15), 2, cv2.LINE_AA)
    output = [header]
    for row in rows:
        canvas = np.full((row.shape[0], width, 3), 242, np.uint8)
        canvas[:, : row.shape[1]] = row
        output.append(canvas)
    return np.vstack(output)


def analyze_register(images: list[np.ndarray], source_rows: list[dict], box: dict, output: Path, pad: int) -> tuple[dict, np.ndarray]:
    crops = [crop_box(image, box, pad) for image in images]
    reference_index = int(np.argmax([sharpness(c) for c in crops]))
    reference = crops[reference_index]
    aligned, logs = [], []
    for index, crop in enumerate(crops):
        if index == reference_index:
            result = {"accepted": True, "ecc": 1.0, "motion": "identity"}
            candidate = crop
        else:
            candidate, result = align_to_reference(reference, crop, "translation", 180, 1e-7)
            refinement_ok = result.get("ecc") is not None and result["ecc"] >= 0.60
            result["local_refinement_accepted"] = refinement_ok
            if not refinement_ok:
                candidate = crop
                result["accepted"] = True
                result["fallback"] = "whole-monitor-registration-only"
        result.update({"observation": index, "source_ordinal": source_rows[index].get("ordinal"), "time_seconds": source_rows[index].get("time_seconds")})
        logs.append(result)
        if result["accepted"]:
            aligned.append((index, candidate))
    if len(aligned) < 6:
        raise RuntimeError(f"Register {box['register']}: only {len(aligned)} local observations")
    accepted_images = [item[1] for item in aligned]
    best_local = int(np.argmax([sharpness(c) for c in accepted_images]))
    stack = np.stack(accepted_images).astype(np.float32)
    median = np.median(stack, axis=0)
    huber = clipped_huber_mean(stack, 1.4, 7)
    build_idx, hold_idx = split_temporal_blocks(len(accepted_images), 6)
    build_median = np.median(np.stack([accepted_images[i] for i in build_idx]).astype(np.float32), axis=0)
    hold_median = np.median(np.stack([accepted_images[i] for i in hold_idx]).astype(np.float32), axis=0)
    stability = np.abs(build_median - hold_median).mean(axis=2)
    width, height = box["width"], box["height"]
    views = {
        "best_observed": core_crop(accepted_images[best_local], pad, width, height),
        "temporal_median": core_crop(median, pad, width, height),
        "huber_mean": core_crop(huber, pad, width, height),
        "split_difference": core_crop(normalize_map(stability), pad, width, height),
    }
    local_deconv_config = {
        "deconvolution": {
            "enabled": True,
            "psf_size": 7,
            "gaussian_sigmas": [0.45, 0.65, 0.85, 1.1],
            "motion_lengths": [],
            "wiener_balances": [0.002, 0.006, 0.018],
            "rl_iterations": [3, 6, 9],
            "noise_sigma_target": float(np.median(stability)),
            "keep_top": 3,
        }
    }
    deconv, sweep, selected = deconvolution_sweep(views["huber_mean"], local_deconv_config)
    if selected:
        views["selected_model"] = deconv[selected]
    register_dir = output / f"register_{box['register']:02d}"
    register_dir.mkdir(parents=True, exist_ok=True)
    for name, image in views.items():
        save_image(register_dir / f"{name}.png", image)
    write_csv(register_dir / "local_alignment.csv", logs)
    write_csv(register_dir / "deconvolution_sweep.csv", sweep)
    panel_order = ["best_observed", "temporal_median", "huber_mean"] + (["selected_model"] if selected else []) + ["split_difference"]
    panel = join_row([labelled_cell(views[name], name) for name in panel_order])
    save_image(register_dir / "comparison.png", panel)
    top_objective = sweep[0]["objective"] if sweep else None
    objective_margin = sweep[1]["objective"] - sweep[0]["objective"] if len(sweep) > 1 else None
    row = {
        "register": box["register"],
        "x": box["x"],
        "y": box["y"],
        "width": width,
        "height": height,
        "local_accepted": len(aligned),
        "reference_observation": aligned[best_local][0],
        "reference_source_ordinal": source_rows[aligned[best_local][0]].get("ordinal"),
        "reference_time_seconds": source_rows[aligned[best_local][0]].get("time_seconds"),
        "split_difference_median": float(np.median(stability)),
        "split_difference_p90": float(np.percentile(stability, 90)),
        "selected_model": selected,
        "selected_objective": top_objective,
        "model_objective_margin": objective_margin,
        "reading": "UNRESOLVED",
    }
    return row, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registered_case", type=Path)
    parser.add_argument("boxes_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pad", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, source_rows = load_case(args.registered_case)
    boxes = load_boxes(args.boxes_csv)
    rows, panels = [], {}
    for box in boxes:
        row, panel = analyze_register(images, source_rows, box, args.output, args.pad)
        rows.append(row)
        panels[box["register"]] = panel
    group_sheets = []
    context_sheets = []
    full_views = {}
    for name, filename in [
        ("best_observed", "01_best_observed.png"),
        ("temporal_median", "02_temporal_median.png"),
        ("huber_mean", "03_huber_mean.png"),
        ("selected_model", "model_rl_gaussian_s0.70_i8.png"),
    ]:
        path = args.registered_case / filename
        if path.exists():
            full_views[name] = read_image(path)
    for key, title, indices in DEFAULT_GROUPS:
        sheet = join_column([panels[i] for i in indices], title)
        save_image(args.output / f"table_{key}.png", sheet)
        group_sheets.append(sheet)
        group_boxes = [box for box in boxes if box["register"] in indices]
        x0 = max(0, min(box["x"] for box in group_boxes) - 55)
        y0 = max(0, min(box["y"] for box in group_boxes) - 45)
        x1 = min(images[0].shape[1], max(box["x"] + box["width"] for box in group_boxes) + 55)
        y1 = min(images[0].shape[0], max(box["y"] + box["height"] for box in group_boxes) + 18)
        context_cells = [labelled_cell(view[y0:y1, x0:x1], name, scale=4) for name, view in full_views.items()]
        if context_cells:
            context_sheet = join_column([join_row(context_cells)], f"{title} - context")
            save_image(args.output / f"context_{key}.png", context_sheet)
            context_sheets.append(context_sheet)
        for row in rows:
            if row["register"] in indices:
                row["group"] = title
                row["T_row"] = f"T-{indices.index(row['register']) + 1}"
    overview = join_column(group_sheets, "AGING TOWER - 20 register block audit")
    save_image(args.output / "00_all_registers_method_atlas.png", overview)
    if context_sheets:
        save_image(args.output / "01_all_table_contexts.png", join_column(context_sheets, "AGING TOWER - table context audit"))
    write_csv(args.output / "register_metrics.csv", rows)
    manifest = {
        "schema": "evidence-media-restoration/monitor-block-audit-0.2",
        "registered_case": str(args.registered_case.resolve()),
        "registered_case_manifest_sha256": sha256_file(args.registered_case / "manifest.json"),
        "boxes_csv": str(args.boxes_csv.resolve()),
        "observations": len(images),
        "registers": len(boxes),
        "groups": [{"key": key, "title": title, "registers": indices} for key, title, indices in DEFAULT_GROUPS],
        "policy": {
            "ocr_used": False,
            "generative_model_used": False,
            "selected_model_class": "MODEL_DEPENDENT",
            "all_readings_initially": "UNRESOLVED",
        },
        "notes": [
            "Whole-monitor registration is followed by independent local translation registration for every register.",
            "PSF/regularization are selected by forward residual diagnostics, never expected digits.",
            "Split difference uses temporally blocked subsets from one dependent source stream.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
