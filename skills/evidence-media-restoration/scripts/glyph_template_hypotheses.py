#!/usr/bin/env python3
"""Generate explicitly non-evidentiary glyph hypotheses from split reconstructions.

Candidates are ranked against deterministic even/odd inverse reconstructions.
They remain MODEL_SUGGESTION until verified in native observation space and
calibrated against matched controls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\ARIALN.TTF"),
    Path(r"C:\Windows\Fonts\ARIALNB.TTF"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    Path(r"C:\Windows\Fonts\verdana.ttf"),
    Path(r"C:\Windows\Fonts\verdanab.ttf"),
    Path(r"C:\Windows\Fonts\micross.ttf"),
]


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cannot write {path}")
    encoded.tofile(str(path))


def latent(path: Path) -> np.ndarray:
    image = imread(path)
    return image[::12, ::12].astype(np.float32) / 255.0


def cell_ranges(width: int) -> list[tuple[int, int]]:
    fractions = [(0.12, 0.245), (0.245, 0.37), (0.37, 0.50), (0.58, 0.715), (0.715, 0.85)]
    return [(max(0, int(round(width * left))), min(width, int(round(width * right)))) for left, right in fractions]


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (8, 96))
    return np.clip((image - low) / max(float(high - low), 1e-5), 0, 1)


def render_digit(font_path: Path, digit: str, height: int, width: int, width_fraction: float, blur: float) -> np.ndarray:
    canvas_size = 192
    font = ImageFont.truetype(str(font_path), 150)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(canvas)
    bounds = draw.textbbox((0, 0), digit, font=font, stroke_width=0)
    draw.text((-bounds[0] + 8, -bounds[1] + 8), digit, fill=255, font=font)
    array = np.asarray(canvas, dtype=np.uint8)
    nonzero = cv2.findNonZero(array)
    if nonzero is None:
        return np.zeros((height, width), np.float32)
    x, y, glyph_width, glyph_height = cv2.boundingRect(nonzero)
    glyph = array[y : y + glyph_height, x : x + glyph_width]
    target_height = max(2, height)
    target_width = max(1, min(width, int(round(width * width_fraction))))
    resized = cv2.resize(glyph, (target_width, target_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((height, width), np.float32)
    left = max(0, (width - target_width) // 2)
    result[:, left : left + target_width] = resized.astype(np.float32) / 255.0
    if blur > 0:
        result = cv2.GaussianBlur(result, (0, 0), blur)
    return normalize(result)


def feature(image: np.ndarray) -> np.ndarray:
    normalized = normalize(image)
    enlarged = cv2.resize(normalized.astype(np.float32), (30, 48), interpolation=cv2.INTER_CUBIC)
    gradient_x = cv2.Sobel(enlarged, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(enlarged, cv2.CV_32F, 0, 1, ksize=3)
    vector = np.concatenate([enlarged.ravel(), 0.35 * gradient_x.ravel(), 0.35 * gradient_y.ravel()])
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scale", type=int, default=3, choices=(2, 3))
    parser.add_argument("--directory")
    args = parser.parse_args()
    output = args.output or args.case / "glyph_hypotheses"
    output.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = args.case / (args.directory or f"inverse_reconstruction_{args.scale}x")

    registers = []
    for register in range(100):
        full_path = reconstruction_dir / f"register_{register:02d}_lambda_0.10.png"
        if not full_path.exists():
            break
        registers.append(
            {
                "full": latent(full_path),
                "even": latent(reconstruction_dir / f"register_{register:02d}_even.png"),
                "odd": latent(reconstruction_dir / f"register_{register:02d}_odd.png"),
            }
        )
    if not registers:
        raise RuntimeError("no inverse reconstructions found")

    empirical_features = {}
    for digit in range(1, 5):
        path = args.case / "known_label_digits" / f"digit_{digit}_empirical_median.png"
        empirical = imread(path).astype(np.float32) / 255.0
        empirical_features[str(digit)] = feature(1.0 - empirical)

    observations = []
    identities = []
    for register, images in enumerate(registers):
        height, width = images["full"].shape
        for position, (left, right) in enumerate(cell_ranges(width)):
            top, bottom = 1, max(2, height - 2)
            crops = [images[name][top:bottom, left:right] for name in ("full", "even", "odd")]
            observations.append(crops)
            identities.append((register, position, left, right, top, bottom))

    parameter_sets = []
    for font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue
        for width_fraction in (0.72, 0.86, 1.0):
            for blur in (0.25, 0.55, 0.85):
                parameter_sets.append((font_path, width_fraction, blur))

    calibration_scores = []
    for font_path, width_fraction, blur in parameter_sets:
        known_scores = []
        for digit in "1234":
            template = render_digit(font_path, digit, 36, 24, width_fraction, blur)
            known_scores.append(float(np.dot(feature(template), empirical_features[digit])))
        aggregate = float(np.mean(known_scores) - 0.20 * np.std(known_scores))
        key = (font_path.name, width_fraction, blur)
        calibration_scores.append((aggregate, key, known_scores))

    calibration_scores.sort(reverse=True)
    best_key = calibration_scores[0][1]
    selected_font = next(path for path in FONT_CANDIDATES if path.name == best_key[0])
    rankings = []
    for crops in observations:
        height, width = crops[0].shape
        observed_features = [feature(crop) for crop in crops]
        candidates = []
        for digit in "0123456789":
            template = render_digit(selected_font, digit, height, width, best_key[1], best_key[2])
            template_feature = feature(template)
            rendered_correlations = [float(np.dot(observed, template_feature)) for observed in observed_features]
            rendered_score = 0.45 * rendered_correlations[0] + 0.275 * rendered_correlations[1] + 0.275 * rendered_correlations[2]
            split_penalty = 0.20 * abs(rendered_correlations[1] - rendered_correlations[2])
            if digit in empirical_features:
                empirical_correlations = [float(np.dot(observed, empirical_features[digit])) for observed in observed_features]
                empirical_score = 0.45 * empirical_correlations[0] + 0.275 * empirical_correlations[1] + 0.275 * empirical_correlations[2]
                score = 0.60 * empirical_score + 0.40 * rendered_score - split_penalty
                bridge = "empirical+rendered"
            else:
                empirical_correlations = []
                score = rendered_score - split_penalty - 0.05
                bridge = "rendered_extrapolation"
            candidates.append((score, digit, rendered_correlations, empirical_correlations, bridge))
        candidates.sort(reverse=True)
        rankings.append(candidates)
    output_rows = []
    for identity, candidates in zip(identities, rankings):
        register, position, left, right, top, bottom = identity
        top_candidates = candidates[:5]
        margin = top_candidates[0][0] - top_candidates[1][0]
        output_rows.append(
            [
                register,
                position,
                "/".join(item[1] for item in top_candidates),
                "/".join(f"{item[0]:.5f}" for item in top_candidates),
                "/".join(item[4] for item in top_candidates),
                margin,
                left,
                right,
                top,
                bottom,
            ]
        )
    with (output / "model_suggestion_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["register", "position", "ranked_digits", "scores", "bridges", "top_margin", "left", "right", "top", "bottom"])
        writer.writerows(output_rows)
    with (output / "font_parameter_ranking.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["calibration_score", "font", "width_fraction", "blur_sigma", "digit_1", "digit_2", "digit_3", "digit_4"])
        for score, key, known_scores in calibration_scores:
            writer.writerow([score, *key, *known_scores])

    empirical_validation = []
    for expected_digit in "1234":
        source_feature = empirical_features[expected_digit]
        ranking = sorted(
            ((float(np.dot(source_feature, candidate_feature)), candidate_digit) for candidate_digit, candidate_feature in empirical_features.items()),
            reverse=True,
        )
        empirical_validation.append({"expected": expected_digit, "ranking": ranking, "correct": ranking[0][1] == expected_digit})

    manifest = {
        "evidence_class": "MODEL_SUGGESTION",
        "registers": len(registers),
        "cells": len(observations),
        "selected_font_parameters": {
            "font": best_key[0],
            "width_fraction": best_key[1],
            "blur_sigma": best_key[2],
        },
        "font_selection_basis": "known T-1 through T-4 label digits on the same HMI screen",
        "empirical_self_validation": empirical_validation,
        "warning": "Candidate rankings are not literal readings and are not calibrated probabilities.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
