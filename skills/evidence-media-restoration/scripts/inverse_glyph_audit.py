#!/usr/bin/env python3
"""Audit native-frame phase diversity and extract repeated screen registers.

This program does not identify characters.  It preserves source-to-screen
homographies, measures native sensor phases before image resampling, detects
repeated dark numeric boxes on the rectified reference screen, and exports
neutral register atlases for later blind glyph modelling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def parse_corners(value: str) -> np.ndarray:
    points = []
    for item in value.split(";"):
        x, y = item.split(",")
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise argparse.ArgumentTypeError("four x,y points required")
    return np.float64(points)


def parse_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x")
    return int(width), int(height)


def parse_region(value: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in value.split(","))
    return x0, y0, x1, y1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(str(path))


def decode_interval(path: Path, start: float, end: float) -> tuple[list[dict], dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(math.floor(start * fps + 1e-6)))
    end_frame = min(frame_count, int(math.ceil(end * fps - 1e-6)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    records = []
    for frame_no in range(start_frame, end_frame):
        ok, image = capture.read()
        if not ok:
            break
        records.append(
            {
                "sequence": len(records),
                "frame_no": frame_no,
                "time_s": frame_no / fps,
                "image": image,
            }
        )
    capture.release()
    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "decoded": len(records),
    }
    return records, metadata


def frame_digest(image: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest().upper()


def estimate_homographies(records: list[dict], reference_index: int, screen_polygon: np.ndarray) -> tuple[list[dict], list[list]]:
    reference = records[reference_index]["image"]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=16000, contrastThreshold=0.015, edgeThreshold=16)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted = []
    log = []
    for index, record in enumerate(records):
        if index == reference_index:
            homography = np.eye(3, dtype=np.float64)
            accepted.append({**record, "H_source_to_reference": homography, "bootstrap_H": [homography]})
            log.append([record["frame_no"], record["time_s"], 9999, 9999, 0.0, "reference"])
            continue
        gray = cv2.cvtColor(record["image"], cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            log.append([record["frame_no"], record["time_s"], 0, 0, None, "no_descriptors"])
            continue
        pairs = matcher.knnMatch(des, des_ref, k=2)
        good = [m for m, n in pairs if m.distance < 0.70 * n.distance]
        good = [
            match
            for match in good
            if cv2.pointPolygonTest(screen_polygon.astype(np.float32), kp_ref[match.trainIdx].pt, False) >= 0
        ]
        if len(good) < 40:
            log.append([record["frame_no"], record["time_s"], len(good), 0, None, "too_few_matches"])
            continue
        source = np.float64([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float64([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 0.90)
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < 32:
            log.append([record["frame_no"], record["time_s"], len(good), inliers, None, "bad_homography"])
            continue
        projected = cv2.perspectiveTransform(source[mask.ravel().astype(bool)], homography)
        residual = float(np.median(np.linalg.norm(projected - target[mask.ravel().astype(bool)], axis=2)))
        inlier_source = source[mask.ravel().astype(bool)]
        inlier_target = target[mask.ravel().astype(bool)]
        random = np.random.default_rng(record["frame_no"])
        bootstraps = []
        subset_size = min(96, len(inlier_source))
        if subset_size >= 16:
            for _ in range(40):
                chosen = random.choice(len(inlier_source), size=subset_size, replace=False)
                candidate, _ = cv2.findHomography(inlier_source[chosen], inlier_target[chosen], 0)
                if candidate is not None:
                    bootstraps.append(candidate)
        accepted.append({**record, "H_source_to_reference": homography, "bootstrap_H": bootstraps or [homography]})
        log.append([record["frame_no"], record["time_s"], len(good), inliers, residual, "accepted"])
    return accepted, log


def alias_matrix(phases: np.ndarray, scale: int) -> np.ndarray:
    columns = []
    for vertical in range(scale):
        for horizontal in range(scale):
            columns.append(np.exp(-2j * np.pi * (horizontal * phases[:, 0] + vertical * phases[:, 1])))
    return np.stack(columns, axis=1)


def phase_audit(records: list[dict], H_reference_to_screen: np.ndarray, anchors: list[tuple[float, float]]) -> list[dict]:
    results = []
    for anchor_no, (screen_x, screen_y) in enumerate(anchors):
        phases = []
        locations = []
        bootstrap_sigmas = []
        for record in records:
            H_source_to_screen = H_reference_to_screen @ record["H_source_to_reference"]
            screen_to_source = np.linalg.inv(H_source_to_screen)
            point = screen_to_source @ np.array([screen_x, screen_y, 1.0])
            point = point[:2] / point[2]
            locations.append(point)
            phases.append(np.mod(point, 1.0))
            bootstrap_locations = []
            for candidate_H in record["bootstrap_H"]:
                candidate_source_to_screen = H_reference_to_screen @ candidate_H
                candidate_screen_to_source = np.linalg.inv(candidate_source_to_screen)
                candidate_point = candidate_screen_to_source @ np.array([screen_x, screen_y, 1.0])
                bootstrap_locations.append(candidate_point[:2] / candidate_point[2])
            bootstrap_locations = np.asarray(bootstrap_locations)
            bootstrap_sigmas.append(float(np.sqrt(np.var(bootstrap_locations[:, 0]) + np.var(bootstrap_locations[:, 1]))))
        phases_array = np.asarray(phases, dtype=np.float64)
        locations_array = np.asarray(locations, dtype=np.float64)
        for scale in (2, 3, 4):
            matrix = alias_matrix(phases_array, scale)
            singular = np.linalg.svd(matrix, compute_uv=False)
            tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
            rank = int(np.count_nonzero(singular > tolerance))
            condition = float(singular[0] / singular[-1]) if singular[-1] > tolerance else None
            bins = np.floor(phases_array * scale).astype(int)
            unique_bins = sorted({(int(x), int(y)) for x, y in bins})
            results.append(
                {
                    "anchor": anchor_no,
                    "screen_xy": [screen_x, screen_y],
                    "scale": scale,
                    "observations": len(records),
                    "unique_phase_bins": len(unique_bins),
                    "required_bins": scale * scale,
                    "phase_bins": unique_bins,
                    "matrix_rank": rank,
                    "required_rank": scale * scale,
                    "condition_number": condition,
                    "bootstrap_native_sigma_px_median": float(np.median(bootstrap_sigmas)),
                    "bootstrap_native_sigma_px_p90": float(np.percentile(bootstrap_sigmas, 90)),
                    "native_x_range": [float(locations_array[:, 0].min()), float(locations_array[:, 0].max())],
                    "native_y_range": [float(locations_array[:, 1].min()), float(locations_array[:, 1].max())],
                }
            )
    return results


def detect_registers(image: np.ndarray, region: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = gray[y0:y1, x0:x1]
    mask = np.uint8(crop < 70) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 38 <= width <= 120 and 9 <= height <= 30 and width / max(height, 1) >= 2.0:
            boxes.append((x + x0, y + y0, width, height))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def save_register_atlas(rectified: list[dict], boxes: list[tuple[int, int, int, int]], output: Path) -> None:
    if not boxes:
        return
    atlas_dir = output / "registers"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (x, y, width, height) in enumerate(boxes):
        patches = [item["rectified"][y : y + height, x : x + width] for item in rectified]
        stack = np.stack(patches, axis=0)
        median = np.uint8(np.median(stack, axis=0))
        mad = np.median(np.abs(stack.astype(np.float32) - median.astype(np.float32)), axis=(0, 3))
        imwrite(atlas_dir / f"register_{number:02d}_median.png", cv2.resize(median, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
        rows.append([number, x, y, width, height, float(np.median(mad)), float(np.percentile(mad, 90))])
    with (output / "register_boxes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["register", "x", "y", "width", "height", "temporal_mad_median", "temporal_mad_p90"])
        writer.writerows(rows)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, homography).reshape(-1, 2)


def local_screen_scale(H_source_to_screen: np.ndarray, screen_center: tuple[float, float]) -> float:
    inverse = np.linalg.inv(H_source_to_screen)
    source_center = project_points(np.asarray([screen_center]), inverse)[0]
    source_basis = np.asarray([source_center, source_center + (1.0, 0.0), source_center + (0.0, 1.0)])
    screen_basis = project_points(source_basis, H_source_to_screen)
    vector_x = screen_basis[1] - screen_basis[0]
    vector_y = screen_basis[2] - screen_basis[0]
    area = abs(float(vector_x[0] * vector_y[1] - vector_x[1] * vector_y[0]))
    return math.sqrt(max(area, 1e-9))


def native_drizzle_register(
    records: list[dict],
    box: tuple[int, int, int, int],
    scale: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x, y, width, height = box
    reference = min(records, key=lambda item: float(np.linalg.norm(item["H_source_to_reference"] - np.eye(3))))
    units_per_source_pixel = local_screen_scale(reference["H_source_to_screen"], (x + width / 2, y + height / 2))
    step = units_per_source_pixel / scale
    target_width = max(2, int(math.ceil(width / step)))
    target_height = max(2, int(math.ceil(height / step)))
    accumulated = np.zeros((target_height, target_width), np.float64)
    weights = np.zeros((target_height, target_width), np.float64)
    screen_polygon = np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64,
    )
    for record in records:
        inverse = np.linalg.inv(record["H_source_to_screen"])
        source_polygon = project_points(screen_polygon, inverse)
        min_x = max(0, int(math.floor(source_polygon[:, 0].min())) - 2)
        max_x = min(record["image"].shape[1] - 1, int(math.ceil(source_polygon[:, 0].max())) + 2)
        min_y = max(0, int(math.floor(source_polygon[:, 1].min())) - 2)
        max_y = min(record["image"].shape[0] - 1, int(math.ceil(source_polygon[:, 1].max())) + 2)
        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        source_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
        projected = project_points(source_points, record["H_source_to_screen"])
        inside = (
            (projected[:, 0] >= x)
            & (projected[:, 0] < x + width)
            & (projected[:, 1] >= y)
            & (projected[:, 1] < y + height)
        )
        if not np.any(inside):
            continue
        source_points = source_points[inside]
        projected = projected[inside]
        luma = cv2.cvtColor(record["image"], cv2.COLOR_BGR2YCrCb)[:, :, 0]
        values = luma[source_points[:, 1].astype(int), source_points[:, 0].astype(int)].astype(np.float64)
        target_x = (projected[:, 0] - x) / step - 0.5
        target_y = (projected[:, 1] - y) / step - 0.5
        base_x = np.floor(target_x).astype(int)
        base_y = np.floor(target_y).astype(int)
        frac_x = target_x - base_x
        frac_y = target_y - base_y
        for offset_x, offset_y, component in (
            (0, 0, (1 - frac_x) * (1 - frac_y)),
            (1, 0, frac_x * (1 - frac_y)),
            (0, 1, (1 - frac_x) * frac_y),
            (1, 1, frac_x * frac_y),
        ):
            destination_x = base_x + offset_x
            destination_y = base_y + offset_y
            valid = (
                (destination_x >= 0)
                & (destination_x < target_width)
                & (destination_y >= 0)
                & (destination_y < target_height)
            )
            np.add.at(accumulated, (destination_y[valid], destination_x[valid]), values[valid] * component[valid])
            np.add.at(weights, (destination_y[valid], destination_x[valid]), component[valid])
    reconstructed = np.divide(accumulated, weights, out=np.zeros_like(accumulated), where=weights > 1e-8)
    return np.uint8(np.clip(reconstructed, 0, 255)), weights, units_per_source_pixel


def normalized_cell(cell: np.ndarray) -> np.ndarray:
    floating = cell.astype(np.float32)
    low, high = np.percentile(floating, (10, 95))
    normalized = np.clip((floating - low) / max(float(high - low), 1.0), 0, 1)
    return cv2.resize(normalized, (24, 36), interpolation=cv2.INTER_CUBIC)


def build_neutral_glyph_atlas(reconstructions: list[np.ndarray], output: Path) -> dict:
    cells = []
    identities = []
    for register, image in enumerate(reconstructions):
        height, width = image.shape
        ranges = [
            (0.12, 0.245),
            (0.245, 0.37),
            (0.37, 0.50),
            (0.58, 0.715),
            (0.715, 0.85),
        ]
        for position, (left_fraction, right_fraction) in enumerate(ranges):
            left = max(0, int(round(width * left_fraction)))
            right = min(width, int(round(width * right_fraction)))
            top = 1
            bottom = max(top + 1, height - 2)
            cells.append(normalized_cell(image[top:bottom, left:right]))
            identities.append((register, position, left, right, top, bottom))
    data = np.asarray([cell.ravel() for cell in cells], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 250, 1e-5)
    compactness, labels, centers = cv2.kmeans(data, 10, None, criteria, 30, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    atlas_dir = output / "neutral_glyph_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    assignments = []
    for index, (identity, label, cell) in enumerate(zip(identities, labels, cells)):
        register, position, left, right, top, bottom = identity
        assignments.append([register, position, int(label), left, right, top, bottom])
        imwrite(atlas_dir / f"r{register:02d}_p{position}_cluster_{int(label):02d}.png", np.uint8(cell * 255))
    with (output / "neutral_glyph_assignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["register", "position", "cluster", "left", "right", "top", "bottom"])
        writer.writerows(assignments)
    center_images = np.uint8(np.clip(centers.reshape(-1, 36, 24) * 255, 0, 255))
    rows = []
    for cluster, center in enumerate(center_images):
        canvas = cv2.copyMakeBorder(center, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(canvas, f"C{cluster}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
        rows.append(canvas)
    atlas = np.concatenate(rows, axis=1)
    imwrite(output / "02_neutral_glyph_cluster_centers.png", cv2.resize(atlas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    counts = {str(cluster): int(np.sum(labels == cluster)) for cluster in range(10)}
    return {"compactness": float(compactness), "cluster_counts": counts, "cells": len(cells)}


def save_native_drizzle(rectified: list[dict], boxes: list[tuple[int, int, int, int]], output: Path) -> tuple[list[dict], dict]:
    target = output / "native_drizzle"
    target.mkdir(parents=True, exist_ok=True)
    metrics = []
    two_x_reconstructions = []
    for number, box in enumerate(boxes):
        for scale in (2, 3):
            reconstruction, support, units_per_source_pixel = native_drizzle_register(rectified, box, scale)
            imwrite(target / f"register_{number:02d}_native_{scale}x_latent.png", reconstruction)
            display = cv2.resize(reconstruction, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
            imwrite(target / f"register_{number:02d}_native_{scale}x.png", display)
            support_normalized = np.uint8(np.clip(support / max(float(support.max()), 1e-9) * 255, 0, 255))
            imwrite(target / f"register_{number:02d}_support_{scale}x.png", cv2.resize(support_normalized, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST))
            metrics.append(
                {
                    "register": number,
                    "scale": scale,
                    "latent_size": [int(reconstruction.shape[1]), int(reconstruction.shape[0])],
                    "screen_units_per_native_pixel": units_per_source_pixel,
                    "support_min_nonzero": float(support[support > 0].min()) if np.any(support > 0) else 0.0,
                    "support_median": float(np.median(support[support > 0])) if np.any(support > 0) else 0.0,
                    "support_zero_fraction": float(np.mean(support <= 0)),
                }
            )
            if scale == 2:
                two_x_reconstructions.append(reconstruction)
    atlas = build_neutral_glyph_atlas(two_x_reconstructions, output)
    return metrics, atlas


def collect_inverse_observations(
    records: list[dict],
    box: tuple[int, int, int, int],
    scale: int,
    psf_sigma_native: float = 0.0,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, int, int, float]:
    x, y, width, height = box
    reference = min(records, key=lambda item: float(np.linalg.norm(item["H_source_to_reference"] - np.eye(3))))
    units_per_source_pixel = local_screen_scale(reference["H_source_to_screen"], (x + width / 2, y + height / 2))
    step = units_per_source_pixel / scale
    target_width = max(2, int(math.ceil(width / step)))
    target_height = max(2, int(math.ceil(height / step)))
    screen_polygon = np.asarray([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], np.float64)
    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    observations: list[float] = []
    frame_ids: list[int] = []
    observation_index = 0
    for frame_id, record in enumerate(records):
        inverse = np.linalg.inv(record["H_source_to_screen"])
        source_polygon = project_points(screen_polygon, inverse)
        min_x = max(0, int(math.floor(source_polygon[:, 0].min())) - 1)
        max_x = min(record["image"].shape[1] - 1, int(math.ceil(source_polygon[:, 0].max())) + 1)
        min_y = max(0, int(math.floor(source_polygon[:, 1].min())) - 1)
        max_y = min(record["image"].shape[0] - 1, int(math.ceil(source_polygon[:, 1].max())) + 1)
        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        source_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
        projected = project_points(source_points, record["H_source_to_screen"])
        inside = (
            (projected[:, 0] >= x)
            & (projected[:, 0] < x + width)
            & (projected[:, 1] >= y)
            & (projected[:, 1] < y + height)
        )
        source_points = source_points[inside]
        projected = projected[inside]
        if len(source_points) == 0:
            continue
        luma = cv2.cvtColor(record["image"], cv2.COLOR_BGR2YCrCb)[:, :, 0]
        values = luma[source_points[:, 1].astype(int), source_points[:, 0].astype(int)].astype(np.float64)
        low, high = np.percentile(values, (7, 96))
        values = np.clip((values - low) / max(float(high - low), 1.0), 0, 1)
        target_x = (projected[:, 0] - x) / step - 0.5
        target_y = (projected[:, 1] - y) / step - 0.5
        base_x = np.floor(target_x).astype(int)
        base_y = np.floor(target_y).astype(int)
        frac_x = target_x - base_x
        frac_y = target_y - base_y
        for sample, value in enumerate(values):
            if psf_sigma_native > 0:
                sigma = psf_sigma_native * scale
                radius = max(2, int(math.ceil(3.0 * sigma)))
                center_x = int(round(target_x[sample]))
                center_y = int(round(target_y[sample]))
                contributions = []
                for row in range(center_y - radius, center_y + radius + 1):
                    for column in range(center_x - radius, center_x + radius + 1):
                        distance_squared = (column - target_x[sample]) ** 2 + (row - target_y[sample]) ** 2
                        weight = math.exp(-0.5 * distance_squared / max(sigma * sigma, 1e-9))
                        contributions.append((column, row, weight))
            else:
                contributions = (
                    (base_x[sample], base_y[sample], (1 - frac_x[sample]) * (1 - frac_y[sample])),
                    (base_x[sample] + 1, base_y[sample], frac_x[sample] * (1 - frac_y[sample])),
                    (base_x[sample], base_y[sample] + 1, (1 - frac_x[sample]) * frac_y[sample]),
                    (base_x[sample] + 1, base_y[sample] + 1, frac_x[sample] * frac_y[sample]),
                )
            total = sum(weight for column, row, weight in contributions if 0 <= column < target_width and 0 <= row < target_height)
            if total <= 1e-8:
                continue
            for column, row, weight in contributions:
                if 0 <= column < target_width and 0 <= row < target_height:
                    rows.append(observation_index)
                    columns.append(row * target_width + column)
                    coefficients.append(float(weight / total))
            observations.append(float(value))
            frame_ids.append(frame_id)
            observation_index += 1
    matrix = sparse.coo_matrix(
        (coefficients, (rows, columns)),
        shape=(len(observations), target_width * target_height),
        dtype=np.float64,
    ).tocsr()
    return matrix, np.asarray(observations), np.asarray(frame_ids), target_width, target_height, units_per_source_pixel


def gradient_matrix(width: int, height: int) -> sparse.csr_matrix:
    rows = []
    columns = []
    values = []
    equation = 0
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if x + 1 < width:
                rows.extend([equation, equation])
                columns.extend([index, index + 1])
                values.extend([1.0, -1.0])
                equation += 1
            if y + 1 < height:
                rows.extend([equation, equation])
                columns.extend([index, index + width])
                values.extend([1.0, -1.0])
                equation += 1
    return sparse.coo_matrix((values, (rows, columns)), shape=(equation, width * height)).tocsr()


def solve_inverse(
    matrix: sparse.csr_matrix,
    observations: np.ndarray,
    gradient: sparse.csr_matrix,
    regularization: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    selected_matrix = matrix if mask is None else matrix[mask]
    selected_observations = observations if mask is None else observations[mask]
    augmented_matrix = sparse.vstack([selected_matrix, math.sqrt(regularization) * gradient], format="csr")
    augmented_observations = np.concatenate([selected_observations, np.zeros(gradient.shape[0])])
    solution = lsqr(augmented_matrix, augmented_observations, atol=1e-8, btol=1e-8, iter_lim=800)[0]
    return np.clip(solution, 0, 1)


def evidence_display(solution: np.ndarray, width: int, height: int) -> np.ndarray:
    image = solution.reshape(height, width)
    low, high = np.percentile(image, (2, 98))
    normalized = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
    return np.uint8(normalized * 255)


def save_inverse_reconstructions(
    records: list[dict],
    boxes: list[tuple[int, int, int, int]],
    output: Path,
    scale: int,
    directory_name: str | None = None,
    psf_sigma_native: float = 0.0,
) -> tuple[list[dict], list[np.ndarray]]:
    target = output / (directory_name or f"inverse_reconstruction_{scale}x")
    target.mkdir(parents=True, exist_ok=True)
    metrics = []
    selected_images = []
    for register, box in enumerate(boxes):
        matrix, observations, frame_ids, width, height, units = collect_inverse_observations(
            records, box, scale, psf_sigma_native
        )
        gradient = gradient_matrix(width, height)
        even = (frame_ids % 2) == 0
        odd = ~even
        chosen = None
        for regularization in (0.03, 0.10, 0.30):
            solution = solve_inverse(matrix, observations, gradient, regularization)
            display = evidence_display(solution, width, height)
            imwrite(
                target / f"register_{register:02d}_lambda_{regularization:.2f}.png",
                cv2.resize(display, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST),
            )
            if abs(regularization - 0.10) < 1e-9:
                chosen = display
        even_solution = solve_inverse(matrix, observations, gradient, 0.10, even)
        odd_solution = solve_inverse(matrix, observations, gradient, 0.10, odd)
        even_display = evidence_display(even_solution, width, height)
        odd_display = evidence_display(odd_solution, width, height)
        stability = np.uint8(np.clip(np.abs(even_solution - odd_solution).reshape(height, width) * 255, 0, 255))
        imwrite(target / f"register_{register:02d}_even.png", cv2.resize(even_display, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST))
        imwrite(target / f"register_{register:02d}_odd.png", cv2.resize(odd_display, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST))
        imwrite(target / f"register_{register:02d}_split_difference.png", cv2.resize(stability, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST))
        odd_residual = np.abs(matrix[odd] @ even_solution - observations[odd])
        even_residual = np.abs(matrix[even] @ odd_solution - observations[even])
        metrics.append(
            {
                "register": register,
                "scale": scale,
                "latent_size": [width, height],
                "observations": int(len(observations)),
                "screen_units_per_native_pixel": units,
                "psf_sigma_native_px": psf_sigma_native,
                "cross_split_median_abs_residual": float(np.median(np.concatenate([odd_residual, even_residual]))),
                "cross_split_p90_abs_residual": float(np.percentile(np.concatenate([odd_residual, even_residual]), 90)),
                "split_solution_median_difference": float(np.median(np.abs(even_solution - odd_solution))),
                "split_solution_p90_difference": float(np.percentile(np.abs(even_solution - odd_solution), 90)),
            }
        )
        selected_images.append(chosen)
    return metrics, selected_images


def build_known_label_digits(
    records: list[dict], boxes: list[tuple[int, int, int, int]], output: Path, psf_sigma_native: float
) -> tuple[list[dict], dict]:
    left_indices = sorted([index for index, box in enumerate(boxes) if box[0] < 150], key=lambda index: boxes[index][1])
    right_indices = sorted([index for index, box in enumerate(boxes) if box[0] >= 150], key=lambda index: boxes[index][1])
    identity_by_register = {}
    for column_indices in (left_indices, right_indices):
        for rank, register in enumerate(column_indices):
            identity_by_register[register] = rank % 4 + 1
    label_boxes = []
    mapping = []
    for register, (x, y, width, height) in enumerate(boxes):
        label_box = (max(0, x - 42), y, 36, height)
        label_boxes.append(label_box)
        mapping.append(
            {
                "register": register,
                "known_digit": identity_by_register.get(register),
                "label_box": list(label_box),
            }
        )
    metrics, images = save_inverse_reconstructions(
        records,
        label_boxes,
        output,
        3,
        directory_name="known_label_inverse_3x_psf055",
        psf_sigma_native=psf_sigma_native,
    )
    def vector(cell: np.ndarray) -> np.ndarray:
        normalized = normalized_cell(cell)
        gradient_x = cv2.Sobel(normalized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(normalized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        values = np.concatenate([normalized.ravel(), 0.3 * gradient_x.ravel(), 0.3 * gradient_y.ravel()])
        values -= values.mean()
        return values / max(float(np.linalg.norm(values)), 1e-8)

    def evaluate_window(left_fraction: float, right_fraction: float) -> tuple[float, float, list[dict]]:
        extracted = []
        for register, image in enumerate(images):
            height, width = image.shape
            left = int(round(width * left_fraction))
            right = int(round(width * right_fraction))
            cell = image[1 : max(2, height - 2), left:right]
            extracted.append((identity_by_register[register], vector(cell)))
        outcomes = []
        margins = []
        correct = 0
        for held_index, (expected, held_vector) in enumerate(extracted):
            scores = []
            for candidate in range(1, 5):
                peers = [item_vector for index, (item_digit, item_vector) in enumerate(extracted) if index != held_index and item_digit == candidate]
                prototype = np.median(np.stack(peers), axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
                scores.append((float(np.dot(held_vector, prototype)), candidate))
            scores.sort(reverse=True)
            is_correct = scores[0][1] == expected
            correct += int(is_correct)
            margins.append(scores[0][0] - scores[1][0])
            outcomes.append({"register": held_index, "expected": expected, "ranking": scores, "correct": is_correct})
        return correct / len(extracted), float(np.median(margins)), outcomes

    window_candidates = []
    for left_fraction in np.arange(0.42, 0.69, 0.03):
        for width_fraction in np.arange(0.18, 0.34, 0.03):
            right_fraction = float(left_fraction + width_fraction)
            if right_fraction > 0.96:
                continue
            accuracy, margin, outcomes = evaluate_window(float(left_fraction), right_fraction)
            window_candidates.append((accuracy, margin, float(left_fraction), right_fraction, outcomes))
    window_candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    loo_accuracy, loo_margin, selected_left, selected_right, loo_outcomes = window_candidates[0]

    samples: dict[int, list[np.ndarray]] = {digit: [] for digit in range(1, 5)}
    sample_dir = output / "known_label_digits"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for register, image in enumerate(images):
        height, width = image.shape
        left = int(round(width * selected_left))
        right = int(round(width * selected_right))
        digit_cell = image[1 : max(2, height - 2), left:right]
        normalized = normalized_cell(digit_cell)
        digit = identity_by_register[register]
        samples[digit].append(normalized)
        imwrite(sample_dir / f"digit_{digit}_from_register_{register:02d}.png", np.uint8(normalized * 255))
    template_summary = {}
    for digit, digit_samples in samples.items():
        stack = np.stack(digit_samples)
        median = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - median[None, ...]), axis=0)
        imwrite(sample_dir / f"digit_{digit}_empirical_median.png", np.uint8(np.clip(median * 255, 0, 255)))
        imwrite(sample_dir / f"digit_{digit}_empirical_mad.png", np.uint8(np.clip(mad * 510, 0, 255)))
        template_summary[str(digit)] = {
            "samples": len(digit_samples),
            "median_pixel_mad": float(np.median(mad)),
            "p90_pixel_mad": float(np.percentile(mad, 90)),
        }
    (output / "known_label_digit_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return metrics, {
        "selected_crop_fractions": [selected_left, selected_right],
        "leave_one_out_accuracy": loo_accuracy,
        "leave_one_out_median_margin": loo_margin,
        "leave_one_out_results": loo_outcomes,
        "templates": template_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--reference-time", required=True, type=float)
    parser.add_argument("--corners", required=True, type=parse_corners)
    parser.add_argument("--size", required=True, type=parse_size)
    parser.add_argument("--register-region", required=True, type=parse_region)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records, video_metadata = decode_interval(args.source, args.start, args.end)
    if not records:
        raise RuntimeError("no frames decoded")
    reference_index = min(range(len(records)), key=lambda index: abs(records[index]["time_s"] - args.reference_time))
    accepted, alignment_log = estimate_homographies(records, reference_index, args.corners)
    width, height = args.size
    screen_corners = np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    H_reference_to_screen = cv2.getPerspectiveTransform(args.corners.astype(np.float32), screen_corners.astype(np.float32)).astype(np.float64)

    rectified = []
    for record in accepted:
        H_source_to_screen = H_reference_to_screen @ record["H_source_to_reference"]
        screen = cv2.warpPerspective(record["image"], H_source_to_screen, args.size, flags=cv2.INTER_LANCZOS4)
        rectified.append({**record, "H_source_to_screen": H_source_to_screen, "rectified": screen})

    reference = next(item for item in rectified if item["frame_no"] == records[reference_index]["frame_no"])
    boxes = detect_registers(reference["rectified"], args.register_region)
    annotated = reference["rectified"].copy()
    for number, (x, y, box_width, box_height) in enumerate(boxes):
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (0, 0, 255), 2)
        cv2.putText(annotated, str(number), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    imwrite(args.output / "01_reference_rectified_registers.png", annotated)
    save_register_atlas(rectified, boxes, args.output)
    drizzle_metrics, glyph_atlas = save_native_drizzle(rectified, boxes, args.output)
    inverse_metrics_2x, inverse_images_2x = save_inverse_reconstructions(rectified, boxes, args.output, 2)
    inverse_metrics_3x, inverse_images_3x = save_inverse_reconstructions(rectified, boxes, args.output, 3)
    inverse_metrics_3x_psf, inverse_images_3x_psf = save_inverse_reconstructions(
        rectified,
        boxes,
        args.output,
        3,
        directory_name="inverse_reconstruction_3x_psf055",
        psf_sigma_native=0.55,
    )
    inverse_atlas_2x = build_neutral_glyph_atlas(inverse_images_2x, args.output / "inverse_atlas_2x")
    inverse_atlas_3x = build_neutral_glyph_atlas(inverse_images_3x, args.output / "inverse_atlas_3x")
    inverse_atlas_3x_psf = build_neutral_glyph_atlas(inverse_images_3x_psf, args.output / "inverse_atlas_3x_psf055")
    known_label_metrics, known_label_templates = build_known_label_digits(rectified, boxes, args.output, 0.55)

    x0, y0, x1, y1 = args.register_region
    anchors = [
        ((x0 + x1) / 2, (y0 + y1) / 2),
        (x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0)),
        (x0 + 0.75 * (x1 - x0), y0 + 0.25 * (y1 - y0)),
        (x0 + 0.25 * (x1 - x0), y0 + 0.75 * (y1 - y0)),
        (x0 + 0.75 * (x1 - x0), y0 + 0.75 * (y1 - y0)),
    ]
    phases = phase_audit(accepted, H_reference_to_screen, anchors)

    digest_groups: dict[str, list[int]] = {}
    for record in records:
        digest_groups.setdefault(frame_digest(record["image"]), []).append(record["frame_no"])
    exact_duplicates = [group for group in digest_groups.values() if len(group) > 1]

    with (args.output / "alignment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_no", "time_s", "matches", "inliers", "median_inlier_residual_px", "status"])
        writer.writerows(alignment_log)
    with (args.output / "homographies.json").open("w", encoding="utf-8") as stream:
        json.dump(
            [
                {
                    "frame_no": item["frame_no"],
                    "time_s": item["time_s"],
                    "H_source_to_reference": item["H_source_to_reference"].tolist(),
                    "H_source_to_screen": item["H_source_to_screen"].tolist(),
                }
                for item in rectified
            ],
            stream,
            indent=2,
        )
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "video": video_metadata,
        "reference_frame": records[reference_index]["frame_no"],
        "reference_time_s": records[reference_index]["time_s"],
        "accepted_homographies": len(accepted),
        "exact_duplicate_frame_groups": exact_duplicates,
        "phase_audit": phases,
        "register_boxes": len(boxes),
        "native_drizzle": drizzle_metrics,
        "neutral_glyph_atlas": glyph_atlas,
        "inverse_reconstruction_2x": inverse_metrics_2x,
        "inverse_reconstruction_3x": inverse_metrics_3x,
        "inverse_reconstruction_3x_psf055": inverse_metrics_3x_psf,
        "inverse_neutral_glyph_atlas_2x": inverse_atlas_2x,
        "inverse_neutral_glyph_atlas_3x": inverse_atlas_3x,
        "inverse_neutral_glyph_atlas_3x_psf055": inverse_atlas_3x_psf,
        "known_label_inverse_3x": known_label_metrics,
        "known_label_digit_templates": known_label_templates,
        "limitations": [
            "No glyph labels or numeric readings are produced by this audit.",
            "Phase rank describes geometric sampling only; blur and codec loss can still destroy glyph information.",
            "Rectified register atlases are diagnostic derived images, not native observations.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"decoded": len(records), "accepted": len(accepted), "registers": len(boxes), "duplicates": exact_duplicates}, indent=2))


if __name__ == "__main__":
    main()
