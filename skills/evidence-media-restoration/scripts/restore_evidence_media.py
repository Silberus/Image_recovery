#!/usr/bin/env python3
"""Deterministic multi-frame restoration with provenance and uncertainty maps.

The program never uses OCR, a language model, inpainting, or a generative model.
It is designed for evidential inspection, not aesthetic restoration.
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


def parse_corners(value: str) -> np.ndarray:
    pts = []
    for item in value.split(";"):
        x, y = item.split(",")
        pts.append((float(x), float(y)))
    if len(pts) != 4:
        raise argparse.ArgumentTypeError("corners must contain four x,y pairs")
    return np.float32(pts)


def parse_size(value: str) -> tuple[int, int]:
    w, h = value.lower().split("x")
    return int(w), int(h)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def decode_video(path: Path, start: float, end: float, rate: float, out: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / native_fps if native_fps else 0.0
    if end <= start:
        end = duration
    rate = min(rate, native_fps) if native_fps else rate
    frames_dir = out / "decoded_source_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sequence, t in enumerate(np.arange(start, end, 1.0 / rate)):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_no = int(round(float(t) * native_fps)) if native_fps else sequence
        target = frames_dir / f"frame_{frame_no:08d}_t{float(t):.6f}.png"
        imwrite(target, frame)
        records.append({"sequence": sequence, "frame_no": frame_no, "time_s": float(t), "path": target, "image": frame})
    cap.release()
    (out / "video_metadata.json").write_text(
        json.dumps(
            {
                "source": str(path.resolve()),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "native_fps": native_fps,
                "width": width,
                "height": height,
                "frame_count": count,
                "nominal_duration_s": duration,
                "requested_interval_s": [start, end],
                "sample_rate_fps": rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return records


def load_images(path: Path) -> list[dict]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = sorted({p for pattern in patterns for p in path.glob(pattern)})
    return [
        {"sequence": i, "frame_no": i, "time_s": None, "path": p, "image": imread(p)}
        for i, p in enumerate(paths)
    ]


def rectify(image: np.ndarray, corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def global_align(records: list[dict], reference_index: int) -> tuple[list[dict], list[list]]:
    reference = records[reference_index]["image"]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02, edgeThreshold=14)
    kp_ref, des_ref = sift.detectAndCompute(reference_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted, log = [], []
    for i, record in enumerate(records):
        if i == reference_index:
            accepted.append({**record, "aligned": record["image"]})
            log.append([record["sequence"], record["frame_no"], record["time_s"], 9999, 9999, "reference"])
            continue
        gray = cv2.cvtColor(record["image"], cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            log.append([record["sequence"], record["frame_no"], record["time_s"], 0, 0, "no_descriptors"])
            continue
        pairs = matcher.knnMatch(des, des_ref, k=2)
        good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
        if len(good) < 30:
            log.append([record["sequence"], record["frame_no"], record["time_s"], len(good), 0, "too_few_matches"])
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if H is None or inliers < 24:
            log.append([record["sequence"], record["frame_no"], record["time_s"], len(good), inliers, "bad_homography"])
            continue
        aligned = cv2.warpPerspective(record["image"], H, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LANCZOS4)
        accepted.append({**record, "aligned": aligned})
        log.append([record["sequence"], record["frame_no"], record["time_s"], len(good), inliers, "accepted"])
    return accepted, log


def local_align(records: list[dict], corners: np.ndarray, size: tuple[int, int], reference_sequence: int) -> tuple[list[dict], list[list]]:
    rectified = [{**r, "rectified": rectify(r["aligned"], corners, size)} for r in records]
    reference = next((r for r in rectified if r["sequence"] == reference_sequence), rectified[len(rectified) // 2])
    ref_gray = cv2.cvtColor(reference["rectified"], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    output, log = [], []
    for record in rectified:
        moving = record["rectified"]
        gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            ecc, warp = cv2.findTransformECC(
                ref_gray,
                gray,
                warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
                None,
                5,
            )
            aligned = cv2.warpAffine(
                moving,
                warp,
                size,
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            ecc, aligned = float("nan"), moving
        item = {
            **record,
            "registered": aligned,
            "dx": float(warp[0, 2]),
            "dy": float(warp[1, 2]),
            "sharpness": sharpness(aligned),
        }
        output.append(item)
        log.append([record["sequence"], record["frame_no"], record["time_s"], float(ecc), item["dx"], item["dy"], item["sharpness"]])
    return output, log


def robust_fusion(stack_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stack = stack_u8.astype(np.float32)
    median = np.median(stack, axis=0)
    mad_rgb = np.median(np.abs(stack - median[None, ...]), axis=0)
    delta = np.maximum(2.0, 1.5 * 1.4826 * mad_rgb)
    huber = median + np.mean(np.clip(stack - median[None, ...], -delta[None, ...], delta[None, ...]), axis=0)
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in stack_u8], axis=0).astype(np.float32)
    gray_med = np.median(gray, axis=0)
    gray_mad = np.median(np.abs(gray - gray_med[None, ...]), axis=0)
    tolerance = np.maximum(4.0, 2.5 * 1.4826 * gray_mad)
    support = np.sum(np.abs(gray - gray_med[None, ...]) <= tolerance[None, ...], axis=0).astype(np.uint16)
    return np.uint8(np.clip(median, 0, 255)), np.uint8(np.clip(huber, 0, 255)), gray_mad, support


def donor_mosaic(records: list[dict], tile: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    stack = np.stack([r["registered"] for r in records], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    h, w = median.shape[:2]
    result = np.empty_like(median)
    donors = np.zeros((math.ceil(h / tile), math.ceil(w / tile)), dtype=np.uint16)
    rows = []
    for iy, y0 in enumerate(range(0, h, tile)):
        for ix, x0 in enumerate(range(0, w, tile)):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            med = median[y0:y1, x0:x1].astype(np.float32)
            scores = []
            for k, record in enumerate(records):
                patch = record["registered"][y0:y1, x0:x1]
                disagreement = float(np.median(np.abs(patch.astype(np.float32) - med)))
                score = sharpness(patch) / (1.0 + 0.35 * disagreement)
                scores.append((score, k, disagreement))
            score, winner, disagreement = max(scores)
            result[y0:y1, x0:x1] = stack[winner, y0:y1, x0:x1]
            donors[iy, ix] = winner
            rows.append(
                {
                    "tile_x": ix,
                    "tile_y": iy,
                    "source_sequence": records[winner]["sequence"],
                    "source_frame_no": records[winner]["frame_no"],
                    "source_time_s": records[winner]["time_s"],
                    "repeatable_detail_score": score,
                    "median_abs_disagreement": disagreement,
                }
            )
    return result, donors, rows


def heatmap(values: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(values.astype(np.float32) / max(vmax, 1e-6), 0, 1)
    return cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)


def metrics(candidate: np.ndarray, holdout: np.ndarray) -> dict:
    residual = np.abs(holdout.astype(np.float32) - candidate.astype(np.float32)[None, ...])
    lap = cv2.Laplacian(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)
    return {
        "holdout_median_abs_residual": float(np.median(residual)),
        "holdout_p90_abs_residual": float(np.percentile(residual, 90)),
        "laplacian_variance": float(lap.var()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Video file or directory of decoded image frames")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corners", type=parse_corners, required=True, help="TL;TR;BR;BL as x,y;x,y;x,y;x,y")
    parser.add_argument("--size", type=parse_size, required=True, help="Rectified screen size, e.g. 1200x850")
    parser.add_argument("--start", type=float, default=0.0, help="Video interval start, seconds")
    parser.add_argument("--end", type=float, default=0.0, help="Video interval end, seconds")
    parser.add_argument("--sample-rate", type=float, default=30.0, help="Frames per second to decode")
    parser.add_argument("--reference-time", type=float, help="Preferred reference time for video")
    parser.add_argument("--reference-index", type=int, help="Reference sequence index")
    parser.add_argument("--tile", type=int, default=32)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.is_dir():
        source_records = load_images(args.input)
        source_description = {
            "type": "image_directory",
            "path": str(args.input.resolve()),
            "files": [{"path": str(r["path"].resolve()), "sha256": file_hash(r["path"])} for r in source_records],
        }
    else:
        source_records = decode_video(args.input, args.start, args.end, args.sample_rate, args.output)
        source_description = {"type": "video", "path": str(args.input.resolve()), "sha256": file_hash(args.input)}
    if len(source_records) < 4:
        raise RuntimeError("At least four source frames are required")

    if args.reference_index is not None:
        reference_index = args.reference_index
    elif args.reference_time is not None:
        reference_index = min(range(len(source_records)), key=lambda i: abs((source_records[i]["time_s"] or 0.0) - args.reference_time))
    else:
        reference_index = len(source_records) // 2
    reference_sequence = source_records[reference_index]["sequence"]

    globally_aligned, global_log = global_align(source_records, reference_index)
    registered, local_log = local_align(globally_aligned, args.corners, args.size, reference_sequence)
    if len(registered) < 4:
        raise RuntimeError("Fewer than four frames passed registration")

    train = registered[::2]
    holdout = registered[1::2]
    if not holdout:
        holdout = train
    train_stack = np.stack([r["registered"] for r in train], axis=0)
    holdout_stack = np.stack([r["registered"] for r in holdout], axis=0)
    best = max(train, key=lambda r: r["sharpness"])["registered"]
    median, huber, mad, support = robust_fusion(train_stack)
    mosaic, donor, donor_rows = donor_mosaic(train, args.tile)

    outputs = {
        "01_best_observed.png": best,
        "02_temporal_median.png": median,
        "03_clipped_huber_mean.png": huber,
        "04_observed_tile_mosaic.png": mosaic,
        "05_support_count.png": heatmap(support, len(train)),
        "06_temporal_mad.png": heatmap(mad, max(4.0, float(np.percentile(mad, 99)))),
    }
    palette = cv2.applyColorMap(np.linspace(0, 255, max(len(train), 2), dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
    outputs["07_donor_map.png"] = cv2.resize(palette[donor], None, fx=args.tile, fy=args.tile, interpolation=cv2.INTER_NEAREST)
    for name, image in outputs.items():
        imwrite(args.output / name, image)

    candidate_metrics = {
        "best_observed": metrics(best, holdout_stack),
        "temporal_median": metrics(median, holdout_stack),
        "clipped_huber_mean": metrics(huber, holdout_stack),
        "observed_tile_mosaic": metrics(mosaic, holdout_stack),
    }
    with (args.output / "global_alignment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "frame_no", "time_s", "matches", "inliers", "status"])
        writer.writerows(global_log)
    with (args.output / "local_alignment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "frame_no", "time_s", "ecc", "dx_px", "dy_px", "sharpness"])
        writer.writerows(local_log)
    with (args.output / "donor_tiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(donor_rows[0]))
        writer.writeheader()
        writer.writerows(donor_rows)

    manifest = {
        "source": source_description,
        "parameters": {
            "corners_tl_tr_br_bl": args.corners.tolist(),
            "rectified_size": list(args.size),
            "reference_sequence": reference_sequence,
            "tile_size": args.tile,
        },
        "counts": {"source": len(source_records), "registered": len(registered), "training": len(train), "holdout": len(holdout)},
        "methods": {
            "global_registration": "SIFT + Lowe ratio + RANSAC homography",
            "local_registration": "ECC translation",
            "fusion": ["temporal median", "clipped Huber mean", "observed-tile donor mosaic"],
            "validation": "even/odd train-holdout split; robust pixel residual",
        },
        "metrics": candidate_metrics,
        "evidence_labels": {
            "best_observed": "OBSERVED after geometric resampling",
            "temporal_median": "DERIVED",
            "clipped_huber_mean": "DERIVED",
            "observed_tile_mosaic": "DERIVED from named source donors",
            "support_count": "DERIVED uncertainty diagnostic",
            "temporal_mad": "DERIVED uncertainty diagnostic",
        },
        "limitations": [
            "No output may be cited as a literal reading unless the glyph repeats across independent source frames and methods.",
            "Interpolation and stacking cannot recover spatial frequencies absent from the source samples.",
            "Different encodes of the same recording are one witness cluster.",
            "No OCR, inpainting, or generative reconstruction was used.",
        ],
        "output_sha256": {name: file_hash(args.output / name) for name in outputs},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "counts": manifest["counts"], "metrics": candidate_metrics}, indent=2))


if __name__ == "__main__":
    main()
