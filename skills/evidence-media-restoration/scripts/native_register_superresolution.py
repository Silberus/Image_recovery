from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _imwrite(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode image: {path}")
    encoded.tofile(str(path))


def _matrix(text: str, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float64).reshape(shape)


def _roi_homography(config: dict[str, Any]) -> np.ndarray:
    quad = np.asarray(config["roi"]["quad"], dtype=np.float32).reshape(4, 2)
    width, height = map(int, config["roi"]["output_size"])
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(quad, target).astype(np.float64)


def _decode_ordinals(video: Path, ordinals: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    wanted = set(ordinals)
    first, last = min(wanted), max(wanted)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: dict[int, np.ndarray] = {}
    ordinal = first
    while ordinal <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if ordinal in wanted:
            frames[ordinal] = frame
        ordinal += 1
    capture.release()
    missing = sorted(wanted.difference(frames))
    if missing:
        raise RuntimeError(f"Frames not decoded: {missing}")
    return frames


def _decode_registered_frames(
    video: Path, case: Path, registration: list[dict[str, str]]
) -> dict[int, np.ndarray]:
    """Decode the exact observations used by the registered case.

    PyAV ordinals are local to the seek/decode window and therefore cannot be
    used as absolute OpenCV frame numbers.  Prefer the preserved packet PTS;
    retain the old absolute-ordinal path for legacy OpenCV cases.
    """
    observations_path = case / "source_observations.csv"
    if observations_path.exists():
        observations = _read_csv(observations_path)
        by_ordinal = {int(row["ordinal"]): row for row in observations}
        wanted: dict[int, int] = {}
        for row in registration:
            ordinal = int(row["source_ordinal"])
            observation = by_ordinal.get(ordinal)
            if observation and observation.get("pts") not in (None, ""):
                wanted[ordinal] = int(observation["pts"])
        if len(wanted) == len(registration):
            try:
                import av  # type: ignore

                pts_to_ordinals: dict[int, list[int]] = {}
                for ordinal, pts in wanted.items():
                    pts_to_ordinals.setdefault(pts, []).append(ordinal)
                first_pts, last_pts = min(pts_to_ordinals), max(pts_to_ordinals)
                decoded: dict[int, np.ndarray] = {}
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    container.seek(first_pts, stream=stream, any_frame=False, backward=True)
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            continue
                        pts = int(frame.pts)
                        if pts in pts_to_ordinals:
                            image = frame.to_ndarray(format="bgr24")
                            for ordinal in pts_to_ordinals[pts]:
                                decoded[ordinal] = image.copy()
                        if pts > last_pts and len(decoded) == len(wanted):
                            break
                missing = sorted(set(wanted).difference(decoded))
                if not missing:
                    return decoded
            except Exception:
                # The legacy absolute-ordinal path below remains deterministic
                # and keeps older OpenCV-generated cases usable.
                pass
    return _decode_ordinals(video, {int(row["source_ordinal"]) for row in registration})


def _huber(stack: np.ndarray, iterations: int = 8, delta: float = 1.5) -> np.ndarray:
    estimate = np.median(stack, axis=0)
    for _ in range(iterations):
        residual = stack - estimate
        scale = 1.4826 * np.median(np.abs(residual), axis=0) + 1.0
        ratio = np.abs(residual) / (delta * scale)
        weights = np.ones_like(ratio, dtype=np.float32)
        mask = ratio > 1.0
        weights[mask] = 1.0 / ratio[mask]
        estimate = np.sum(weights * stack, axis=0) / np.maximum(np.sum(weights, axis=0), 1e-6)
    return estimate


def _normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low, high = np.percentile(gray, [2.0, 98.0])
    if high <= low + 1.0:
        return np.clip(gray, 0, 255).astype(np.uint8)
    return np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 4)).apply(gray)
    smooth = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    return np.clip(clahe.astype(np.float32) + 0.65 * (clahe.astype(np.float32) - smooth), 0, 255).astype(np.uint8)


def _local_align(reference: np.ndarray, moving: np.ndarray, scale: int) -> tuple[np.ndarray, float, float, float]:
    ref = _normalize_gray(reference).astype(np.float32) / 255.0
    mov = _normalize_gray(moving).astype(np.float32) / 255.0
    shift, response = cv2.phaseCorrelate(mov, ref)
    dx, dy = float(shift[0]), float(shift[1])
    if not np.isfinite(dx + dy + response) or abs(dx) > 2.5 * scale or abs(dy) > 2.5 * scale or response < 0.05:
        return moving, 0.0, 0.0, float(response if np.isfinite(response) else 0.0)
    transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    aligned = cv2.warpAffine(
        moving,
        transform,
        (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, dx, dy, float(response)


def _phase(matrix: np.ndarray, rectified_point: tuple[float, float]) -> tuple[float, float]:
    inverse = np.linalg.inv(matrix)
    point = np.asarray([rectified_point[0], rectified_point[1], 1.0], dtype=np.float64)
    native = inverse @ point
    native /= native[2]
    return float(native[0] % 1.0), float(native[1] % 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct native-frame register reconstruction with phase audit")
    parser.add_argument("--registered-case", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--margin", type=int, default=4, help="Margin in rectified pixels")
    args = parser.parse_args()

    case = args.registered_case.resolve()
    video = args.video.resolve()
    boxes_path = args.boxes.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((case / "resolved_config.json").read_text(encoding="utf-8"))
    registration = [row for row in _read_csv(case / "registration.csv") if row.get("accepted") == "True"]
    boxes = _read_csv(boxes_path)
    source_frames = _decode_registered_frames(video, case, registration)
    roi_matrix = _roi_homography(config)
    out_width, out_height = map(int, config["roi"]["output_size"])
    high_scale = np.diag([args.scale, args.scale, 1.0]).astype(np.float64)

    transforms: list[dict[str, Any]] = []
    for row in registration:
        global_matrix = _matrix(row["global_homography"], (3, 3))
        local_affine = _matrix(row["warp"], (2, 3))
        local_matrix = np.vstack([local_affine, [0.0, 0.0, 1.0]])
        native_to_rectified = np.linalg.inv(local_matrix) @ roi_matrix @ global_matrix
        transforms.append({"row": row, "matrix": native_to_rectified})

    reference_index = next((i for i, item in enumerate(transforms) if item["row"].get("reference") == "True"), 0)
    metric_rows: list[dict[str, Any]] = []
    atlas_rows: list[np.ndarray] = []
    atlas_labels: list[str] = []

    for box_row in boxes:
        register = int(box_row["register"])
        x, y, width, height = (int(box_row[key]) for key in ("x", "y", "width", "height"))
        margin = args.margin
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(out_width, x + width + margin), min(out_height, y + height + margin)
        crop_width, crop_height = (x1 - x0) * args.scale, (y1 - y0) * args.scale
        crop_shift = np.asarray(
            [[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        crop_scale = high_scale @ crop_shift
        observations: list[np.ndarray] = []
        phase_rows: list[dict[str, Any]] = []
        center = (x + width / 2.0, y + height / 2.0)
        for item in transforms:
            row = item["row"]
            native_to_rectified = item["matrix"]
            direct = crop_scale @ native_to_rectified
            ordinal = int(row["source_ordinal"])
            crop = cv2.warpPerspective(
                source_frames[ordinal],
                direct,
                (crop_width, crop_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            observations.append(crop)
            px, py = _phase(native_to_rectified, center)
            phase_rows.append(
                {
                    "register": register,
                    "source_ordinal": ordinal,
                    "time_seconds": row.get("time_seconds"),
                    "phase_x": px,
                    "phase_y": py,
                    "phase_bin_x": min(args.scale - 1, int(px * args.scale)),
                    "phase_bin_y": min(args.scale - 1, int(py * args.scale)),
                }
            )

        crop_ranges = [float(np.percentile(crop, 99) - np.percentile(crop, 1)) for crop in observations]
        nonzero_fractions = [float(np.mean(np.any(crop != 0, axis=2))) for crop in observations]
        if max(nonzero_fractions, default=0.0) < 0.01 or max(crop_ranges, default=0.0) < 1.0:
            raise RuntimeError(
                f"Register {register} produced an empty/constant crop; check timestamp decoding and ROI coordinates"
            )

        reference = observations[reference_index]
        aligned: list[np.ndarray] = []
        alignment_rows: list[dict[str, Any]] = []
        for observation, item in zip(observations, transforms):
            registered, dx, dy, response = _local_align(reference, observation, args.scale)
            aligned.append(registered)
            alignment_rows.append(
                {
                    "register": register,
                    "source_ordinal": int(item["row"]["source_ordinal"]),
                    "local_dx_highres": dx,
                    "local_dy_highres": dy,
                    "phase_response": response,
                }
            )

        stack = np.asarray(aligned, dtype=np.float32)
        median = np.median(stack, axis=0)
        huber = _huber(stack)
        first_half = _huber(stack[::2])
        second_half = _huber(stack[1::2])
        split_difference = np.mean(np.abs(first_half - second_half), axis=2)
        temporal_mad = np.median(
            np.abs(stack - np.median(stack, axis=0, keepdims=True)), axis=(0, 3)
        )

        inner_x0, inner_y0 = margin * args.scale, margin * args.scale
        inner_x1, inner_y1 = inner_x0 + width * args.scale, inner_y0 + height * args.scale
        median_inner = np.clip(median[inner_y0:inner_y1, inner_x0:inner_x1], 0, 255).astype(np.uint8)
        huber_inner = np.clip(huber[inner_y0:inner_y1, inner_x0:inner_x1], 0, 255).astype(np.uint8)
        split_inner = split_difference[inner_y0:inner_y1, inner_x0:inner_x1]
        mad_inner = temporal_mad[inner_y0:inner_y1, inner_x0:inner_x1]
        gray = _normalize_gray(huber_inner)
        enhanced = _enhance(gray)

        register_dir = output / f"register_{register:02d}"
        register_dir.mkdir(exist_ok=True)
        _imwrite(register_dir / "native_huber_with_margin_x4.png", np.clip(huber, 0, 255).astype(np.uint8))
        _imwrite(register_dir / "native_median_x4.png", median_inner)
        _imwrite(register_dir / "native_huber_x4.png", huber_inner)
        _imwrite(register_dir / "native_huber_gray_normalized_x4.png", gray)
        _imwrite(register_dir / "native_huber_enhanced_x4.png", enhanced)
        split_view = np.clip(split_inner / max(float(np.percentile(split_inner, 99)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        _imwrite(register_dir / "split_difference_x4.png", split_view)
        mad_view = np.clip(mad_inner / max(float(np.percentile(mad_inner, 99)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        _imwrite(register_dir / "temporal_mad_x4.png", mad_view)
        _write_csv(register_dir / "phase_coverage.csv", phase_rows)
        _write_csv(register_dir / "local_alignment.csv", alignment_rows)

        # Preserve every aligned observation. A downstream decoder must be
        # able to separate a changing clock/counter from static text instead
        # of averaging incompatible glyph states into a plausible-looking lie.
        all_dir = register_dir / "aligned_observations"
        all_dir.mkdir(exist_ok=True)
        all_rows: list[dict[str, Any]] = []
        for registered, item in zip(aligned, transforms):
            ordinal = int(item["row"]["source_ordinal"])
            time_seconds = float(item["row"]["time_seconds"])
            inner = np.clip(
                registered[inner_y0:inner_y1, inner_x0:inner_x1], 0, 255
            ).astype(np.uint8)
            name = f"src_{ordinal:05d}_t_{time_seconds:010.6f}.png"
            _imwrite(all_dir / name, inner)
            all_rows.append(
                {
                    "source_ordinal": ordinal,
                    "time_seconds": time_seconds,
                    "file": name,
                    "classification": "DERIVED_DETERMINISTIC",
                }
            )
        _write_csv(all_dir / "observations.csv", all_rows)

        observed_tiles: list[np.ndarray] = []
        tile_width, tile_height = width * 8, height * 8
        ranked_observations: list[tuple[float, int, np.ndarray]] = []
        for registered, item in zip(aligned, transforms):
            inner = registered[inner_y0:inner_y1, inner_x0:inner_x1]
            gray_inner = _normalize_gray(inner)
            core = gray_inner[
                max(1, args.scale) : max(1, gray_inner.shape[0] - args.scale),
                max(1, args.scale) : max(1, gray_inner.shape[1] - args.scale),
            ]
            sharpness = float(cv2.Laplacian(core, cv2.CV_32F).var())
            ranked_observations.append((sharpness, int(item["row"]["source_ordinal"]), inner.copy()))
            observed = cv2.resize(inner, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
            tile = np.full((tile_height + 26, tile_width, 3), 245, dtype=np.uint8)
            label = f"src {int(item['row']['source_ordinal'])}  t={float(item['row']['time_seconds']):.3f}s"
            cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            tile[26:, :] = observed
            observed_tiles.append(tile)
        columns = 4
        rows = []
        for start in range(0, len(observed_tiles), columns):
            row_tiles = observed_tiles[start : start + columns]
            while len(row_tiles) < columns:
                row_tiles.append(np.full_like(observed_tiles[0], 245))
            rows.append(np.hstack(row_tiles))
        _imwrite(register_dir / "native_observation_atlas.png", np.vstack(rows))

        # Preserve the strongest source observations separately. These remain
        # OBSERVED-derived views and are essential for checking whether fusion
        # has blurred a genuine glyph feature.
        top_dir = register_dir / "top_observations"
        top_dir.mkdir(exist_ok=True)
        top_rows: list[dict[str, Any]] = []
        for rank, (sharpness, ordinal, inner) in enumerate(
            sorted(ranked_observations, key=lambda item: item[0], reverse=True)[:6], start=1
        ):
            normalized = _normalize_gray(inner)
            _imwrite(top_dir / f"rank_{rank:02d}_src_{ordinal}_color_x4.png", inner)
            _imwrite(top_dir / f"rank_{rank:02d}_src_{ordinal}_gray_x4.png", normalized)
            _imwrite(top_dir / f"rank_{rank:02d}_src_{ordinal}_enhanced_x4.png", _enhance(normalized))
            top_rows.append(
                {
                    "rank": rank,
                    "source_ordinal": ordinal,
                    "sharpness_laplacian_variance": sharpness,
                    "classification": "DERIVED_DETERMINISTIC",
                }
            )
        _write_csv(top_dir / "ranking.csv", top_rows)

        occupied = {(row["phase_bin_x"], row["phase_bin_y"]) for row in phase_rows}
        metric_rows.append(
            {
                "register": register,
                "frames": len(aligned),
                "phase_bins_occupied": len(occupied),
                "phase_bins_total": args.scale * args.scale,
                "phase_coverage_fraction": len(occupied) / float(args.scale * args.scale),
                "split_mae": float(np.mean(split_inner)),
                "split_p90": float(np.percentile(split_inner, 90)),
                "temporal_mad_mean": float(np.mean(mad_inner)),
                "temporal_mad_p90": float(np.percentile(mad_inner, 90)),
                "crop_dynamic_range_p99_p01_median": float(np.median(crop_ranges)),
                "crop_nonzero_fraction_median": float(np.median(nonzero_fractions)),
                "reading": "UNRESOLVED",
                "evidence_class": "DERIVED_DETERMINISTIC",
            }
        )

        enlarged_color = cv2.resize(huber_inner, (width * 12, height * 12), interpolation=cv2.INTER_NEAREST)
        enlarged_enhanced = cv2.resize(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), (width * 12, height * 12), interpolation=cv2.INTER_NEAREST)
        separator = np.full((height * 12, 12, 3), 255, dtype=np.uint8)
        atlas_rows.append(np.hstack([enlarged_color, separator, enlarged_enhanced]))
        atlas_labels.append(f"register {register:02d} | native Huber x{args.scale} | deterministic contrast view")

    max_width = max(image.shape[1] for image in atlas_rows)
    rendered_rows: list[np.ndarray] = []
    for label, image in zip(atlas_labels, atlas_rows):
        canvas = np.full((image.shape[0] + 34, max_width, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[34 : 34 + image.shape[0], : image.shape[1]] = image
        rendered_rows.append(canvas)
    _imwrite(output / "00_native_register_atlas.png", np.vstack(rendered_rows))
    _write_csv(output / "native_register_metrics.csv", metric_rows)

    manifest = {
        "schema": "evidence-media-restoration/native-register-superresolution/0.1",
        "source_video": {"path": str(video), "sha256": _sha256(video)},
        "registered_case": {"path": str(case), "manifest_sha256": _sha256(case / "manifest.json")},
        "register_boxes": {"path": str(boxes_path), "sha256": _sha256(boxes_path)},
        "scale": args.scale,
        "margin_rectified_pixels": args.margin,
        "frames": len(transforms),
        "reference_source_ordinal": int(transforms[reference_index]["row"]["source_ordinal"]),
        "method": "single native-to-high-resolution homography per frame; local phase refinement; temporal median and Huber fusion",
        "policy": {
            "ocr_used": False,
            "generative_model_used": False,
            "reading_status": "UNRESOLVED until topology audit",
            "classification": "DERIVED_DETERMINISTIC",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
