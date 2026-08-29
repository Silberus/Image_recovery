from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core import FrameRecord, sharpness


def parse_quad(value: Any) -> np.ndarray | None:
    if value in (None, [], ""):
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(4, 2)
    return arr


def rectify(image: np.ndarray, quad: np.ndarray | None, size: tuple[int, int] | None) -> np.ndarray:
    if quad is None:
        return image.copy() if size is None else cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    if size is None:
        w = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        size = (max(32, w), max(32, h))
    dst = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _gray_float(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float32) / 255.0


def align_to_reference(reference: np.ndarray, moving: np.ndarray, motion: str, iterations: int, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    ref = _gray_float(reference)
    mov = _gray_float(moving)
    shift, phase_response = cv2.phaseCorrelate(mov, ref)
    if motion == "translation":
        model = cv2.MOTION_TRANSLATION
        warp = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
    elif motion == "homography":
        model = cv2.MOTION_HOMOGRAPHY
        warp = np.eye(3, dtype=np.float32)
        warp[0, 2], warp[1, 2] = shift
    else:
        model = cv2.MOTION_AFFINE
        warp = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(epsilon))
    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, model, criteria, None, 5)
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
        if model == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(moving, warp, (reference.shape[1], reference.shape[0]), flags=flags, borderMode=cv2.BORDER_REFLECT)
        else:
            aligned = cv2.warpAffine(moving, warp, (reference.shape[1], reference.shape[0]), flags=flags, borderMode=cv2.BORDER_REFLECT)
        return aligned, {
            "accepted": True,
            "ecc": float(cc),
            "phase_response": float(phase_response),
            "motion": motion,
            "warp": np.asarray(warp).tolist(),
        }
    except cv2.error as exc:
        return moving, {"accepted": False, "ecc": None, "phase_response": float(phase_response), "motion": motion, "error": str(exc).splitlines()[0]}


def align_full_frame_homography(reference: np.ndarray, moving: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=int(config.get("sift_features", 5000)))
    kp_mov, des_mov = detector.detectAndCompute(gray_mov, None)
    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    if des_mov is None or des_ref is None or len(kp_mov) < 12 or len(kp_ref) < 12:
        return moving, {"global_homography_accepted": False, "global_error": "insufficient SIFT descriptors"}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_mov, des_ref, k=2)
    ratio = float(config.get("lowe_ratio", 0.76))
    good = [first for first, second in pairs if first.distance < ratio * second.distance]
    minimum = int(config.get("min_feature_matches", 18))
    if len(good) < minimum:
        return moving, {"global_homography_accepted": False, "global_matches": len(good), "global_error": f"matches<{minimum}"}
    source = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, float(config.get("ransac_threshold", 3.0)))
    if matrix is None or mask is None:
        return moving, {"global_homography_accepted": False, "global_matches": len(good), "global_error": "homography failed"}
    inliers = int(mask.ravel().sum())
    if inliers < int(config.get("min_homography_inliers", 14)):
        return moving, {"global_homography_accepted": False, "global_matches": len(good), "global_inliers": inliers, "global_error": "too few inliers"}
    projected = cv2.perspectiveTransform(source, matrix)
    residual = np.linalg.norm(projected.reshape(-1, 2) - target.reshape(-1, 2), axis=1)
    inlier_residual = residual[mask.ravel().astype(bool)]
    warped = cv2.warpPerspective(moving, matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return warped, {
        "global_homography_accepted": True,
        "global_matches": len(good),
        "global_inliers": inliers,
        "global_inlier_ratio": float(inliers / max(len(good), 1)),
        "global_reprojection_median": float(np.median(inlier_residual)),
        "global_reprojection_p90": float(np.percentile(inlier_residual, 90)),
        "global_homography": matrix.tolist(),
    }


def register(records: list[FrameRecord], config: dict[str, Any]) -> tuple[list[FrameRecord], list[dict[str, Any]], int]:
    quad = parse_quad(config.get("roi", {}).get("quad"))
    size_value = config.get("roi", {}).get("output_size")
    size = tuple(map(int, size_value)) if size_value else None
    rcfg = config.get("registration", {})
    fixed_rectified = [rectify(r.image, quad, size) for r in records]
    scores = [sharpness(i) for i in fixed_rectified]
    reference_time = rcfg.get("reference_time_seconds")
    if reference_time is not None and any(r.time_seconds is not None for r in records):
        reference_index = min(range(len(records)), key=lambda i: abs((records[i].time_seconds if records[i].time_seconds is not None else 1e30) - float(reference_time)))
    else:
        reference_index = int(np.argmax(scores))
    global_logs: list[dict[str, Any]] = []
    if quad is not None and bool(rcfg.get("global_feature_alignment", True)):
        reference_source = records[reference_index].image
        rectified = []
        for i, record in enumerate(records):
            if i == reference_index:
                globally_aligned = record.image
                global_result = {"global_homography_accepted": True, "global_matches": None, "global_inliers": None, "global_homography": np.eye(3).tolist()}
            else:
                globally_aligned, global_result = align_full_frame_homography(reference_source, record.image, rcfg)
            rectified.append(rectify(globally_aligned, quad, size))
            global_logs.append(global_result)
    else:
        rectified = fixed_rectified
        global_logs = [{"global_homography_accepted": None} for _ in records]
    reference = rectified[reference_index]
    motion = str(rcfg.get("motion", "affine"))
    min_ecc = float(rcfg.get("min_ecc", 0.78))
    max_frames = int(rcfg.get("max_accepted_frames", 120))
    aligned_records: list[FrameRecord] = []
    log: list[dict[str, Any]] = []
    for i, (record, image) in enumerate(zip(records, rectified)):
        if i == reference_index:
            aligned, result = image, {"accepted": True, "ecc": 1.0, "phase_response": 1.0, "motion": "identity", "warp": [[1, 0, 0], [0, 1, 0]]}
        else:
            aligned, result = align_to_reference(reference, image, motion, int(rcfg.get("iterations", 120)), float(rcfg.get("epsilon", 1e-6)))
            if result.get("ecc") is not None and result["ecc"] < min_ecc:
                result["accepted"] = False
                result["rejection"] = f"ecc<{min_ecc}"
        result.update(global_logs[i])
        result.update({"source_ordinal": record.ordinal, "time_seconds": record.time_seconds, "sharpness_before": scores[i], "reference": i == reference_index})
        log.append(result)
        if result["accepted"] and len(aligned_records) < max_frames:
            aligned_records.append(FrameRecord(record.ordinal, aligned, record.time_seconds, record.pts, record.key_frame, record.pict_type, record.source_name, record.backend))
    return aligned_records, log, reference_index
