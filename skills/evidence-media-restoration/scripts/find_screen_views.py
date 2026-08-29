#!/usr/bin/env python3
"""Find every public-video view of a known planar screen by feature matching."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_corners(value: str) -> np.ndarray:
    points = []
    for item in value.split(";"):
        x, y = item.split(",")
        points.append((float(x), float(y)))
    return np.float32(points)


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cannot write {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-time", required=True, type=float)
    parser.add_argument("--corners", required=True, type=parse_corners)
    parser.add_argument("--sample-rate", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.reference_time * fps)))
    ok, reference = capture.read()
    if not ok:
        raise RuntimeError("cannot read reference frame")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(reference_gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, args.corners.astype(np.int32), 255)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015, edgeThreshold=16)
    kp_reference, des_reference = sift.detectAndCompute(reference_gray, mask)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    interval = max(1, int(round(fps / args.sample_rate)))
    rows = []
    accepted = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_no = 0
    while frame_no < frame_count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            rows.append([frame_no, frame_no / fps, 0, 0, 0, 0, "no_descriptors"])
            frame_no += interval
            continue
        pairs = matcher.knnMatch(des, des_reference, k=2)
        good = [match for match, second in pairs if match.distance < 0.70 * second.distance]
        if len(good) < 18:
            rows.append([frame_no, frame_no / fps, len(good), 0, 0, 0, "too_few_matches"])
            frame_no += interval
            continue
        source = np.float32([kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([kp_reference[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        H_frame_to_reference, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 1.5)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if H_frame_to_reference is None or inliers < 14:
            rows.append([frame_no, frame_no / fps, len(good), inliers, 0, 0, "bad_homography"])
            frame_no += interval
            continue
        H_reference_to_frame = np.linalg.inv(H_frame_to_reference)
        projected = cv2.perspectiveTransform(args.corners.reshape(-1, 1, 2), H_reference_to_frame).reshape(-1, 2)
        area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        destination = np.float32([[0, 0], [599, 0], [599, 424], [0, 424]])
        H_frame_to_rectified = cv2.getPerspectiveTransform(projected.astype(np.float32), destination)
        rectified = cv2.warpPerspective(gray, H_frame_to_rectified, (600, 425), flags=cv2.INTER_LANCZOS4)
        sharpness = float(cv2.Laplacian(rectified, cv2.CV_32F).var())
        score = area * max(sharpness, 1e-6)
        rows.append([frame_no, frame_no / fps, len(good), inliers, area, sharpness, "accepted"])
        accepted.append((score, area, sharpness, frame_no, image, projected))
        frame_no += interval
    capture.release()

    with (args.output / "screen_views.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_no", "time_s", "matches", "inliers", "projected_screen_area_px2", "rectified_sharpness", "status"])
        writer.writerows(rows)
    accepted.sort(reverse=True, key=lambda item: item[0])
    tiles = []
    for rank, (_, area, sharpness, frame_no, image, projected) in enumerate(accepted[:16]):
        annotated = image.copy()
        cv2.polylines(annotated, [projected.astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"#{rank + 1} t={frame_no / fps:.3f}s area={area:.0f} sharp={sharpness:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        tile = cv2.resize(annotated, (640, 360), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
        imwrite(args.output / f"rank_{rank + 1:02d}_frame_{frame_no:06d}.png", image)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows_of_tiles = [np.concatenate(tiles[index : index + 4], axis=1) for index in range(0, len(tiles), 4)]
        imwrite(args.output / "best_screen_views_contact.png", np.concatenate(rows_of_tiles, axis=0))
    print(f"accepted={len(accepted)} best_time_s={(accepted[0][3] / fps) if accepted else 'none'}")


if __name__ == "__main__":
    main()
