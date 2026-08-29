#!/usr/bin/env python3
"""Rectify a photographed screen from four explicitly supplied corners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    data.tofile(path)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corners", required=True, help="tlx,tly;trx,try;brx,bry;blx,bly")
    parser.add_argument("--size", default="1200x850")
    args = parser.parse_args()
    points = np.asarray(
        [[float(value) for value in pair.split(",")] for pair in args.corners.split(";")],
        dtype=np.float32,
    )
    if points.shape != (4, 2):
        raise RuntimeError("Exactly four corners are required in TL,TR,BR,BL order")
    width, height = [int(value) for value in args.size.lower().split("x")]
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    source = _read(args.image.resolve())
    rectified = cv2.warpPerspective(source, matrix, (width, height), flags=cv2.INTER_LANCZOS4)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "rectified_screen.png", rectified)
    status = rectified[int(height * 0.925) : height]
    _write(args.output / "status_bar.png", status)
    manifest = {
        "schema": "evidence-media-restoration/rectified-still-screen/0.1",
        "source": str(args.image.resolve()),
        "source_sha256": _hash(args.image.resolve()),
        "source_shape": list(source.shape[:2]),
        "corners_tl_tr_br_bl": points.tolist(),
        "output_size": [width, height],
        "interpolation": "LANCZOS4",
        "classification": "DERIVED_DETERMINISTIC",
        "reading_status": "UNRESOLVED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
