from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact auditable glyph sheet for ddd.dd HMI fields")
    parser.add_argument("--register-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", default="20:50,49:79,78:108,118:148,147:177")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = [tuple(map(int, value.split(":"))) for value in args.windows.split(",")]

    cells = []
    for register in range(20):
        image = _read(args.register_root / f"register_{register:02d}" / "native_huber_gray_normalized_x4.png")
        if image.shape != (60, 200):
            image = cv2.resize(image, (200, 60), interpolation=cv2.INTER_CUBIC)
        for position, (left, right) in enumerate(windows, start=1):
            glyph = image[4:56, left:right]
            view = cv2.resize(glyph, (150, 104), interpolation=cv2.INTER_NEAREST)
            cell = np.full((132, 150, 3), 245, dtype=np.uint8)
            cell[28:] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.putText(cell, f"r{register:02d} p{position}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 180), 1, cv2.LINE_AA)
            cells.append(cell)
    rows = [np.hstack(cells[start : start + 10]) for start in range(0, len(cells), 10)]
    _write(args.output / "00_compact_glyph_sheet.png", np.vstack(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
