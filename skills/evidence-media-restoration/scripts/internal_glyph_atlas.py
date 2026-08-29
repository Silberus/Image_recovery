from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _generic_label(register: int) -> str:
    """Never bake a case-specific interpretation into a neutral atlas."""
    return f"block {register // 4 + 1} row {register % 4 + 1}"


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


def _normalize(glyph: np.ndarray) -> np.ndarray:
    glyph = glyph.astype(np.float32)
    low, high = np.percentile(glyph, [2, 98])
    glyph = np.clip((glyph - low) * (255.0 / max(high - low, 1.0)), 0, 255)
    return glyph.astype(np.uint8)


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.resize(first, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    second = cv2.resize(second, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    first = (first - first.mean()) / max(first.std(), 1e-6)
    second = (second - second.mean()) / max(second.std(), 1e-6)
    return float(np.mean(first * second))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="20:50,49:79,78:108,118:148,147:177")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = [tuple(map(int, item.split(":"))) for item in args.windows.split(",")]

    glyphs: dict[str, np.ndarray] = {}
    atlas_rows: list[np.ndarray] = []
    for register in range(20):
        image = _read(args.register_root / f"register_{register:02d}" / "native_huber_gray_normalized_x4.png")
        cells: list[np.ndarray] = []
        for position, (left, right) in enumerate(windows):
            glyph = _normalize(image[:, left:right])
            key = f"r{register:02d}p{position + 1}"
            glyphs[key] = glyph
            view = cv2.resize(glyph, (220, 180), interpolation=cv2.INTER_NEAREST)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.putText(view, key, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cells.append(view)
        row = np.hstack(cells)
        label = np.full((28, row.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(label, f"r{register:02d}  {_generic_label(register)}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        atlas_rows.append(np.vstack([label, row]))
    _write(args.output / "00_internal_glyph_atlas.png", np.vstack(atlas_rows))

    keys = sorted(glyphs)
    similarity = np.eye(len(keys), dtype=np.float32)
    for i, first in enumerate(keys):
        for j in range(i + 1, len(keys)):
            value = _corr(glyphs[first], glyphs[keys[j]])
            similarity[i, j] = similarity[j, i] = value
    rows = []
    for i, key in enumerate(keys):
        order = np.argsort(similarity[i])[::-1]
        neighbours = [j for j in order if j != i][:8]
        row = {"glyph": key}
        for rank, neighbour in enumerate(neighbours, start=1):
            row[f"match_{rank}"] = keys[neighbour]
            row[f"corr_{rank}"] = float(similarity[i, neighbour])
        rows.append(row)
    with (args.output / "glyph_nearest_neighbours.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    heat = np.clip((similarity + 1.0) * 127.5, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    heat = cv2.resize(heat, (1200, 1200), interpolation=cv2.INTER_NEAREST)
    _write(args.output / "01_similarity_matrix.png", heat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
