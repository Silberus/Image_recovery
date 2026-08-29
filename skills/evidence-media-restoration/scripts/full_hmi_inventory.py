from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Canonical 1200 x 850 registered-screen coordinates. The first pass uses
# deliberately generous rectangles; narrow sub-ROIs are derived only after
# the full inventory map is visually checked.
BLOCKS = [
    (1, "upper_left_status", 8, 112, 170, 222),
    (2, "temperature_hot_oil_zone_1", 14, 210, 165, 337),
    (3, "temperature_hot_oil_zone_2", 17, 334, 170, 449),
    (4, "temperature_aging_zone_3", 20, 444, 174, 555),
    (5, "temperature_cool_down_zone_4", 23, 548, 177, 668),
    (6, "temperature_zone_4_one_hour_average", 164, 535, 320, 664),
    (7, "indicator_top_left_green_green_yellow", 235, 112, 370, 210),
    (8, "ager_column_and_zone_labels", 345, 165, 515, 590),
    (9, "stream_labels_left_of_ager", 205, 205, 355, 510),
    (10, "stream_labels_above_ager", 425, 105, 790, 210),
    (11, "pink_banner", 775, 175, 955, 222),
    (12, "cb_product_selection", 715, 220, 1005, 315),
    (13, "single_yellow_01", 558, 225, 705, 285),
    (14, "single_yellow_02", 555, 282, 710, 343),
    (15, "single_yellow_03", 590, 338, 750, 405),
    (16, "single_yellow_04", 595, 405, 760, 472),
    (17, "single_yellow_05", 615, 485, 780, 555),
    (18, "single_yellow_06", 635, 570, 805, 645),
    (19, "single_green_top_right", 995, 195, 1145, 280),
    (20, "indicator_lower_right_green_green_yellow", 965, 432, 1115, 545),
    (21, "stream_labels_right", 930, 270, 1175, 650),
    (22, "bottom_center_equipment_and_streams", 365, 520, 850, 730),
    (23, "bottom_status_bar", 0, 775, 1200, 850),
    (24, "screen_title", 780, 105, 1090, 180),
]


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _panel(image: np.ndarray, box: tuple[int, int, int, int], label: str, width: int = 1200) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    scale = min(width / max(crop.shape[1], 1), 6.0)
    rendered = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((rendered.shape[0] + 34, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (20, 20, 180), 1, cv2.LINE_AA)
    canvas[34:, : rendered.shape[1]] = rendered
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-screen HMI block inventory")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--last-observed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    images = {
        "last observed source frame 1375": _read(args.last_observed),
        "best observed": _read(args.case / "01_best_observed.png"),
        "temporal median": _read(args.case / "02_temporal_median.png"),
        "Huber mean": _read(args.case / "03_huber_mean.png"),
    }
    base_shape = next(iter(images.values())).shape[:2]
    for key, image in list(images.items()):
        if image.shape[:2] != base_shape:
            images[key] = cv2.resize(image, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_CUBIC)

    inventory = images["last observed source frame 1375"].copy()
    rows = []
    for number, name, x0, y0, x1, y1 in BLOCKS:
        color = (0, 0, 220) if number <= 6 else (0, 130, 255)
        cv2.rectangle(inventory, (x0, y0), (x1, y1), color, 2)
        cv2.putText(inventory, str(number), (x0 + 4, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        panels = [
            _panel(image, (x0, y0, x1, y1), label)
            for label, image in images.items()
        ]
        last_crop = images["last observed source frame 1375"][y0:y1, x0:x1]
        last_view = cv2.resize(
            last_crop,
            (last_crop.shape[1] * 6, last_crop.shape[0] * 6),
            interpolation=cv2.INTER_NEAREST,
        )
        _write(args.output / f"block_{number:02d}_{name}_last_observed_x6.png", last_view)
        max_width = max(panel.shape[1] for panel in panels)
        panels = [
            np.pad(panel, ((0, 0), (0, max_width - panel.shape[1]), (0, 0)), constant_values=245)
            for panel in panels
        ]
        _write(args.output / f"block_{number:02d}_{name}.png", np.vstack(panels))
        rows.append(
            {
                "block": number,
                "name": name,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "transcription": "UNRESOLVED",
                "evidence_class": "OBSERVED_PENDING_AUDIT",
            }
        )
    _write(args.output / "00_inventory_map_last_observed.png", inventory)
    _write_csv(args.output / "inventory.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
