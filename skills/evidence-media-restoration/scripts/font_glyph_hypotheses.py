from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    Path(r"C:\Windows\Fonts\verdana.ttf"),
    Path(r"C:\Windows\Fonts\verdanab.ttf"),
    Path(r"C:\Windows\Fonts\micross.ttf"),
]


@dataclass(frozen=True)
class Config:
    font: Path
    size: int
    sigma: float


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


def _standardize(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32)
    value -= value.mean()
    value /= max(float(value.std()), 1e-6)
    return value


def _features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intensity = _standardize(image)
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = _standardize(cv2.magnitude(gx, gy))
    return intensity, gradient


@lru_cache(maxsize=None)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=None)
def _render(digit: int, width: int, height: int, config: Config, dx: int, dy: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    font = _font(str(config.font), config.size)
    text = str(digit)
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = (width - text_width) // 2 - bounds[0] + dx
    y = (height - text_height) // 2 - bounds[1] + dy
    draw.text((x, y), text, fill=255, font=font)
    rendered = np.asarray(canvas, dtype=np.uint8)
    rendered = cv2.GaussianBlur(rendered, (0, 0), config.sigma)
    return rendered


@lru_cache(maxsize=None)
def _template_features(
    digit: int,
    width: int,
    height: int,
    config: Config,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _features(_render(digit, width, height, config, dx, dy))


def _score(
    observed_features: tuple[np.ndarray, np.ndarray],
    template_features: tuple[np.ndarray, np.ndarray],
) -> float:
    intensity = float(np.mean(observed_features[0] * template_features[0]))
    gradient = float(np.mean(observed_features[1] * template_features[1]))
    return 0.72 * intensity + 0.28 * gradient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="20:76,68:128,120:182")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = [tuple(map(int, item.split(":"))) for item in args.windows.split(",")]

    glyphs: dict[str, np.ndarray] = {}
    for register in range(20):
        image = _read(args.register_root / f"register_{register:02d}" / "native_huber_gray_normalized_x4.png")
        for position, (left, right) in enumerate(windows, start=1):
            glyphs[f"r{register:02d}p{position}"] = image[:, left:right]
    observed_features = {key: _features(value) for key, value in glyphs.items()}

    configs = [
        Config(font, size, sigma)
        for font in FONT_PATHS if font.exists()
        for size in (30, 34, 38, 42, 46)
        for sigma in (3.0, 5.0, 7.0, 9.0)
    ]
    config_results: list[tuple[float, Config, dict[str, list[tuple[float, int, int, int]]]]] = []
    for config in configs:
        per_glyph: dict[str, list[tuple[float, int, int, int]]] = {}
        best_scores: list[float] = []
        for key, glyph in glyphs.items():
            candidates: list[tuple[float, int, int, int]] = []
            height, width = glyph.shape
            for digit in range(10):
                best = (-1e9, digit, 0, 0)
                for dx in (-5, 0, 5):
                    for dy in (-4, 0, 4):
                        template_features = _template_features(digit, width, height, config, dx, dy)
                        score = _score(observed_features[key], template_features)
                        if score > best[0]:
                            best = (score, digit, dx, dy)
                candidates.append(best)
            candidates.sort(reverse=True)
            per_glyph[key] = candidates
            best_scores.append(candidates[0][0])
        global_score = float(np.median(best_scores))
        config_results.append((global_score, config, per_glyph))
    config_results.sort(key=lambda item: item[0], reverse=True)
    top_configs = config_results[:12]

    config_rows = [
        {"rank": rank, "global_median_best_score": score, "font": config.font.name, "size": config.size, "sigma": config.sigma}
        for rank, (score, config, _) in enumerate(top_configs, start=1)
    ]
    with (args.output / "global_config_ranking.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(config_rows[0]))
        writer.writeheader()
        writer.writerows(config_rows)

    rows = []
    atlas_rows = []
    for key in sorted(glyphs):
        per_digit = {digit: [] for digit in range(10)}
        best_render = None
        for _, config, result in top_configs:
            for score, digit, dx, dy in result[key]:
                per_digit[digit].append(score)
        aggregated = sorted(((float(np.median(scores)), digit) for digit, scores in per_digit.items()), reverse=True)
        top = aggregated[:3]
        best_digit = top[0][1]
        best_config = top_configs[0][1]
        candidates_for_best = top_configs[0][2][key]
        chosen = next(item for item in candidates_for_best if item[1] == best_digit)
        best_render = _render(best_digit, glyphs[key].shape[1], glyphs[key].shape[0], best_config, chosen[2], chosen[3])
        rows.append(
            {
                "glyph": key,
                "top1_digit": top[0][1], "top1_score": top[0][0],
                "top2_digit": top[1][1], "top2_score": top[1][0],
                "top3_digit": top[2][1], "top3_score": top[2][0],
                "margin_top1_top2": top[0][0] - top[1][0],
                "classification": "MODEL_SUGGESTION",
            }
        )
        observed = cv2.resize(glyphs[key], (220, 180), interpolation=cv2.INTER_NEAREST)
        rendered = cv2.resize(best_render, (220, 180), interpolation=cv2.INTER_NEAREST)
        observed = cv2.cvtColor(observed, cv2.COLOR_GRAY2BGR)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
        canvas = np.full((206, 440, 3), 245, dtype=np.uint8)
        canvas[26:, :220] = observed
        canvas[26:, 220:] = rendered
        label = f"{key}  top={top[0][1]}  alt={top[1][1]}/{top[2][1]}  margin={top[0][0]-top[1][0]:.3f}  MODEL_SUGGESTION"
        cv2.putText(canvas, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 180), 1, cv2.LINE_AA)
        atlas_rows.append(canvas)
    with (args.output / "glyph_hypotheses.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write(args.output / "00_glyph_hypothesis_atlas.png", np.vstack(atlas_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
