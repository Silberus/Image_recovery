from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_ROOT = Path(r"C:\Windows\Fonts")


@dataclass(frozen=True)
class Config:
    font: str
    size: int
    sigma: float
    dx: int
    dy: int


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def _conform(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


@lru_cache(maxsize=None)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _feature_channels(image: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    value = image.astype(np.float32)
    gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    result = []
    for channel in (value, gx, gy):
        vector = channel[mask].astype(np.float32)
        vector -= float(vector.mean())
        result.append(vector)
    return result


def _layout_components(width: int, height: int, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return blurred digit components [5,10,H,W] and the fixed decimal point."""
    font_path = FONT_ROOT / config.font
    font = _font(str(font_path), config.size)
    probe = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(probe)
    pattern = "000.00"
    bounds = draw.textbbox((0, 0), pattern, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    origin_x = (width - text_width) / 2.0 - bounds[0] + config.dx
    origin_y = (height - text_height) / 2.0 - bounds[1] + config.dy
    advances = [float(draw.textlength(pattern[:i], font=font)) for i in range(len(pattern))]
    digit_char_indices = [0, 1, 2, 4, 5]

    components = np.zeros((5, 10, height, width), dtype=np.float32)
    for position, char_index in enumerate(digit_char_indices):
        for digit in range(10):
            canvas = Image.new("L", (width, height), 0)
            ImageDraw.Draw(canvas).text(
                (origin_x + advances[char_index], origin_y), str(digit), fill=255, font=font
            )
            component = np.asarray(canvas, dtype=np.float32)
            components[position, digit] = cv2.GaussianBlur(component, (0, 0), config.sigma)

    fixed_canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(fixed_canvas).text(
        (origin_x + advances[3], origin_y), ".", fill=255, font=font
    )
    fixed = cv2.GaussianBlur(np.asarray(fixed_canvas, dtype=np.float32), (0, 0), config.sigma)
    return components, fixed


def _candidate_digits() -> np.ndarray:
    numbers = np.arange(100000, dtype=np.int32)
    return np.column_stack(
        [
            numbers // 10000,
            (numbers // 1000) % 10,
            (numbers // 100) % 10,
            (numbers // 10) % 10,
            numbers % 10,
        ]
    )


class FastScorer:
    def __init__(self, config: Config, width: int, height: int, mask: np.ndarray, digits: np.ndarray):
        components, fixed = _layout_components(width, height, config)
        self.digits = digits
        self.channel_models: list[dict[str, np.ndarray | float]] = []
        for channel_index in range(3):
            comp_vectors = np.empty((5, 10, int(mask.sum())), dtype=np.float32)
            for position in range(5):
                for digit in range(10):
                    comp_vectors[position, digit] = _feature_channels(
                        components[position, digit], mask
                    )[channel_index]
            fixed_vector = _feature_channels(fixed, mask)[channel_index]
            candidate_norm2 = np.full(len(digits), float(np.dot(fixed_vector, fixed_vector)), dtype=np.float32)
            for position in range(5):
                self_norm = np.einsum("ij,ij->i", comp_vectors[position], comp_vectors[position])
                fixed_cross = comp_vectors[position] @ fixed_vector
                candidate_norm2 += self_norm[digits[:, position]]
                candidate_norm2 += 2.0 * fixed_cross[digits[:, position]]
            for first in range(5):
                for second in range(first + 1, 5):
                    pair_cross = comp_vectors[first] @ comp_vectors[second].T
                    candidate_norm2 += 2.0 * pair_cross[digits[:, first], digits[:, second]]
            self.channel_models.append(
                {
                    "components": comp_vectors,
                    "fixed": fixed_vector,
                    "candidate_norm": np.sqrt(np.maximum(candidate_norm2, 1e-6)),
                }
            )

    def scores(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observed_channels = _feature_channels(image, mask)
        total = np.zeros(len(self.digits), dtype=np.float32)
        weights = (0.70, 0.15, 0.15)
        for observed, model, weight in zip(observed_channels, self.channel_models, weights):
            components = model["components"]
            fixed = model["fixed"]
            numerator = np.full(len(self.digits), float(np.dot(observed, fixed)), dtype=np.float32)
            for position in range(5):
                digit_dots = components[position] @ observed
                numerator += digit_dots[self.digits[:, position]]
            denominator = max(float(np.linalg.norm(observed)), 1e-6) * model["candidate_norm"]
            total += weight * numerator / denominator
        return total


def _format_candidate(index: int) -> str:
    return f"{index // 100:03d}.{index % 100:02d}"


def _read_configs(path: Path, limit: int, shifts_x: list[int], shifts_y: list[int]) -> list[Config]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[:limit]
    configs = []
    for row in rows:
        for dx in shifts_x:
            for dy in shifts_y:
                configs.append(Config(row["font"], int(row["size"]), float(row["sigma"]), dx, dy))
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Whole-field ddd.dd forward-model hypotheses")
    parser.add_argument("--register-root", required=True, type=Path)
    parser.add_argument("--config-ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-limit", type=int, default=12)
    parser.add_argument("--top-configs", type=int, default=8)
    parser.add_argument("--candidate-union", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw_huber = [
        _read(args.register_root / f"register_{register:02d}" / "native_huber_gray_normalized_x4.png")
        for register in range(20)
    ]
    width = int(round(float(np.median([image.shape[1] for image in raw_huber]))))
    height = int(round(float(np.median([image.shape[0] for image in raw_huber]))))
    huber = [_conform(image, width, height) for image in raw_huber]
    mask = np.zeros((height, width), dtype=bool)
    mask[max(4, height // 12) : height - max(4, height // 12), max(10, width // 16) : width - max(10, width // 16)] = True
    digits = _candidate_digits()
    configs = _read_configs(args.config_ranking, args.config_limit, [-4, 0, 4], [-3, 0, 3])

    config_rows: list[dict[str, Any]] = []
    candidate_sets = [set() for _ in range(20)]
    for number, config in enumerate(configs, start=1):
        scorer = FastScorer(config, width, height, mask, digits)
        best_scores = []
        for register, image in enumerate(huber):
            scores = scorer.scores(image, mask)
            top = np.argpartition(scores, -args.candidate_union)[-args.candidate_union :]
            candidate_sets[register].update(int(index) for index in top)
            best_scores.append(float(scores[top].max()))
        config_rows.append(
            {
                "search_order": number,
                "font": config.font,
                "size": config.size,
                "sigma": config.sigma,
                "dx": config.dx,
                "dy": config.dy,
                "median_best_field_score": float(np.median(best_scores)),
                "classification": "MODEL_SUGGESTION",
            }
        )
    config_rows.sort(key=lambda row: float(row["median_best_field_score"]), reverse=True)
    _write_csv(args.output / "field_config_ranking.csv", config_rows)
    top_configs = [
        Config(row["font"], int(row["size"]), float(row["sigma"]), int(row["dx"]), int(row["dy"]))
        for row in config_rows[: args.top_configs]
    ]
    scorers = {config: FastScorer(config, width, height, mask, digits) for config in top_configs}

    result_rows: list[dict[str, Any]] = []
    atlas_rows: list[np.ndarray] = []
    for register in range(20):
        observation_paths = [
            args.register_root / f"register_{register:02d}" / "native_huber_gray_normalized_x4.png"
        ] + sorted(
            (args.register_root / f"register_{register:02d}" / "top_observations").glob("rank_*_gray_x4.png")
        )
        observations = [_conform(_read(path), width, height) for path in observation_paths]
        candidates = np.asarray(sorted(candidate_sets[register]), dtype=np.int32)
        best_by_observation = np.full((len(observations), len(candidates)), -np.inf, dtype=np.float32)
        for config in top_configs:
            scorer = scorers[config]
            for obs_index, observation in enumerate(observations):
                all_scores = scorer.scores(observation, mask)
                best_by_observation[obs_index] = np.maximum(
                    best_by_observation[obs_index], all_scores[candidates]
                )
        aggregate = np.median(best_by_observation, axis=0)
        order = np.argsort(aggregate)[::-1]
        top_order = order[:10]
        winner_counts = np.bincount(
            np.argmax(best_by_observation, axis=1), minlength=len(candidates)
        )
        for rank, local_index in enumerate(top_order, start=1):
            candidate = int(candidates[local_index])
            next_score = float(aggregate[top_order[1]]) if rank == 1 and len(top_order) > 1 else ""
            result_rows.append(
                {
                    "register": register,
                    "rank": rank,
                    "candidate": _format_candidate(candidate),
                    "median_score_across_observations": float(aggregate[local_index]),
                    "winner_observation_count": int(winner_counts[local_index]),
                    "observations": len(observations),
                    "top1_margin": float(aggregate[local_index] - next_score) if rank == 1 else "",
                    "classification": "MODEL_SUGGESTION",
                }
            )

        best_candidate = int(candidates[top_order[0]])
        best_config = top_configs[0]
        components, fixed = _layout_components(width, height, best_config)
        value_digits = digits[best_candidate]
        rendered = fixed.copy()
        for position in range(5):
            rendered += components[position, value_digits[position]]
        rendered = np.clip(rendered, 0, 255).astype(np.uint8)
        observed_view = cv2.resize(huber[register], (800, 240), interpolation=cv2.INTER_NEAREST)
        rendered_view = cv2.resize(rendered, (800, 240), interpolation=cv2.INTER_NEAREST)
        row = np.full((280, 1600, 3), 245, dtype=np.uint8)
        row[40:, :800] = cv2.cvtColor(observed_view, cv2.COLOR_GRAY2BGR)
        row[40:, 800:] = cv2.cvtColor(rendered_view, cv2.COLOR_GRAY2BGR)
        label = (
            f"r{register:02d} observed | candidate {_format_candidate(best_candidate)} "
            f"margin={float(aggregate[top_order[0]] - aggregate[top_order[1]]):.4f} MODEL_SUGGESTION"
        )
        cv2.putText(row, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 180), 1, cv2.LINE_AA)
        atlas_rows.append(row)

    _write_csv(args.output / "field_hypotheses.csv", result_rows)
    _write_image(args.output / "00_field_hypothesis_atlas.png", np.vstack(atlas_rows))
    manifest = {
        "schema": "evidence-media-restoration/field-forward-hypotheses/0.1",
        "format_assumption": "ddd.dd",
        "candidate_space": 100000,
        "classification": "MODEL_SUGGESTION",
        "promotion_rule": "No candidate becomes a reading without independent topology or source evidence.",
        "registers": 20,
        "configurations_tested": len(configs),
        "top_configurations_used_for_temporal_check": len(top_configs),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
