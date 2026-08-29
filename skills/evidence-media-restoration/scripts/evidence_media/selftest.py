from __future__ import annotations

import json
import copy
from pathlib import Path

import cv2
import numpy as np

from .core import psnr, save_image, write_json
from .pipeline import run_pipeline


def make_fixture(root: Path, seed: int = 7) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    frames = root / "fixture_frames"
    frames.mkdir(parents=True, exist_ok=True)
    truth = np.full((300, 640, 3), 232, np.uint8)
    cv2.rectangle(truth, (18, 18), (622, 282), (85, 85, 85), 3)
    cv2.putText(truth, "AGING TOWER", (145, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (35, 35, 35), 2, cv2.LINE_AA)
    for i, text in enumerate(["HOT OIL ZONE 1", "AGING ZONE 3", "T-1  271.69 deg F", "R2 LOCKHOPPER", "CB PRODUCT R530"]):
        cv2.putText(truth, text, (55, 112 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 45, 35), 1, cv2.LINE_AA)
    save_image(root / "truth.png", truth)
    for i in range(16):
        dx, dy = rng.uniform(-2.2, 2.2, 2)
        angle = rng.uniform(-0.25, 0.25)
        m = cv2.getRotationMatrix2D((320, 150), angle, 1.0)
        m[:, 2] += (dx, dy)
        image = cv2.warpAffine(truth, m, (640, 300), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        sigma = 0.65 + 0.5 * rng.random()
        image = cv2.GaussianBlur(image, (0, 0), sigma)
        noise = rng.normal(0, 5.5, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        encode = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(65 + 20 * rng.random())])[1]
        decoded = cv2.imdecode(encode, cv2.IMREAD_COLOR)
        save_image(frames / f"frame_{i:03d}.png", decoded)
    return frames, root / "truth.png"


def run_selftest(output: Path, config: dict) -> dict:
    fixture, truth_path = make_fixture(output)
    result_dir = output / "result"
    test_config = copy.deepcopy(config)
    test_config.setdefault("roi", {})["quad"] = []
    test_config["roi"]["output_size"] = []
    test_config.setdefault("registration", {})["min_ecc"] = 0.65
    manifest = run_pipeline(fixture, result_dir, test_config)
    truth = cv2.imread(str(truth_path), cv2.IMREAD_COLOR)
    best = cv2.imread(str(result_dir / "01_best_observed.png"), cv2.IMREAD_COLOR)
    median = cv2.imread(str(result_dir / "02_temporal_median.png"), cv2.IMREAD_COLOR)
    huber = cv2.imread(str(result_dir / "03_huber_mean.png"), cv2.IMREAD_COLOR)
    metrics = {"best_psnr": psnr(truth, best), "median_psnr": psnr(truth, median), "huber_psnr": psnr(truth, huber), "registered": manifest["counts"]["registered"]}
    passed = metrics["registered"] >= 10 and max(metrics["median_psnr"], metrics["huber_psnr"]) > metrics["best_psnr"]
    summary = {"passed": bool(passed), "metrics": metrics, "criterion": "registered>=10 and robust fusion PSNR exceeds best observed PSNR"}
    write_json(output / "selftest.json", summary)
    if not passed:
        raise RuntimeError("Self-test failed: " + json.dumps(summary))
    return summary
