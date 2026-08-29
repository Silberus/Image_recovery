from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _crossing(profile: np.ndarray, level: float, descending: bool) -> float | None:
    for index in range(len(profile) - 1):
        first, second = float(profile[index]), float(profile[index + 1])
        hit = first >= level > second if descending else first <= level < second
        if hit and second != first:
            return index + (level - first) / (second - first)
    return None


def _edge_sigma(profile: np.ndarray, descending: bool) -> float | None:
    smooth = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (1, 0), 0.7).ravel()
    low = float(np.percentile(smooth, 10))
    high = float(np.percentile(smooth, 90))
    if high - low < 30:
        return None
    normalized = (smooth - low) / (high - low)
    p90 = _crossing(normalized, 0.9, descending)
    p10 = _crossing(normalized, 0.1, descending)
    if p90 is None or p10 is None:
        return None
    width = abs(p10 - p90)
    if not 1.0 <= width <= 30.0:
        return None
    return width / 2.5631031311


def _estimate(full: np.ndarray, margin: int) -> tuple[list[float], list[float]]:
    h, w = full.shape
    sigmas_x: list[float] = []
    sigmas_y: list[float] = []
    for y in range(margin + 10, h - margin - 10):
        left = _edge_sigma(full[y, : margin + 28], descending=True)
        right = _edge_sigma(full[y, w - margin - 28 :][::-1], descending=True)
        if left is not None:
            sigmas_x.append(left)
        if right is not None:
            sigmas_x.append(right)
    # Use the blank strips immediately inside the left/right field margins.
    for x in list(range(margin + 2, margin + 14)) + list(range(w - margin - 14, w - margin - 2)):
        top = _edge_sigma(full[: margin + 26, x], descending=True)
        bottom = _edge_sigma(full[h - margin - 26 :, x][::-1], descending=True)
        if top is not None:
            sigmas_y.append(top)
        if bottom is not None:
            sigmas_y.append(bottom)
    return sigmas_x, sigmas_y


def _gaussian_psf(sigma_x: float, sigma_y: float) -> np.ndarray:
    radius_x = max(2, int(np.ceil(3.5 * sigma_x)))
    radius_y = max(2, int(np.ceil(3.5 * sigma_y)))
    x = np.arange(-radius_x, radius_x + 1, dtype=np.float32)
    y = np.arange(-radius_y, radius_y + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
    psf /= float(psf.sum())
    return psf.astype(np.float32)


def _normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low + 1e-6:
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _wiener(image: np.ndarray, psf: np.ndarray, balance: float) -> np.ndarray:
    h, w = image.shape
    kernel = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    kernel[:kh, :kw] = psf
    kernel = np.roll(kernel, -kh // 2, axis=0)
    kernel = np.roll(kernel, -kw // 2, axis=1)
    transfer = np.fft.fft2(kernel)
    observed = np.fft.fft2(image.astype(np.float32))
    restored = np.fft.ifft2(np.conj(transfer) * observed / (np.abs(transfer) ** 2 + balance)).real
    return restored.astype(np.float32)


def _richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    observed = np.maximum(image.astype(np.float32), 0.0) + 1e-3
    estimate = np.maximum(cv2.GaussianBlur(observed, (0, 0), 0.6), 1e-3)
    flipped = psf[::-1, ::-1]
    for _ in range(iterations):
        convolution = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        relative = observed / np.maximum(convolution, 1e-3)
        estimate *= cv2.filter2D(relative, -1, flipped, borderType=cv2.BORDER_REFLECT)
        estimate = np.clip(estimate, 0, 1024)
    return estimate


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate effective PSF from display edges and deconvolve registers")
    parser.add_argument("--register-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-highres", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    edge_rows: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    full_images = []
    for register in range(20):
        full = _read(args.register_root / f"register_{register:02d}" / "native_huber_with_margin_x4.png")
        full_images.append(full)
        values_x, values_y = _estimate(full, args.margin_highres)
        all_x.extend(values_x)
        all_y.extend(values_y)
        edge_rows.append(
            {
                "register": register,
                "edge_samples_x": len(values_x),
                "effective_sigma_x_highres_median": float(np.median(values_x)) if values_x else "",
                "edge_samples_y": len(values_y),
                "effective_sigma_y_highres_median": float(np.median(values_y)) if values_y else "",
                "classification": "DERIVED_DETERMINISTIC",
            }
        )
    sigma_x = float(np.median(all_x)) if all_x else 6.0
    sigma_y = float(np.median(all_y)) if all_y else sigma_x
    # Avoid pretending the rounded HMI border is a perfect optical knife edge.
    # The sweep around the measured effective value is therefore preserved.
    factors = (0.70, 0.85, 1.00, 1.15)
    atlas_rows = []
    for register, full in enumerate(full_images):
        margin = args.margin_highres
        h, w = full.shape
        inner = full[margin : h - margin, margin : w - margin]
        panels = [cv2.resize(_normalize(inner), (800, 240), interpolation=cv2.INTER_NEAREST)]
        labels = ["observed Huber x4"]
        for factor in factors:
            psf = _gaussian_psf(max(1.0, sigma_x * factor), max(1.0, sigma_y * factor))
            restored = _wiener(full, psf, balance=0.018)
            panels.append(cv2.resize(_normalize(restored[margin : h - margin, margin : w - margin]), (800, 240), interpolation=cv2.INTER_NEAREST))
            labels.append(f"Wiener edge-PSF x{factor:.2f}")
        psf = _gaussian_psf(max(1.0, sigma_x * 0.85), max(1.0, sigma_y * 0.85))
        for iterations in (3, 6, 10):
            restored = _richardson_lucy(full, psf, iterations)
            panels.append(cv2.resize(_normalize(restored[margin : h - margin, margin : w - margin]), (800, 240), interpolation=cv2.INTER_NEAREST))
            labels.append(f"RL edge-PSF i{iterations}")
        row = np.full((len(panels) * 270 + 34, 800, 3), 245, dtype=np.uint8)
        cv2.putText(row, f"register {register:02d}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        offset = 34
        for label, panel in zip(labels, panels):
            cv2.putText(row, label, (8, offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 180), 1, cv2.LINE_AA)
            row[offset + 30 : offset + 270] = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            offset += 270
        atlas_rows.append(row)
        _write(args.output / f"register_{register:02d}_edge_psf_methods.png", row)

    _write_csv(args.output / "edge_psf_estimates.csv", edge_rows)
    manifest = {
        "schema": "evidence-media-restoration/edge-psf-deconvolution/0.1",
        "effective_sigma_x_highres": sigma_x,
        "effective_sigma_y_highres": sigma_y,
        "highres_scale": 4,
        "sigma_x_rectified_pixels": sigma_x / 4.0,
        "sigma_y_rectified_pixels": sigma_y / 4.0,
        "edge_samples_x": len(all_x),
        "edge_samples_y": len(all_y),
        "interpretation": "effective edge spread including optics, resampling, display antialiasing and rounded-border bias",
        "classification": "DERIVED_DETERMINISTIC",
        "reading_status": "UNRESOLVED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
