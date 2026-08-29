from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy import signal


@dataclass
class FrameRecord:
    ordinal: int
    image: np.ndarray
    time_seconds: float | None = None
    pts: int | None = None
    key_frame: bool | None = None
    pict_type: str | None = None
    source_name: str | None = None
    backend: str = "opencv"

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("image", None)
        return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def entropy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def blockiness(image: np.ndarray, block: int = 8) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    a = gray.astype(np.float32)
    if min(a.shape[:2]) <= block:
        return 0.0
    vb = np.abs(a[:, block::block] - a[:, block - 1 :: block]).mean()
    hb = np.abs(a[block::block, :] - a[block - 1 :: block, :]).mean()
    vi = np.abs(a[:, 1:] - a[:, :-1]).mean()
    hi = np.abs(a[1:, :] - a[:-1, :]).mean()
    return float(max(0.0, (vb + hb) / 2 - (vi + hi) / 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else float(20 * math.log10(255.0 / math.sqrt(mse)))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - ux) * (y - uy)).mean()
    return float(((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))


def residual_diagnostics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    r = observed.astype(np.float32) - predicted.astype(np.float32)
    gray = r.mean(axis=2) if r.ndim == 3 else r
    centered = gray - gray.mean()
    denom = float(np.sum(centered * centered)) + 1e-9
    ac_x = float(np.sum(centered[:, 1:] * centered[:, :-1]) / denom)
    ac_y = float(np.sum(centered[1:, :] * centered[:-1, :]) / denom)
    f = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(f) ** 2
    h, w = power.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    low = power[rr < min(h, w) * 0.12].mean() if np.any(rr < min(h, w) * 0.12) else 0
    high = power[rr > min(h, w) * 0.32].mean() if np.any(rr > min(h, w) * 0.32) else 0
    return {
        "mae": float(np.mean(np.abs(r))),
        "rmse": float(np.sqrt(np.mean(r * r))),
        "std": float(r.std()),
        "lag1_autocorr_x": ac_x,
        "lag1_autocorr_y": ac_y,
        "low_high_spectral_ratio": float(low / (high + 1e-9)),
    }


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    size = max(3, int(size) | 1)
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return k / k.sum()


def motion_psf(size: int, angle_degrees: float) -> np.ndarray:
    size = max(3, int(size) | 1)
    k = np.zeros((size, size), np.float32)
    cv2.line(k, (0, size // 2), (size - 1, size // 2), 1.0, 1)
    m = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_degrees, 1.0)
    k = cv2.warpAffine(k, m, (size, size))
    return k / max(float(k.sum()), 1e-9)


def convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        return signal.fftconvolve(arr, psf, mode="same")
    return np.stack([signal.fftconvolve(arr[..., c], psf, mode="same") for c in range(arr.shape[2])], axis=2)


def wiener_deconvolution(image: np.ndarray, psf: np.ndarray, balance: float) -> tuple[np.ndarray, float]:
    arr = image.astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    pad = np.zeros((h, w), np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-kh // 2, -kw // 2), axis=(0, 1))
    H = np.fft.fft2(pad)
    inv = np.conj(H) / (np.abs(H) ** 2 + float(balance))
    chans = [arr] if arr.ndim == 2 else [arr[..., c] for c in range(arr.shape[2])]
    out = [np.real(np.fft.ifft2(np.fft.fft2(c) * inv)) for c in chans]
    raw = out[0] if arr.ndim == 2 else np.stack(out, axis=2)
    clipping = float(np.mean((raw < 0) | (raw > 1)))
    return np.clip(raw * 255.0, 0, 255), clipping


def richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
    arr = image.astype(np.float32) / 255.0
    chans = [arr] if arr.ndim == 2 else [arr[..., c] for c in range(arr.shape[2])]
    flipped = psf[::-1, ::-1]
    restored = []
    clipping = 0.0
    for c in chans:
        estimate = np.maximum(c, 1e-4)
        for _ in range(max(1, int(iterations))):
            relative = c / np.maximum(signal.fftconvolve(estimate, psf, mode="same"), 1e-5)
            estimate *= signal.fftconvolve(relative, flipped, mode="same")
        clipping += float(np.mean((estimate < 0) | (estimate > 1)))
        restored.append(np.clip(estimate, 0, 1))
    raw = restored[0] if arr.ndim == 2 else np.stack(restored, axis=2)
    return raw * 255.0, clipping / len(chans)


def make_contact_sheet(images: list[np.ndarray], labels: list[str], columns: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("No images for contact sheet")
    thumb_w = min(640, max(i.shape[1] for i in images))
    thumb_h = int(thumb_w * max(i.shape[0] / i.shape[1] for i in images))
    rows = math.ceil(len(images) / columns)
    canvas = np.full((rows * (thumb_h + 42), columns * thumb_w, 3), 245, np.uint8)
    for idx, (im, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, columns)
        scale = min(thumb_w / im.shape[1], thumb_h / im.shape[0])
        resized = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y = r * (thumb_h + 42)
        x = c * thumb_w
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(canvas, label, (x + 8, y + thumb_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas
