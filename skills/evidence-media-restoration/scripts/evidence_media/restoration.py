from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core import convolve, gaussian_psf, motion_psf, residual_diagnostics, richardson_lucy, sharpness, wiener_deconvolution


def deterministic_denoise(image: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    dcfg = config.get("denoise", {})
    outputs: dict[str, np.ndarray] = {}
    if dcfg.get("nlm", True):
        h = float(dcfg.get("nlm_h", 4.0))
        outputs["denoise_nlm"] = cv2.fastNlMeansDenoisingColored(image.astype(np.uint8), None, h, h, 7, 21)
    if dcfg.get("bilateral", False):
        outputs["denoise_bilateral"] = cv2.bilateralFilter(image.astype(np.uint8), 7, 20, 3)
    return outputs


def deconvolution_sweep(image: np.ndarray, config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str | None]:
    dcfg = config.get("deconvolution", {})
    if not dcfg.get("enabled", True):
        return {}, [], None
    candidates: dict[str, np.ndarray] = {}
    metrics: list[dict[str, Any]] = []
    psfs: list[tuple[str, np.ndarray]] = []
    for sigma in dcfg.get("gaussian_sigmas", [0.7, 1.0, 1.4]):
        psfs.append((f"gaussian_s{float(sigma):.2f}", gaussian_psf(int(dcfg.get("psf_size", 9)), float(sigma))))
    for length in dcfg.get("motion_lengths", []):
        for angle in dcfg.get("motion_angles", [0, 45, 90, 135]):
            psfs.append((f"motion_l{int(length)}_a{int(angle)}", motion_psf(int(length), float(angle))))
    balances = [float(v) for v in dcfg.get("wiener_balances", [0.002, 0.006, 0.02])]
    rl_iterations = [int(v) for v in dcfg.get("rl_iterations", [4, 8, 12])]
    noise_target = float(dcfg.get("noise_sigma_target", 3.0))
    for psf_name, psf in psfs:
        for balance in balances:
            name = f"wiener_{psf_name}_b{balance:g}"
            result, clipping = wiener_deconvolution(image, psf, balance)
            predicted = np.clip(convolve(result, psf), 0, 255)
            diag = residual_diagnostics(image, predicted)
            objective = abs(diag["std"] - noise_target) + 2.0 * (abs(diag["lag1_autocorr_x"]) + abs(diag["lag1_autocorr_y"])) + 80.0 * clipping
            metrics.append({"name": name, "method": "wiener", "psf": psf_name, "parameter": balance, "clipping_fraction": clipping, "sharpness": sharpness(result), "objective": float(objective), **diag})
            candidates[name] = result
        for iterations in rl_iterations:
            name = f"rl_{psf_name}_i{iterations}"
            result, clipping = richardson_lucy(image, psf, iterations)
            predicted = np.clip(convolve(result, psf), 0, 255)
            diag = residual_diagnostics(image, predicted)
            objective = abs(diag["std"] - noise_target) + 2.0 * (abs(diag["lag1_autocorr_x"]) + abs(diag["lag1_autocorr_y"])) + 80.0 * clipping
            metrics.append({"name": name, "method": "richardson_lucy", "psf": psf_name, "parameter": iterations, "clipping_fraction": clipping, "sharpness": sharpness(result), "objective": float(objective), **diag})
            candidates[name] = result
    metrics.sort(key=lambda x: x["objective"])
    selected = metrics[0]["name"] if metrics else None
    keep = {selected: candidates[selected]} if selected else {}
    count = int(dcfg.get("keep_top", 3))
    for row in metrics[:count]:
        keep[row["name"]] = candidates[row["name"]]
    return keep, metrics, selected
