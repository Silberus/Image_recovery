from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from .core import entropy, make_contact_sheet, psnr, residual_diagnostics, save_image, sha256_file, sharpness, ssim_global, write_csv, write_json
from .fusion import fuse
from .io_media import decode_frames, inspect_source
from .registration import register
from .restoration import deconvolution_sweep, deterministic_denoise


def _normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def _validation_rows(outputs: dict[str, np.ndarray], holdout: list[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for name, image in outputs.items():
        if image.ndim != 3 or not holdout:
            continue
        residuals = [residual_diagnostics(h, image) for h in holdout]
        rows.append({
            "name": name,
            "evidence_class": "OBSERVED" if name == "best_observed" else ("DERIVED_DETERMINISTIC" if not name.startswith(("wiener_", "rl_")) else "MODEL_DEPENDENT"),
            "sharpness": sharpness(image),
            "entropy": entropy(image),
            "holdout_mae_median": float(np.median([r["mae"] for r in residuals])),
            "holdout_rmse_median": float(np.median([r["rmse"] for r in residuals])),
            "holdout_lag1_abs_median": float(np.median([abs(r["lag1_autocorr_x"]) + abs(r["lag1_autocorr_y"]) for r in residuals])),
            "note": "Pseudo-holdout from the same source stream; not an independent witness.",
        })
    return sorted(rows, key=lambda r: r["holdout_mae_median"])


def _report_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Evidence Media Restoration report",
        "",
        f"- Created: `{manifest['created_utc']}`",
        f"- Source: `{manifest['source']['path']}`",
        f"- Source SHA-256: `{manifest['source'].get('sha256', 'directory sequence')}`",
        f"- Decode backend: `{manifest['source'].get('decode_backend', manifest['source'].get('backend'))}`",
        f"- Decoded / registered: `{manifest['counts']['decoded']}` / `{manifest['counts']['registered']}`",
        f"- Evidence mode: `{manifest['policy']['evidence_mode']}`",
        "",
        "## Interpretation boundary",
        "",
        "The report distinguishes observed pixels, deterministic multi-observation derivatives, model-dependent deconvolution, and optional neural suggestions. A sharper-looking result is not proof of a glyph. No OCR, inpainting, or generative completion is performed by the core pipeline.",
        "",
        "## Validation ranking",
        "",
        "| Output | Class | Sharpness | Holdout MAE | Holdout residual correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in manifest["validation"]:
        lines.append(f"| {row['name']} | {row['evidence_class']} | {row['sharpness']:.2f} | {row['holdout_mae_median']:.3f} | {row['holdout_lag1_abs_median']:.4f} |")
    lines += [
        "",
        "## Deconvolution selection",
        "",
        f"Selected only by discrepancy/residual diagnostics: `{manifest['restoration'].get('selected_deconvolution')}`. This selection is independent of expected text and does not establish character identity.",
        "",
        "## Required reading order",
        "",
        "1. Inspect `00_contact_sheet.png` and the source-frame ledger.",
        "2. Use `best_observed.png` for literal observation.",
        "3. Compare median/Huber/mosaic and support/MAD maps.",
        "4. Treat deconvolution as a model-dependent diagnostic only.",
        "5. Promote a symbol only when its strokes are stable across independent processing routes and alternatives are separable.",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline(input_path: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scfg = config.get("source", {})
    records, source_info = decode_frames(input_path, scfg.get("start_seconds"), scfg.get("end_seconds"), int(scfg.get("stride", 1)), bool(scfg.get("prefer_pyav", True)))
    if len(records) < 1:
        raise RuntimeError("No observation was decoded")
    aligned, registration_log, reference_index = register(records, config)
    required = 1 if source_info.get("kind") == "image" else int(config.get("registration", {}).get("min_accepted_frames", 4))
    if len(aligned) < required:
        raise RuntimeError(f"Only {len(aligned)} frames passed registration gates")
    images = [r.image for r in aligned]
    if len(images) == 1:
        h, w = images[0].shape[:2]
        fused = {
            "best_observed": images[0],
            "temporal_median": images[0].copy(),
            "huber_mean": images[0].copy(),
            "observed_tile_mosaic": images[0].copy(),
            "temporal_mad": np.zeros((h, w), np.float32),
            "support_count": np.ones((h, w), np.float32),
            "donor_map": np.ones((h, w), np.uint16),
        }
        fusion_meta = {"build_indices": [0], "holdout_indices": [], "donor_rows": [], "best_build_index": 0, "single_observation": True}
    else:
        fused, fusion_meta = fuse(images, config)
    holdout = [images[i] for i in fusion_meta["holdout_indices"]]
    outputs: dict[str, np.ndarray] = {k: v for k, v in fused.items() if v.ndim == 3}
    base_name = str(config.get("restoration", {}).get("base", "huber_mean"))
    base = outputs.get(base_name, outputs["huber_mean"])
    outputs.update(deterministic_denoise(base, config.get("restoration", {})))
    deconv, deconv_metrics, selected = deconvolution_sweep(base, config.get("restoration", {}))
    outputs.update(deconv)
    name_map = {
        "best_observed": "01_best_observed.png",
        "temporal_median": "02_temporal_median.png",
        "huber_mean": "03_huber_mean.png",
        "observed_tile_mosaic": "04_observed_tile_mosaic.png",
        "denoise_nlm": "05_denoise_nlm.png",
        "denoise_bilateral": "06_denoise_bilateral.png",
    }
    written = []
    for name, image in outputs.items():
        filename = name_map.get(name, f"model_{name}.png")
        save_image(output_dir / filename, image)
        written.append({"name": name, "file": filename, "sha256": sha256_file(output_dir / filename)})
    mad = fused["temporal_mad"]
    support = fused["support_count"]
    donor = fused["donor_map"]
    save_image(output_dir / "07_temporal_mad.png", cv2.applyColorMap(cv2.normalize(mad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_MAGMA))
    save_image(output_dir / "08_support_count.png", cv2.applyColorMap(cv2.normalize(support, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
    save_image(output_dir / "09_donor_map.png", cv2.applyColorMap(cv2.normalize(donor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO))
    contact_names = ["best_observed", "temporal_median", "huber_mean", "observed_tile_mosaic"] + ([selected] if selected else [])
    contact_names = [n for n in contact_names if n in outputs]
    save_image(output_dir / "00_contact_sheet.png", make_contact_sheet([outputs[n] for n in contact_names], contact_names, 2))
    frame_dir = output_dir / "source_observations"
    frame_rows = []
    for i, record in enumerate(aligned):
        filename = f"obs_{i:04d}_source_{record.ordinal:08d}.png"
        save_image(frame_dir / filename, record.image)
        frame_rows.append({**record.public(), "registered_index": i, "file": str(Path("source_observations") / filename), "sha256": sha256_file(frame_dir / filename), "sharpness": sharpness(record.image)})
    validation = _validation_rows(outputs, holdout)
    write_csv(output_dir / "registration.csv", registration_log)
    write_csv(output_dir / "source_observations.csv", frame_rows)
    write_csv(output_dir / "donor_tiles.csv", fusion_meta["donor_rows"])
    write_csv(output_dir / "validation.csv", validation)
    write_csv(output_dir / "deconvolution_sweep.csv", deconv_metrics)
    write_json(output_dir / "resolved_config.json", config)
    manifest = {
        "schema": "evidence-media-restoration/0.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_info,
        "counts": {"decoded": len(records), "registered": len(aligned), "build": len(fusion_meta["build_indices"]), "holdout": len(fusion_meta["holdout_indices"])},
        "reference_decoded_index": reference_index,
        "policy": {
            "evidence_mode": bool(config.get("policy", {}).get("evidence_mode", True)),
            "ocr_used": False,
            "inpainting_used": False,
            "generative_model_used": False,
            "holdout_is_independent_witness": False,
            "classes": ["OBSERVED", "DERIVED_DETERMINISTIC", "MODEL_DEPENDENT", "MODEL_SUGGESTION"],
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "fusion": {k: v for k, v in fusion_meta.items() if k != "donor_rows"},
        "restoration": {"base": base_name, "selected_deconvolution": selected, "selection_rule": "noise discrepancy + residual lag-1 correlation + clipping penalty; no OCR/readability term"},
        "validation": validation,
        "outputs": written,
        "warnings": [
            "Same-stream frames and alternate encodes are dependent observations, not independent witnesses.",
            "Interpolation and deconvolution do not create new observed strokes.",
            "Low forward residual is necessary but insufficient for literal glyph identification.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "REPORT.md").write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest
