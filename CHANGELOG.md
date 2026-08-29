# Changelog

## 0.2.1

- Added single-resampling native-frame reconstruction for tiny HMI registers.
- Added per-register source-observation ranking and subpixel phase coverage.
- Added effective PSF estimation from non-semantic display edges and bounded
  Wiener/Richardson-Lucy diagnostics.
- Added compact repeated-glyph sheets and whole-field `ddd.dd` hypothesis
  testing with mandatory `MODEL_SUGGESTION` classification.
- Added fail-closed guards for incorrect glyph counts, weak candidate margins,
  dependent video encodes and fusion-induced stroke loss.

## 0.2.0 — 2026-08-29

- Added a unified YAML-configured CLI for video, image sequences, and single images.
- Added optional PyAV decoding with PTS/time-base/key-frame/picture-type records and an explicit OpenCV fallback warning.
- Added ROI rectification, phase/ECC registration, rejection gates, temporally blocked pseudo-holdout, robust median/Huber fusion, donor mosaic, support and MAD maps.
- Added deterministic denoise and independent Wiener/Richardson-Lucy PSF/regularization sweeps with forward residual, lag correlation, spectral ratio and clipping diagnostics.
- Added three reusable profiles plus a tested UOP Aging Tower example.
- Added isolated runtime setup, synthetic regression self-test, single-image and real-HMI smoke tests.
- Added library/license matrix and eight-book algorithm crosswalk.
- Kept OCR, inpainting and generative completion outside the evidentiary core.
