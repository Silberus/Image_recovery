# Evidence Media Restoration 0.2.1

A configurable, evidence-preserving image/video restoration plugin. It keeps observed pixels, deterministic derivatives, model-dependent deconvolution, and optional neural suggestions in separate evidence classes.

## Quick start

```powershell
.\scripts\setup-runtime.ps1 -Profile extended
.\scripts\run-restoration.ps1 -InputPath input.mp4 -OutputPath case-output -Profile hmi-screen
```

Copy and edit the chosen YAML profile before a case. The run produces a contact sheet, source observations, registered-frame ledger, robust fusions, donor/support/MAD maps, deconvolution sweep, validation table, resolved configuration, manifest, hashes, and report.

No OCR, inpainting, or generative completion is used by the core pipeline.

## Tiny HMI registers

For values only a few source pixels high, process the full screen first and the
registers second:

1. register the complete screen with a fixed reference frame;
2. map native decoded pixels directly into each register ROI once;
3. preserve the best individual observations and the complete aligned ledger
   alongside median/Huber fusion;
4. inspect a full status row before splitting its date, clock, or counter, and
   segment changing fields into persistent states before fusion;
5. estimate effective blur from nearby display borders, never from an expected
   digit;
6. run a bounded PSF sweep and retain every parameter/result; if no valid edge
   exists, mark the sweep `UNCALIBRATED_FALLBACK` and do not use it as evidence;
7. use whole-field forward hypotheses only as `MODEL_SUGGESTION` and reject
   them when candidate margins or temporal stability are weak.

Specialized scripts are in
`skills/evidence-media-restoration/scripts/`:

- `native_register_superresolution.py` — direct native-frame mapping, phase
  census, source-observation ranking and robust fusion;
- `temporal_field_topology.py` — fail-closed persistent-state and topology
  analysis for static or changing fields;
- `rectify_still_screen.py` — deterministic rectification of supplied stills
  from four explicit screen corners;
- `edge_psf_deconvolution.py` — independent edge-spread estimate plus bounded
  Wiener/Richardson-Lucy diagnostics;
- `compact_glyph_sheet.py` — auditable repeated-glyph sheet;
- `font_glyph_hypotheses.py` and `field_forward_hypotheses.py` — explicitly
  non-evidentiary inverse-rendering hypotheses.
