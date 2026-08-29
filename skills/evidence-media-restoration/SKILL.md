---
name: evidence-media-restoration
description: "Restore and analyze degraded photos, video frames, screens, documents, and fragmented visual evidence without inventing content. Use for frame extraction, codec/GOP inspection, registration, lucky imaging, robust stacking, shift-and-add or drizzle-like reconstruction, image mosaics, donor/support/uncertainty maps, cross-file manifestation comparison, and analytical mosaic/data-fusion work where every reading needs provenance and an explicit confidence boundary."
---

# Evidence Media Restoration

## Non-negotiable rule

Never turn the expected answer into visible evidence. Preserve the original, work on copies, log every transformation, and keep observed pixels separate from derived estimates and interpretive hypotheses.

Do not use generative fill, neural hallucination, content-aware inpainting, or prompt-guided restoration to read evidence. If a user separately requests an illustrative reconstruction, isolate and label it `MODEL_SUGGESTION — NOT EVIDENCE`; do not use it to support a factual reading.

## Evidence classes

Use these labels consistently:

- `OBSERVED`: directly visible in one identified source frame or file.
- `DERIVED_DETERMINISTIC`: deterministic transformation of identified sources, such as rectification, median, or a support map.
- `RECONSTRUCTED_MINIMUM`: the smallest structure forced by multiple independent observations.
- `MODEL_SUGGESTION`: a non-evidentiary illustration or machine hypothesis.
- `UNRESOLVED`: information not supported at source resolution.

Different encodes, screenshots, or crops of one recording form one witness cluster. They can provide different compression artefacts but are not independent corroboration.

## Workflow

1. Create a case directory. Record path, byte length, SHA-256, container/codec, dimensions, frame rate, duration, and extraction interval.
2. Inspect the codec and GOP only to understand decoding dependencies. An I-frame is not automatically the sharpest or most legible frame.
3. Compare alternate files across the whole timeline using low-frequency frame fingerprints, stable time offsets, and pixel similarity. Cluster dependent manifestations before counting support.
4. Decode every frame in the relevant interval. Preserve decoded frames and exact times/frame numbers.
5. Select geometry from visible screen or document corners. Do not tune a region around an expected word or digit.
6. Register in two stages: global feature/RANSAC alignment, then local ECC or phase-correlation refinement. Reject weak transforms and retain the alignment log.
7. Produce at least four views: best observed frame, temporal median, robust clipped/Huber mean, and an observed-tile mosaic. Every tile mosaic must include a donor map and donor table.
8. Produce uncertainty views: source support count, temporal median absolute deviation, and transform residuals.
9. Split frames into training and holdout sets. Compare reconstruction residuals on holdout frames. Sharpness alone never chooses the evidentiary result.
10. Treat shift-and-add, drizzle-like oversampling, Wiener/Richardson-Lucy deconvolution, or super-resolution as derived diagnostics. Accept a character only if its topology is stable across source frames, registration perturbations, and at least two non-equivalent deterministic methods.
11. Inspect original-resolution outputs visually. Record which strokes are stable and which remain ambiguous. Use `?` for an unresolved character; never silently complete it from process expectations.
12. Deliver originals/provenance, accepted frames, method comparison, donor/support/MAD maps, metrics, parameter log, and a statement of the information ceiling.

For tiny instrument values, labels, licence plates, or screen registers, a readable enlargement is not the decision surface. Read [references/glyph-inverse-decoding.md](references/glyph-inverse-decoding.md) and perform recognition in observation space: test candidate glyphs through the measured camera/display degradation model against the native decoded frames. First establish whether the frames contain independent sub-pixel phases; do not estimate this from images that have already been rectified or resampled.

### Tiny-register fail-closed sequence

1. Rectify/register the whole screen to establish geometry and labels.
2. Compose the native-frame-to-register transform and resample each source
   frame only once. Use `native_register_superresolution.py`; retain the six
   strongest source observations because fusion can erase a decisive stroke.
3. Determine the true field format and glyph count before splitting. A wrong
   split can create high-scoring but meaningless digit matches.
4. Estimate an effective PSF from nearby non-semantic edges with
   `edge_psf_deconvolution.py`. Treat the estimate as optics + display
   antialiasing + resampling, not as a pure lens measurement.
5. Compare observed, Wiener and Richardson-Lucy topology over a bounded sweep.
   A stroke that appears only at one parameter is not evidence.
6. Use `compact_glyph_sheet.py` to compare repeated glyphs across registers.
7. Candidate-font and whole-field searches remain `MODEL_SUGGESTION`. Reject
   the numeric output when top-candidate margins are small, candidates change
   across source frames, or the assumed font/format is not independently known.
8. Report `UNRESOLVED` rather than imposing process-plausible values.

## Working tool

Use `scripts/evidence_media_tool.py` as the default entry point. It accepts a video, one image, or an image-sequence directory and a YAML profile. The complete run performs source inventory, exact PyAV decoding when available, ROI rectification, phase/ECC registration, quality rejection, temporally blocked build/holdout split, robust fusion, donor/support/MAD products, independent deconvolution sweeps, residual diagnostics, hashes, CSV ledgers, a manifest, and a Markdown report.

Create the isolated runtime once from the plugin root:

```powershell
.\scripts\setup-runtime.ps1 -Profile extended
```

Inspect a source without changing it:

```powershell
.\.runtime\Scripts\python.exe .\skills\evidence-media-restoration\scripts\evidence_media_tool.py inspect input.mp4
```

Run a profile:

```powershell
.\scripts\run-restoration.ps1 -InputPath input.mp4 -OutputPath case-output -Profile hmi-screen
```

Edit a copy of `assets/profiles/hmi-screen.yaml` to set the interval and four visible ROI corners. Use `archival-film.yaml` for stabilized film segments and `document-text.yaml` for multiple photographs of one flat document. Before trusting an installation, run:

```powershell
.\.runtime\Scripts\python.exe .\skills\evidence-media-restoration\scripts\evidence_media_tool.py selftest .selftest --config .\skills\evidence-media-restoration\assets\profiles\hmi-screen.yaml
```

Keep `scripts/restore_evidence_media.py`, `inverse_glyph_audit.py`, `glyph_template_hypotheses.py`, `native_register_superresolution.py`, `edge_psf_deconvolution.py`, `compact_glyph_sheet.py`, `font_glyph_hypotheses.py`, `field_forward_hypotheses.py`, and `find_screen_views.py` for specialized investigations. Successful execution is never proof of legibility.

## Dependency and model boundary

Read [references/library-and-repository-matrix.md](references/library-and-repository-matrix.md) before enabling optional packages or external repositories. Core and extended dependencies are version-ranged, not silently downloaded during analysis. External neural repositories and weights are not bundled. Their outputs, if explicitly enabled later, must be isolated as `MODEL_SUGGESTION` and may not resolve a factual glyph.

## Analytical mosaic

When the task combines visual fragments with documents, posts, metadata, or technical sources, read `references/analytical-mosaic.md`. Build a claim-evidence graph, preserve source dependence, test alternatives, and report the minimal supported reconstruction rather than the most coherent story.

## Platform-media provenance

When asked who filmed, edited, published, or stored an online video, read
`references/platform-media-provenance.md`. Keep the camera operator, production
company, editor, account/post author, corporate commissioning team, and platform
transcoder as separate identities. Container encoder tags and CDN asset times
normally describe processing or publication, not capture. Confirm a creator or
vendor only from an explicit credit, authoritative production record, matching
original-file metadata, or a dated public statement; recurring visual style is
discovery evidence only.

## Theory and method choice

Read `references/methods-and-math.md` before choosing restoration parameters, claiming super-resolution, or explaining why a value remains unreadable. It defines the forward model, registration/fusion equations, sampling ceiling, and validation gates.

For the direct mapping from the eight supplied books to implemented controls, read [references/book-to-algorithm-crosswalk.md](references/book-to-algorithm-crosswalk.md).
