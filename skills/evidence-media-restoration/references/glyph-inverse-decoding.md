# Inverse decoding of tiny screen text

Use this mode when a character is too small or blurred for direct visual reading. The objective is not a sharper illustration; it is a calibrated set of character hypotheses supported by the native observations.

## 1. Information census before reconstruction

Decode the native Y, Cb and Cr planes without intermediate JPEG export. Record frame type, presentation time, GOP membership, repeated/near-repeated frames and local quantization or blocking diagnostics when the codec exposes them. P/B frames are valid decoded observations, but prediction dependencies mean that consecutive frames are not automatically independent noise samples.

Estimate each source-frame mapping directly to a common physical screen plane. Preserve the mapping matrices. Do not rectify every frame first and then infer phase diversity from the rectified images: the first interpolation can create the apparent fractional shifts.

For a high-resolution screen coordinate `u`, map its projected location in native frame `k` to the fractional sensor phase

`phi_k(u) = frac(P_k(u))`.

Report phase-bin occupancy, effective sample weights and the conditioning of the local sampling operator. A nominal 4x grid needs phase support across that grid; many samples in one bin do not replace missing bins. Select the largest scale whose local operator is sufficiently supported and stable under frame removal.

## 2. Screen-specific degradation model

Use the source-plane model

`y_k = C_k Q_k D_k H_k R_k S_theta(g) + n_k`,

where `g` is a candidate glyph or string, `S_theta` renders it on the display lattice, `R_k` models screen-pixel/subpixel response and viewing geometry, `H_k` models optical and motion blur, `D_k` camera sampling, `Q_k` codec quantization and `C_k` photometric response.

Estimate nuisance parameters from the same screen: border edges for blur and geometry, flat black/gray fields for noise and codec structure, and large known text for screen-pixel and font-scale clues. Treat font identity as an uncertain parameter unless an exact HMI asset or matching clean screenshot establishes it.

Process luma and colour evidence separately. H.264 4:2:0 chroma normally has less spatial resolution than luma; colour contrast can still help segmentation but must not be counted as equal-resolution character evidence.

Do not assume one shift-invariant PSF for the whole monitor. Camera rotation, perspective and local focus can make motion blur spatially varying. Estimate edge-spread evidence in several screen regions. Use a global kernel only if the local estimates are mutually compatible; otherwise use a bounded patchwise or projective-motion family.

Treat the residual as coloured until tested. Display moire, refresh banding, ISP and inter-frame coding violate an independent white-noise assumption. Parameter-choice rules that require a noise norm or distribution must use an empirically estimated residual model.

## 3. Self-calibrating glyph atlas

Exploit repeated UI structure without assigning meanings prematurely:

1. Detect equal-size numeric boxes and decimal positions from geometry.
2. Segment character cells using the common box layout, not a desired number.
3. Cluster recurring cell observations across registers and time after native-plane mapping.
4. Build an empirical atlas of stroke-probability maps with support and disagreement maps.
5. Label a cluster only from a clearly observed occurrence or a separately established HMI/font asset. Until then use neutral cluster IDs.

This is analogous to deciphering a repeated alphabet: repetition can reconstruct glyph classes even when one occurrence is weak. It cannot attach a digit name without a bridge.

### Empirical blur dictionary and topology bridge

When a symbol is unambiguous in one part of the same display, preserve its
appearance across frames as an empirical degradation dictionary. Compare a
target to the whole sequence or to even/odd reconstructions, not merely to one
sharp-looking enlargement. Use the known occurrence as an anchor only after it
passes leave-one-location-out recognition on other known anchors.

Normalize every glyph to an explicit `(height, width)` contract. Check the
array shape before clustering: swapping width and height can produce stable but
case-location-specific clusters. For weight or font variations, a topology
view may combine a centered soft mask, skeleton, distance transform and axis
projections. Select its parameters on held-out known glyphs, never on the
unknown target.

If the alphabet is known in advance, retain that fixed-alphabet model in the
report even when an unsupervised score prefers fewer clusters. A lower selected
cluster count may mean that some symbols are absent, but it may also mean that
blur merged distinct glyphs. Do not use it to shrink a known alphabet silently.

### Constrained word decipherment

For degraded labels, keep the visual likelihood and the word constraint as
separate terms. A candidate lexicon, UI vocabulary, repeated headings or known
field grammar may rank visually plausible strings, but cannot create a missing
stroke. Report the unconstrained per-position candidate sets alongside the
lexicon-ranked words. Accept a word only when its letters remain stable across
frame splits and at least one alternative deterministic reconstruction; when
the lexicon is the deciding evidence, label the result `MODEL_SUGGESTION`.

Use confirmed labels to add new glyph anchors iteratively. Re-run the held-out
anchor test after every expansion, and stop propagation when one inferred
anchor would be used to validate itself.

## 4. Candidate testing in observation space

For candidate glyph `g`, render a high-resolution template, project and degrade it separately for every frame, then compare the prediction with the native pixel patch. Fit only declared nuisance parameters such as sub-pixel offset, contrast and a bounded blur kernel.

Use a robust likelihood, for example

`L(g) = min_eta sum_k sum_p w_kp rho(y_kp - F_k(S_theta(g); eta_k)_p)`.

Compare candidates by likelihood ratios or posterior mass, not by which enlarged glyph looks most familiar. Produce the complete ranked candidate set for each position. If several candidates explain the observations similarly, return the set, for example `{2,7,9}`, rather than one digit.

For a numeric string, combine cell likelihoods only after checking that segmentation errors are negligible. A temporal smoothness model may test consistency of successive readings, but process plausibility, expected temperature, neighbouring values and engineering intuition are not pixel evidence and must be reported separately.

## 5. Calibration and falsification

Before accepting real readings, measure the method's error on controls that match the target:

- Render every allowed glyph with each defensible font candidate.
- Pass controls through measured perspective, blur, sampling, colour subsampling and compression.
- Include small perturbations and mismatched blur kernels.
- Run the complete blind decoder and build a confusion matrix.
- Add negative controls: wrong digits, shuffled phases, missing frames and deliberately under-resolved glyphs.

Choose an abstention threshold from the control error rate. A target result is reportable only if it survives leave-one-GOP-out, leave-one-viewpoint-out, registration perturbation and plausible degradation-model changes. Pixel residuals alone are insufficient; require a calibrated candidate margin and stable stroke topology.

Select regularization strength and iterative stopping independently of the desired reading. If a defensible noise norm is available, use a discrepancy-principle check; otherwise use GCV and residual-spectrum diagnostics, with an L-curve as a supporting diagnostic. Report whether these criteria disagree. Do not tune a parameter to maximize OCR confidence.

For each candidate, forward-project the reconstruction through every admissible degradation model. A low residual is necessary but not sufficient: if two glyphs remain within the calibrated error envelope, the correct result is `AMBIGUOUS {…}`.

## 6. Evidence output

For every field deliver:

- native frame/time/GOP witnesses;
- projected native patches without enhancement;
- phase-coverage and conditioning diagnostics;
- empirical glyph/stroke support maps;
- ranked candidates and likelihood margins;
- synthetic-control confusion matrix;
- sensitivity and ablation results;
- final status: `CONFIRMED`, `AMBIGUOUS {…}`, or `UNRESOLVED`.

Neural or generative text super-resolution may be shown only as `MODEL_SUGGESTION`. It cannot supply missing strokes or validate a literal reading.

PDE/variational inpainting is subject to the same restriction. Its continuation of isophotes or edges is a mathematical interpolation, not evidence that the completed stroke existed on the display.

## Method anchors

- Farsiu et al., *Fast and Robust Multiframe Super Resolution*, IEEE TIP 13(10), 2004, DOI `10.1109/TIP.2004.834669`.
- Wronski et al., *Handheld Multi-Frame Super-Resolution*, ACM TOG 38, 2019.
- Bhat et al., *Deep Burst Super-Resolution*, CVPR 2021.
- Chen et al., *Scene Text Telescope*, CVPR 2021. Text-focused neural priors are diagnostic/model suggestions, not evidence authority.
- Rajagopalan and Chellappa (eds.), *Motion Deblurring: Algorithms and Systems*, Cambridge University Press, 2014. Spatially varying and projective-motion blur models.
- Hansen, *Discrete Inverse Problems: Insight and Algorithms*, SIAM, 2010. Discrepancy principle, GCV, L-curve and residual analysis.
- Bertero, Boccacci and De Mol, *Introduction to Inverse Problems in Imaging*, 2nd ed., CRC Press, 2022. Ill-posedness, regularization and iterative stopping.
- Hansen, Nagy and O'Leary, *Deblurring Images: Matrices, Spectra, and Filtering*, SIAM, 2006. PSF centering, boundary conditions and spectral filtering.
- Bhandari, Kadambi and Raskar, *Computational Imaging*, MIT Press, 2022. Forward-model-first computational imaging.
- Barbu, *Novel Diffusion-Based Models for Image Restoration and Interpolation*, Springer, 2019. PDE diffusion and inpainting; interpolation is not observational evidence.
- OSAC, *Standard Guide for Forensic Digital Video Examination Workflow*, version 2.0.
- SWGDE, *Best Practices for Digital Forensic Video Analysis*.
