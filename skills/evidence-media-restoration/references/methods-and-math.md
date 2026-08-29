# Methods and mathematical limits

## Forward model

For frame `k`, use

`y_k = Q_k D_k H_k W_k x + n_k + c_k`,

where `x` is the latent scene, `W_k` geometric motion, `H_k` optical/motion blur, `D_k` sampling, `Q_k` quantization/codec loss, `n_k` noise, and `c_k` compression artefacts. Restoration estimates `x`; it does not observe it directly.

A robust multi-frame estimate can be written

`x_hat = argmin_x sum_k w_k rho(y_k - Q_k D_k H_k W_k x) + lambda R(x)`.

Every regularizer `R` inserts assumptions. A visually sharp solution is not necessarily the best-supported solution.

## Registration

Estimate a projective mapping `p' ~ H p` with feature matches and RANSAC, then refine the local region with translation or affine ECC:

`H* = argmax_H ECC(I_ref, W(I_k; H))`.

Reject transforms with too few inliers, implausible geometry, or unstable results under small parameter changes. Retain match/inlier counts and residual shifts.

## Robust fusion

Median:

`x_med(p) = median_k y_k'(p)`.

Robust clipped mean around the median:

`x_h(p) = x_med(p) + mean_k clip(y_k'(p)-x_med(p), -delta(p), delta(p))`,

with `delta(p)` derived from the median absolute deviation. This suppresses transient codec blocks and occlusions but may blur fine text.

For each pixel report temporal dispersion

`MAD(p) = median_k |y_k'(p)-x_med(p)|`

and support count, the number of frames within a stated tolerance of the median. Low dispersion is necessary but not sufficient for correct reading.

## Lucky imaging and donor mosaics

Rank local patches, not whole frames. A donor mosaic must copy each output tile from one registered observed frame and record the donor frame/time. Penalize tiles that disagree with the temporal median; sharpness alone rewards codec ringing.

Visible seams are acceptable in an analysis artifact because they expose source boundaries. Do not hide them unless a second labelled display copy is created.

## Shift-and-add and drizzle-like oversampling

Sub-pixel offsets can improve sampling only when the source frames contain genuinely different phase samples. A generic weighted oversampling model is

`x(u) = sum_k w_k(u) y_k'(u) / sum_k w_k(u)`.

Call an interpolation-and-shift implementation `drizzle-like` unless it implements the Fruchter-Hook footprint/drop operator and documents `pixfrac`. Always down-project the result to the source grid and compare it with held-out frames.

## Deconvolution

Wiener filtering in the frequency domain:

`X_hat = H*Y / (|H|^2 + K)`.

Richardson-Lucy iteration:

`x_(t+1) = x_t [h_flip * (y / (h*x_t))]`.

Both require a defensible point-spread function. Wrong blur kernels or too many iterations manufacture ringing that resembles character strokes. Use them as sensitivity tests, never as the only basis for a literal reading.

## Sampling ceiling

If character strokes are not sampled above the effective Nyquist limit, or were removed by motion blur and quantization, no deterministic enhancement can uniquely recover them. Multiple frames help only when they add different sub-pixel samples or independent noise realizations.

Measure phase diversity on the mappings from the latent screen plane into the native sensor grid. Fractional translations measured after rectification do not prove independent sampling because the rectification interpolation itself introduces fractional values. Assess local identifiability from the rank/conditioning of the native sampling operator and select the reconstruction scale from supported phase coverage rather than a requested enlargement factor.

For literal text, the statistically correct comparison is often not between enhanced images but between forward projections of competing glyph hypotheses. See `glyph-inverse-decoding.md` for candidate likelihoods, empirical glyph atlases, synthetic confusion controls and abstention rules.

## Validation without ground truth

- Split source frames into train and holdout sets.
- Build candidates only from train frames.
- Compare median and 90th-percentile absolute residual on holdout frames.
- Check edge residuals and stability under frame subset/registration perturbation.
- Compare against the best observed frame.
- Require topological stroke stability across at least two non-equivalent methods.
- Report unresolved characters explicitly.

## Primary references

- Baker, Matthews. *Lucas-Kanade 20 Years On: A Unifying Framework*. CMU Robotics Institute.
- Evangelidis, Psarakis. *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*. IEEE TPAMI.
- Fruchter, Hook. *Drizzle: A Method for the Linear Reconstruction of Undersampled Images*. PASP. DOI 10.1086/338393.
- Law, Mackay, Baldwin. *Lucky imaging: high angular resolution imaging in the visible from the ground*. A&A. DOI 10.1051/0004-6361:20053695.
- Farsiu et al. *Fast and Robust Multiframe Super Resolution*. IEEE TIP, 2004.
- Szeliski. *Image Alignment and Stitching: A Tutorial*. Microsoft Research.
- SWGDE. *Image Processing Guidelines*.
- ENFSI. *Best Practice Manual for Forensic Image and Video Enhancement*.
- ITU-T H.264 recommendation.
