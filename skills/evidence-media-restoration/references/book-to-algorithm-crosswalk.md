# Eight-book method crosswalk

This file records how the eight user-supplied books changed the executable design. It is not a substitute for the books and contains no copied chapters.

| Source | Operational lesson | Implemented control |
|---|---|---|
| Rajagopalan & Chellappa, *Motion Deblurring: Algorithms and Systems* (2014) | Blur and motion estimation are coupled; spatially varying motion invalidates a single blind global kernel. | affine/homographic registration choices; Gaussian/motion PSF sweeps; model outputs separated from observations |
| Read & Meyer, *Restoration of Motion Picture Film* (2000) | Diagnose film damage and preserve the character of the source before cosmetic restoration. | defect-specific YAML profiles; source hashes; no silent inpainting or content completion |
| Kokaram, *Motion Picture Restoration* (1998) | Temporal restoration depends on motion compensation and outlier handling; dust/blotches are not ordinary Gaussian noise. | registration gates, robust median/Huber fusion, temporal MAD and support maps |
| Hansen, *Discrete Inverse Problems* (2010) | Ill-posed inversion needs explicit regularization and parameter-choice diagnostics. | parameter sweeps; discrepancy/residual objective; no readability term |
| Bertero, Boccacci & De Mol, *Introduction to Inverse Problems in Imaging*, 2nd ed. (2022) | Forward models, priors and data errors must remain explicit; a fit does not prove uniqueness. | forward re-blur residual, residual autocorrelation/spectrum, model-dependent class |
| Bhandari, Kadambi & Raskar, *Computational Imaging* (2022) | Capture geometry and computation form one system; sampling limits cannot be repaired by display enlargement. | projective ROI rectification; provenance of native observations; support for exact time metadata |
| Hansen, Nagy & O'Leary, *Deblurring Images* (2006) | Boundary conditions, PSF choice, ringing and regularization control deconvolution credibility. | reflected registration borders, clipping penalty, PSF/regularization ledger, top-candidate retention |
| Barbu, *Variational and PDE Methods in Image Processing* (2019) | Variational/PDE smoothing and inpainting encode assumptions; inpainted structure is not observed structure. | deterministic denoise allowed; PDE/content-aware inpainting excluded from the evidentiary core |

## Items deliberately not automated

- Literal OCR selection or tuning against an expected word.
- Blind spatially varying PSF estimation without adequate calibration features.
- PDE or neural inpainting of missing strokes.
- Treating adjacent H.264 frames, alternate encodes, or model outputs as independent witnesses.
- Promoting the visually sharpest candidate without residual and stability checks.

