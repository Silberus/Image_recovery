# Library and repository matrix

Checked 2026-08-29. Repository links are research references, not code vendored by this plugin. Verify the current license of any optional repository and each downloaded weight before use.

## Installed by the plugin runtime

| Layer | Project | Role | License boundary | Evidence use |
|---|---|---|---|---|
| Core | [NumPy](https://github.com/numpy/numpy) | arrays, FFT data handling | BSD-3-Clause | deterministic |
| Core | [SciPy](https://github.com/scipy/scipy) | convolution and numerical routines | BSD-3-Clause | deterministic |
| Core | [OpenCV](https://github.com/opencv/opencv) | decode fallback, perspective geometry, phase correlation, ECC, image filters | Apache-2.0 from OpenCV 4.5 | deterministic; log interpolation and transform |
| Core | [PyYAML](https://github.com/yaml/pyyaml) | profile loading | MIT | configuration only |
| Core | [Pillow](https://github.com/python-pillow/Pillow) | image interoperability | HPND | deterministic I/O |
| Extended | [PyAV](https://github.com/PyAV-Org/PyAV) | container/stream/packet/frame access, PTS, time base, key-frame and picture type | BSD 3-clause text; linked FFmpeg has its own build-dependent obligations | preferred exact decode path |
| Extended | [scikit-image](https://github.com/scikit-image/scikit-image) | additional denoise, Wiener, Richardson-Lucy and quality tools | project files primarily BSD-3-Clause; inspect bundled notices | deterministic/model-dependent |
| Extended | [PyWavelets](https://github.com/PyWavelets/pywt) | wavelet denoising and multiscale diagnostics | MIT | deterministic |
| Extended | [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | packaged FFmpeg fallback | BSD-2-Clause wrapper; FFmpeg binary license depends on build | decode/encode helper only |

FFmpeg itself is mainly LGPL-2.1-or-later, but a build can become GPL when GPL components are enabled. Record the exact binary/build configuration rather than assuming a license from the wrapper.

## External optional research adapters

These are not installed, imported, cloned, or redistributed by the plugin. They are listed so an operator can make an explicit, separately reviewed choice.

| Project | Capability | Repository license observed | Allowed result class |
|---|---|---|---|
| [BasicSR](https://github.com/XPixelGroup/BasicSR) | image/video SR, denoise, deblur, JPEG artifact removal; includes BasicVSR family | Apache-2.0 | `MODEL_SUGGESTION` |
| [MMagic](https://github.com/open-mmlab/mmagic) | framework and inference recipes for restoration including BasicVSR | Apache-2.0 | `MODEL_SUGGESTION` |
| [RVRT](https://github.com/JingyunLiang/RVRT) | recurrent video restoration for SR/deblur/denoise | CC BY-NC 4.0 in repository root; non-commercial restriction | `MODEL_SUGGESTION`; never bundle by default |
| [KAIR](https://github.com/cszn/KAIR) | DPIR, USRNet, DnCNN, FFDNet, SwinIR experiments | MIT | `MODEL_SUGGESTION` |
| [Kornia](https://github.com/kornia/kornia) | differentiable geometry, filters and registration | Apache-2.0 | deterministic if no learned model; otherwise `MODEL_SUGGESTION` |
| [PIQ](https://github.com/photosynthesis-team/piq) | PyTorch image-quality metrics | Apache-2.0 | metric only; never proof of text identity |
| [pytorch-msssim](https://github.com/VainF/pytorch-msssim) | differentiable SSIM/MS-SSIM | MIT | metric only |

## Selection rules derived from the eight supplied books

1. Model the acquisition chain before choosing a filter: display/scene, optics, perspective, sensor sampling, demosaic, resampling, codec.
2. Use local or projectively varying PSF when one global kernel is contradicted by the geometry.
3. Treat residuals as colored until residual-spectrum and lag diagnostics show otherwise.
4. Choose regularization/stopping by discrepancy, GCV/L-curve or residual diagnostics, never by which setting makes the expected word look best.
5. A low forward residual is necessary but insufficient. Literal reading also requires separation of competing glyph hypotheses under perturbation.
6. PDE or learned inpainting creates a continuation, not an observed stroke; keep it outside the evidentiary layer.

## Provenance requirement for an optional model

Record repository URL, commit, environment lock, weight URL/hash/license, input hash, full command/config, random seed, hardware, output hash, and result class. Compare against observed and deterministic outputs. Do not allow a neural output to vote as another independent witness.

