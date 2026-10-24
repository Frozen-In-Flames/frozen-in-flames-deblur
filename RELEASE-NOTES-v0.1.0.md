# Frozen In Flames Deblur v0.1.0

First release. A classical (no AI, no ML) deblurring and enhancement
toolkit with a setting per blur type, built from the owner's pasted
pipeline and improvement notes.

## What's in it

* **Blur-type settings system** — `auto`, `motion`, `focus`, `gaussian`,
  `lowres`, as data (`Settings` + `PRESETS`), with manual pinning of
  length/angle/radius/sigma that skips the search.
* **Spectral PSF identification** — candidates are swept (coarse +
  refinement) and scored by a null-fingerprint match with a chance
  baseline plus a radial profile residual; auto mode gates motion on
  fingerprint strength. Every CLI run prints the chosen type, PSF
  parameters, top candidates, and scores for audit.
* **Pipeline in the measured order** — light denoise → deconvolution at
  native resolution on linear luminance (Wiener / Richardson-Lucy /
  RL-with-TV) with reflect padding → Lanczos upscale to ~4K long side →
  one gentle unsharp mask → CLAHE last at clip 1.8.
* **CLI + library** — `python -m deblur in out --type ...`, and
  `enhance_image(image, settings) -> (image, Report)`.
* **Tests** — 14 pytest oracles with negative controls on synthetic
  ground truth generated in code (no external assets).

## Why

The starting point was a single fixed recipe (one generic PSF, sharpen,
CLAHE, upscale). The upgrade that mattered most was giving each blur
type its own identifiable settings: blur can only be inverted by the
kernel that actually caused it, so the tool first identifies the kernel
from the image's spectrum, then inverts it. The identification method
itself went through three measured designs; the first two (restoration
sharpness, then steepness × reblur fidelity) provably reward wrong
answers — ghost edges and timid kernels — and the report says so with
numbers (TEST-REPORT.md).

## Measured results (summary)

* Motion (15 px, 30°): recovered **exactly** in auto and motion mode;
  output steepness ×2.87 over the blurred input; true PSF beats the
  60°-wrong control by 0.32 of candidate score.
* Focus (disk r4): recovered r5.0 (tolerance ±1.5); output ×2.78.
* Gaussian (σ2.0): recovered σ2.8 (tolerance ±1.0; a documented
  +0.4…0.8 spectral-fit bias); output ×2.76, ×1.95 over sharpen-only.
* Lowres (4× downscale): reaches the 400 px target; ×1.49 steepness
  over plain Lanczos.
* Identity control: a sharp image is not degraded (ratio 1.360 inside
  the [0.85, 1.60] band, dimensions unchanged).

## Honest limits

* No detail is invented: heavy or spatially-varying blur is only
  partially recoverable by any classical method, this one included.
* In auto, disk defocus and matched-strength gaussian softness can be
  confused (scores within noise); the restorations are near-equivalent
  in output quality, and pinning `--type` removes the ambiguity.
* Gaussian/focus sweeps read ~+0.5–1.0 strong on the synthetic suite
  (spectral-fit bias, documented with numbers in TEST-REPORT.md).
* Found during development and routed around: scikit-image 0.26's
  `unsharp_mask(channel_axis=-1)` returns a destroyed image (wrong-axis
  slicing bug); the pipeline applies the 2D function per channel and a
  regression test guards it.

## License

Apache-2.0, Copyright 2026 Aries Hilton.
