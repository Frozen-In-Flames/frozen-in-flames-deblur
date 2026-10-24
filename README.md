# Frozen In Flames Deblur

Classical image deblurring and enhancement. **No AI, no machine learning,
no trained models** — just signal processing: deconvolution with
point-spread functions that are identified from the image's own spectrum,
plus unsharp masking, CLAHE, and high-quality upscaling. Built on OpenCV,
scikit-image, and numpy.

It removes the three common single-image blurs and handles the fourth
common problem, low resolution:

| Setting (`--type`) | Problem | How it is undone |
|---|---|---|
| `motion` | Camera or subject moved during exposure | Line PSF sweep (length × angle), deconvolution |
| `focus` | Out of focus (defocus) | Disk PSF sweep (radius), deconvolution |
| `gaussian` | General lens softness | Gaussian PSF sweep (sigma), deconvolution |
| `lowres` | Image is just small | No deconvolution: Lanczos upscale to target + gentle sharpen + CLAHE |
| `auto` | You don't know which one you have | Sweeps all families; motion must prove itself with a spectral fingerprint (see below) |

## Browser demo

**Live page: https://frozen-in-flames.github.io/frozen-in-flames-deblur/**

Drop in any blurred photo and get the restored photo back, right in the
browser. There is no server and no upload: the page loads the engine
once and runs this repository's own Python package under Pyodide
(Python on WebAssembly, pinned to version 0.29.5, with numpy, scipy,
OpenCV and scikit-image from the Pyodide package set), so your image
never leaves your device.

Two staging choices exist only because browser math is slow, and both
live in `deblur/browser.py`:

* Blur estimation (the PSF sweep) runs on a working copy capped at
  900 px on its long side. The winning parameters are scaled back to
  native size and pinned, then the normal pipeline restores the full
  resolution image once. At or under the cap the scale factor is 1.0
  and the browser path returns exactly what the CLI returns.
* If the OpenCV build in the browser lacks
  `fastNlMeansDenoisingColored`, the denoise step is skipped and the
  run report says so. (The pinned build has it; the guard has never
  triggered in testing.)

First load is heavy: the runtime and libraries total about 65 MB of
downloads, which can take a minute or more on a slow connection. The
browser caches them, so later visits start much faster. Restoration
of a large photo still takes a while, and photos over 40 megapixels
are refused with a plain message. Details and the measured parity
numbers (browser engine vs native, same fixture) are in
RELEASE-NOTES-v0.2.0.md; the page itself is `docs/index.html`, and
`tools/sync_web_engine.py` keeps the copy of the package under
`docs/deblur/` identical to `deblur/`, file for file.

## Install

```bash
pip install -r requirements.txt
```

## Use

```bash
# Let it figure out the blur:
python -m deblur photo.jpg photo_fixed.png --type auto

# You know it's motion blur; let it find the length and angle:
python -m deblur photo.jpg photo_fixed.png --type motion

# Pin exact parameters yourself (skips the search entirely):
python -m deblur photo.jpg photo_fixed.png --type motion --length 15 --angle 30
python -m deblur photo.jpg photo_fixed.png --type focus --radius 4
python -m deblur photo.jpg photo_fixed.png --type gaussian --sigma 2

# Small image: upscale path only, to a 3840 px long side (the default):
python -m deblur small.jpg big.png --type lowres

# Options: --method wiener|rl-tv|rl (default rl-tv), --upscale 3840|none,
#          --no-denoise
```

Every run prints what it decided, so a run can be audited:

```
input        : sample_blurred.png  400x400
blur type    : motion
psf params   : {'length': 15.0, 'angle': 30.0}
method       : rl-tv
top candidates (family, params, score, null_match):
  motion    {'length': 15.0, 'angle': 30.0}  score=+0.1144  null=+0.270
  motion    {'length': 15.0, 'angle': 27.0}  score=+0.1029  null=+0.258
  ...
score in/out : 0.52504 -> 1.50668
steps        : denoise: fastNlMeans h=3; deconv[rl-tv] family=motion ...; unsharp ...; clahe clip=1.8
output       : sample_out.png  400x400
```

As a library:

```python
from deblur import enhance_image, Settings, PRESETS

result, report = enhance_image("photo.jpg", Settings(blur_type="auto"))
result, report = enhance_image(image_array_bgr_uint8, PRESETS["motion"])
print(report.blur_type, report.psf_params, report.candidates)
```

Settings are data: one `Settings` object (and the `PRESETS` dict) drives
the whole pipeline — blur type, method, denoise, upscale target, CLAHE clip,
unsharp strength, and any pinned PSF parameters.

## How the blur is identified

Deconvolution only works when the PSF matches the real blur, so the sweep
scores every candidate **spectrally** — it never trusts how a restoration
"looks" to a sharpness metric:

* **Null fingerprint.** A blur kernel's transfer function has nulls:
  motion blur carves parallel null stripes into the spectrum, a defocus
  disk carves rings, a gaussian carves none. The candidate's null map is
  cosine-matched against the observed image's null map, minus a chance
  baseline (the same match with the candidate's map displaced), so dense
  kernels can't win by overlapping everything a little.
* **Radial profile residual.** After removing the candidate's radial
  log-spectrum from the observed one, what remains should be a straight
  line (natural spectra are roughly power laws). The weighted RMS of the
  non-line remainder measures how wrong the candidate's strength is.
* **Motion gate.** In `auto`, motion is only chosen if its fingerprint
  match clears a gate (`null_match >= 0.18`, calibrated on the synthetic
  suite: true motion 0.23–0.32, strongest false positives 0.11–0.16).
  Otherwise the decision falls to focus vs gaussian.

Sharpness-of-restoration scoring was tried first and measured to fail:
a wrong-PSF deconvolution manufactures ghost edges that sharpness metrics
reward, and any "how close is the reblur" measure is gamed by timid
kernels that barely change the image. Details and numbers: TEST-REPORT.md.

## Pipeline order, and why

1. **Light denoise** — noise is deconvolution's enemy; a mild
   fastNlMeans pass comes first.
2. **Deconvolve at native resolution, luminance only** — the PSF is only
   known at native scale, and blur is linear, so it is undone on a linear
   luminance channel (not LAB L\*, whose nonlinear remap biases the
   recovered parameters). Color follows by the luminance ratio, which
   preserves hue and avoids color fringing. Reflect padding prevents
   border ringing. Methods: Wiener, Richardson-Lucy, or RL with TV
   denoising between chunks (default; suppresses RL's ringing).
3. **Upscale** (Lanczos) to the target long side, if requested.
4. **One gentle unsharp mask** (amount 1.2; the 2.5 in early drafts halos).
5. **CLAHE last**, clip limit 1.8 — strong CLAHE earlier would amplify
   deconvolution artifacts along with the image.

## Honest expectations

* Mild to moderate, roughly uniform blur (slight defocus, small camera
  shake, soft lens): this works well.
* Classical processing sharpens and clarifies what is in the image. It
  **cannot invent real detail that heavy blur destroyed**, and blur that
  varies across the image (one side sharp, one side smeared) is only
  partially helped — the PSF is estimated as one global kernel.
* Disk defocus and gaussian softness of matched strength are genuinely
  hard to tell apart from the spectrum alone; in `auto` they can be
  confused. On the test suite the confusion is measurable but costs
  nothing in output quality (the two restorations score within ~3% of
  each other). Pin `--type` when you know which one you have.
* The gaussian sweep reads high by +0.4 to +0.8 sigma on the synthetic
  suite (spectral-fit bias); the focus sweep is unbiased within ±1 px
  and motion recovery is exact on the suite.
* If you have several shots of the same scene, aligning and stacking
  them (multi-frame super-resolution) beats any single-image method,
  including this one.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

Synthetic ground truth (edges, text-like bars, gradients — generated in
code, no external images) is blurred with known PSFs; the suite asserts
parameter recovery within stated tolerances, improvement over the blurred
input, and wins over negative controls (wrong-PSF and sharpen-only
restorations). One regression test guards a real upstream defect found
during development: scikit-image 0.26's `unsharp_mask(channel_axis=-1)`
slices the wrong axis and returns a destroyed image; the pipeline loops
channels itself (see `deblur/pipeline.py: unsharp_rgb`). Full numbers in
TEST-REPORT.md.

## License

Apache-2.0, Copyright 2026 Aries Hilton.
