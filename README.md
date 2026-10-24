# Frozen In Flames Deblur

Classical image deblurring and enhancement. **No AI, no machine learning,
no trained models** — just signal processing: deconvolution with
point-spread functions that are identified from the image's own spectrum,
plus unsharp masking, CLAHE, and high-quality upscaling. Built on OpenCV,
scikit-image, and numpy.

It removes the three common single-image blurs, handles low
resolution, treats noise-dominated photos without pretending they are
blurred, and can restore foreground and background separately when
the blur changes across one photo. Parameters are zoom-aware (a
photo taken at a digital zoom reports its blur in sensor terms), a
recorded-video mode deblurs whole clips frame by frame, and a live
clarify view serves camera glasses and phone cameras as an
accessibility aid (see "Camera glasses" below):

| Setting (`--type`) | Problem | How it is undone |
|---|---|---|
| `motion` | Camera or subject moved during exposure | Line PSF sweep (length × angle), deconvolution |
| `focus` | Out of focus (defocus) | Disk PSF sweep (radius), deconvolution |
| `gaussian` | General lens softness | Gaussian PSF sweep (sigma), deconvolution |
| `lowres` | Image is just small | No deconvolution: Lanczos upscale to target + gentle sharpen + CLAHE |
| `noise` | Heavy grain, no real blur | Strong denoise (strength scales with the measured noise), **no deconvolution**, calmed sharpen/CLAHE tail |
| `segmented` | Foreground and background blurred differently | Edge-mask segmentation into two regions, a PSF estimated per region, the frame restored once per region and composited through the feathered mask (see below) |
| `auto` | You don't know which one you have | Sweeps all families; motion must prove itself with a spectral fingerprint (see below); picks the noise path when no blur family earns a deconvolution, and reports `identity` when the photo already reads as sharp |

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
  and the browser path returns exactly what the CLI returns. The
  `noise` and `segmented` settings skip staging and analyze at full
  size, exactly as the CLI does (segmented is slower in a browser;
  the page says so).
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
RELEASE-NOTES-v0.3.0.md; the page itself is `docs/index.html`, and
`tools/sync_web_engine.py` keeps the copy of the package under
`docs/deblur/` identical to `deblur/`, file for file.

The companion **live clarify view** (`docs/live.html`, linked from
the photo page) runs on a camera feed instead of a file: temporal
noise reduction, CLAHE contrast, a de-smear step keyed to the
camera motion it measures between frames by phase correlation, and
sharpening that auto-scales with its zoom slider, all in plain
JavaScript at a capped processing resolution, with a large instant
on/off toggle and a button that captures a still and hands it to
the full engine on the photo page. It is a clarify chain, not the
engine, and it says so on the page. Its math lives in
`docs/live-math.js` and is tested in Node like the rest.

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

# Grainy photo, no real blur: strong denoise, no deconvolution:
python -m deblur noisy.jpg clean.png --type noise

# Foreground and background blurred differently: restore each region
# with its own measured PSF:
python -m deblur portrait.jpg fixed.png --type segmented

# Photo taken at 2x digital zoom: the report (and any pin) speaks
# sensor space, so the numbers mean the same blur as an unzoomed shot:
python -m deblur zoomed.jpg fixed.png --type auto --zoom 2
python -m deblur zoomed.jpg fixed.png --type motion --length 15 --angle 30 --zoom 2

# A recorded clip (e.g. from camera glasses), deblurred frame by frame:
python -m deblur video clip.mp4 clip_fixed.mp4
python -m deblur video clip.mp4 clip_fixed.mp4 --sample-every 6 --method wiener

# Options: --method wiener|rl-tv|rl (default rl-tv), --upscale 3840|none,
#          --no-denoise, --noise-h H (noise path denoise floor, default 10),
#          --zoom Z (digital zoom the image was taken at, default 1).
#          Video adds: --sample-every N (default 12),
#          --estimate-max-side PX (default 480); its --method default
#          is wiener, the fast method, and there is no upscale.
```

Every run prints what it decided, so a run can be audited:

```
input        : sample_blurred.png  400x400
blur type    : motion
psf params   : {'length': 15.0, 'angle': 30.0}
noise sigma  : 0.0 gray levels (Immerkaer estimate on the original)
method       : rl-tv
top candidates (family, params, score, null_match):
  motion    {'length': 15.0, 'angle': 30.0}  score=+0.0289  null=+0.287
  motion    {'length': 11.0, 'angle': 21.0}  score=-0.0011  null=+0.186
  ...
score in/out : 0.52504 -> 1.50668
steps        : denoise: fastNlMeans h=3; deconv[rl-tv] family=motion ...; unsharp ...; clahe clip=1.8
output       : sample_out.png  400x400
```

Segmented runs additionally print the mode (`per-region` or
`fallback`), the foreground coverage, and each region's family,
parameters and coverage.

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
"looks" to a sharpness metric. Two preparation steps come first, because
both were measured failure modes:

* **Estimation reads a denoised copy.** Noise is measured on the
  untouched input (Immerkaer's estimator, averaged over the flattest
  half of the image so texture cannot inflate it), then the sweep runs
  on a copy denoised at a strength scaled to that measurement.
  Restoration still works on the original pixels. Heavy sensor noise
  otherwise buries the fingerprint entirely.
* **Scoring reads the detail layer.** The scorer sees the estimation
  image minus its sigma-2 gaussian blur. Flat regions and smooth
  gradients have no detail layer, so only edge-rich regions vote, and
  the low frequencies where content-layout fringes live are removed
  while the band carrying the blur fingerprint is kept. (A spatially
  multiplied edge mask was tried for this and measured to corrupt the
  read — the mask's own spectrum convolves into the fingerprint band.
  The edge mask itself is used where it is sound: segmented mode.)

Then every candidate is scored on:

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
* **The auto decision.** Motion is chosen only if its winner passes
  three guards: its null match clears a floor (0.10); its match is
  *specific* — it beats the longest sweepable kernel (31 px) at the
  same angle by at least 0.05, because chance correlation grows with
  kernel length and piles up at the longest length with no interior
  peak, while a true kernel's match peaks at its own length and falls;
  and the isotropic evidence stays below its cap, so a strong
  focus/gaussian explanation is not overridden by a coincidental
  directional match. The isotropic winner is accepted when it reduces
  the radial residual by at least 0.05 versus a near-identity kernel.
  When nothing clears its bar, the image is treated as **noise** (if
  the measured noise is heavy, or no blur model explains the image at
  all) or reported as **identity** (already sharp): no deconvolution
  is invented for an image that has not earned one.

Sharpness-of-restoration scoring was tried first and measured to fail:
a wrong-PSF deconvolution manufactures ghost edges that sharpness metrics
reward, and any "how close is the reblur" measure is gamed by timid
kernels that barely change the image. Details and numbers: TEST-REPORT.md.

## Segmented mode: foreground and background, deblurred both

The design, in the owner's words: "there should be some sort of edge
mask that you use to sort background from foreground to deblur both."
The algorithm is classical throughout:

1. **Tile and classify.** The estimation luminance is tiled (about
   160 px tiles, fewer on small images). Each tile is classified by
   spectral scoring over the coarse candidate grids, on its detail
   layer. Tiles whose mean edge density is too low to carry evidence
   are marked weak and do not vote.
2. **Region growing.** Confident tiles join 4-neighbours whose
   parameters are similar (same family; motion within 15 degrees and
   6 px; focus within 2 px; gaussian within 1.2 sigma). Connected
   components are the candidate regions; the two largest win. Every
   remaining tile joins the region its own classification matches
   (when it matches exactly one), else the nearest component.
3. **Foreground vs background.** The region whose tiles touch the
   image border least is the foreground (subjects are interior,
   backgrounds reach the frame edge); ties go to the smaller coverage.
4. **Per-region estimation.** Each region's PSF is estimated by a
   full sweep on its single purest, most evidence-bearing tile whose
   classification agrees with the region's representative class. (A
   masked full-frame composite was tried and measured to fail: local
   mean fill keeps the other region's mid-frequency fingerprint, and
   any usefully wide crop mixes both regions, whose fingerprints
   cancel.)
5. **Restore and composite.** If the two estimates agree within the
   suite tolerances (motion ±4 px / ±10 degrees, focus ±1.5 px,
   gaussian ±1.0 sigma), the blur is effectively uniform: segmented
   falls back to the single global restoration and the report says
   so. Otherwise the full frame is restored once per region and the
   two restorations are composited through the feathered mask (the
   tile label map upsampled, gaussian feather of a sixth of a tile),
   so the seam is a blend, not a cut.

The run report (CLI and the `Report` object) exposes the mask's
foreground coverage %, each region's family and parameters, and
whether per-region restoration or the fallback was used.

## Camera glasses: zoom, recorded video, and the live view

The toolkit's glasses architecture (full note: `docs/glasses.md`)
is the split the open camera-glasses class is built on: the glasses
are the camera, the paired device does the processing, and there
are two modes at two honestly labeled speeds.

**Zoom.** Digital zoom crops the sensor image and enlarges the
crop, so a sensor blur of B px spans z*B px in the view at zoom z.
With `--zoom z` (or `Settings.zoom`), estimates are still measured
on the image as given (view space), but the report converts them to
**sensor space** (view / z), and manual pins are read as sensor
space and restored with the view-space kernel pin × z. Angles never
scale. On a swept run the zoom changes no pixels, only what the
numbers mean; on a pinned run it is what makes the pin the same
physical blur at any zoom. Without it, a 2x-zoomed photo's blur
reads about twice the sensor truth, which is exactly the miss the
zoom factor exists to remove.

**Recorded video.** `python -m deblur video in.mp4 out.mp4`
deblurs a whole clip (the wearer's recorded content, processed
afterward on the companion). The PSF is estimated on sampled
frames only (`--sample-every`, default 12) on a downscaled working
copy; the samples are median-smoothed within runs of one family and
interpolated into a per-frame track, so the restoration kernel
cannot jitter frame to frame (family changes step at the midpoint
between samples, and frames the estimator calls noise or identity
get those treatments locally, not a held blur kernel). Each frame
is then restored from the pipeline's own pieces with the fast
method by default (`--method wiener`; rl-tv is available at its
per-frame price). Frames are written at native size; segmented is
a stills setting and is refused for video. The summary prints the
sampled estimates and the measured seconds per frame. Measured on
the calibration clip (448×320, 30 frames, this project's test VM):
0.59 s/frame overall, track correlation 0.99 against the known
blur curve. This is an afterward job, not a live one.

**Live view.** `docs/live.html` (see Browser demo above) is the
accessibility mode: a fast clarify chain on the live camera feed
with an instant toggle, a zoom slider whose clarity strength
auto-scales, a phase-correlation readout of the camera's measured
motion, and a capture button that hands a frozen frame to the full
engine. What it is not is the engine running live; no honest chain
at this classical cost is.

**The named next step** is IMU-aided PSF: glasses carry gyros, and
motion blur during an exposure is angular velocity × exposure time.
The smear can be computed at its source instead of estimated from
pixels, with the pixel estimator as fallback and arbiter. See
`docs/glasses.md`.

## Pipeline order, and why

0. **Measure the noise** on the untouched input (Immerkaer). The
   number drives the estimation denoise strength, the restoration
   denoise strength, and the auto noise decision.
1. **Denoise** — noise is deconvolution's enemy. The restoration base
   gets a fastNlMeans pass whose strength scales with the measured
   noise (h=3 on clean images, the long-standing light pass, ramping
   to h=10 at heavy noise); the separate estimation copy is denoised
   by the same rule. The noise path instead denoises at the
   configured floor (h=10) raised to 1.4× the measured sigma when
   the noise is heavier — fastNlMeans only bites when its strength
   is comparable to the noise it hunts.
2. **Deconvolve at native resolution, luminance only** — the PSF is only
   known at native scale, and blur is linear, so it is undone on a linear
   luminance channel (not LAB L\*, whose nonlinear remap biases the
   recovered parameters). Color follows by the luminance ratio, which
   preserves hue and avoids color fringing. Reflect padding prevents
   border ringing. Methods: Wiener, Richardson-Lucy, or RL with TV
   denoising between chunks (default; suppresses RL's ringing).
   The noise and identity outcomes skip this step entirely.
3. **Upscale** (Lanczos) to the target long side, if requested.
4. **One gentle unsharp mask** (amount 1.2; the 2.5 in early drafts halos).
5. **CLAHE last**, clip limit 1.8 — strong CLAHE earlier would amplify
   deconvolution artifacts along with the image. On the noise path the
   tail is calmed (unsharp ≤ 0.7, CLAHE ≤ 1.2): measured on a flat
   patch, the standard tail re-amplifies residual grain about 4.5×.

## Honest expectations

* Mild to moderate, roughly uniform blur (slight defocus, small camera
  shake, soft lens): this works well.
* Classical processing sharpens and clarifies what is in the image. It
  **cannot invent real detail that heavy blur destroyed**.
* Blur that varies across the image is what `segmented` is for, within
  its limits: two regions maximum, boundaries resolved at tile
  resolution (about 160 px), and the purity of each region's
  restoration near the boundary is limited by the mask upsampling
  (the segmented oracle keeps at least 0.80× of the matching global
  restoration's steepness inside each region, and beats the wrong
  region's kernel by 3× there). Smoothly varying blur (a tilt-shift
  gradient, depth falloff) is approximated, not solved.
* The noise decision keys on pixel-scale grain (the Immerkaer
  measurement) or on no blur model explaining the image. A photo whose
  grain was already smoothed by a phone's own noise reduction into
  low-frequency mottling measures near zero pixel noise; the
  no-blur-model branch is what routes it correctly (see the test
  report's portrait2 after-check).
* Motion buried in heavy noise can still fall under the fingerprint
  guards: on the owner's motion portrait the fingerprint measures
  essentially zero even after denoised, detail-layer estimation, and
  `auto` returns a mild gaussian restoration (which the owner judged
  the cleaner result on that photo). Pin `--type motion` when you
  know; the sweep then finds the kernel (27 px / 45 degrees on that
  photo).
* Sharp but very flat synthetic-style content (large flat regions,
  few hard edges, no texture) can draw a small spurious motion pick
  in `auto`; the restoration it produces stays inside the identity
  no-harm band. Textured sharp photos read as `identity`.
* Disk defocus and gaussian softness of matched strength are genuinely
  hard to tell apart from the spectrum alone; in `auto` they can be
  confused. On the test suite the confusion is measurable but costs
  nothing in output quality (the two restorations score within ~3% of
  each other). Pin `--type` when you know which one you have.
* On the synthetic suite through the current estimator: the gaussian
  sweep reads sigma 2.2 for a true 2.0 in gaussian mode; the focus
  sweep reads radius 3.0 for a true 4 (within its ±1.5 tolerance);
  motion recovery is exact, including under sigma-12 noise and on
  the 800 px gradient fixture that read 12 degrees off in v0.2.0.
* The estimator's proven envelope is motion kernels up to about
  21 px in the space it measures. At 25-30 px native it under-reads
  or declines on the suite fixtures (a 30 px kernel has destroyed
  most of the fingerprint band). This bounds zoom too: at 2x zoom a
  15 px sensor blur is a 30 px view kernel, so the zoom conversion
  applies exactly but the underlying auto read can be weak; pin the
  parameters when you know them. Measured values: TEST-REPORT.md.
* Video mode inherits every stills limit, plus its own: the track
  assumes the blur changes slowly between sampled frames (a whip
  pan or a cut inside a clip will be tracked at sample resolution,
  and a cut's frames step at the midpoint between samples), and the
  per-frame cost is the stills cost divided by nothing. The live
  clarify chain sharpens and contrasts the captured feed; it does
  not recover detail the sensor never recorded.
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
restorations). The v0.3.0 suite (`tests/test_v030.py`) adds the noise
oracle, the three estimation failure shapes rebuilt synthetically, and
the segmented composite oracle with its wrong-kernel controls and seam
discipline. The v0.4.0 suite (`tests/test_v040.py`) adds the zoom
oracles (sensor-space reporting inside the estimator's proven envelope,
exact pin conversion, the no-zoom control missing by ~2×) and the
recorded-video oracle (a synthetic clip with a known time-varying blur:
output sharpness, track recovery against the truth curve, and temporal
discipline against an unsmoothed per-frame control). The live page's
math is tested in Node (`web-test/live_math.test.mjs`: phase-correlation
shift recovery, the de-smear step measured by physical edge width,
the zoom/temporal policy rules) and both pages carry static checks
(`web-test/static_checks.mjs`, `web-test/live_static_checks.mjs`).
One regression test guards a real upstream defect found
during development: scikit-image 0.26's `unsharp_mask(channel_axis=-1)`
slices the wrong axis and returns a destroyed image; the pipeline loops
channels itself (see `deblur/pipeline.py: unsharp_rgb`). Full numbers in
TEST-REPORT.md.

## License

Apache-2.0, Copyright 2026 Aries Hilton.
