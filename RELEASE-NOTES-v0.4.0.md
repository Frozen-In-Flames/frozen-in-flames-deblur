# Frozen In Flames Deblur v0.4.0 — deblur for camera glasses

This release takes the toolkit to open-source AR glasses with
cameras, in the two ways glasses content actually exists: **clips
the wearer recorded**, deblurred afterward, and **the live view**,
clarified as it happens for wearers who need the environment
clearer (words, faces, bodies) to read it at all. The architecture
is the one that class of glasses is built on: the glasses are the
camera, the paired device does the processing, and nothing leaves
the user's device. It is written up in `docs/glasses.md`.

## What's new, and why

**Zoom-aware parameters (`--zoom`, `Settings.zoom`).** Camera
glasses zoom by cropping and enlarging, so a photo taken at 2x
zoom carries its blur at twice the pixel size, and every reading
taken from it is inflated by the zoom. Now the report speaks
**sensor space**: estimates are converted (view / zoom), manual
pins are accepted in sensor space and restored at pin x zoom, and
the kernel actually applied is recorded alongside. A blur means
the same physical thing at any zoom level, and a pin that was
right unzoomed stays right zoomed. On a swept run the zoom changes
no pixels (asserted in the suite); on a pinned run it is the
difference between restoring with the true kernel and with half
of it (1.62x output steepness on the oracle fixture).

**Recorded-video mode (`python -m deblur video`).** Recorded
glasses content is video, so the engine now eats video: the PSF
is estimated on sampled frames (every 12th by default) on a
downscaled working copy, the samples are median-smoothed and
interpolated into a per-frame track so the restoration kernel
cannot jitter frame to frame, and each frame is restored from the
pipeline's own pieces with the fast Wiener method by default.
Family changes step at the midpoint between samples (never a
blended hybrid kernel), and frames the estimator calls noise or
identity get those treatments locally. The summary prints the
sampled estimates and the measured cost.

**Live clarify view (`docs/live.html`).** The accessibility mode:
a fast chain on the camera feed, in plain JavaScript, at a capped
640 px processing width: motion-gated temporal noise reduction,
CLAHE contrast, one directional de-smear step keyed to the camera
motion the page measures between frames by phase correlation
(readout on the page, in px/frame with direction), and sharpening
that auto-scales as you zoom. A large toggle (or the E key) drops
to the raw view instantly, because an accessibility tool you
cannot A/B is a bad tool; a capture button freezes the framed
view at full resolution and hands it to the full engine on the
photo page, through on-device browser storage only. The photo
page links here, and its top card explains the split.

## Measured

* Zoom oracle (sensor blur 10 px, 2x zoomed view): reports 11.5 px
  sensor / 27 degrees against truth 10 / 30; the same view without
  the zoom factor reports 23 px, the ~2x miss the factor removes.
  Pin conversion is exact (sensor 15 at zoom 2 restores with the
  30 px view kernel, output identical to pinning 30 unzoomed).
* Video oracle (synthetic clip, blur easing 10 → 20 px at 30
  degrees, noise, mp4): sampled estimates track the truth curve
  (correlation 0.988, mean error 0.72 px); output steepness
  3.10x the input's; the smoothed track's largest frame-to-frame
  kernel change is 0.67 px against 2.00 px for unsmoothed
  per-frame estimation. Cost on the test VM: **0.59 s/frame** at
  448x320, reported by the tool itself on every run.
* Live math (Node tests): phase correlation recovers synthetic
  frame shifts to 0.03 px; one de-smear step narrows a 9 px
  smeared edge from 8 px to 6 px transition width at the
  operating strength, with 0.05 ringing.
* Suites: pytest **41/41** (25 carried, 16 new); index static
  checks 13/13; live static checks 25/25; live-math tests 18/18;
  browser glue and parity re-verified after the staging change.

## Limits, stated

* The stills estimator's proven envelope is kernels up to about
  21 px in the space it measures; at 25–30 px it under-reads or
  declines (measured, in TEST-REPORT.md). Strong blur at high
  zoom lands there: the zoom conversion stays exact, the auto
  read may not. Pins are the answer when the blur is known.
* The live chain is clarification, not blind deconvolution: it
  buys back about a quarter of a smear's edge width and cleans
  the feed; it does not recover detail the sensor never recorded.
  Anything that matters goes through the capture-still handoff to
  the full engine.
* Video mode assumes blur changes slowly between sampled frames
  and costs two full decodes of the clip. It is an afterward job
  and says so.
* Named next step: IMU-aided PSF. Glasses carry gyros; motion
  blur during an exposure is angular velocity x exposure time,
  computable at its source instead of estimated from pixels.
