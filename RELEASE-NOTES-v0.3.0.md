# Frozen In Flames Deblur v0.3.0

The estimation release. v0.1.0/v0.2.0 measured three real failures
against real photos, and the owner supplied the design that answers
the third one. This release rebuilds how the blur is identified
(denoised estimation copy, detail-layer scoring, a motion gate that
demands specificity), adds noise as a first-class setting with an
auto noise decision, and ships the owner's edge-mask design as the
new segmented setting: foreground and background sorted by an edge
mask, each measured and deblurred with its own PSF.

## What's new, and why

* **Noise is a first-class setting** (`--type noise`, and auto can
  choose it). Why: a noise-dominated photo was being deconvolved —
  auto picked focus r9 and sharpened the grain — when the right
  treatment is the owner's manual recipe: strong denoise, no
  deconvolution, gentle tail. The setting formalizes it. The
  denoise strength is the configured floor (`--noise-h`, default
  10, the strength that worked on the owner's photo) raised to 1.4×
  the measured noise sigma when the noise is heavier, because
  fastNlMeans only bites when its strength is comparable to the
  noise (measured at sigma 15: h=3 leaves a flat patch untouched,
  h=20 clears it). The tail is calmed on this path (unsharp ≤ 0.7,
  CLAHE ≤ 1.2): the standard tail re-amplifies residual grain about
  4.5× on a flat patch. Auto reaches the noise path when no blur
  family clears its evidence bar and either the measured noise
  (Immerkaer's estimator, flat-region weighted so texture cannot
  inflate it) reaches 5 gray levels, or no blur model explains the
  image better than near-identity at all (negative evidence — the
  signature of phone noise-reduction smear, which measures near
  zero pixel noise: 0.08 gray levels on the owner's photo).

* **Estimation runs on a denoised copy, scored on the detail
  layer.** Why: heavy sensor noise buried the motion fingerprint
  (null match 0.030, vetoed), flat content diluted it (a flat
  synthetic with true motion 15/30 was read as focus r8), and a
  large smooth gradient biased the angle (13/42 against truth
  15/30). Estimation now measures noise on the untouched input,
  denoises a copy at a strength scaled to that measurement, and
  scores candidates against the copy's detail layer (the image
  minus its sigma-2 gaussian blur), so only edge-rich regions vote
  and low-frequency layout fringes cannot. An edge mask multiplied
  into the image was tried first and rejected on measurement: the
  mask's own spectrum corrupts the fingerprint (an exact 15/30 read
  became 9/9). The mask lives where it is sound: segmented mode.

* **The motion gate now demands specificity.** Why: on the new
  scorer, chance correlation grows with kernel length, so spurious
  motion picks pile up at the longest kernel (31 px) with null
  matches up to 0.30 — above any absolute gate — on clean,
  isotropic, and noise-only images alike. A true kernel's match
  peaks at its own length and falls (+0.11…+0.28 specificity
  measured); chance gives ~0. Motion must now clear three guards:
  null-match floor 0.10, specificity ≥ 0.05, and isotropic evidence
  below 0.095 (a strong focus/gaussian explanation is not
  overridden by a coincidental directional match). When nothing
  earns a deconvolution, auto reports **identity** (already sharp)
  instead of deconvolving anyway: measured on the sharp suite
  fixture, even a sigma-0.5 deconvolution plus the tail overshoots
  the identity no-harm band, while the tail alone sits inside it.

* **Segmented mode** (`--type segmented`): the owner's design.
  "There should be some sort of edge mask that you use to sort
  background from foreground to deblur both." Tiles are classified
  spectrally, region-grown over a parameter-similarity graph into
  foreground and background (the region touching the frame border
  least is the foreground); each region's PSF is estimated by a
  full sweep on its purest agreeing tile; the frame is restored
  once per region and composited through the feathered mask. If
  the two estimates agree within tolerance, it falls back to the
  single global restoration and says so. The report (CLI, library
  `Report`, and the browser page) exposes the mode, the mask's
  foreground coverage %, and each region's family, parameters and
  coverage. Two measured dead ends are documented in the test
  report: masked full-frame estimation composites (the fill keeps
  the other region's fingerprint; mixed crops cancel to null match
  −0.01 for the true kernel) and centroid-anchored crops (a ring's
  centroid is the other region).

* **Browser page** gains both settings in its blur-type select and
  shows the noise sigma and segmented block in the run report. The
  engine copies under `docs/deblur/` are re-synced byte for byte;
  noise and segmented skip the 900 px staging and analyze at full
  size, exactly as the CLI does. Parity re-verified: native and
  Pyodide engines agree exactly on the suite fixture (motion
  15/30; sharpening ratios 2.870 vs 2.866).

## Measured results

Full numbers in TEST-REPORT.md. Headlines:

* Test suite: 25/25 (14 carried, 11 new). Web: static checks
  13/13, glue test pass, parity pass.
* The flat synthetic that read as focus r8 now reads **motion
  15/30 exact**; the 800 px gradient fixture's angle error went
  from 12 degrees to **3** (15/33); motion under sigma-12 noise
  reads 15/24 (within the ±4 / ±10 tolerances).
* Noise oracle (sharp + sigma-15 noise, no blur): auto selects
  the noise path; flat-patch noise std drops to **0.298×** the
  input; textured-patch steepness stays at 1.374× the clean
  original (band [0.65, 1.50]).
* Segmented oracle (foreground motion 21/0, background 21/90):
  both regions recovered **exactly**; per-region output beats the
  wrong-kernel global control by **×3.26** (foreground) and
  **×3.17** (background) inside each region, and keeps ×0.998 /
  ×0.834 of the right-kernel control; seam band max gradient
  ×1.042 of the better control (bar 1.5×). On a uniformly blurred
  image, segmented falls back and matches the plain motion mode's
  output score exactly (×1.000).
* The owner's noise photo now routes to the noise path in auto
  (was: focus r9 deconvolution). The owner's motion portrait still
  does not auto-detect its motion (fingerprint null match 0.028
  against the 0.10 floor — buried even after denoised detail-layer
  estimation); auto returns a mild gaussian σ1.0, the style of
  result the owner judged cleaner on that photo, and `--type
  motion` continues to find 27 px / 45 degrees when told.

## Honest limits

* Segmented resolves **two** regions at tile resolution (about
  160 px): on the oracle, true foreground coverage 36.2% is
  reported as 43.8%, and background purity near the boundary costs
  up to ~17% steepness against the matching global restoration.
  Smaller than 192 px on the short side, or a single evidence
  region, falls back to global restoration.
* The noise measurement reads pixel-scale grain. Damage already
  smoothed by a phone into low-frequency mottling measures near
  zero and is routed by the negative-evidence branch instead; an
  image that is both mottled and truly blurred would be
  under-deblurred in auto. Pin `--type` when you know.
* Sharp, very flat synthetic-style content can still draw a small
  spurious motion pick in auto (measured 5 px / 138 degrees; its
  output stays inside the no-harm band). Textured sharp photos
  read as identity.
* Family-mode reads shifted slightly with the detail layer:
  gaussian mode σ2.2 for truth 2.0 (bias reduced from +0.4…+0.8
  to +0.2); focus mode r3.0 for truth 4 (within tolerance).

Apache-2.0, Copyright 2026 Aries Hilton.
