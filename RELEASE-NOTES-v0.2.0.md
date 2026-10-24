# Frozen In Flames Deblur v0.2.0

The browser release. The toolkit gets a public page where anyone can
drop in a blurred photo and get the restored photo back, running the
same Python engine that ships in this repository. The command line
tool and the pipeline are unchanged.

## What's in it

* **Browser upload page** (`docs/index.html`, served by GitHub Pages
  from the `docs/` folder):
  https://frozen-in-flames.github.io/frozen-in-flames-deblur/
  Drag and drop or file picker for any image, the same controls as
  the CLI (blur type auto / motion / focus / gaussian / lowres,
  restoration method, upscale target, denoise toggle), before and
  after views side by side, the engine's own run report, and a PNG
  download button. Staged progress text ("loading engine",
  "estimating blur", "restoring", "done") runs throughout, and the
  engine works in a Web Worker so the page itself never freezes.
* **The engine rule, kept.** The page is not a reimplementation. It
  loads Pyodide 0.29.5 (Python 3.13 on WebAssembly) from the jsDelivr
  CDN, loads numpy, scipy, OpenCV and scikit-image from the Pyodide
  package set, then loads this repository's own `deblur` package,
  copied file for file into `docs/deblur/` by
  `tools/sync_web_engine.py` (which also has a `--check` mode that
  fails on drift), and runs the actual pipeline.
* **Privacy by construction.** There is no server. Your image never
  leaves your device: the page states it plainly, and the static
  checks in `web-test/` verify it (no XMLHttpRequest, no beacons, no
  WebSocket, no POST, no forms, no trackers; the only fetches are the
  engine's own `.py` files from the same origin, and the only external
  host is the Pyodide CDN).
* **Browser staging** (`deblur/browser.py`, also usable from the
  CLI's Python): parameter estimation runs the pipeline's own sweep
  on a working copy capped at 900 px on its long side, the winning
  PSF parameters are scaled back to native size and pinned, and the
  pipeline then restores the full resolution image once.

## Why

v0.1.0 proved the engine on synthetic ground truth, but using it
meant cloning a repo and running a command line. The owner's
standards-first doctrine says a tool should meet people where they
already are, and for photos that is a web page. Running the same
Python under WebAssembly, instead of porting the math to JavaScript,
keeps one engine, one set of measured claims, and one place where a
fix lands for every user.

## Measured parity (browser engine vs native)

Verified with `web-test/parity.mjs`: the pinned Pyodide stack (numpy
2.2.5, scipy 1.14.1, OpenCV 4.11.0.86, scikit-image 0.25.2) against
the native environment (numpy 2.5.4, OpenCV 5.0.0, scikit-image
0.26.0), on the pytest suite's synthetic motion fixture (truth:
length 15, angle 30), for both the full pipeline and the staged
browser path. All 23 parity assertions pass.

| Quantity | Native | Pyodide (browser) |
|---|---|---|
| Recovered family | motion | motion |
| Recovered length | 15 | 15 |
| Recovered angle | 30 | 30 |
| Input steepness | 0.52504 | 0.52514 |
| Output steepness | 1.50668 | 1.50502 |
| Improvement ratio | 2.870x | 2.866x |

The parameters are identical. The steepness scores differ in the
fourth decimal (about 0.1 percent), which is the OpenCV version
difference (4.11 in the browser build vs 5.0 natively) showing up in
low-level rounding, not a math change. The page's own glue code was
also executed end to end under Pyodide (`web-test/glue_test.mjs`,
9 checks): file bytes in through the same decode the browser uses,
estimate returned motion 15 / 30 with null match 0.270, and an
800 px upscale restore wrote back a real 800x800 PNG. The pytest
suite is unchanged and still passes 14 of 14.

## WASM adaptations (complete list)

* **Staging, described above.** Estimation on a capped copy, restore
  pinned at full size. This is invocation, not math: sweep, scoring
  and deconvolution are the pipeline's own functions.
* **Denoise feature detect (dormant).** `deblur/browser.py` checks
  for `cv2.fastNlMeansDenoisingColored`; if a build lacked it, the
  browser path would skip denoise and record that in the run report.
  In the pinned build the function exists (verified by probe and by
  the parity run) and the guard does not trigger.
* **PNG encode fallback (dormant).** The page's glue tries
  `cv2.imencode` first and falls back to Pillow if a build ever
  lacked it. The pinned build encodes fine (verified in the glue
  test); the fallback has never run.
* **Decode from bytes.** The page hands the engine the file's bytes
  and decodes with `cv2.imdecode`, since a browser has no file path
  for `cv2.imread`. Same OpenCV decoder.

## Honest limits

* **First load is about 65 MB** (measured: 12.3 MB runtime core plus
  52.4 MB of library wheels, uncompressed). In the test environment
  the library download and load took about a minute; a slow
  connection will take longer. The browser caches everything, so
  repeat visits skip most of it.
* **Speed.** Browser WebAssembly is several times slower than native
  Python. A small photo restores in seconds; a large photo at a
  3840 px upscale target can take minutes. The stage line keeps
  moving, and the tab stays usable, but patience is part of the deal.
* **Estimation on very large photos** reads the blur from a copy at
  most 900 px wide, so extremely large kernels are estimated more
  coarsely than the CLI would at native size. Pin the blur type when
  you know it, or use the CLI for critical work.
* **Memory cap.** Photos over 40 megapixels are refused with a plain
  message; the CLI has no such cap.
* Everything from the v0.1.0 notes still applies: classical methods
  cannot invent detail that heavy blur destroyed, and disk defocus
  vs matched gaussian softness can still be confused in auto mode.

## License

Apache-2.0, Copyright 2026 Aries Hilton.

## Live verification (October 24, 2026, main-side, real browser on the published page)

- First live run caught a real defect: GitHub Pages dropped `deblur/__init__.py` (Jekyll ignores underscore files), so the in-browser engine could not load. Fixed with `docs/.nojekyll` (commit 9fb4a1e) and retested.
- Parity, third confirmation: on an 800 px textured fixture blurred at motion 15 px / 30 degrees, the live page reported motion, length 13, angle 42; the native staged estimator on the identical file reports motion, length 13, angle 42. The browser engine and the desktop engine agree exactly; the angle offset on this fixture is an estimator property (its large smooth gradient biases the read), documented here rather than hidden, and it is part of what the v0.2.1 edge-masked estimation work addresses. On the 400 px suite fixture the same estimator recovers 15 / 30 exactly (pytest).
- On a flat, low-texture fixture the estimator (browser and native identically) reads a 15 px / 30 degree motion blur as focus radius 8. Low-texture content is a documented estimator limit, not a browser defect.
- The completed live runs produced visibly sharper textured detail and an enabled PNG download, as designed.
