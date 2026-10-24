# Deblur for camera glasses: the architecture

v0.4.0 extends the toolkit from stills to open-source AR glasses
with cameras. This note states the architecture the code implements,
so the claims in the README and on the pages can be checked against
it.

## The split: glasses are the camera, the companion does the work

Open-source camera glasses are built thin: a camera, a small
processor, a radio. The heavy compute lives on the paired device
(phone or computer) the glasses already talk to. This toolkit
follows that shape exactly:

* **Capture on the glasses.** The camera (the glasses', or a phone
  camera standing in for any camera a browser can open) produces
  stills, clips, and the live feed.
* **Processing on the companion.** Everything in this repository
  runs on the companion device, on-device from the user's point of
  view: no server, no upload, in the browser or from the command
  line.

## Two modes, two speeds, honestly labeled

The full engine (the Python package) identifies an unknown blur
from the image's own spectrum and inverts it. That costs seconds
per frame. A live view needs an answer per frame. Pretending one
chain is the other would be a lie in both directions, so there are
two, and each is labeled as what it is:

1. **Recorded content (stills and video).** Photos and clips the
   wearer recorded on the glasses are deblurred afterward with the
   full engine. For video (`deblur/video.py`,
   `python -m deblur video`), the PSF is estimated on sampled
   frames, temporally smoothed into a track (median filter within
   family runs, interpolation between samples, family changes step
   at the midpoint), and every frame is restored with its track
   parameters using the fast method by default. The run reports its
   measured seconds per frame. On the calibration clip (448x320,
   this project's test VM) that was 0.59 s/frame overall: recorded
   content is an afterward job, priced honestly.

2. **Live view (accessibility).** `docs/live.html` runs a fast
   **clarify** chain on the camera feed for wearers who need the
   environment clearer as it happens (words, faces, bodies): motion-
   gated temporal noise reduction, CLAHE contrast, one measured
   directional de-smear step, and sharpening. It is clarification,
   not blind deconvolution, and the page says so. Two properties
   are treated as requirements, not features: the enhancement
   toggles off instantly (one big button, or the E key), because an
   accessibility tool the wearer cannot A/B against reality is a
   bad tool; and the page shows its real frame rate and processing
   resolution (capped at 640 px wide), because performance claims
   for a live tool are measurements or they are nothing. A capture
   button freezes the framed view at full camera resolution and
   hands it to mode 1 (the photo page and its full engine), so the
   two speeds meet in one tool.

## The zoom model

Camera glasses zoom by cropping the sensor image and enlarging the
crop (digital zoom). A blur that spans B px on the sensor therefore
spans z*B px in the zoomed view at zoom z, and any measurement made
on the view file is in view pixels. `Settings.zoom` (CLI `--zoom`)
encodes exactly that:

* estimates are measured in **view space** (the pixels handed in);
* reports and region parameters are stated in **sensor space**
  (view / zoom), so a reading means the same physical blur at any
  zoom level;
* manual pins are stated in **sensor space** and restored with the
  view-space kernel pin x zoom;
* restoration always deconvolves the handed-in image with the
  view-space kernel, so on a swept run zoom relabels the report and
  changes no pixels (the suite asserts the outputs are identical),
  while on a pinned run it is what makes the pin correct;
* angles never scale. Noise is measured on the view as given (zoom
  magnifies grain along with everything else; the number is a view
  measurement and the notes say so).

The live page applies the same model in miniature: its sharpening
strength auto-scales with the zoom slider
(`clarityForZoom`: x(1 + 0.4(z-1)), capped at 2.2x), because zoom
magnifies blur and noise together.

One measured boundary of the model: the stills estimator's proven
envelope is kernels up to about 21 px in the space it measures.
A 2x zoom of a 15 px sensor blur is a 30 px view kernel, outside
that envelope: the conversion still applies exactly (the suite
asserts it), but the underlying auto read can under-read or decline
at that size, at any zoom. The numbers are in TEST-REPORT.md.

## The named next step: measure the blur at its source

Everything above estimates blur from pixels. Glasses carry
gyroscopes, and the dominant live-view blur is the wearer's own
head motion during each exposure. That blur does not need to be
estimated from the image at all: angular velocity x exposure time
gives the smear's direction and length directly, per frame, from
the IMU. The natural evolution of both modes is IMU-aided PSF:
the live de-smear keyed to the gyro instead of (or checked against)
the phase-correlation readout, and the recorded-video track seeded
from the clip's motion metadata where the glasses provide it. The
pixel estimator remains the fallback for content with no motion
data, and the arbiter when the two disagree.
