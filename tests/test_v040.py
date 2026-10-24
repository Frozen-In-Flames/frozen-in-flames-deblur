"""v0.4.0 oracles: zoom-aware reporting/pinning, and recorded-video
mode (sampled estimation, temporally smoothed PSF track, fast
per-frame restoration).

Thresholds were fixed from the calibration measurements recorded in
TEST-REPORT.md before this suite ran against the final code.

ZOOM. A digital zoom of z crops the sensor image and enlarges the
crop, so a sensor blur of B px spans z*B px in the zoomed view file.
The pipeline measures in view space, reports in sensor space
(view / zoom), and accepts manual pins in sensor space (restored
with pin * zoom).

  * End-to-end, inside the estimator's proven envelope (kernels up
    to ~21 px view space; see the documented limit): textured
    source blurred 10 px / 30 deg on the sensor, center crop
    upscaled 2x (effective view blur 20 px). With zoom=2 the
    reported sensor length is within +/-4 of 10 and the angle
    within +/-10 of 30; the view-space kernel is within +/-6 of
    20. The SAME view at zoom=1 (the no-zoom control) reports the
    view-space length as-is, missing the sensor truth by >= 8 px:
    that ~2x miss is what the zoom factor removes.
  * Sweep discipline: zoom changes no pixels on a swept run. The
    zoom=1 and zoom=2 outputs are identical, the family agrees, and
    the zoom=2 sensor reading is exactly the zoom=1 reading / 2.
    This also covers the brief's 15 px case (effective 30 px view
    kernel), where the conversion is asserted exactly while the
    absolute read is documented as outside the estimator envelope.
  * Pin path, exact: pinning the sensor truth (15 / 30) at zoom=2
    on the 15 px fixture reports {15, 30}, restores with view
    {30, 30}, produces output identical to pinning 30 at zoom=1,
    and beats the under-restored control (the same pin at zoom=1,
    i.e. ignoring the zoom) by >= 1.25x output steepness.

VIDEO. Synthetic clip: a textured scene, slow pan, motion blur
length easing 10 -> 20 px at a fixed 30 degrees (smoothstep), mild
noise, mp4-encoded and decoded by the tool itself.

  (i)   mean edge steepness of output frames >= 1.5x the decoded
        input's (calibration measured 3.1x);
  (ii)  sampled angles within +/-10 deg of truth; the smoothed
        per-frame length track correlates with the truth curve at
        r >= 0.85 (measured 0.99) with mean abs error <= 4 px
        (measured 0.7);
  (iii) temporal discipline: the smoothed track's largest
        frame-to-frame length change is <= 1.5 px (measured 0.67),
        while an unsmoothed per-frame-estimation control on the
        same decoded frames jumps at least 1.5x as much (measured
        ~3x).
"""

import subprocess
import sys

import cv2
import numpy as np
import pytest

from deblur.pipeline import enhance_image
from deblur.presets import Settings
from deblur.psf import apply_blur, motion_psf
from deblur.scoring import score_image
from deblur.video import build_track, enhance_video, estimate_frame_psf

from conftest import to_bgr
from test_v030 import _textured_source

ANGLE_TOL = 10.0
LENGTH_TOL = 4.0


def angle_diff(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _zoomed_view(sensor_length, angle=30.0):
    """The brief's simulation: blur on the sensor, center crop,
    upscale 2x. Returns the view (float gray) whose effective blur
    is 2x the sensor blur."""
    src = _textured_source(800)
    blurred = apply_blur(src, motion_psf(sensor_length, angle))
    crop = blurred[200:600, 200:600]
    return cv2.resize(crop, (800, 800), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Zoom: settings validation
# ---------------------------------------------------------------------------

def test_zoom_must_be_positive():
    with pytest.raises(ValueError):
        Settings(zoom=0)
    with pytest.raises(ValueError):
        Settings(zoom=-2.0)
    assert Settings().zoom == 1.0


# ---------------------------------------------------------------------------
# Zoom: end-to-end estimation inside the proven envelope
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zoom10_runs():
    view = to_bgr(_zoomed_view(10.0))
    out1, rep1 = enhance_image(view, Settings(blur_type="auto",
                                              upscale=None))
    out2, rep2 = enhance_image(view, Settings(blur_type="auto",
                                              upscale=None, zoom=2.0))
    return out1, rep1, out2, rep2


def test_zoom_estimation_reports_sensor_terms(zoom10_runs):
    _, _, _, rep2 = zoom10_runs
    assert rep2.blur_type == "motion", (rep2.blur_type, rep2.psf_params)
    assert abs(rep2.psf_params["length"] - 10.0) <= LENGTH_TOL, rep2.psf_params
    assert angle_diff(rep2.psf_params["angle"], 30.0) <= ANGLE_TOL
    # the kernel actually applied lives in view space: ~20 px
    assert abs(rep2.view_psf_params["length"] - 20.0) <= 6.0, \
        rep2.view_psf_params


def test_zoom_control_without_zoom_misestimates(zoom10_runs):
    _, rep1, _, _ = zoom10_runs
    assert rep1.blur_type == "motion", (rep1.blur_type, rep1.psf_params)
    # read in view space and reported as-is: ~2x the sensor truth
    assert abs(rep1.psf_params["length"] - 10.0) >= 8.0, rep1.psf_params


def test_zoom_sweep_conversion_is_exact_and_pixel_neutral(zoom10_runs):
    out1, rep1, out2, rep2 = zoom10_runs
    assert np.array_equal(out1, out2)
    assert rep1.blur_type == rep2.blur_type
    assert rep2.psf_params["length"] == rep1.psf_params["length"] / 2.0
    assert rep2.view_psf_params == rep1.psf_params


# ---------------------------------------------------------------------------
# Zoom: the brief's 15 px case (effective 30 px view kernel)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zoom15_view():
    return to_bgr(_zoomed_view(15.0))


def test_zoom_pin_path_exact(zoom15_view):
    out_a, rep_a = enhance_image(
        zoom15_view, Settings(blur_type="motion", upscale=None,
                              length=15, angle=30, zoom=2.0))
    out_b, rep_b = enhance_image(
        zoom15_view, Settings(blur_type="motion", upscale=None,
                              length=30, angle=30, zoom=1.0))
    out_c, rep_c = enhance_image(
        zoom15_view, Settings(blur_type="motion", upscale=None,
                              length=15, angle=30, zoom=1.0))
    assert rep_a.psf_params == {"length": 15.0, "angle": 30.0}
    assert rep_a.view_psf_params == {"length": 30.0, "angle": 30.0}
    assert np.array_equal(out_a, out_b)
    # ignoring the zoom under-restores: the same pin at zoom=1 uses
    # half the true view kernel and lands measurably softer
    assert rep_a.output_score >= 1.25 * rep_c.output_score, (
        rep_a.output_score, rep_c.output_score)


def test_zoom_sweep_conversion_on_15px_fixture(zoom15_view):
    out1, rep1 = enhance_image(zoom15_view, Settings(blur_type="auto",
                                                     upscale=None))
    out2, rep2 = enhance_image(zoom15_view, Settings(blur_type="auto",
                                                     upscale=None,
                                                     zoom=2.0))
    assert np.array_equal(out1, out2)
    assert rep1.blur_type == rep2.blur_type == "motion"
    assert rep2.view_psf_params == rep1.psf_params
    assert rep2.psf_params["length"] == rep1.psf_params["length"] / 2.0


def test_cli_zoom_smoke(tmp_path, zoom15_view):
    src = tmp_path / "zoomed.png"
    cv2.imwrite(str(src), zoom15_view)
    r = subprocess.run(
        [sys.executable, "-m", "deblur", str(src),
         str(tmp_path / "out.png"), "--type", "motion",
         "--length", "15", "--angle", "30", "--zoom", "2",
         "--upscale", "none"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "psf params   : {'length': 15.0, 'angle': 30.0}" in r.stdout
    assert "zoom         : 2x" in r.stdout, r.stdout
    assert "'length': 30.0" in r.stdout  # the view-space kernel used


# ---------------------------------------------------------------------------
# Video: track construction units
# ---------------------------------------------------------------------------

def test_track_interpolates_and_holds():
    samples = [(0, "motion", {"length": 10.0, "angle": 30.0}),
               (10, "motion", {"length": 20.0, "angle": 30.0})]
    track = build_track(samples, 14)
    assert track[0] == ("motion", {"length": 10.0, "angle": 30.0})
    assert track[5][1]["length"] == pytest.approx(15.0)
    assert track[13][1]["length"] == 20.0  # holds after the last sample


def test_track_angle_wraps_the_short_way():
    samples = [(0, "motion", {"length": 10.0, "angle": 175.0}),
               (10, "motion", {"length": 10.0, "angle": 5.0})]
    track = build_track(samples, 11)
    mid = track[5][1]["angle"]
    assert min(mid, 180.0 - mid) <= 2.0, mid  # passes through 0, not 90


def test_track_family_change_steps_at_midpoint():
    samples = [(0, "motion", {"length": 10.0, "angle": 30.0}),
               (10, "focus", {"radius": 4.0})]
    track = build_track(samples, 11)
    assert track[4][0] == "motion"
    assert track[5][0] == "focus"
    assert track[5][1] == {"radius": 4.0}  # never a blended hybrid


def test_track_median_smoothing_kills_spikes():
    samples = [(0, "motion", {"length": 10.0, "angle": 30.0}),
               (5, "motion", {"length": 30.0, "angle": 30.0}),
               (10, "motion", {"length": 12.0, "angle": 30.0})]
    track = build_track(samples, 11)
    assert track[5][1]["length"] == 12.0  # median of (10, 30, 12)


# ---------------------------------------------------------------------------
# Video: the synthetic-clip oracle
# ---------------------------------------------------------------------------

CLIP_FRAMES = 30
CLIP_ANGLE = 30.0


def _truth_length(i, n=CLIP_FRAMES):
    t = i / (n - 1)
    return 10.0 + 10.0 * (t * t * (3 - 2 * t))


@pytest.fixture(scope="module")
def video_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("video")
    scene = _textured_source(720, seed=9)
    rng = np.random.default_rng(4242)
    h, w = 320, 448
    src = folder / "clip.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (w, h))
    for i in range(CLIP_FRAMES):
        crop = scene[150:150 + h, 60 + 2 * i:60 + 2 * i + w]
        blurred = apply_blur(crop, motion_psf(_truth_length(i), CLIP_ANGLE))
        noisy = np.clip(blurred + rng.normal(0, 3.0 / 255.0, blurred.shape),
                        0, 1)
        writer.write(to_bgr(noisy))
    writer.release()

    settings = Settings(blur_type="auto", method="wiener", upscale=None)
    out = folder / "clip_out.mp4"
    report = enhance_video(src, out, settings, sample_every=6)

    def decode(path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames

    in_frames, out_frames = decode(src), decode(out)
    # the unsmoothed control: estimate every decoded frame on its own
    control = [estimate_frame_psf(f, settings, max_side=480)
               for f in in_frames]
    return {"report": report, "in": in_frames, "out": out_frames,
            "control": control}


def _mean_steepness(frames):
    return float(np.mean([
        score_image(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    .astype(np.float64) / 255.0)
        for f in frames
    ]))


def test_video_output_sharper(video_case):
    si = _mean_steepness(video_case["in"])
    so = _mean_steepness(video_case["out"])
    assert len(video_case["out"]) == CLIP_FRAMES
    assert so >= 1.5 * si, (si, so)


def test_video_track_recovers_truth(video_case):
    rep = video_case["report"]
    assert rep.frames == CLIP_FRAMES
    assert rep.samples, "no samples recorded"
    for idx, family, _params in rep.samples:
        assert family == "motion", (idx, family)
    for idx, _family, _params in rep.samples:
        ang = rep.track[idx][1]["angle"]
        assert angle_diff(ang, CLIP_ANGLE) <= ANGLE_TOL, (idx, ang)
    lengths = np.array([p["length"] for _f, p in rep.track])
    truth = np.array([_truth_length(i) for i in range(CLIP_FRAMES)])
    corr = float(np.corrcoef(lengths, truth)[0, 1])
    assert corr >= 0.85, corr
    assert float(np.mean(np.abs(lengths - truth))) <= LENGTH_TOL


def test_video_temporal_discipline(video_case):
    rep = video_case["report"]
    lengths = np.array([p["length"] for _f, p in rep.track])
    smooth_max = float(np.max(np.abs(np.diff(lengths))))
    assert smooth_max <= 1.5, smooth_max
    ctrl = np.array([p.get("length", np.nan)
                     for _fam, p, _n in video_case["control"]], float)
    assert not np.isnan(ctrl).any()
    ctrl_max = float(np.nanmax(np.abs(np.diff(ctrl))))
    assert ctrl_max >= 1.5 * smooth_max, (ctrl_max, smooth_max)


def test_video_rejects_segmented_and_missing(tmp_path):
    with pytest.raises(ValueError):
        enhance_video(tmp_path / "nope.mp4", tmp_path / "o.mp4",
                      Settings(blur_type="segmented"))
    with pytest.raises(ValueError):
        enhance_video(tmp_path / "nope.mp4", tmp_path / "o.mp4",
                      Settings(blur_type="auto"))


def test_cli_video_smoke(tmp_path):
    scene = _textured_source(400, seed=4)
    src = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (320, 240))
    for i in range(12):
        crop = scene[40:280, 20 + i:20 + i + 320]
        writer.write(to_bgr(apply_blur(crop, motion_psf(12, 30))))
    writer.release()
    out = tmp_path / "tiny_out.mp4"
    r = subprocess.run(
        [sys.executable, "-m", "deblur", "video", str(src), str(out),
         "--sample-every", "4"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "s/frame" in r.stdout, r.stdout
    assert "samples" in r.stdout and ": motion" in r.stdout, r.stdout
    cap = cv2.VideoCapture(str(out))
    n = 0
    while True:
        ok, _f = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    assert n == 12
    # segmented is refused for video, in words
    r = subprocess.run(
        [sys.executable, "-m", "deblur", "video", str(src), str(out),
         "--type", "segmented"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "segmented is a stills setting" in r.stderr
