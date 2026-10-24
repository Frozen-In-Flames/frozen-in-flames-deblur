"""v0.3.0 oracles: noise as a first-class setting, estimation on a
denoised copy read through the detail layer, and segmented
(per-region) restoration.

Thresholds were fixed from the design targets and the calibration
battery before this suite ran against the final code; measured values
live in TEST-REPORT.md.

  * noise estimator (Immerkaer, flat-weighted): a clean fixture reads
    < 2.0 gray levels; a fixture with gaussian noise sigma 15 reads
    within [10, 20].
  * noise oracle: sharp fixture + noise sigma 15, no blur. Auto must
    select the noise path (no deconvolution); the flat-patch noise
    std of the output must drop to <= 0.50x the input's; edge
    steepness on a textured patch must stay within [0.65, 1.50]x of
    the CLEAN fixture's (the denoise must not melt what it keeps).
  * estimation oracles (denoise-first + detail layer):
      (a) motion 15/30 + noise sigma 12: auto identifies motion
          within +/-4 length / +/-10 degrees;
      (b) flat-style fixture (large flat regions, sparse shapes) +
          motion 15/30: family motion, same tolerances (v0.2.0 read
          this shape of content as focus);
      (c) 800 px textured fixture with a large smooth gradient +
          motion 15/30: angle error <= 10 degrees (v0.2.0: 12).
  * segmented oracle: one sharp source composited from two layers,
    foreground disk blurred with motion 21/0, background with motion
    21/90, feathered composite:
      (i)   region estimates recover each region's angle within
            +/-10 degrees and length within +/-4;
      (ii)  the per-region output's steepness inside each region
            beats the WRONG-kernel global control in that region by
            >= 1.15x, and keeps >= 0.80x of the RIGHT-kernel global
            control in that region (the tile-grid mask costs some
            purity near boundaries; that is the stated price);
      (iii) seam discipline: the max gradient magnitude in the true
            mask's boundary band of the segmented output is
            <= 1.5x the better (lower) of the two global controls'.
  * controls: segmented on a uniformly blurred image falls back and
    matches the plain mode's output score within 5%; the noise
    setting on a clean sharp image changes steepness only within
    [0.70, 1.50]x with dimensions unchanged.
"""

import subprocess
import sys

import cv2
import numpy as np
import pytest

from deblur.pipeline import enhance_image
from deblur.presets import Settings
from deblur.psf import apply_blur, motion_psf
from deblur.scoring import edge_steepness, estimate_noise_sigma

from conftest import make_gt, to_bgr

ANGLE_TOL = 10.0
LENGTH_TOL = 4.0
FLAT_STD_FACTOR_MAX = 0.50
STEEP_BAND = (0.65, 1.50)
WRONG_KERNEL_MIN = 1.15
OWN_KERNEL_MIN = 0.80
SEAM_MAX = 1.5

FLAT_PATCH = (slice(270, 320), slice(280, 350))   # constant 0.15 in the GT
CHECKER_PATCH = (slice(30, 90), slice(280, 340))  # checkerboard in the GT


def angle_diff(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def add_noise(gray, sigma, rng):
    return np.clip(gray + rng.normal(0, sigma / 255.0, gray.shape), 0, 1)


# ---------------------------------------------------------------------------
# Noise measurement + the noise path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def noisy_gt(gt):
    rng = np.random.default_rng(20261024)
    return add_noise(gt, 15, rng)


def test_noise_sigma_estimator(gt, noisy_gt):
    clean = estimate_noise_sigma(gt)
    noisy = estimate_noise_sigma(noisy_gt)
    assert clean < 2.0, clean
    assert 10.0 <= noisy <= 20.0, noisy


def test_auto_selects_noise_path(gt, noisy_gt):
    out, rep = enhance_image(to_bgr(noisy_gt), Settings(blur_type="auto",
                                                        upscale=None))
    assert rep.blur_type == "noise", (rep.blur_type, rep.psf_params)
    assert any("deconv skipped (noise)" in s for s in rep.steps)
    g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    std_in = float(noisy_gt[FLAT_PATCH].std())
    std_out = float(g_out[FLAT_PATCH].std())
    assert std_out <= FLAT_STD_FACTOR_MAX * std_in, (std_in, std_out)
    steep_out = edge_steepness(g_out[CHECKER_PATCH])
    steep_clean = edge_steepness(gt[CHECKER_PATCH])
    lo, hi = STEEP_BAND
    assert lo * steep_clean <= steep_out <= hi * steep_clean, (
        steep_out, steep_clean)


def test_noise_mode_on_clean_no_harm(gt):
    out, rep = enhance_image(to_bgr(gt), Settings(blur_type="noise",
                                                  upscale=None))
    assert rep.blur_type == "noise"
    assert rep.output_size == rep.input_size
    ratio = rep.output_score / rep.input_score
    assert 0.70 <= ratio <= 1.50, ratio


# ---------------------------------------------------------------------------
# Denoise-first + detail-layer estimation on the failure shapes
# ---------------------------------------------------------------------------

def test_estimation_motion_buried_in_noise(gt):
    rng = np.random.default_rng(20261024)
    img = add_noise(apply_blur(gt, motion_psf(15, 30)), 12, rng)
    _, rep = enhance_image(to_bgr(img), Settings(blur_type="auto",
                                                  upscale=None))
    assert rep.blur_type == "motion", (rep.blur_type, rep.psf_params)
    assert abs(rep.psf_params["length"] - 15.0) <= LENGTH_TOL
    assert angle_diff(rep.psf_params["angle"], 30.0) <= ANGLE_TOL


def _flat_fixture():
    flat = np.full((512, 512), 0.78, np.float32)
    cv2.rectangle(flat, (60, 60), (200, 180), 0.16, -1)
    cv2.circle(flat, (360, 140), 70, 0.35, -1)
    cv2.rectangle(flat, (280, 300), (460, 420), 0.51, -1)
    cv2.circle(flat, (120, 380), 45, 0.24, -1)
    return flat


def test_estimation_flat_fixture():
    img = apply_blur(_flat_fixture(), motion_psf(15, 30))
    _, rep = enhance_image(to_bgr(img), Settings(blur_type="auto",
                                                 upscale=None))
    assert rep.blur_type == "motion", (rep.blur_type, rep.psf_params)
    assert abs(rep.psf_params["length"] - 15.0) <= LENGTH_TOL
    assert angle_diff(rep.psf_params["angle"], 30.0) <= ANGLE_TOL


def test_estimation_gradient_fixture_angle():
    img = apply_blur(make_gt(800), motion_psf(15, 30))
    _, rep = enhance_image(to_bgr(img), Settings(blur_type="auto",
                                                  upscale=None))
    assert rep.blur_type == "motion", (rep.blur_type, rep.psf_params)
    assert abs(rep.psf_params["length"] - 15.0) <= LENGTH_TOL
    assert angle_diff(rep.psf_params["angle"], 30.0) <= ANGLE_TOL


# ---------------------------------------------------------------------------
# Segmented (per-region) restoration
# ---------------------------------------------------------------------------

def _textured_source(size=640, seed=11):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    gx = np.linspace(0.3, 0.5, size, dtype=np.float32)[None, :]
    gy = np.linspace(0.0, 0.1, size, dtype=np.float32)[:, None]
    img[:] = gx + gy
    for _ in range(260):
        x0, y0 = int(rng.integers(4, size - 30)), int(rng.integers(4, size - 30))
        s = int(rng.integers(5, 22))
        img[y0:y0 + s, x0:x0 + s] = float(rng.uniform(0.08, 0.95))
    for i in range(8):
        y = 30 + i * 76
        for j in range(4):
            x0 = int(rng.integers(10, 60)) + j * 160
            img[y:y + 9, x0:x0 + int(rng.integers(90, 140))] = (
                0.9 if (i + j) % 2 else 0.12)
    return img


@pytest.fixture(scope="module")
def segmented_case():
    size = 640
    src = _textured_source(size)
    cy = cx = size // 2
    radius = int(size * 0.34)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(yy - cy, xx - cx)
    mask = np.clip((radius + 6 - dist) / 12.0, 0, 1).astype(np.float32)
    comp = (mask * apply_blur(src, motion_psf(21, 0))
            + (1 - mask) * apply_blur(src, motion_psf(21, 90)))
    img = to_bgr(comp)
    out, rep = enhance_image(img, Settings(blur_type="segmented",
                                           upscale=None))
    controls = {}
    for name, kw in (("fg", {"length": 21, "angle": 0}),
                     ("bg", {"length": 21, "angle": 90})):
        o, _ = enhance_image(img, Settings(blur_type="motion",
                                           upscale=None, **kw))
        controls[name] = cv2.cvtColor(
            o, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return {
        "rep": rep,
        "out_gray": cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64)
        / 255.0,
        "controls": controls,
        "dist": dist,
        "radius": radius,
    }


FG_PATCH = (slice(260, 380), slice(260, 380))  # deep inside the disk
BG_PATCH = (slice(30, 150), slice(30, 150))    # a background corner


def test_segmented_region_estimates(segmented_case):
    rep = segmented_case["rep"]
    assert rep.segment_mode == "per-region", rep.regions
    by_name = {r["name"]: r for r in rep.regions}
    fg, bg = by_name["foreground"], by_name["background"]
    assert fg["family"] == "motion" and bg["family"] == "motion", rep.regions
    assert abs(fg["params"]["length"] - 21.0) <= LENGTH_TOL
    assert abs(bg["params"]["length"] - 21.0) <= LENGTH_TOL
    assert angle_diff(fg["params"]["angle"], 0.0) <= ANGLE_TOL
    assert angle_diff(bg["params"]["angle"], 90.0) <= ANGLE_TOL
    assert rep.mask_coverage is not None
    assert 25.0 <= rep.mask_coverage <= 65.0, rep.mask_coverage


def test_segmented_beats_wrong_kernel_controls(segmented_case):
    seg = segmented_case["out_gray"]
    fg_ctl = segmented_case["controls"]["fg"]
    bg_ctl = segmented_case["controls"]["bg"]
    seg_fg = edge_steepness(seg[FG_PATCH])
    seg_bg = edge_steepness(seg[BG_PATCH])
    # each region beats the control that used the OTHER region's PSF
    assert seg_bg >= WRONG_KERNEL_MIN * edge_steepness(fg_ctl[BG_PATCH]), (
        seg_bg, edge_steepness(fg_ctl[BG_PATCH]))
    assert seg_fg >= WRONG_KERNEL_MIN * edge_steepness(bg_ctl[FG_PATCH]), (
        seg_fg, edge_steepness(bg_ctl[FG_PATCH]))
    # and keeps most of the control that used its OWN PSF
    assert seg_fg >= OWN_KERNEL_MIN * edge_steepness(fg_ctl[FG_PATCH]), (
        seg_fg, edge_steepness(fg_ctl[FG_PATCH]))
    assert seg_bg >= OWN_KERNEL_MIN * edge_steepness(bg_ctl[BG_PATCH]), (
        seg_bg, edge_steepness(bg_ctl[BG_PATCH]))


def test_segmented_seam_discipline(segmented_case):
    dist, radius = segmented_case["dist"], segmented_case["radius"]
    band = np.abs(dist - radius) <= 6

    def band_max(gray):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.hypot(gx, gy)[band].max())

    seg_band = band_max(segmented_case["out_gray"])
    better_control = min(band_max(segmented_case["controls"]["fg"]),
                         band_max(segmented_case["controls"]["bg"]))
    assert seg_band <= SEAM_MAX * better_control, (seg_band, better_control)


def test_segmented_fallback_on_uniform_blur(gt):
    img = to_bgr(apply_blur(gt, motion_psf(15, 30)))
    _, rep_seg = enhance_image(img, Settings(blur_type="segmented",
                                             upscale=None))
    _, rep_mot = enhance_image(img, Settings(blur_type="motion",
                                             upscale=None))
    assert rep_seg.segment_mode == "fallback", rep_seg.regions
    assert rep_seg.output_score >= 0.95 * rep_mot.output_score, (
        rep_seg.output_score, rep_mot.output_score)


# ---------------------------------------------------------------------------
# CLI smoke for the two new settings
# ---------------------------------------------------------------------------

def test_cli_noise_and_segmented(tmp_path, gt, noisy_gt):
    noisy_path = tmp_path / "noisy.png"
    cv2.imwrite(str(noisy_path), to_bgr(noisy_gt))
    r = subprocess.run(
        [sys.executable, "-m", "deblur", str(noisy_path),
         str(tmp_path / "out_noise.png"), "--type", "noise",
         "--upscale", "none"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert "blur type    : noise" in r.stdout, r.stdout

    motion_path = tmp_path / "motion.png"
    cv2.imwrite(str(motion_path),
                to_bgr(apply_blur(gt, motion_psf(15, 30))))
    r = subprocess.run(
        [sys.executable, "-m", "deblur", str(motion_path),
         str(tmp_path / "out_seg.png"), "--type", "segmented",
         "--upscale", "none"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    assert "segmented    : fallback" in r.stdout, r.stdout
