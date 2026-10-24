"""Oracle tests with negative controls.

Thresholds were fixed from the design targets before the suite was run
against the final scorer; measured values live in TEST-REPORT.md.

  * motion (truth length 15, angle 30): recovered length within +/-4,
    angle within +/-10 degrees, in both auto and motion mode.
  * focus (truth radius 4): recovered radius within +/-1.5 (focus mode).
  * gaussian (truth sigma 2.0): recovered sigma within +/-1.0 (gaussian
    mode; the spectral radial fit has a measured +0.4..0.8 upward bias
    on this fixture family, documented in TEST-REPORT.md).
  * deblurred output steepness >= 1.30x the blurred input (all cases).
  * negative controls: right-parameter output beats wrong-parameter
    output by >= 1.15x steepness (motion, focus); deconvolution beats
    sharpen-only by >= 1.15x (gaussian); candidate-score margins as
    stated per test.
  * lowres: output long side reaches the target exactly; steepness
    >= 1.10x plain Lanczos upscale.
  * identity: sharp image, upscale off: dimensions unchanged and the
    output/input steepness ratio stays in [0.85, 1.60].
"""

import subprocess
import sys

import cv2
import numpy as np
import pytest

from deblur.pipeline import _center_crop, _luminance, enhance_image, unsharp_rgb
from deblur.presets import Settings
from deblur.psf import apply_blur, build_psf, disk_psf, gaussian_psf, motion_psf
from deblur.scoring import SpectrumScorer, score_image

from conftest import to_bgr

ANGLE_TOL = 10.0
LENGTH_TOL = 4.0
RADIUS_TOL = 1.5
SIGMA_TOL = 1.0
IMPROVE_MIN = 1.30
CONTROL_MIN = 1.15


def angle_diff(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


# ---------------------------------------------------------------------------
# Unit oracles
# ---------------------------------------------------------------------------

def test_psf_normalization_and_shape():
    for psf in (motion_psf(15, 30), disk_psf(4), gaussian_psf(2.0)):
        assert psf.ndim == 2 and psf.shape[0] == psf.shape[1]
        assert psf.shape[0] % 2 == 1
        assert abs(float(psf.sum()) - 1.0) < 1e-5
        assert float(psf.min()) >= 0.0


def test_motion_psf_orientation():
    k = motion_psf(15, 0)
    mid = k.shape[0] // 2
    assert float(k[mid, :].sum()) > 0.9  # angle 0: mass on the middle row


def test_unsharp_regression_mean_preserved():
    # Guards the scikit-image 0.26 channel_axis=-1 defect: the broken
    # path returned an image with mean ~0.003 from a healthy input.
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(0.5, 0.15, (60, 60, 3)), 0, 1)
    img[20:40, 20:40] = 0.9
    out = unsharp_rgb(img, radius=2.0, amount=1.2)
    assert abs(float(out.mean()) - float(img.mean())) < 0.02
    assert float(out.std()) > float(img.std())  # it actually sharpened


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def motion_blurred(gt):
    return apply_blur(gt, motion_psf(15, 30))


def test_motion_recovery_auto_and_mode(motion_blurred):
    for mode in ("auto", "motion"):
        _, rep = enhance_image(
            to_bgr(motion_blurred), Settings(blur_type=mode, upscale=None)
        )
        assert rep.blur_type == "motion", (mode, rep.blur_type, rep.psf_params)
        assert abs(rep.psf_params["length"] - 15.0) <= LENGTH_TOL
        assert angle_diff(rep.psf_params["angle"], 30.0) <= ANGLE_TOL
        assert rep.output_score >= IMPROVE_MIN * rep.input_score


def test_motion_candidate_score_controls(motion_blurred):
    L = _luminance(to_bgr(motion_blurred))
    sc = SpectrumScorer(_center_crop(L))
    right = sc.score(motion_psf(15, 30), "motion")["score"]
    wrong_angle = sc.score(motion_psf(15, 90), "motion")["score"]  # 60 deg off
    wrong_family = sc.score(disk_psf(4), "focus")["score"]
    assert right - wrong_angle >= 0.10
    assert right - wrong_family >= 0.10


def test_motion_wrong_pin_control(motion_blurred):
    _, right = enhance_image(
        to_bgr(motion_blurred), Settings(blur_type="motion", upscale=None)
    )
    _, wrong = enhance_image(
        to_bgr(motion_blurred),
        Settings(blur_type="motion", length=15, angle=90, upscale=None),
    )
    assert wrong.manual
    assert right.output_score >= CONTROL_MIN * wrong.output_score


# ---------------------------------------------------------------------------
# Focus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def focus_blurred(gt):
    return apply_blur(gt, disk_psf(4))


def test_focus_recovery(focus_blurred):
    _, rep = enhance_image(
        to_bgr(focus_blurred), Settings(blur_type="focus", upscale=None)
    )
    assert rep.blur_type == "focus"
    assert abs(rep.psf_params["radius"] - 4.0) <= RADIUS_TOL
    assert rep.output_score >= IMPROVE_MIN * rep.input_score


def test_focus_candidate_and_pin_controls(focus_blurred):
    L = _luminance(to_bgr(focus_blurred))
    sc = SpectrumScorer(_center_crop(L))
    right = sc.score(disk_psf(4), "focus")["score"]
    wrong = sc.score(gaussian_psf(1.0), "gaussian")["score"]
    assert right > wrong  # positive margin (measured ~0.03; see report)

    _, right_rep = enhance_image(
        to_bgr(focus_blurred), Settings(blur_type="focus", upscale=None)
    )
    _, wrong_rep = enhance_image(
        to_bgr(focus_blurred),
        Settings(blur_type="gaussian", sigma=1.0, upscale=None),
    )
    assert right_rep.output_score >= CONTROL_MIN * wrong_rep.output_score


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gaussian_blurred(gt):
    return apply_blur(gt, gaussian_psf(2.0))


def test_gaussian_recovery(gaussian_blurred):
    _, rep = enhance_image(
        to_bgr(gaussian_blurred), Settings(blur_type="gaussian", upscale=None)
    )
    assert rep.blur_type == "gaussian"
    assert abs(rep.psf_params["sigma"] - 2.0) <= SIGMA_TOL
    assert rep.output_score >= IMPROVE_MIN * rep.input_score


def test_gaussian_beats_sharpen_only(gaussian_blurred):
    _, deconv_rep = enhance_image(
        to_bgr(gaussian_blurred), Settings(blur_type="gaussian", upscale=None)
    )
    _, sharp_rep = enhance_image(
        to_bgr(gaussian_blurred), Settings(blur_type="lowres", upscale=None)
    )
    assert deconv_rep.output_score >= CONTROL_MIN * sharp_rep.output_score


# ---------------------------------------------------------------------------
# Low resolution
# ---------------------------------------------------------------------------

def test_lowres_upscale(gt):
    small = cv2.resize(gt, (100, 100), interpolation=cv2.INTER_AREA)
    out, rep = enhance_image(
        to_bgr(small), Settings(blur_type="lowres", upscale=400)
    )
    assert max(rep.output_size) == 400
    plain = cv2.resize(small, (400, 400), interpolation=cv2.INTER_LANCZOS4)
    out_score = score_image(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0)
    assert out_score >= 1.10 * score_image(plain)


# ---------------------------------------------------------------------------
# Identity control + manual override + CLI
# ---------------------------------------------------------------------------

def test_identity_no_degradation(gt):
    _, rep = enhance_image(
        to_bgr(gt), Settings(blur_type="auto", upscale=None)
    )
    assert rep.output_size == rep.input_size
    ratio = rep.output_score / rep.input_score
    assert 0.85 <= ratio <= 1.60


def test_manual_override_skips_sweep(motion_blurred):
    _, rep = enhance_image(
        to_bgr(motion_blurred),
        Settings(blur_type="motion", length=15, angle=30, upscale=None),
    )
    assert rep.manual
    assert rep.psf_params == {"length": 15.0, "angle": 30.0}
    assert rep.candidates == []


def test_cli_smoke(tmp_path, motion_blurred):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    assert cv2.imwrite(str(src), to_bgr(motion_blurred))
    proc = subprocess.run(
        [sys.executable, "-m", "deblur", str(src), str(dst),
         "--type", "motion", "--upscale", "none"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert dst.exists()
    assert "blur type    : motion" in proc.stdout
    assert "psf params" in proc.stdout and "'length':" in proc.stdout
