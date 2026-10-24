"""Point-spread-function (PSF) builders for the three blur families.

A PSF is the kernel a sharp image was convolved with to produce the blur.
Deconvolution needs a PSF that matches the real blur, so the toolkit builds
PSFs for motion blur (a line), defocus (a disk), and gaussian softness
(a bell), exactly as specced in the project notes.
"""

import cv2
import numpy as np


def motion_psf(length, angle_deg, size=None):
    """Line PSF for linear motion blur.

    length: blur length in pixels. angle_deg: direction of motion, degrees.
    The kernel is a horizontal line rotated about its center, normalized
    to sum to 1.
    """
    if length < 1:
        raise ValueError("motion length must be >= 1")
    size = size or int(length) | 1
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1
    m = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1)
    k = cv2.warpAffine(k, m, (size, size))
    total = k.sum()
    if total <= 0:
        raise ValueError("degenerate motion PSF")
    return (k / total).astype(np.float32)


def disk_psf(radius):
    """Disk PSF for out-of-focus (defocus) blur of the given radius in px."""
    if radius <= 0:
        raise ValueError("disk radius must be > 0")
    size = int(2 * radius + 1) | 1
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (size // 2, size // 2), int(radius), 1, -1)
    total = k.sum()
    if total <= 0:
        raise ValueError("degenerate disk PSF")
    return (k / total).astype(np.float32)


def gaussian_psf(sigma):
    """Gaussian PSF for lens softness / general blur, std-dev sigma in px."""
    if sigma <= 0:
        raise ValueError("gaussian sigma must be > 0")
    size = int(6 * sigma + 1) | 1
    g = cv2.getGaussianKernel(size, sigma)
    k = (g @ g.T).astype(np.float32)
    return k / k.sum()


BUILDERS = {
    "motion": lambda p: motion_psf(p["length"], p["angle"]),
    "focus": lambda p: disk_psf(p["radius"]),
    "gaussian": lambda p: gaussian_psf(p["sigma"]),
}


def build_psf(family, params):
    """Build a PSF for a family name + parameter dict."""
    if family not in BUILDERS:
        raise ValueError(f"unknown blur family: {family!r}")
    return BUILDERS[family](params)


def apply_blur(channel, psf):
    """Convolve a single float channel with a PSF (reflect borders).

    Used for synthetic ground-truth generation in tests and for any caller
    that needs to simulate a blur.
    """
    return cv2.filter2D(channel, -1, psf, borderType=cv2.BORDER_REFLECT)
