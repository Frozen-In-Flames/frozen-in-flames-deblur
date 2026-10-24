"""Frozen In Flames Deblur: classical (no AI) image deblurring.

Removes motion blur, defocus, and gaussian softness by deconvolution
with swept, scored PSF settings; handles low-resolution inputs with
an upscale + gentle sharpen path; treats noise-dominated images with
a dedicated strong-denoise setting (no deconvolution); and restores
foreground and background separately in segmented mode, each with
its own estimated PSF. Built on OpenCV, scikit-image, numpy.
"""

from .browser import enhance_image_browser, estimate_settings
from .pipeline import Report, enhance_image, segment_image, select_psf
from .presets import PRESETS, Settings
from .psf import build_psf, disk_psf, gaussian_psf, motion_psf
from .scoring import estimate_noise_sigma, score_breakdown, score_image

__version__ = "0.3.0"

__all__ = [
    "enhance_image",
    "enhance_image_browser",
    "estimate_settings",
    "select_psf",
    "segment_image",
    "Report",
    "Settings",
    "PRESETS",
    "build_psf",
    "motion_psf",
    "disk_psf",
    "gaussian_psf",
    "score_image",
    "score_breakdown",
    "estimate_noise_sigma",
]
