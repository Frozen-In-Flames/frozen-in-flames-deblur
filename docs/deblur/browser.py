"""Browser entry points: the same engine, staged for a slow runtime.

The web page (docs/index.html) runs this package under Pyodide, where
a full-resolution PSF sweep would be needlessly slow. The staging here
changes WHEN work happens, never the math:

  1. estimate_settings() runs the pipeline's own select_psf() sweep on
     a downscaled working copy (long side capped, default 900 px) and
     returns a Settings object with the winning PSF pinned, its
     parameters scaled back to native resolution (lengths, radii and
     sigmas scale linearly with image size; angles do not change).
  2. enhance_image() is then called once on the full-resolution image
     with those pinned parameters, so restoration happens exactly as
     the CLI does it, sweep skipped by the pin.

For images at or under the cap the scale factor is 1.0 and the pinned
parameters are exactly what the native sweep returns. Manual pins and
the lowres path skip estimation entirely and behave identically to
the CLI.

One environment adaptation lives here, behind a feature detect: if
cv2.fastNlMeansDenoisingColored is missing from the OpenCV build in
use (it is present in the pinned Pyodide build, opencv-python
4.11.0.86), estimation and restoration both drop the denoise step and
the returned info records that in "notes" so the page can show it.
The pipeline's own denoise call is left untouched; the detect here
only decides the Settings the browser hands to it.
"""

from dataclasses import replace

import cv2
import numpy as np

from .pipeline import _luminance, enhance_image, select_psf
from .presets import Settings

ESTIMATE_MAX_SIDE = 900  # px, long side of the sweep working copy

_SCALE_KEYS = {"motion": ("length",), "focus": ("radius",), "gaussian": ("sigma",)}


def _denoise_available():
    return hasattr(cv2, "fastNlMeansDenoisingColored")


def _scale_params(family, params, k):
    out = dict(params)
    for key in _SCALE_KEYS.get(family, ()):
        if key in out:
            out[key] = float(out[key]) * k
    return out


def _pinned_settings(settings, family, params):
    """A copy of settings with exactly family's parameters pinned."""
    pins = {"length": None, "angle": None, "radius": None, "sigma": None}
    for key, value in params.items():
        pins[key] = float(value)
    return replace(settings, blur_type=family, **pins)


def estimate_settings(image, settings=None, max_side=ESTIMATE_MAX_SIDE):
    """Stage 1: identify the blur on a capped working copy.

    image: BGR uint8 ndarray (as enhance_image accepts).
    Returns (pinned_settings, info). info records the working size,
    the scale factor, the winning family and parameters at both
    scales, the top candidates (parameters scaled to native scale),
    and any adaptation notes. If estimation does not apply (manual
    pins or lowres), pinned_settings is the settings unchanged and
    info["estimated"] is False.
    """
    settings = settings or Settings()
    info = {"estimated": False, "notes": []}
    if settings.blur_type == "lowres" or settings.manual_family() is not None:
        return settings, info

    img = np.asarray(image)
    h, w = img.shape[:2]
    long_side = max(h, w)
    k = long_side / float(max_side) if long_side > max_side else 1.0
    if k > 1.0:
        work = cv2.resize(
            img,
            (int(round(w / k)), int(round(h / k))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = img
    info["working_size"] = (work.shape[1], work.shape[0])
    info["scale_factor"] = k

    est_settings = settings
    if settings.denoise and not _denoise_available():
        est_settings = replace(settings, denoise=False)
        info["notes"].append(
            "denoise unavailable in this OpenCV build; estimation ran without it"
        )
    if est_settings.denoise:
        work = cv2.fastNlMeansDenoisingColored(work, None, 3, 3, 7, 21)

    family, work_params, ranked, null_match = select_psf(_luminance(work), est_settings)
    params = _scale_params(family, work_params, k)
    info.update(
        estimated=True,
        blur_type=family,
        working_params=work_params,
        psf_params=params,
        null_match=null_match,
        candidates=[
            (fam, _scale_params(fam, p, k), s, nm) for fam, p, s, nm in ranked[:5]
        ],
    )
    return _pinned_settings(settings, family, params), info


def enhance_image_browser(image, settings=None, max_side=ESTIMATE_MAX_SIDE):
    """Stage 1 + 2: estimate on the capped copy, restore at full size.

    Returns (bgr_uint8_result, Report, info) where Report is the
    pipeline's own report for the full-resolution restoration and info
    is estimate_settings' audit dict. When denoise is unavailable in
    the running OpenCV build, the restoration Settings carry the same
    adaptation (denoise off) and info["notes"] says so.
    """
    settings = settings or Settings()
    pinned, info = estimate_settings(image, settings, max_side=max_side)
    if settings.denoise and not _denoise_available():
        pinned = replace(pinned, denoise=False)
        note = "denoise unavailable in this OpenCV build; restoration ran without it"
        if note not in info["notes"]:
            info["notes"].append(note)
    result, report = enhance_image(image, pinned)
    return result, report, info
