"""Recorded-video enhancement: deblur a whole clip, frame by frame.

This is the recorded-content mode for camera glasses (mode 1 of
docs/glasses.md): the wearer records video on the glasses, and the
clip is deblurred afterward on a companion device, where seconds
per frame are affordable. The live mode is a different, much
cheaper chain (docs/live.html); nothing here pretends to be
real-time.

Design (the stills engine, adapted to a frame sequence):

  * TWO PASSES. Pass 1 (analysis) decodes the clip and estimates the
    PSF on SAMPLED frames only (every `sample_every` frames), on a
    downscaled working copy, using the pipeline's own select_psf on
    the usual denoised estimation luminance. Pass 2 (render) decodes
    again and restores each frame with its track parameters.
    Decoding twice is the honest price of knowing the whole track
    before rendering any of it.
  * TEMPORAL DISCIPLINE. Raw per-frame estimates jitter: adjacent
    frames of one continuous shot can read several px apart. The
    track therefore smooths the samples (median of 3 within a run of
    one family) and linearly interpolates parameters between
    samples, so the kernel a frame is restored with can only drift
    as fast as the samples drift. Family changes are never blended
    (a motion kernel averaged with a disk is neither): the track
    steps to the new family at the midpoint between samples.
    "noise" and "identity" samples are honoured locally: frames
    nearer such a sample get the noise treatment (strong denoise,
    no deconvolution) or the plain tail, not a held blur kernel.
  * PER-FRAME RESTORATION reuses the pipeline pieces: the same
    noise-scaled denoise, the same luminance deconvolution
    (_deconvolve_bgr), the same unsharp + CLAHE tail. The method
    defaults to wiener on the CLI because it is the fast method;
    rl-tv is available and costs what it costs. Upscaling is not
    applied in video mode: frames are written at native size.
    Segmented mode is a stills setting and is rejected here.
  * COST IS REPORTED, not hidden: the report carries the analysis
    and render seconds and the overall seconds per frame measured
    on the machine that ran it.
"""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .pipeline import (
    NOISE_TAIL_CLAHE,
    NOISE_TAIL_UNSHARP,
    _deconvolve_bgr,
    _denoise_colored,
    _denoise_h_for,
    _estimation_luminance,
    _luminance,
    _scale_psf_params,
    select_psf,
    unsharp_rgb,
)
from .presets import Settings
from .scoring import estimate_noise_sigma

BLUR_FAMILIES = ("motion", "focus", "gaussian")

_FOURCC_BY_SUFFIX = {".mp4": "mp4v", ".mov": "mp4v", ".avi": "MJPG",
                     ".mkv": "mp4v"}


@dataclass
class VideoReport:
    input_path: str = ""
    output_path: str = ""
    frames: int = 0
    fps: float = 0.0
    size: tuple = ()                # (width, height)
    sample_every: int = 0
    samples: list = field(default_factory=list)
    # raw estimates: [(frame_index, family, params)] at native scale
    track: list = field(default_factory=list)
    # per-frame [(family, params)] after smoothing + interpolation
    clip_noise_sigma: float = 0.0   # median of the sampled readings
    method: str = ""
    analysis_seconds: float = 0.0
    render_seconds: float = 0.0
    seconds_per_frame: float = 0.0  # (analysis + render) / frames


# ---------------------------------------------------------------------------
# Estimation on one frame
# ---------------------------------------------------------------------------

def estimate_frame_psf(frame_bgr, settings, max_side=480):
    """Estimate one frame's PSF. Returns (family, params, noise_sigma).

    The frame is downscaled to `max_side` on its long side for the
    sweep (the same staging the browser build uses), the parameters
    are scaled back to native frame scale, and the noise sigma is
    measured on the working copy: downscaling understates grain
    somewhat, so treat the number as a per-clip guide, not a
    per-pixel measurement. Manual pins short-circuit the sweep and
    are already at native scale.
    """
    img = np.asarray(frame_bgr)
    noise_sigma = estimate_noise_sigma(_luminance(img))
    manual = settings.manual_family()
    if manual is not None:
        family = manual if settings.blur_type in ("auto", manual) \
            else settings.blur_type
        return family, settings.manual_params(family), noise_sigma
    h, w = img.shape[:2]
    long_side = max(h, w)
    k = long_side / float(max_side) if long_side > max_side else 1.0
    if k > 1.0:
        work = cv2.resize(
            img, (int(round(w / k)), int(round(h / k))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = img
    est_L = _estimation_luminance(work, noise_sigma)
    family, params, _, _ = select_psf(est_L, settings,
                                      noise_sigma=noise_sigma)
    if k > 1.0:
        params = _scale_psf_params(family, params, k)
    return family, params, noise_sigma


# ---------------------------------------------------------------------------
# The PSF track: smoothing + interpolation
# ---------------------------------------------------------------------------

def _angle_delta(a, b):
    """Signed smallest rotation from a to b on the 180-degree circle
    (motion angles are axial: 0 and 180 are the same direction)."""
    return ((b - a + 90.0) % 180.0) - 90.0


def _lerp_angle(a, b, t):
    return (a + t * _angle_delta(a, b)) % 180.0


def _unwrap_angles(angles):
    out = [float(angles[0])]
    for a in angles[1:]:
        out.append(out[-1] + _angle_delta(out[-1], a))
    return out


def _smooth_run(run):
    """Median-of-3 over the samples of one same-family run (shorter
    runs pass through: two samples cannot be median-filtered, and
    interpolation still bounds their per-frame drift)."""
    if len(run) < 3:
        return list(run)
    keys = list(run[0][2].keys())
    cols = {}
    for key in keys:
        vals = [float(p[key]) for _, _, p in run]
        if key == "angle":
            vals = _unwrap_angles(vals)
        cols[key] = vals
    out = []
    for i, (idx, fam, _params) in enumerate(run):
        lo, hi = max(0, i - 1), min(len(run), i + 2)
        newp = {}
        for key in keys:
            m = float(np.median(cols[key][lo:hi]))
            newp[key] = m % 180.0 if key == "angle" else m
        out.append((idx, fam, newp))
    return out


def _smooth_samples(samples):
    """Median-smooth within maximal runs of one blur family."""
    out = []
    i = 0
    while i < len(samples):
        fam = samples[i][1]
        if fam in BLUR_FAMILIES:
            j = i
            while j < len(samples) and samples[j][1] == fam:
                j += 1
            out.extend(_smooth_run(samples[i:j]))
            i = j
        else:
            out.append(samples[i])
            i += 1
    return out


def _lerp_params(pa, pb, t):
    out = {}
    for key, va in pa.items():
        vb = pb[key]
        out[key] = _lerp_angle(va, vb, t) if key == "angle" \
            else va + t * (vb - va)
    return out


def build_track(samples, total_frames):
    """Per-frame [(family, params)] from raw sampled estimates.

    Samples are median-smoothed within same-family runs, parameters
    are linearly interpolated between bracketing samples of the
    same family, and anything else (family change, or a noise /
    identity neighbour) steps at the midpoint between samples.
    Frames outside the sampled range hold the nearest sample.
    """
    if not samples:
        return [("identity", {})] * total_frames
    sm = _smooth_samples(samples)
    track = []
    for f in range(total_frames):
        prev = None
        nxt = None
        for s in sm:
            if s[0] <= f:
                prev = s
            if s[0] >= f and nxt is None:
                nxt = s
        if prev is not None and prev[0] == f:
            track.append((prev[1], dict(prev[2])))
        elif prev is not None and nxt is not None:
            if prev[1] == nxt[1] and prev[1] in BLUR_FAMILIES:
                t = (f - prev[0]) / float(nxt[0] - prev[0])
                track.append((prev[1], _lerp_params(prev[2], nxt[2], t)))
            else:
                mid = (prev[0] + nxt[0]) / 2.0
                chosen = prev if f < mid else nxt
                track.append((chosen[1], dict(chosen[2])))
        elif prev is not None:
            track.append((prev[1], dict(prev[2])))
        else:
            track.append((nxt[1], dict(nxt[2])))
    return track


# ---------------------------------------------------------------------------
# Per-frame restoration (pipeline pieces, no upscale, no Report)
# ---------------------------------------------------------------------------

def _finish_frame(img, settings, calm=False):
    """The pipeline's finishing tail for one frame: one gentle
    unsharp mask, then CLAHE last. `calm` applies the noise path's
    capped tail values, as the stills pipeline does."""
    amount = settings.unsharp_amount
    clip = settings.clahe_clip
    if calm:
        amount = min(amount, NOISE_TAIL_UNSHARP)
        clip = min(clip, NOISE_TAIL_CLAHE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sharp = unsharp_rgb(rgb, radius=settings.unsharp_radius, amount=amount)
    img = cv2.cvtColor(
        np.clip(sharp, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR
    )
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)),
                        cv2.COLOR_LAB2BGR)


def _restore_frame(frame, family, params, settings, clip_noise_sigma):
    if family == "noise":
        base = frame
        if settings.denoise:
            h_eff = max(settings.noise_h, 1.4 * clip_noise_sigma)
            base = _denoise_colored(frame, h_eff)
        return _finish_frame(base, settings, calm=True)
    base = frame
    if settings.denoise:
        base = _denoise_colored(frame, _denoise_h_for(clip_noise_sigma))
    if family in BLUR_FAMILIES:
        base = _deconvolve_bgr(base, family, params, settings)
    return _finish_frame(base, settings, calm=False)


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------

def _open_capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {path!r}")
    return cap


def enhance_video(input_path, output_path, settings=None,
                  sample_every=12, estimate_max_side=480, progress=None):
    """Enhance a recorded video. Returns a VideoReport.

    input_path / output_path: video files (the output container is
    chosen by suffix: .mp4/.mov/.mkv or .avi). settings: the stills
    Settings (blur type, method, denoise, pins); upscale is ignored
    (frames are written at native size) and blur_type "segmented"
    is rejected (per-region analysis is a stills setting).
    sample_every: estimate one frame in every N. progress, if given,
    is called as progress(frames_done, total_frames) during the
    render pass.
    """
    settings = replace(settings or Settings(), upscale=None)
    if settings.blur_type == "segmented":
        raise ValueError(
            "segmented is a stills setting; video mode restores one "
            "PSF per frame from the smoothed track")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    report = VideoReport(input_path=str(input_path),
                         output_path=str(output_path),
                         sample_every=sample_every,
                         method=settings.method)

    # Pass 1: analysis. Decode once, estimate the sampled frames.
    t0 = time.perf_counter()
    cap = _open_capture(input_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    samples, noises = [], []
    total = 0
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        if total % sample_every == 0:
            family, params, noise = estimate_frame_psf(
                frame, settings, max_side=estimate_max_side)
            samples.append((total, family, params))
            noises.append(noise)
        total += 1
    cap.release()
    if total == 0 or size is None:
        raise ValueError(f"no frames decoded from {input_path!r}")
    report.analysis_seconds = time.perf_counter() - t0
    report.frames, report.fps, report.size = total, fps, size
    report.samples = samples
    report.clip_noise_sigma = float(np.median(noises)) if noises else 0.0
    report.track = build_track(samples, total)

    # Pass 2: render. Decode again, restore frame by frame.
    fourcc_name = _FOURCC_BY_SUFFIX.get(Path(str(output_path)).suffix.lower())
    if fourcc_name is None:
        raise ValueError(
            f"unsupported output suffix for {output_path!r} "
            f"(use one of {sorted(_FOURCC_BY_SUFFIX)})")
    cap = _open_capture(input_path)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*fourcc_name),
        fps, size)
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"could not open video writer for {output_path!r}")
    t0 = time.perf_counter()
    try:
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"frame {i} failed to decode on the render pass "
                    f"(the analysis pass read {total})")
            family, params = report.track[i]
            writer.write(_restore_frame(frame, family, params, settings,
                                        report.clip_noise_sigma))
            if progress is not None:
                progress(i + 1, total)
    finally:
        cap.release()
        writer.release()
    report.render_seconds = time.perf_counter() - t0
    report.seconds_per_frame = (
        (report.analysis_seconds + report.render_seconds) / total)
    return report
