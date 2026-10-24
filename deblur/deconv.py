"""Single-channel deconvolution: Wiener, Richardson-Lucy, RL with TV.

All methods deconvolve one float channel in [0, 1]. The channel is
reflect-padded before deconvolution and cropped after, because FFT-based
deconvolution assumes wrap-around borders and otherwise paints ghost
bands at the edges.
"""

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle, richardson_lucy, wiener


def _pad(channel, psf):
    pad = max(psf.shape) // 2 + 2
    padded = cv2.copyMakeBorder(
        channel, pad, pad, pad, pad, cv2.BORDER_REFLECT
    )
    return padded, pad


def _finish(padded, pad):
    out = padded[pad:-pad, pad:-pad] if pad else padded
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def deconv_wiener(channel, psf, balance=0.01):
    """Wiener deconvolution; balance trades sharpness against noise."""
    padded, pad = _pad(channel, psf)
    out = wiener(padded, psf, balance, clip=False)
    return _finish(out, pad)


def deconv_rl(channel, psf, num_iter=25):
    """Plain Richardson-Lucy. Fast, but amplifies noise and rings."""
    padded, pad = _pad(channel, psf)
    out = richardson_lucy(np.clip(padded, 0.0, 1.0), psf, num_iter=num_iter, clip=False)
    return _finish(out, pad)


def deconv_rl_tv(channel, psf, num_iter=15, tv_weight=0.01):
    """Richardson-Lucy in chunks with TV denoising between chunks.

    The TV passes suppress the ringing plain RL grows, at some cost in
    fine texture. Default restoration method for the pipeline.
    """
    padded, pad = _pad(channel, psf)
    out = np.clip(padded, 0.0, 1.0)
    step = 5
    done = 0
    while done < num_iter:
        n = min(step, num_iter - done)
        out = richardson_lucy(out, psf, num_iter=n, clip=False)
        out = np.clip(out, 0.0, 1.0)
        done += n
        if done < num_iter:
            out = denoise_tv_chambolle(out, weight=tv_weight)
    return _finish(out, pad)


METHODS = ("wiener", "rl-tv", "rl")


def deconvolve(channel, psf, method="rl-tv", **kwargs):
    """Dispatch to a deconvolution method by name."""
    if method == "wiener":
        return deconv_wiener(channel, psf, **kwargs)
    if method == "rl":
        return deconv_rl(channel, psf, **kwargs)
    if method == "rl-tv":
        return deconv_rl_tv(channel, psf, **kwargs)
    raise ValueError(f"unknown deconvolution method: {method!r} (use one of {METHODS})")
