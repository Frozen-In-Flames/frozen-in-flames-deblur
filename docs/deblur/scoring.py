"""Auditable scoring: spectral fingerprints for PSF selection,
edge steepness for before/after reporting.

How the sweep scores a candidate PSF (all classical signal processing):

* Null fingerprint match. A blur kernel's transfer function has nulls:
  motion blur carves parallel null stripes into the spectrum, a defocus
  disk carves rings, a gaussian carves none. For the candidate, build
  its null map (where its log spectrum dips below its own radial mean)
  and cosine-match it against the observed image's null map. Raw
  matching rewards dense kernels (a long motion kernel's many stripes
  overlap any structure by chance), so the score subtracts a chance
  baseline: the same match with the candidate's null map displaced
  (rotated 90 degrees for the isotropic families, whose maps are
  rotation-symmetric; rolled half an array for motion, whose stripes
  must pass through the spectrum center to be real). Only alignment
  that beats its own chance level counts.

* Radial profile residual. Radially average the log spectrum of the
  observation and of the candidate; subtract; remove the best-fit line
  in log-frequency (a natural image spectrum is roughly a power law, so
  what remains after the correct kernel is removed should be a line);
  the weighted RMS of the remainder measures how badly the candidate's
  strength/shape misses. This carries most of the parameter-strength
  discrimination.

    candidate score = null_match - radial_resid

* Motion gate (auto mode only). Motion is chosen in auto mode only if
  the best motion candidate's null_match clears NULL_GATE: a real
  motion fingerprint or nothing. Below the gate, auto decides between
  focus and gaussian by score. Calibrated on the synthetic suite:
  true motion scores 0.23-0.32, the strongest isotropic spoilers reach
  0.11-0.14. Mild real-world motion blur can fall under the gate; pin
  --type motion when you know.

Why not score restorations by sharpness (the obvious approach): it was
tried first and measured to fail. A wrong-PSF deconvolution manufactures
ghost edges that every sharpness metric rewards, and weak kernels win
any fidelity-to-observation measure by barely changing the image.
The numbers are in TEST-REPORT.md.

score_image (edge steepness, sum|grad|^3 / sum|grad|^2) is used only
for before/after reporting on the same image, where it is honest:
a step edge of height h smeared over w pixels contributes ~h/w, so
steepness tracks how concentrated the gradients are.
"""

import cv2
import numpy as np

NULL_GATE = 0.18
_BORDER = 16  # px cropped before steepness: border effects live there


# ---------------------------------------------------------------------------
# Reporting score
# ---------------------------------------------------------------------------

def _prep(gray):
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim == 3:
        g = cv2.cvtColor(g.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if g.shape[0] > 2 * _BORDER + 8 and g.shape[1] > 2 * _BORDER + 8:
        g = g[_BORDER:-_BORDER, _BORDER:-_BORDER]
    return g


def edge_steepness(gray):
    """sum(|grad|^3) / sum(|grad|^2): gradient concentration."""
    g = _prep(gray)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return float((mag ** 3).sum() / ((mag ** 2).sum() + 1e-12))


def score_image(gray):
    """Sharpness of a single image (edge steepness). Higher = sharper."""
    return edge_steepness(gray)


def score_breakdown(gray):
    g = _prep(gray)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return {
        "steepness": float((mag ** 3).sum() / ((mag ** 2).sum() + 1e-12)),
        "grad_energy": float(np.mean(mag)),
        "score": edge_steepness(gray),
    }


# ---------------------------------------------------------------------------
# Candidate scoring (spectral)
# ---------------------------------------------------------------------------

def _cosine(a, b, w):
    aw, bw = a * w, b * w
    d = np.sqrt((aw * aw).sum() * (bw * bw).sum())
    return float((aw * bw).sum() / d) if d > 0 else 0.0


class SpectrumScorer:
    """Scores candidate PSFs against one observed single-channel image.

    The observed channel must be a square float array (the pipeline
    center-crops); the 90-degree chance baseline assumes it.
    """

    def __init__(self, observed):
        img = np.asarray(observed, dtype=np.float64)
        if img.shape[0] != img.shape[1]:
            side = min(img.shape)
            y, x = (img.shape[0] - side) // 2, (img.shape[1] - side) // 2
            img = img[y:y + side, x:x + side]
        self.shape = img.shape
        h, w = img.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
        self.power = F.real ** 2 + F.imag ** 2
        fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
        self.r = np.hypot(fx, fy)
        outer = self.r >= 0.42
        self.floor = float(np.median(self.power[outer])) if np.any(outer) \
            else float(np.median(self.power))
        self.cells = (self.r >= 0.02) & (self.r <= 0.40)
        self.redges = np.logspace(np.log10(0.008), np.log10(0.45), 46)
        self.rbin = np.clip(np.digitize(self.r, self.redges) - 1, 0, 44)
        self.rmask = (self.r >= 0.008) & (self.r <= 0.45)
        self.rx = np.log10((self.redges[:-1] + self.redges[1:]) / 2)
        self.logp = np.log10(self.power + self.floor)
        self.wcell = np.sqrt(self.power) * self.cells
        self.obs_null = self._null_map(self.logp)
        self._radial_w = self._radial_weights()

    # -- pieces -------------------------------------------------------------
    def _psf_power(self, psf):
        kh, kw = psf.shape
        pad = np.zeros(self.shape, np.float64)
        pad[:kh, :kw] = psf
        pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
        H = np.fft.fftshift(np.fft.fft2(pad))
        return H.real ** 2 + H.imag ** 2

    def _radial_profile(self, logp, mask=None):
        mask = self.rmask if mask is None else mask
        sums = np.bincount(self.rbin[mask], weights=logp[mask], minlength=45)
        cnts = np.bincount(self.rbin[mask], minlength=45)
        return sums / np.maximum(cnts, 1)

    def _radial_weights(self):
        w = np.bincount(self.rbin[self.rmask], weights=self.power[self.rmask],
                        minlength=45)
        return np.sqrt(w / w.max())

    def _null_map(self, logp):
        sums = np.bincount(self.rbin.ravel(), weights=logp.ravel(), minlength=45)
        cnts = np.bincount(self.rbin.ravel(), minlength=45)
        radial_mean = (sums / np.maximum(cnts, 1))[self.rbin]
        return np.clip(radial_mean - logp, 0, None)

    def _radial_resid(self, cand_logp):
        resid = self._radial_profile(self.logp) - self._radial_profile(cand_logp)
        w = self._radial_w
        A = np.vstack([np.ones_like(self.rx), self.rx]).T
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, resid * w, rcond=None)
        err = (resid - A @ coef) * w
        return float(np.sqrt(np.sum(err ** 2) / np.sum(w ** 2)))

    # -- public ---------------------------------------------------------------
    def score(self, psf, family):
        """Return dict(score, null_match, radial_resid) for one candidate."""
        cand_logp = np.log10(self._psf_power(psf) + self.floor)
        cn = self._null_map(cand_logp)
        aligned = _cosine(self.obs_null, cn, self.wcell)
        if family == "motion":
            h, w = cn.shape
            chance_map = np.roll(cn, shift=(h // 2, w // 2), axis=(0, 1))
        else:
            chance_map = np.rot90(cn)
        null_match = aligned - _cosine(self.obs_null, chance_map, self.wcell)
        radial = self._radial_resid(cand_logp)
        return {
            "score": null_match - radial,
            "null_match": null_match,
            "radial_resid": radial,
        }


def score_candidate(observed, psf, family="gaussian", balance=None):
    """One-off candidate score (builds a SpectrumScorer for the call)."""
    return SpectrumScorer(observed).score(psf, family)
