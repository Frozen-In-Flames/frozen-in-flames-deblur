"""The enhancement pipeline, in the order the measurements support:

  1. light denoise          (noise is deconvolution's enemy)
  2. deconvolve at native resolution, on a LINEAR luminance channel
     (the PSF is only known at native scale, and blur is a linear
     process: it must be undone in a linear space, not in LAB L*,
     whose nonlinear remap biases the recovered parameters. Color is
     preserved by scaling RGB by the luminance ratio.)
  3. upscale (Lanczos) to the target long side, if requested
  4. one gentle unsharp mask
  5. CLAHE last, clip limit ~1.8 (strong CLAHE earlier amplifies artifacts)

PSF selection: unless parameters are pinned manually, each blur family
is swept (coarse grid, then a refinement around the best) and every
candidate is scored spectrally against a square center crop (see
scoring.py). In auto mode, motion must additionally clear the null
fingerprint gate or the decision falls to the isotropic families.
The final restoration uses the selected method (default RL with TV).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import unsharp_mask as _skimage_unsharp

from .deconv import deconvolve
from .presets import Settings
from .psf import build_psf
from .scoring import NULL_GATE, SpectrumScorer, score_image

_SWEEP_CROP = 384  # px; sweep runs on a square center crop at native resolution


def unsharp_rgb(rgb, radius, amount):
    """Unsharp mask on a float RGB image, one channel at a time.

    skimage's unsharp_mask(channel_axis=-1) is broken in scikit-image
    0.26: its channel loop slices with _shared.utils.slice_at_axis,
    which builds (slice(None),) * axis and so turns axis=-1 into
    (channel, Ellipsis), selecting row planes instead of channels and
    leaving most of the result array uninitialized (measured: a healthy
    image came back with mean 0.003). Looping the 2D function per
    channel is exactly what channel_axis was supposed to do.
    """
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[:, :, c] = _skimage_unsharp(rgb[:, :, c], radius=radius, amount=amount)
    return out


# --------------------------------------------------------------------------
# PSF sweep
# --------------------------------------------------------------------------

def _coarse_candidates(family):
    if family == "motion":
        return [
            {"length": float(L), "angle": float(a)}
            for L in (9, 15, 21, 27)
            for a in range(0, 180, 15)
        ]
    if family == "focus":
        return [{"radius": float(r)} for r in (2, 3, 4, 5, 6, 8)]
    if family == "gaussian":
        return [{"sigma": float(s)} for s in (0.8, 1.2, 1.6, 2.0, 2.6, 3.2)]
    raise ValueError(family)


def _refine_candidates(family, best):
    out = []
    if family == "motion":
        for dl in (-4, -2, 0, 2, 4):
            L = best["length"] + dl
            if L < 3:
                continue
            for da in (-12, -9, -6, -3, 0, 3, 6, 9, 12):
                out.append({"length": float(L), "angle": float((best["angle"] + da) % 180)})
    elif family == "focus":
        for dr in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
            r = best["radius"] + dr
            if r >= 1.0:
                out.append({"radius": float(r)})
    elif family == "gaussian":
        for ds in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            s = best["sigma"] + ds
            if s >= 0.5:
                out.append({"sigma": float(round(s, 3))})
    return out


def _key(family, params):
    return (family, tuple(sorted((k, round(float(v), 3)) for k, v in params.items())))


def _center_crop(img, size=_SWEEP_CROP):
    """Square center crop (the spectral scorer's chance baseline needs it)."""
    h, w = img.shape[:2]
    side = min(h, w, size)
    y, x = (h - side) // 2, (w - side) // 2
    return img[y:y + side, x:x + side]


def sweep_family(scorer, family):
    """Score coarse + refined candidates for one family.

    Returns (family, params, score, null_match) entries, best first.
    """
    results, seen = [], set()

    def evaluate(params_list):
        best_local = None
        for params in params_list:
            k = _key(family, params)
            if k in seen:
                continue
            seen.add(k)
            psf = build_psf(family, params)
            r = scorer.score(psf, family)
            entry = (family, params, r["score"], r["null_match"])
            results.append(entry)
            if best_local is None or entry[2] > best_local[2]:
                best_local = entry
        return best_local

    best = evaluate(_coarse_candidates(family))
    if best is not None:
        evaluate(_refine_candidates(family, best[1]))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def select_psf(L, settings):
    """Return (family, params, ranked_candidates, null_match_of_winner).

    ranked_candidates covers every family tried, best first, so the
    choice is auditable. Manual parameters short-circuit the sweep.
    Auto mode applies the motion gate: motion wins only if its best
    candidate's null fingerprint match clears NULL_GATE; otherwise the
    best isotropic (focus/gaussian) candidate wins.
    """
    manual = settings.manual_family()
    if manual is not None:
        family = manual if settings.blur_type in ("auto", manual) else settings.blur_type
        return family, settings.manual_params(family), [], None
    scorer = SpectrumScorer(_center_crop(L))
    if settings.blur_type == "auto":
        families = ("motion", "focus", "gaussian")
    else:
        families = (settings.blur_type,)
    per_family = {fam: sweep_family(scorer, fam) for fam in families}
    ranked = sorted(
        (entry for fam in families for entry in per_family[fam]),
        key=lambda r: r[2], reverse=True,
    )
    if settings.blur_type != "auto":
        family, params, _, null_match = ranked[0]
        return family, params, ranked, null_match
    motion_best = per_family["motion"][0]
    if motion_best[3] >= NULL_GATE:
        return motion_best[0], motion_best[1], ranked, motion_best[3]
    iso = [per_family["focus"][0], per_family["gaussian"][0]]
    family, params, _, null_match = max(iso, key=lambda r: r[2])
    return family, params, ranked, null_match


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class Report:
    blur_type: str
    psf_params: dict = field(default_factory=dict)
    method: str = ""
    manual: bool = False
    null_match: object = None  # winner's null fingerprint match (None if manual)
    input_score: float = 0.0
    output_score: float = 0.0
    input_size: tuple = ()
    output_size: tuple = ()
    candidates: list = field(default_factory=list)  # top sweep entries
    steps: list = field(default_factory=list)


def _luminance(bgr_u8):
    """Linear luminance (Rec. 601) of a BGR uint8 image, float in [0, 1]."""
    b, g, r = cv2.split(bgr_u8.astype(np.float32) / 255.0)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _gray_float(bgr_u8):
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


def enhance_image(image, settings=None):
    """Enhance one image. Returns (bgr_uint8_result, Report).

    image: BGR uint8 ndarray, or a path to an image file.
    """
    settings = settings or Settings()
    if isinstance(image, (str, bytes)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"could not load {image!r}")
    else:
        img = np.asarray(image)
        if img.dtype != np.uint8 or img.ndim != 3:
            raise ValueError("expected a BGR uint8 image array or a file path")
        img = img.copy()

    report = Report(blur_type=settings.blur_type, method=settings.method)
    report.input_size = (img.shape[1], img.shape[0])
    report.input_score = score_image(_gray_float(img))

    # 1. light denoise
    if settings.denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        report.steps.append("denoise: fastNlMeans h=3")

    # 2. deconvolution on linear luminance at native resolution;
    #    color follows by the luminance ratio (hue-preserving).
    if settings.blur_type != "lowres":
        L = _luminance(img)
        family, params, ranked, null_match = select_psf(L, settings)
        report.blur_type = family
        report.psf_params = params
        report.manual = settings.manual_family() is not None
        report.null_match = null_match
        report.candidates = ranked[:5]
        psf = build_psf(family, params)
        L2 = deconvolve(L, psf, method=settings.method)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ratio = np.clip(L2 / np.clip(L, 1e-4, None), 0.0, 4.0)
        img = cv2.cvtColor(
            np.clip(rgb * ratio[:, :, None], 0.0, 1.0).astype(np.float32),
            cv2.COLOR_RGB2BGR,
        )
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        report.steps.append(
            f"deconv[{settings.method}] family={family} params={params}"
        )
    else:
        report.steps.append("deconv skipped (lowres)")

    # 3. upscale
    if settings.upscale:
        h, w = img.shape[:2]
        scale = settings.upscale / max(h, w)
        if scale > 1.0:
            img = cv2.resize(
                img, (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )
            report.steps.append(f"upscale x{scale:.2f} -> long side {settings.upscale}")

    # 4. gentle unsharp mask (per channel; see unsharp_rgb for why)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sharp = unsharp_rgb(rgb, radius=settings.unsharp_radius,
                        amount=settings.unsharp_amount)
    img = cv2.cvtColor(
        np.clip(sharp, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR
    )
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    report.steps.append(
        f"unsharp radius={settings.unsharp_radius} amount={settings.unsharp_amount}"
    )

    # 5. CLAHE last, on L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=settings.clahe_clip, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    report.steps.append(f"clahe clip={settings.clahe_clip}")

    report.output_size = (img.shape[1], img.shape[0])
    report.output_score = score_image(_gray_float(img))
    return img, report
