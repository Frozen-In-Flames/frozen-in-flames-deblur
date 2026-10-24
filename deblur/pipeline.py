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

v0.3.0 estimation changes (each one answers a measured failure):

  * Estimation runs on a DENOISED COPY whose denoise strength scales
    with the measured noise (Immerkaer sigma on the untouched input),
    never on the raw pixels: heavy sensor noise buries the spectral
    fingerprint (a real motion photo measured null_match 0.030, far
    under the gate, and was misread as mild gaussian softness).
  * The scorer sees the DETAIL LAYER of the estimation copy
    (scoring.py: the image minus its sigma-2 gaussian blur). Flat
    regions and smooth gradients have no detail layer, so only
    edge-rich regions vote on the PSF; content layout fringes live
    at low frequencies the detail layer removes, while the blur
    fingerprint lives in the band it keeps. Measured: a flat
    synthetic with true motion 15/30 (read as focus r8 in v0.2.0)
    and an 800 px gradient fixture (read 13/42 against truth 15/30)
    both read exactly 15/30 through the detail layer. A spatially
    multiplied edge mask was tried first and rejected on
    measurement: the mask's own spectrum convolves into the
    fingerprint band and corrupts the read (15/30 became 9/9 on
    the texture fixture). The edge mask itself lives where it is
    sound: defining and compositing the segmented regions.
  * The motion gate gains SPECIFICITY: the winning motion
    candidate's null match must beat the null match of the longest
    kernel at the same angle by SPEC_MIN. Chance correlation grows
    with kernel length, so spurious motion picks pile up at the
    longest length with no interior peak; a true kernel's match
    peaks at its own length and falls. Measured: true motion peaks
    give specificity +0.11..+0.28; clean, isotropic and noise-only
    images give ~0. An isotropic-evidence cap completes the guard:
    when a focus/gaussian winner explains the radial spectrum
    strongly, a coincidental directional match does not override it.
  * Auto gains a NOISE decision: when no blur family clears its
    evidence bar, the image is treated as noise, not blur: strong
    denoise, no deconvolution. The trigger is either measured noise
    at NOISE_SIGMA_GATE, or negative isotropic evidence (no blur
    model explains the image better than near-identity, the
    signature of smearing that is not a deconvolvable PSF).
    Deconvolving noise was the measured failure on a noise-dominated
    photo (auto chose focus r9 and sharpened the grain).
  * blur_type "segmented" implements the owner's design: an edge
    mask sorts foreground from background, each region's PSF is
    estimated separately, the frame is restored once per region and
    composited through the feathered mask. See segment_image().

v0.4.0 zoom awareness: Settings.zoom states the digital zoom of the
view the image came from. A zoom of z enlarges a center crop, so a
blur of B px on the sensor measures z*B px in the view. Estimates
are measured on the image as handed in (view space); the Report
converts them to sensor space (view / zoom) so readings are
comparable across zoom levels, records the view-space kernel that
was actually restored with, and manual pins are accepted in sensor
space and restored at pin * zoom. See _split_sensor_view().
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from skimage.filters import unsharp_mask as _skimage_unsharp

from .deconv import deconvolve
from .presets import Settings
from .psf import build_psf, motion_psf
from .scoring import (
    NULL_GATE,
    SpectrumScorer,
    detail_layer,
    edge_density,
    estimate_noise_sigma,
    score_image,
)

_SWEEP_CROP = 384  # px; sweep runs on a square center crop at native resolution

# Auto decision bars (stated here, calibrated on the fixture battery
# in TEST-REPORT.md; the battery's measured values sit on both sides
# of each bar with the margins recorded there).
#
# Motion specificity: the winning motion candidate's null match must
# exceed the null match of the longest sweepable kernel (31 px) at
# the same angle by this much. True kernels peak at their own
# length (+0.11..+0.28 measured); chance correlation grows with
# kernel length and peaks AT the longest length (~0 measured).
SPEC_MIN = 0.05
MOTION_MAX_LENGTH = 31.0  # coarse grid top (27) + refinement reach (4)
# Isotropic evidence: the winning focus/gaussian candidate's radial
# profile residual reduction versus a near-identity gaussian
# (sigma 0.5) on the same scorer. At/above ISO_EVIDENCE_MIN the
# isotropic winner is accepted as real blur. At/above
# MOTION_EVIDENCE_CAP the isotropic explanation is strong enough
# that a directional coincidence may not override it (true motion
# measured at most +0.068; true isotropic blur at least +0.124).
ISO_EVIDENCE_MIN = 0.05
MOTION_EVIDENCE_CAP = 0.095
# Noise decision: noise is chosen in auto mode when no blur family
# clears its bar AND (the measured noise sigma, Immerkaer in
# 0..255 gray levels, flat-region weighted, reaches this gate:
# clean and blurred images measure <= 0.1, a sigma-12 noisy image
# measures 12.0) OR (the isotropic evidence is negative: no blur
# model explains the image better than near-identity).
NOISE_SIGMA_GATE = 5.0
# Noise path tail caps: the shared tail (unsharp 1.2, CLAHE 1.8)
# re-amplifies residual grain about 4.5x on a flat patch (measured:
# flat-patch std 0.006 after denoise, 0.027 after the tail, on a
# sigma-15 input of 0.0575). The noise path therefore caps the
# tail's unsharp amount and CLAHE clip at these calmer values when
# it finishes (a caller's lower values are kept).
NOISE_TAIL_UNSHARP = 0.7
NOISE_TAIL_CLAHE = 1.2


def _denoise_colored(img, h):
    """fastNlMeans color denoise at strength h (h also drives the
    color-component strength, the pairing the original recipe used).
    No-ops if the running OpenCV build lacks the function, so the
    engine never dies on a platform quirk; callers that must know
    (the browser staging) feature-detect separately."""
    if not hasattr(cv2, "fastNlMeansDenoisingColored"):
        return img
    return cv2.fastNlMeansDenoisingColored(img, None, float(h), float(h), 7, 21)


def _denoise_h_for(noise_sigma):
    """Denoise strength as a function of measured noise: 3.0 (the
    long-standing light pass) on clean images, ramping linearly to
    10.0 at sigma 16+. One rule, used for the estimation copy and for
    the restoration base, so a noisy photo is measured and restored
    at the strength its noise actually needs."""
    if noise_sigma is None:
        return 3.0
    return float(min(10.0, max(3.0, 3.0 + (noise_sigma - 2.0) / 2.0)))


def _angle_diff(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


_PSF_SCALE_KEYS = {"motion": ("length",), "focus": ("radius",),
                   "gaussian": ("sigma",)}


def _scale_psf_params(family, params, k):
    """One family's parameters scaled by k. Lengths, radii and
    sigmas scale with image size and with zoom; angles never do."""
    out = dict(params)
    for key in _PSF_SCALE_KEYS.get(family, ()):
        if key in out:
            out[key] = float(out[key]) * k
    return out


def _split_sensor_view(settings, family, raw_params):
    """Split one select_psf result into (sensor-space, view-space).

    A swept estimate is measured on the image as handed in, so the
    raw parameters ARE the view-space kernel and the sensor-space
    reading is raw / zoom. A manual pin is stated by the caller in
    sensor space, so the raw parameters are the sensor reading and
    the view-space kernel is raw * zoom. Either way the restoration
    uses the view-space entry. At zoom 1.0 the two are identical.
    """
    if settings.manual_family() is not None:
        return dict(raw_params), _scale_psf_params(
            family, raw_params, settings.zoom)
    return _scale_psf_params(family, raw_params, 1.0 / settings.zoom), \
        dict(raw_params)


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


def select_psf(L, settings, noise_sigma=None):
    """Return (family, params, ranked_candidates, null_match_of_winner).

    L is the estimation luminance: a denoised copy of the input (see
    _estimation_luminance). The scorer is built on the detail layer
    of its center crop, so flat regions do not vote. ranked_candidates
    covers every family tried, best first, so the choice is auditable.
    Manual parameters short-circuit the sweep.

    Auto decision order:
      1. motion, if its best candidate passes all three guards: null
         match at least NULL_GATE, specificity at least SPEC_MIN
         (its match beats the longest kernel at the same angle), and
         the isotropic evidence below MOTION_EVIDENCE_CAP;
      2. the best isotropic (focus/gaussian) candidate, if its radial
         evidence reaches ISO_EVIDENCE_MIN;
      3. "noise", if noise_sigma (measured on the untouched input)
         reaches NOISE_SIGMA_GATE, or the isotropic evidence is
         negative (nothing deconvolvable explains the image);
      4. otherwise "identity": no blur model earned a deconvolution,
         so the caller skips it and applies only the gentle tail.
         (Pre-v0.3.0 the fallback deconvolved with the best isotropic
         candidate anyway; measured on the sharp suite fixture, even
         a sigma-0.5 deconvolution plus the tail oversharpens past
         the identity no-harm band, while the tail alone sits inside
         it. The ranked candidates still record what would have won.)
    Passing noise_sigma=None disables step 3 (region estimation in
    segmented mode uses that: the noise decision is a whole-frame
    judgment, regions always get a PSF; an "identity" region result
    is mapped to the near-identity reference kernel there).
    """
    manual = settings.manual_family()
    if manual is not None:
        family = manual if settings.blur_type in ("auto", manual) else settings.blur_type
        return family, settings.manual_params(family), [], None
    scorer = SpectrumScorer(detail_layer(_center_crop(L)))
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
    iso = [per_family["focus"][0], per_family["gaussian"][0]]
    winner = max(iso, key=lambda r: r[2])
    ref = scorer.score(build_psf("gaussian", {"sigma": 0.5}), "gaussian")
    win = scorer.score(build_psf(winner[0], winner[1]), winner[0])
    evidence = ref["radial_resid"] - win["radial_resid"]
    longest = scorer.score(
        motion_psf(MOTION_MAX_LENGTH, motion_best[1]["angle"]), "motion"
    )
    specificity = motion_best[3] - longest["null_match"]
    if (motion_best[3] >= NULL_GATE and specificity >= SPEC_MIN
            and evidence < MOTION_EVIDENCE_CAP):
        return motion_best[0], motion_best[1], ranked, motion_best[3]
    if evidence >= ISO_EVIDENCE_MIN:
        return winner[0], winner[1], ranked, winner[3]
    if noise_sigma is not None and (noise_sigma >= NOISE_SIGMA_GATE
                                     or evidence < 0.0):
        return "noise", {}, ranked, None
    return "identity", {}, ranked, None


def _estimation_luminance(img_bgr, noise_sigma):
    """The luminance PSF estimation is allowed to see: a denoised
    copy of the input (strength scaled to the measured noise, see
    _denoise_h_for), converted to linear luminance. Restoration never
    uses this copy; it works on the original pixels. The estimation
    denoise runs regardless of the restoration denoise toggle:
    measuring through noise was the measured failure, and the toggle
    governs what happens to the output, not how the blur is read."""
    est = _denoise_colored(img_bgr, _denoise_h_for(noise_sigma))
    return _luminance(est)


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
    # v0.3.0 additions
    noise_sigma: object = None  # Immerkaer sigma of the input, gray levels
    segment_mode: str = ""      # "" | "per-region" | "fallback"
    mask_coverage: object = None  # foreground coverage %, segmented only
    regions: list = field(default_factory=list)  # per-region estimate dicts
    # v0.4.0 additions. psf_params (and regions[].params) are stated
    # in SENSOR space when zoom != 1 (see _split_sensor_view);
    # view_psf_params records the kernel the restoration of the
    # handed-in image actually used (identical at zoom 1.0).
    zoom: float = 1.0
    view_psf_params: dict = field(default_factory=dict)


def _luminance(bgr_u8):
    """Linear luminance (Rec. 601) of a BGR uint8 image, float in [0, 1]."""
    b, g, r = cv2.split(bgr_u8.astype(np.float32) / 255.0)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _gray_float(bgr_u8):
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


def _deconvolve_bgr(base_bgr, family, params, settings):
    """One full-frame restoration of a BGR image with one PSF:
    deconvolve the linear luminance, carry color by the luminance
    ratio (hue-preserving). This is pipeline step 2 as a unit, so
    segmented mode can run it once per region and composite."""
    L = _luminance(base_bgr)
    L2 = deconvolve(L, build_psf(family, params), method=settings.method)
    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ratio = np.clip(L2 / np.clip(L, 1e-4, None), 0.0, 4.0)
    out = cv2.cvtColor(
        np.clip(rgb * ratio[:, :, None], 0.0, 1.0).astype(np.float32),
        cv2.COLOR_RGB2BGR,
    )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Segmented restoration: the owner's edge-mask design
#
# "There should be some sort of edge mask that you use to sort
# background from foreground to deblur both."
#
# Algorithm (classical throughout; no learned models):
#   1. Tile the estimation luminance (side ~160 px, fewer on small
#      images). Classify each tile by spectral scoring over the
#      coarse candidate grids (all three families), on the tile's
#      detail layer. Tiles whose mean edge density is too low to
#      carry evidence are marked weak and do not vote.
#   2. Region growing: build the graph of confident tiles joined to
#      4-neighbours whose parameters are similar (same family; motion
#      within 15 degrees and 6 px; focus within 2 px; gaussian
#      within 1.2 sigma). Connected components of that graph are the
#      candidate regions. The two largest components win; every
#      remaining tile joins the region its own classification matches
#      (when it matches exactly one), else the nearest component.
#   3. Foreground vs background: the region whose tiles touch the
#      image border least is the foreground (subjects are interior,
#      backgrounds reach the frame edge). Ties go to the smaller
#      coverage.
#   4. Each region's PSF is re-estimated properly: a full sweep on
#      the region's single most evidence-bearing tile. (A masked
#      full-frame composite was tried and measured to fail: the
#      local-mean fill keeps the other region's mid-frequency
#      fingerprint, and any usefully wide crop mixes both regions,
#      whose fingerprints cancel. A pure tile carries one region's
#      blur only.)
#   5. If the two estimates agree within the suite tolerances, the
#      blur is effectively uniform: fall back to one global
#      restoration and say so. Otherwise restore the full frame once
#      per region and composite through the feathered mask (tile
#      label map upsampled, gaussian feather of a sixth of a tile),
#      so the seam is a blend, not a cut.
# --------------------------------------------------------------------------

_SEG_MIN_DIM = 192       # below this, an image is too small to segment
_SEG_TILE_CAP = 160      # px, tile side on large images
_SEG_MAX_TILES = 120     # grid budget; tile side grows past this


def _tile_side(h, w):
    side = int(min(_SEG_TILE_CAP, max(96, min(h, w) // 3)))
    while ((h + side - 1) // side) * ((w + side - 1) // side) > _SEG_MAX_TILES:
        side = int(side * 1.25) + 1
    return side


def _params_similar(fam_a, pa, fam_b, pb):
    if fam_a != fam_b:
        return False
    if fam_a == "motion":
        return _angle_diff(pa["angle"], pb["angle"]) <= 15.0 \
            and abs(pa["length"] - pb["length"]) <= 6.0
    if fam_a == "focus":
        return abs(pa["radius"] - pb["radius"]) <= 2.0
    return abs(pa["sigma"] - pb["sigma"]) <= 1.2


def _params_agree(fam_a, pa, fam_b, pb):
    """The suite's recovery tolerances: inside them, two estimates
    are the same blur for restoration purposes."""
    if fam_a != fam_b:
        return False
    if fam_a == "motion":
        return _angle_diff(pa["angle"], pb["angle"]) <= 10.0 \
            and abs(pa["length"] - pb["length"]) <= 4.0
    if fam_a == "focus":
        return abs(pa["radius"] - pb["radius"]) <= 1.5
    return abs(pa["sigma"] - pb["sigma"]) <= 1.0


def _classify_tile(tile_L):
    """Coarse spectral classification of one tile.
    Returns (family, params, score) of the best coarse candidate."""
    scorer = SpectrumScorer(detail_layer(tile_L))
    best = None
    for fam in ("motion", "focus", "gaussian"):
        for params in _coarse_candidates(fam):
            r = scorer.score(build_psf(fam, params), fam)
            if best is None or r["score"] > best[2]:
                best = (fam, params, r["score"])
    return best


def segment_image(est_L, settings):
    """Segment the estimation luminance into two blur regions.

    Returns a dict:
      mode: "per-region" | "fallback"
      mask: soft foreground mask, float [0, 1], full frame (or None)
      regions: [{"name", "family", "params", "coverage"}] with the
               per-region estimates (both modes, when segmentation
               produced two regions)
      reason: why fallback was chosen, when it was
    The caller restores: per-region mode composites two restorations
    through mask; fallback restores once with the global estimate.
    """
    h, w = est_L.shape
    if min(h, w) < _SEG_MIN_DIM:
        return {"mode": "fallback", "mask": None, "regions": [],
                "reason": "image too small to segment"}
    side = _tile_side(h, w)
    rows = max(2, int(round(h / side)))
    cols = max(2, int(round(w / side)))

    def spans(n, count):
        edges = [int(round(i * n / count)) for i in range(count + 1)]
        return [(edges[i], edges[i + 1]) for i in range(count)]

    row_spans, col_spans = spans(h, rows), spans(w, cols)
    density = edge_density(est_L)

    # 1. classify tiles; weak (evidence-free) tiles do not vote
    tiles = {}
    confs = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = row_spans[r]
            x0, x1 = col_spans[c]
            conf = float(density[y0:y1, x0:x1].mean())
            confs.append(conf)
            tiles[(r, c)] = {"conf": conf, "cls": None}
    conf_floor = 0.5 * float(np.percentile(confs, 90)) if confs else 0.0
    for (r, c), t in tiles.items():
        if t["conf"] < conf_floor:
            continue
        y0, y1 = row_spans[r]
        x0, x1 = col_spans[c]
        t["cls"] = _classify_tile(est_L[y0:y1, x0:x1])

    # 2. region growing over the similarity graph
    confident = [k for k, t in tiles.items() if t["cls"] is not None]
    if len(confident) < 2:
        return {"mode": "fallback", "mask": None, "regions": [],
                "reason": "fewer than two evidence-bearing tiles"}
    parent = {k: k for k in confident}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (r, c) in confident:
        for nb in ((r + 1, c), (r, c + 1)):
            if nb in parent:
                a, b = tiles[(r, c)]["cls"], tiles[nb]["cls"]
                if _params_similar(a[0], a[1], b[0], b[1]):
                    parent[find((r, c))] = find(nb)
    comps = {}
    for k in confident:
        comps.setdefault(find(k), []).append(k)
    comp_list = sorted(comps.values(), key=len, reverse=True)
    comp_list = [c for c in comp_list if len(c) >= 2] or comp_list[:1]
    if len(comp_list) < 2:
        return {"mode": "fallback", "mask": None, "regions": [],
                "reason": "tiles agree on one blur region"}
    region_tiles = comp_list[:2]

    # Every remaining tile joins a region: by its own measured blur
    # when its classification matches exactly one region's
    # representative classification (a tile's evidence outranks its
    # address), otherwise by nearest region tile on the grid.
    label = {}
    for i, comp in enumerate(region_tiles):
        for k in comp:
            label[k] = i
    reps = []
    for comp in region_tiles:
        rep_tile = max(comp, key=lambda k: tiles[k]["conf"])
        reps.append(tiles[rep_tile]["cls"])
    for r in range(rows):
        for c in range(cols):
            if (r, c) in label:
                continue
            cls = tiles[(r, c)]["cls"]
            if cls is not None:
                sim = [
                    _params_similar(cls[0], cls[1], rep[0], rep[1])
                    if rep is not None else False
                    for rep in reps
                ]
                if sim.count(True) == 1:
                    label[(r, c)] = sim.index(True)
                    continue
            best_i, best_d = 0, None
            for i, comp in enumerate(region_tiles):
                d = min(abs(r - kr) + abs(c - kc) for kr, kc in comp)
                if best_d is None or d < best_d:
                    best_i, best_d = i, d
            label[(r, c)] = best_i

    # 3. foreground = the region touching the border least
    def border_fraction(comp):
        n = sum(1 for (r, c) in comp
                if r == 0 or c == 0 or r == rows - 1 or c == cols - 1)
        return n / len(comp)

    order = sorted(range(2),
                   key=lambda i: (border_fraction(region_tiles[i]),
                                  len(region_tiles[i])))
    fg_idx = order[0]

    # 4. per-region PSF estimation from each region's best tile
    soft_src = np.zeros((rows, cols), np.float64)
    for (r, c), i in label.items():
        soft_src[r, c] = 1.0 if i == fg_idx else 0.0
    regions = []
    region_settings = replace(settings, blur_type="auto")

    def purity(k):
        """Fraction of a tile's in-grid neighbours sharing its label:
        interior tiles are pure, boundary tiles straddle both blurs."""
        r, c = k
        same = tot = 0
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in label:
                tot += 1
                same += label[nb] == label[k]
        return same / tot if tot else 1.0

    for i, name in ((fg_idx, "foreground"), (1 - fg_idx, "background")):
        # The region's PSF is estimated from one pure tile at full
        # sweep. Two measured reasons not to estimate from a masked
        # full-frame composite: the local-mean fill keeps the other
        # region's mid-frequency fingerprint (a sigma-16 fill does
        # not erase it), and any crop wide enough to be useful mixes
        # both regions, whose fingerprints cancel (a 59%-pure
        # background crop measured null_match -0.01 for its true
        # kernel). A pure tile carries one region's blur only. The
        # anchor is the region's purest tile whose coarse
        # classification agrees with the region component's
        # representative classification (a boundary tile straddling
        # both blurs reads as mush even when it is confident).
        comp = region_tiles[i]
        rep = max(comp, key=lambda k: tiles[k]["conf"])
        rep_cls = tiles[rep]["cls"]
        labeled = [k for k, v in label.items() if v == i]
        agreeing = [
            k for k in labeled
            if tiles[k]["cls"] is not None and rep_cls is not None
            and _params_similar(tiles[k]["cls"][0], tiles[k]["cls"][1],
                                rep_cls[0], rep_cls[1])
        ]
        pool = agreeing or [k for k in labeled if tiles[k]["cls"] is not None] \
            or labeled
        anchor = max(pool, key=lambda k: (purity(k), tiles[k]["conf"]))
        ar, ac = anchor
        y0, y1 = row_spans[ar]
        x0, x1 = col_spans[ac]
        fam, params, _, _ = select_psf(
            est_L[y0:y1, x0:x1], region_settings, noise_sigma=None,
        )
        if fam == "identity":
            # A region with no blur evidence is restored with the
            # near-identity reference kernel (a no-op restoration),
            # so regions always carry a real PSF.
            fam, params = "gaussian", {"sigma": 0.5}
        regions.append({"name": name, "family": fam, "params": params,
                        "coverage": 0.0})
    mask = cv2.resize(soft_src, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), side / 6.0), 0.0, 1.0)
    fg_cov = float(mask.mean() * 100.0)
    for reg in regions:
        reg["coverage"] = fg_cov if reg["name"] == "foreground" else 100.0 - fg_cov

    a, b = regions
    if _params_agree(a["family"], a["params"], b["family"], b["params"]):
        return {"mode": "fallback", "mask": None, "regions": regions,
                "reason": "region estimates agree within tolerance"}
    return {"mode": "per-region", "mask": mask, "regions": regions,
            "reason": ""}


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

    report = Report(blur_type=settings.blur_type, method=settings.method,
                    zoom=settings.zoom)
    report.input_size = (img.shape[1], img.shape[0])
    report.input_score = score_image(_gray_float(img))
    # Noise is measured on the untouched input, before any denoise
    # can erase the very signal being measured.
    report.noise_sigma = estimate_noise_sigma(_luminance(img))
    orig = img

    # The noise path stands alone: strong denoise, no deconvolution
    # (there is no blur to invert), then the shared tail. Auto
    # reaches it through select_psf's decision. The strength is the
    # configured noise_h as a floor, raised to 1.4x the measured
    # sigma when the measured noise is heavier: fastNlMeans only
    # bites when h is comparable to the noise (measured on the
    # suite scene: sigma 15 needs h~20 to clear a flat patch; h=10
    # leaves it nearly untouched, while on the owner's noise photo
    # h=10 was the recipe that worked, hence the floor).
    def run_noise_path():
        h_eff = settings.noise_h
        if report.noise_sigma is not None:
            h_eff = max(h_eff, 1.4 * report.noise_sigma)
        out = _denoise_colored(orig, h_eff)
        report.steps.append(
            f"denoise: fastNlMeans h={h_eff:g} (noise path, "
            f"measured sigma {report.noise_sigma:.1f})"
        )
        report.steps.append("deconv skipped (noise)")
        return out

    def finish_noise_path():
        calm = replace(
            settings,
            unsharp_amount=min(settings.unsharp_amount, NOISE_TAIL_UNSHARP),
            clahe_clip=min(settings.clahe_clip, NOISE_TAIL_CLAHE),
        )
        if (calm.unsharp_amount, calm.clahe_clip) != (
                settings.unsharp_amount, settings.clahe_clip):
            report.steps.append(
                f"noise path tail calmed: unsharp amount "
                f"{calm.unsharp_amount:g}, clahe clip {calm.clahe_clip:g}"
            )
        return _finish_tail(run_noise_path(), calm, report)

    if settings.blur_type == "noise":
        return finish_noise_path()

    if settings.blur_type == "identity":
        # Pinned by the browser staging when auto found no blur
        # evidence on the working copy: gentle tail only.
        report.steps.append("deconv skipped (identity: no blur evidence)")
        return _finish_tail(img, settings, report)

    # 1. restoration base denoise, strength scaled to measured
    # noise. Lazy: paths that end up not deconvolving (noise,
    # identity) never pay for it, and their step log stays truthful.
    _base = []

    def get_base():
        if not _base:
            b = orig
            if settings.denoise:
                h_base = _denoise_h_for(report.noise_sigma)
                b = _denoise_colored(b, h_base)
                report.steps.append(f"denoise: fastNlMeans h={h_base:g}")
            _base.append(b)
        return _base[0]

    # 2. deconvolution on linear luminance at native resolution;
    #    color follows by the luminance ratio (hue-preserving).
    #    Estimation reads a separate denoised copy's detail layer.
    if settings.blur_type == "lowres":
        img = get_base()
        report.steps.append("deconv skipped (lowres)")
    elif settings.blur_type == "segmented":
        est_L = _estimation_luminance(orig, report.noise_sigma)
        seg = segment_image(est_L, settings)
        report.segment_mode = seg["mode"]
        # Region estimates are measured on the image as handed in
        # (view space); the report states them in sensor space.
        report.regions = [
            {**r, "params": _scale_psf_params(
                r["family"], r["params"], 1.0 / settings.zoom)}
            for r in seg["regions"]
        ]
        report.mask_coverage = (
            next((r["coverage"] for r in seg["regions"]
                  if r["name"] == "foreground"), None)
            if seg["regions"] else None
        )
        if seg["mode"] == "per-region":
            fg = next(r for r in seg["regions"] if r["name"] == "foreground")
            bg = next(r for r in seg["regions"] if r["name"] == "background")
            base = get_base()
            rest_fg = _deconvolve_bgr(base, fg["family"], fg["params"], settings)
            rest_bg = _deconvolve_bgr(base, bg["family"], bg["params"], settings)
            m = seg["mask"][:, :, None]
            img = np.clip(
                m * rest_fg.astype(np.float32)
                + (1.0 - m) * rest_bg.astype(np.float32),
                0, 255,
            ).astype(np.uint8)
            report.psf_params = _scale_psf_params(
                fg["family"], fg["params"], 1.0 / settings.zoom)
            report.view_psf_params = fg["params"]
            report.steps.append(
                f"deconv[{settings.method}] segmented per-region: "
                f"foreground={fg['family']} {fg['params']}, "
                f"background={bg['family']} {bg['params']}, "
                f"mask feathered, foreground coverage "
                f"{report.mask_coverage:.1f}%"
                + (f" (region params view-space; report states "
                   f"sensor-space at zoom {settings.zoom:g})"
                   if settings.zoom != 1.0 else "")
            )
        else:
            family, params, ranked, null_match = select_psf(
                est_L, replace(settings, blur_type="auto"),
                noise_sigma=report.noise_sigma,
            )
            if settings.zoom != 1.0 and settings.manual_family() is None:
                ranked = [
                    (fam, _scale_psf_params(fam, p, 1.0 / settings.zoom),
                     s, nm)
                    for fam, p, s, nm in ranked
                ]
            if family == "noise":
                report.blur_type = "noise"
                report.candidates = ranked[:5]
                return finish_noise_path()
            if family == "identity":
                report.blur_type = "identity"
                report.candidates = ranked[:5]
                report.steps.append(
                    "deconv skipped (identity: no blur evidence; "
                    f"segmented fallback: {seg['reason']})"
                )
                return _finish_tail(img, settings, report)
            params_sensor, params_view = _split_sensor_view(
                settings, family, params)
            report.blur_type = family
            report.psf_params = params_sensor
            report.view_psf_params = params_view
            report.null_match = null_match
            report.candidates = ranked[:5]
            img = _deconvolve_bgr(get_base(), family, params_view, settings)
            report.steps.append(
                f"deconv[{settings.method}] family={family} "
                f"params={params_view} "
                f"(segmented fallback: {seg['reason']})"
            )
    else:
        est_L = _estimation_luminance(orig, report.noise_sigma)
        family, params, ranked, null_match = select_psf(
            est_L, settings, noise_sigma=report.noise_sigma
        )
        report.manual = settings.manual_family() is not None
        report.null_match = null_match
        if settings.zoom != 1.0 and settings.manual_family() is None:
            ranked = [
                (fam, _scale_psf_params(fam, p, 1.0 / settings.zoom), s, nm)
                for fam, p, s, nm in ranked
            ]
        report.candidates = ranked[:5]
        if family == "noise":
            report.blur_type = "noise"
            return finish_noise_path()
        if family == "identity":
            report.blur_type = "identity"
            report.steps.append(
                "deconv skipped (identity: no blur evidence)"
            )
            return _finish_tail(img, settings, report)
        params_sensor, params_view = _split_sensor_view(
            settings, family, params)
        report.blur_type = family
        report.psf_params = params_sensor
        report.view_psf_params = params_view
        img = _deconvolve_bgr(get_base(), family, params_view, settings)
        step = f"deconv[{settings.method}] family={family} params={params_view}"
        if settings.zoom != 1.0:
            step += f" (sensor-space {params_sensor}, zoom {settings.zoom:g})"
        report.steps.append(step)

    return _finish_tail(img, settings, report)


def _finish_tail(img, settings, report):
    """Pipeline steps 3-5, shared by every path: upscale, one gentle
    unsharp mask, CLAHE last; then the output measurements."""

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
