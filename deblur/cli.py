"""Command line interface.

    python -m deblur input output
        --type auto|motion|focus|gaussian|lowres|noise|segmented
        [--length N --angle DEG --radius R --sigma S]
        [--method wiener|rl-tv|rl] [--upscale 3840|none] [--no-denoise]
        [--noise-h H]

Every run prints the chosen blur type, the PSF parameters, the
measured noise, the region estimates in segmented mode, and the
scores, so the run can be audited after the fact.
"""

import argparse
import sys

import cv2

from .pipeline import enhance_image
from .presets import Settings


def build_parser():
    p = argparse.ArgumentParser(
        prog="deblur",
        description="Classical (no AI) deblurring and enhancement: "
                    "motion, focus, gaussian, low-resolution, noise, "
                    "and segmented (per-region) settings.",
    )
    p.add_argument("input", help="input image path")
    p.add_argument("output", help="output image path")
    p.add_argument(
        "--type", default="auto",
        choices=["auto", "motion", "focus", "gaussian", "lowres",
                 "noise", "segmented"],
        help="blur type setting (default: auto = sweep all families; "
             "auto also picks the noise path when grain dominates)",
    )
    p.add_argument("--length", type=float, default=None,
                   help="motion: pin blur length in px (skips the sweep)")
    p.add_argument("--angle", type=float, default=None,
                   help="motion: pin blur angle in degrees (skips the sweep)")
    p.add_argument("--radius", type=float, default=None,
                   help="focus: pin defocus disk radius in px (skips the sweep)")
    p.add_argument("--sigma", type=float, default=None,
                   help="gaussian: pin sigma in px (skips the sweep)")
    p.add_argument("--method", default="rl-tv",
                   choices=["wiener", "rl-tv", "rl"],
                   help="deconvolution method (default: rl-tv)")
    p.add_argument("--upscale", default="3840",
                   help="target long side in px, or 'none' (default: 3840)")
    p.add_argument("--no-denoise", action="store_true",
                   help="skip the restoration denoise pass (blur "
                        "estimation still measures on a denoised copy)")
    p.add_argument("--noise-h", type=float, default=None,
                   help="strong denoise strength for the noise path "
                        "(default: 10)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    settings = Settings(
        blur_type=args.type,
        method=args.method,
        denoise=not args.no_denoise,
        upscale=None if str(args.upscale).lower() == "none" else int(args.upscale),
        length=args.length,
        angle=args.angle,
        radius=args.radius,
        sigma=args.sigma,
    )
    if args.noise_h is not None:
        settings.noise_h = args.noise_h
    img = cv2.imread(args.input)
    if img is None:
        print(f"error: could not load {args.input!r}", file=sys.stderr)
        return 2
    out, report = enhance_image(img, settings)
    if not cv2.imwrite(args.output, out):
        print(f"error: could not write {args.output!r}", file=sys.stderr)
        return 2

    print(f"input        : {args.input}  {report.input_size[0]}x{report.input_size[1]}")
    print(f"blur type    : {report.blur_type}"
          + ("  (manual parameters)" if report.manual else ""))
    if report.psf_params:
        print(f"psf params   : {report.psf_params}")
    if report.noise_sigma is not None:
        print(f"noise sigma  : {report.noise_sigma:.1f} gray levels "
              f"(Immerkaer estimate on the original)")
    if report.segment_mode:
        print(f"segmented    : {report.segment_mode}"
              + (f"  (foreground coverage {report.mask_coverage:.1f}%)"
                 if report.mask_coverage is not None else ""))
        for reg in report.regions:
            print(f"  {reg['name']:10s} : {reg['family']} {reg['params']}  "
                  f"coverage {reg['coverage']:.1f}%")
    print(f"method       : {report.method}")
    if report.candidates:
        print("top candidates (family, params, score, null_match):")
        for fam, params, s, nm in report.candidates:
            print(f"  {fam:9s} {params}  score={s:+.4f}  null={nm:+.3f}")
    print(f"score in/out : {report.input_score:.5f} -> {report.output_score:.5f}"
          + ("  (output scored at output resolution; steepness is "
             "scale-dependent, compare at equal size)"
             if report.output_size != report.input_size else ""))
    print(f"steps        : {'; '.join(report.steps)}")
    print(f"output       : {args.output}  {report.output_size[0]}x{report.output_size[1]}")
    return 0
