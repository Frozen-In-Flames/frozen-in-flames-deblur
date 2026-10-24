"""Command line interface.

Stills:

    python -m deblur input output
        --type auto|motion|focus|gaussian|lowres|noise|segmented
        [--length N --angle DEG --radius R --sigma S]
        [--method wiener|rl-tv|rl] [--upscale 3840|none] [--no-denoise]
        [--noise-h H] [--zoom Z]

Recorded video (the camera-glasses recorded-content mode; the PSF
is estimated on sampled frames, temporally smoothed, and each frame
is restored with the fast method by default):

    python -m deblur video input.mp4 output.mp4
        [--type auto|motion|focus|gaussian|lowres|noise]
        [--method wiener|rl-tv|rl] [--sample-every N]
        [--estimate-max-side PX] [--length N --angle DEG ...]
        [--no-denoise] [--noise-h H]

Every run prints the chosen blur type, the PSF parameters, the
measured noise, the region estimates in segmented mode, and the
scores, so the run can be audited after the fact.
"""

import argparse
import sys

import cv2

from .pipeline import enhance_image
from .presets import Settings
from .video import enhance_video


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
    p.add_argument("--zoom", type=float, default=None,
                   help="digital zoom of the view this image came "
                        "from (default: 1). Reported parameters are "
                        "converted to sensor space (view / zoom); "
                        "pinned parameters are read as sensor space "
                        "and restored at pin x zoom")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "video":
        return _video_main(argv[1:])
    return _image_main(argv)


def _image_main(argv):
    args = build_parser().parse_args(argv)
    if args.zoom is not None and not (args.zoom > 0):
        print(f"error: --zoom must be > 0, got {args.zoom}", file=sys.stderr)
        return 2
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
    if args.zoom is not None:
        settings.zoom = args.zoom
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
    if report.zoom != 1.0:
        print(f"zoom         : {report.zoom:g}x (psf params above are "
              f"sensor-space; restoration used view-space "
              f"{report.view_psf_params})")
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


# ---------------------------------------------------------------------------
# Recorded video
# ---------------------------------------------------------------------------

def build_video_parser():
    p = argparse.ArgumentParser(
        prog="deblur video",
        description="Deblur a recorded video (for example a clip "
                    "from camera glasses) frame by frame: the PSF "
                    "is estimated on sampled frames, temporally "
                    "smoothed into a track, and every frame is "
                    "restored with its track parameters.",
    )
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output video path (.mp4, .mov, "
                                  ".mkv or .avi)")
    p.add_argument(
        "--type", default="auto",
        choices=["auto", "motion", "focus", "gaussian", "lowres",
                 "noise", "segmented"],
        help="blur type setting (default: auto; segmented is a "
             "stills setting and is rejected for video)",
    )
    p.add_argument("--length", type=float, default=None,
                   help="motion: pin blur length in px (skips "
                        "estimation; the track is constant)")
    p.add_argument("--angle", type=float, default=None,
                   help="motion: pin blur angle in degrees")
    p.add_argument("--radius", type=float, default=None,
                   help="focus: pin defocus disk radius in px")
    p.add_argument("--sigma", type=float, default=None,
                   help="gaussian: pin sigma in px")
    p.add_argument("--method", default="wiener",
                   choices=["wiener", "rl-tv", "rl"],
                   help="per-frame deconvolution method (default: "
                        "wiener, the fast method; rl-tv is the "
                        "stills default and costs far more per frame)")
    p.add_argument("--sample-every", type=int, default=12,
                   help="estimate the PSF on one frame in every N "
                        "(default: 12); the track interpolates "
                        "between samples")
    p.add_argument("--estimate-max-side", type=int, default=480,
                   help="long side of the working copy each sampled "
                        "frame is estimated on (default: 480)")
    p.add_argument("--no-denoise", action="store_true",
                   help="skip the per-frame denoise pass")
    p.add_argument("--noise-h", type=float, default=None,
                   help="strong denoise strength for frames the "
                        "estimator calls noise (default: 10)")
    return p


def _video_main(argv):
    args = build_video_parser().parse_args(argv)
    if args.type == "segmented":
        print("error: segmented is a stills setting; video mode "
              "restores one PSF per frame", file=sys.stderr)
        return 2
    if args.sample_every < 1:
        print("error: --sample-every must be >= 1", file=sys.stderr)
        return 2
    settings = Settings(
        blur_type=args.type,
        method=args.method,
        denoise=not args.no_denoise,
        upscale=None,
        length=args.length,
        angle=args.angle,
        radius=args.radius,
        sigma=args.sigma,
    )
    if args.noise_h is not None:
        settings.noise_h = args.noise_h

    def progress(done, total):
        if done == total or done % max(1, total // 10) == 0:
            print(f"  frame {done}/{total}")

    try:
        report = enhance_video(
            args.input, args.output, settings,
            sample_every=args.sample_every,
            estimate_max_side=args.estimate_max_side,
            progress=progress,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    w, h = report.size
    print(f"input        : {args.input}  {w}x{h}, {report.fps:g} fps, "
          f"{report.frames} frames")
    print(f"samples      : every {report.sample_every} frames; "
          f"raw estimates (native frame scale):")
    for idx, family, params in report.samples:
        print(f"  frame {idx:5d}  : {family} {params}")
    print(f"track        : median-smoothed within family runs, "
          f"interpolated per frame; family changes step at the "
          f"midpoint between samples")
    print(f"noise sigma  : {report.clip_noise_sigma:.1f} gray levels "
          f"(median of the sampled frames' readings)")
    print(f"method       : {report.method} (per frame; no upscale "
          f"in video mode)")
    print(f"time         : analysis {report.analysis_seconds:.1f}s, "
          f"render {report.render_seconds:.1f}s, "
          f"{report.seconds_per_frame:.2f} s/frame overall on this "
          f"machine")
    print(f"output       : {args.output}  {w}x{h}, {report.frames} frames")
    return 0
