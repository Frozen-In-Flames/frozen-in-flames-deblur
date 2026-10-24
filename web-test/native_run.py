"""Native side of the parity test: run the engine with CPython.

Prints one JSON object: the full enhance_image result and the staged
browser-path result on the synthetic motion-blurred fixture (truth:
motion, length 15, angle 30), the same fixture the pytest suite uses.
Run with the repo venv:  ../.venv/bin/python native_run.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import make_gt, to_bgr  # noqa: E402

from deblur.browser import enhance_image_browser  # noqa: E402
from deblur.pipeline import enhance_image  # noqa: E402
from deblur.presets import Settings  # noqa: E402
from deblur.psf import apply_blur, motion_psf  # noqa: E402


def summarize(rep):
    return {
        "blur_type": rep.blur_type,
        "psf_params": rep.psf_params,
        "input_score": rep.input_score,
        "output_score": rep.output_score,
        "ratio": rep.output_score / rep.input_score,
    }


def main():
    blurred = apply_blur(make_gt(), motion_psf(15, 30))
    img = to_bgr(blurred)
    settings = Settings(blur_type="auto", upscale=None)
    _, full = enhance_image(img, settings)
    _, staged, info = enhance_image_browser(img, settings)
    print(json.dumps({
        "full": summarize(full),
        "staged": summarize(staged),
        "staged_info": {
            "estimated": info["estimated"],
            "psf_params": info["psf_params"],
            "scale_factor": info["scale_factor"],
            "notes": info["notes"],
        },
    }))


if __name__ == "__main__":
    main()
