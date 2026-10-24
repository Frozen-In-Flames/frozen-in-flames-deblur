"""Settings as data: one Settings object drives the whole pipeline.

PRESETS names the blur-type entry points so callers pick a setting by
name instead of scattering flags. Manual PSF parameters (length/angle,
radius, sigma) pin the sweep off and restore with exactly those values.
"""

from dataclasses import dataclass
from typing import Optional

BLUR_TYPES = ("auto", "motion", "focus", "gaussian", "lowres", "noise",
              "segmented")


@dataclass
class Settings:
    blur_type: str = "auto"          # auto | motion | focus | gaussian | lowres
                                     # | noise | segmented
    method: str = "rl-tv"            # wiener | rl-tv | rl
    denoise: bool = True             # fastNlMeans pass before deconvolution;
                                     # strength scales with measured noise
    upscale: Optional[int] = 3840    # target long side in px; None disables
    clahe_clip: float = 1.8          # CLAHE clip limit, applied last
    unsharp_radius: float = 2.0
    unsharp_amount: float = 1.2      # gentle; 2.5 (the original paste) halos
    wiener_balance: float = 0.01
    noise_h: float = 10.0            # noise path denoise strength floor;
                                     # the path raises it to 1.4x the
                                     # measured noise sigma when heavier
                                     # (fastNlMeans only bites when the
                                     # strength is comparable to the noise)
    # Manual PSF overrides (pin these to skip the sweep)
    length: Optional[float] = None   # motion
    angle: Optional[float] = None    # motion, degrees
    radius: Optional[float] = None   # focus (defocus disk)
    sigma: Optional[float] = None    # gaussian
    # Digital zoom of the view the image was taken from (1.0 = no
    # zoom). Semantics: a camera that zooms digitally by z crops the
    # sensor image and enlarges the crop, so a blur that spans B px
    # on the sensor spans z*B px in the zoomed view. Everything the
    # estimator measures on the image as handed in is therefore in
    # VIEW space; everything a user pins by hand (length / radius /
    # sigma above) is stated in SENSOR space, and the report speaks
    # sensor space too, so numbers stay comparable across zoom
    # levels: sensor = view / zoom, and a pin is restored with the
    # view-space kernel pin * zoom. Angles never scale. Restoration
    # always deconvolves the image as given with the view-space
    # kernel, so with a swept estimate zoom relabels the report and
    # changes nothing in the pixels; with a manual pin it is what
    # makes the pin mean the same physical blur at any zoom.
    zoom: float = 1.0

    def __post_init__(self):
        if not (self.zoom > 0):
            raise ValueError(f"zoom must be > 0, got {self.zoom!r}")

    def manual_family(self):
        """Family implied by pinned parameters, or None if sweeping."""
        if self.length is not None or self.angle is not None:
            return "motion"
        if self.radius is not None:
            return "focus"
        if self.sigma is not None:
            return "gaussian"
        return None

    def manual_params(self, family):
        if family == "motion":
            return {
                "length": float(self.length if self.length is not None else 15),
                "angle": float(self.angle if self.angle is not None else 0),
            }
        if family == "focus":
            return {"radius": float(self.radius)}
        if family == "gaussian":
            return {"sigma": float(self.sigma)}
        raise ValueError(f"no manual parameters for family {family!r}")


PRESETS = {
    "motion": Settings(blur_type="motion", method="rl-tv"),
    "focus": Settings(blur_type="focus", method="rl-tv"),
    "gaussian": Settings(blur_type="gaussian", method="rl-tv"),
    "lowres": Settings(blur_type="lowres", method="rl-tv", upscale=3840),
    "noise": Settings(blur_type="noise", method="rl-tv"),
    "segmented": Settings(blur_type="segmented", method="rl-tv"),
    "auto": Settings(blur_type="auto", method="rl-tv"),
}
