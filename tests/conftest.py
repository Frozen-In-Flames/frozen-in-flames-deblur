"""Synthetic ground truth, generated in code (no external assets).

Sharp image with smooth gradients, flat patches, text-like bars at
irregular offsets/widths, a rotated rectangle, circles, a jittered
checker patch, and scattered texture squares. Deliberately NOT
periodic: strictly regular gratings produce spectral ray artifacts
that real photos do not have, and the PSF scorer is spectral.
"""

import cv2
import numpy as np
import pytest


def make_gt(size=400, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    gx = np.linspace(0.25, 0.55, size, dtype=np.float32)[None, :]
    gy = np.linspace(0.0, 0.15, size, dtype=np.float32)[:, None]
    img[:] = gx + gy
    img[40:110, 40:150] = 0.85
    img[250:340, 260:370] = 0.15
    y = 148
    for i in range(5):
        x0 = int(rng.integers(35, 80))
        wbar = int(rng.integers(120, 230))
        hbar = int(rng.integers(7, 12))
        img[y:y + hbar, x0:x0 + wbar] = 0.95 if i % 2 == 0 else 0.05
        y += int(rng.integers(15, 24))
    img[300:304, 40:200] = 0.9
    img[120:220, 300:304] = 0.9
    cell = 10
    for i in range(6):
        for j in range(6):
            y0, x0 = 30 + i * cell, 280 + j * cell
            img[y0:y0 + cell, x0:x0 + cell] = 0.8 if (i + j) % 2 == 0 else 0.2
    pts = np.array([[210, 60], [300, 85], [285, 130], [195, 105]], np.int32)
    cv2.fillPoly(img, [pts], 0.65)
    cv2.circle(img, (200, 320), 28, 0.8, -1)
    cv2.circle(img, (330, 160), 18, 0.1, -1)
    for _ in range(14):
        x0, y0 = int(rng.integers(10, 370)), int(rng.integers(10, 370))
        s = int(rng.integers(4, 10))
        img[y0:y0 + s, x0:x0 + s] = float(rng.uniform(0.1, 0.95))
    return img


def to_bgr(gray):
    return cv2.cvtColor(
        np.clip(gray * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )


@pytest.fixture(scope="module")
def gt():
    return make_gt()
