"use strict";
/*
 * Pure logic for the live clarify page (docs/live.html), factored
 * out of the page so Node can test it (web-test/live_math.test.mjs).
 * No DOM, no network: just the math the page runs per frame.
 *
 *   phaseCorrelation(prev, cur, w, h)
 *     Translation between two grayscale frames, by FFT phase
 *     correlation. Returns {dx, dy, response}: the content of `cur`
 *     is the content of `prev` moved by (dx, dy) pixels (at the
 *     analysis resolution), and `response` is the correlation peak
 *     height (1.0 for identical frames, near 0 when there is no
 *     coherent translation).
 *
 *   directionalDesmear(channel, w, h, dx, dy, strength)
 *     One first-order de-smear step along a measured motion vector:
 *     out = I + strength * (I - boxBlurAlong(I, direction)), where
 *     the box length matches the measured shift. A box blur B has
 *     the first-order inverse I + (I - B); this is directional
 *     unsharp masking along the motion, an approximation step, NOT
 *     deconvolution, and the page labels it that way. Measured on a
 *     synthetic step edge smeared 9 px (web-test/live_math.test.mjs
 *     checks this): one step at strength 0.5 narrows the edge's
 *     10-90% transition from 8 px to 6 px, with ringing of about
 *     +/-0.05 past the plateaus; higher strengths narrow no further
 *     and ring harder, so the page operates at 0.5. About a quarter
 *     of the smear width, honestly bought.
 *
 *   clarityForZoom(base, zoom)
 *     The clarity auto-scale rule: zooming magnifies blur and noise
 *     together, so the unsharp amount grows with zoom,
 *     base * (1 + 0.4 * (zoom - 1)), capped at 2.2x base.
 *
 *   temporalAlpha(motionMag)
 *     History weight for the temporal noise reduction: still pixels
 *     blend with the previous frame (weight up to 0.55), moving
 *     pixels do not (the weight decays with measured motion), so
 *     the blend cannot smear what is actually moving.
 *
 *   edgeSteepness(gray, w, h)
 *     The same gradient-concentration measure the Python engine
 *     reports (sum|g|^3 / sum|g|^2, Sobel gradients), so numbers
 *     mean the same thing on both sides of the project.
 */

const LiveMath = (() => {

  // ---- FFT (iterative radix-2; sizes must be powers of two) --------
  function fft1d(re, im, invert) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (2 * Math.PI / len) * (invert ? 1 : -1);
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0;
        for (let j = 0; j < len / 2; j++) {
          const ur = re[i + j], ui = im[i + j];
          const vr = re[i + j + len / 2] * cwr - im[i + j + len / 2] * cwi;
          const vi = re[i + j + len / 2] * cwi + im[i + j + len / 2] * cwr;
          re[i + j] = ur + vr; im[i + j] = ui + vi;
          re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
          const nwr = cwr * wr - cwi * wi;
          cwi = cwr * wi + cwi * wr;
          cwr = nwr;
        }
      }
    }
    if (invert) {
      for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
  }

  function fft2(re, im, w, h, invert) {
    const rowRe = new Float64Array(w), rowIm = new Float64Array(w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        rowRe[x] = re[y * w + x]; rowIm[x] = im[y * w + x];
      }
      fft1d(rowRe, rowIm, invert);
      for (let x = 0; x < w; x++) {
        re[y * w + x] = rowRe[x]; im[y * w + x] = rowIm[x];
      }
    }
    const colRe = new Float64Array(h), colIm = new Float64Array(h);
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        colRe[y] = re[y * w + x]; colIm[y] = im[y * w + x];
      }
      fft1d(colRe, colIm, invert);
      for (let y = 0; y < h; y++) {
        re[y * w + x] = colRe[y]; im[y * w + x] = colIm[y];
      }
    }
  }

  function hannWindow(n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
    return out;
  }

  // ---- phase correlation -------------------------------------------
  function phaseCorrelation(prev, cur, w, h) {
    const n = w * h;
    const aRe = new Float64Array(n), aIm = new Float64Array(n);
    const bRe = new Float64Array(n), bIm = new Float64Array(n);
    const wx = hannWindow(w), wy = hannWindow(h);
    let meanA = 0, meanB = 0;
    for (let i = 0; i < n; i++) { meanA += prev[i]; meanB += cur[i]; }
    meanA /= n; meanB /= n;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x, win = wx[x] * wy[y];
        aRe[i] = (prev[i] - meanA) * win;
        bRe[i] = (cur[i] - meanB) * win;
      }
    }
    fft2(aRe, aIm, w, h, false);
    fft2(bRe, bIm, w, h, false);
    // cross-power spectrum conj(A) * B, normalized: if cur is prev
    // moved by (dx, dy) this is exp(-i 2pi (fx dx + fy dy)) and its
    // inverse FFT peaks at (dx, dy).
    const rRe = new Float64Array(n), rIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const cr = aRe[i] * bRe[i] + aIm[i] * bIm[i];
      const ci = aRe[i] * bIm[i] - aIm[i] * bRe[i];
      const mag = Math.hypot(cr, ci);
      if (mag > 1e-12) { rRe[i] = cr / mag; rIm[i] = ci / mag; }
    }
    fft2(rRe, rIm, w, h, true);
    let peak = -Infinity, px = 0, py = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = rRe[y * w + x];
        if (v > peak) { peak = v; px = x; py = y; }
      }
    }
    // parabolic subpixel refinement on wrapped neighbours
    function refine(get, p, size) {
      const vm = get((p - 1 + size) % size), v0 = get(p), vp = get((p + 1) % size);
      const denom = vm - 2 * v0 + vp;
      if (denom >= 0) return 0;
      const d = 0.5 * (vm - vp) / denom;
      return Math.max(-1, Math.min(1, d));
    }
    const dxSub = refine((x) => rRe[py * w + x], px, w);
    const dySub = refine((y) => rRe[y * w + px], py, h);
    let dx = px + dxSub, dy = py + dySub;
    if (dx > w / 2) dx -= w;
    if (dy > h / 2) dy -= h;
    return { dx, dy, response: peak };
  }

  // ---- directional de-smear -----------------------------------------
  function boxBlurAlong(channel, w, h, ux, uy, halfLen) {
    const out = new Float32Array(channel.length);
    const taps = 2 * halfLen + 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let t = -halfLen; t <= halfLen; t++) {
          const sx = x + ux * t, sy = y + uy * t;
          const x0 = Math.floor(sx), y0 = Math.floor(sy);
          const fx = sx - x0, fy = sy - y0;
          let acc = 0;
          for (let oy = 0; oy <= 1; oy++) {
            for (let ox = 0; ox <= 1; ox++) {
              const cx = Math.min(w - 1, Math.max(0, x0 + ox));
              const cy = Math.min(h - 1, Math.max(0, y0 + oy));
              const wgt = (ox ? fx : 1 - fx) * (oy ? fy : 1 - fy);
              acc += channel[cy * w + cx] * wgt;
            }
          }
          sum += acc;
        }
        out[y * w + x] = sum / taps;
      }
    }
    return out;
  }

  function directionalDesmear(channel, w, h, dx, dy, strength) {
    const mag = Math.hypot(dx, dy);
    if (mag < 0.75 || strength <= 0) return Float32Array.from(channel);
    const length = Math.min(11, Math.max(2, Math.round(mag)));
    const halfLen = Math.max(1, Math.round(length / 2));
    const blurred = boxBlurAlong(channel, w, h, dx / mag, dy / mag, halfLen);
    const out = new Float32Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      out[i] = channel[i] + strength * (channel[i] - blurred[i]);
    }
    return out;
  }

  // ---- small policy rules --------------------------------------------
  function clarityForZoom(base, zoom) {
    if (zoom <= 1) return base;
    return Math.min(base * (1 + 0.4 * (zoom - 1)), base * 2.2);
  }

  function temporalAlpha(motionMag, base = 0.55) {
    const a = base / (1 + Math.max(0, motionMag) / 2.5);
    return Math.min(0.8, Math.max(0, a));
  }

  // ---- the engine's reporting measure, mirrored -----------------------
  function edgeSteepness(gray, w, h) {
    const border = Math.min(16, Math.floor((Math.min(w, h) - 8) / 2));
    const b = Math.max(0, border);
    let s3 = 0, s2 = 0;
    for (let y = b; y < h - b; y++) {
      for (let x = b; x < w - b; x++) {
        const gx = -gray[(y - 1) * w + x - 1] - 2 * gray[y * w + x - 1] - gray[(y + 1) * w + x - 1]
                   + gray[(y - 1) * w + x + 1] + 2 * gray[y * w + x + 1] + gray[(y + 1) * w + x + 1];
        const gy = -gray[(y - 1) * w + x - 1] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + x + 1]
                   + gray[(y + 1) * w + x - 1] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + x + 1];
        const m = Math.hypot(gx, gy);
        s3 += m * m * m;
        s2 += m * m;
      }
    }
    return s3 / (s2 + 1e-12);
  }

  return {
    phaseCorrelation,
    boxBlurAlong,
    directionalDesmear,
    clarityForZoom,
    temporalAlpha,
    edgeSteepness,
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LiveMath;
}
