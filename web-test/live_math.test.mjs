// Tests for docs/live-math.js, the pure logic of the live clarify
// page (docs/live.html). Thresholds were fixed from the calibration
// measurements recorded in TEST-REPORT.md before this suite ran:
//
//   * phase correlation recovers synthetic translations to within
//     1.0 px per axis at the 128x64 analysis size (calibration:
//     0.03 px), with response >= 0.5 on coherent texture, and a
//     zero-shift control reads |shift| <= 0.6 px;
//   * the directional de-smear step, measured PHYSICALLY (edge
//     transition width, not a sharpness score: sharpness scores
//     reward sharpening in any direction, including the wrong one,
//     as a wrong-direction control in calibration showed), narrows
//     a 9 px smeared step edge from a 10-90% width of 8 px to
//     <= 7 px at the page's operating strength 0.5, with ringing
//     <= 0.08 past the plateaus;
//   * clarityForZoom is the identity at zoom 1, grows monotonically,
//     and caps at 2.2x base; temporalAlpha starts at its base and
//     decays monotonically with motion, inside [0, 0.8].
//
// Run:  node live_math.test.mjs

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const HERE = dirname(fileURLToPath(import.meta.url));
const LM = require(join(HERE, "..", "docs", "live-math.js"));

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? "PASS" : "FAIL") + "  " + label +
    (detail ? "  (" + detail + ")" : ""));
  if (!ok) failures += 1;
}

function lcg(seed) {
  let s = seed;
  return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}

// ---- phase correlation: shift recovery --------------------------------
const W = 128, H = 64;
{
  const rand = lcg(7);
  let img = new Float32Array(W * H);
  for (let i = 0; i < img.length; i++) img[i] = rand();
  for (let p = 0; p < 3; p++) {
    img = LM.boxBlurAlong(img, W, H, 1, 0, 1);
    img = LM.boxBlurAlong(img, W, H, 0, 1, 1);
  }
  const shifted = (dx, dy) => {
    const out = new Float32Array(img.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const sx = ((x - dx) % W + W) % W, sy = ((y - dy) % H + H) % H;
      out[y * W + x] = img[sy * W + sx];
    }
    return out;
  };
  for (const [dx, dy] of [[5, -3], [-7, 4], [2, 6]]) {
    const r = LM.phaseCorrelation(img, shifted(dx, dy), W, H);
    check(`shift (${dx},${dy}) recovered within 1.0 px`,
      Math.abs(r.dx - dx) <= 1.0 && Math.abs(r.dy - dy) <= 1.0,
      `got (${r.dx.toFixed(2)}, ${r.dy.toFixed(2)})`);
    check(`shift (${dx},${dy}) response >= 0.5`, r.response >= 0.5,
      r.response.toFixed(3));
  }
  const zero = LM.phaseCorrelation(img, img, W, H);
  check("zero shift control reads ~0",
    Math.abs(zero.dx) <= 0.6 && Math.abs(zero.dy) <= 0.6,
    `got (${zero.dx.toFixed(2)}, ${zero.dy.toFixed(2)})`);
  const flat = LM.phaseCorrelation(
    new Float32Array(W * H).fill(0.5), new Float32Array(W * H).fill(0.5),
    W, H);
  check("flat frames give no coherent shift (response < 0.5)",
    flat.response < 0.5, flat.response.toFixed(3));
}

// ---- directional de-smear: physical edge-width measurement -------------
{
  const S = 200;
  const scene = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    scene[y * S + x] = x < 100 ? 0.15 : 0.85;
  }
  const blurred = LM.boxBlurAlong(scene, S, S, 1, 0, 4); // 9 px smear
  const edgeWidth = (img) => {
    let total = 0, n = 0;
    for (let y = 40; y < 160; y += 7) {
      let x10 = -1, x90 = -1;
      for (let x = 60; x < 140; x++) {
        const v = img[y * S + x];
        if (x10 < 0 && v >= 0.15 + 0.1 * 0.7) x10 = x;
        if (v >= 0.15 + 0.9 * 0.7) { x90 = x; break; }
      }
      if (x10 >= 0 && x90 >= 0) { total += x90 - x10; n++; }
    }
    return total / n;
  };
  check("9 px smear widens the edge to ~8 px",
    Math.abs(edgeWidth(blurred) - 8) <= 1.5, edgeWidth(blurred).toFixed(2));
  const fixed = LM.directionalDesmear(blurred, S, S, 9, 0, 0.5);
  check("de-smear at strength 0.5 narrows the edge to <= 7 px",
    edgeWidth(fixed) <= 7.0, edgeWidth(fixed).toFixed(2));
  let lo = Infinity, hi = -Infinity;
  for (let y = 90; y < 110; y++) for (let x = 60; x < 140; x++) {
    lo = Math.min(lo, fixed[y * S + x]);
    hi = Math.max(hi, fixed[y * S + x]);
  }
  check("ringing stays within 0.08 past the plateaus",
    lo >= 0.15 - 0.08 && hi <= 0.85 + 0.08,
    `profile range [${lo.toFixed(3)}, ${hi.toFixed(3)}]`);
  const still = LM.directionalDesmear(blurred, S, S, 0.2, 0, 0.5);
  check("sub-pixel measured motion leaves the frame alone",
    edgeWidth(still) === edgeWidth(blurred));
}

// ---- policy rules -------------------------------------------------------
{
  const z = [1, 2, 3, 4].map((v) => LM.clarityForZoom(1.0, v));
  check("clarityForZoom(1) is the identity", z[0] === 1.0);
  check("clarityForZoom grows monotonically with zoom",
    z[1] > z[0] && z[2] > z[1] && z[3] > z[2], z.map((v) => v.toFixed(2)).join(", "));
  check("clarityForZoom caps at 2.2x base",
    LM.clarityForZoom(1.0, 4) <= 2.2 + 1e-9 &&
    LM.clarityForZoom(3.0, 4) <= 6.6 + 1e-9);
  const a = [0, 2, 5, 10].map((m) => LM.temporalAlpha(m));
  check("temporalAlpha starts at its base (0.55)", a[0] === 0.55);
  check("temporalAlpha decays monotonically with motion",
    a[1] < a[0] && a[2] < a[1] && a[3] < a[2], a.map((v) => v.toFixed(3)).join(", "));
  check("temporalAlpha stays inside [0, 0.8]",
    a.every((v) => v >= 0 && v <= 0.8));
}

if (failures) {
  console.error("\n" + failures + " live-math test(s) FAILED");
  process.exit(1);
}
console.log("\nall live-math tests passed");
