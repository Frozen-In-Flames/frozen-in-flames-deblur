// End to end test of the exact glue code embedded in docs/index.html.
//
// The worker glue (glue_estimate / glue_restore) is extracted from the
// page source line by line, executed under the pinned Pyodide, and run
// against a real encoded PNG (the flow a browser takes: file bytes in,
// PNG bytes out). This guards against the page shipping glue that only
// looks right.
//
// Run:  node glue_test.mjs    (after npm install)

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? "PASS" : "FAIL") + "  " + label + (detail ? "  (" + detail + ")" : ""));
  if (!ok) failures += 1;
}

// ---- extract the glue python from docs/index.html -------------------
const html = readFileSync(join(ROOT, "docs", "index.html"), "utf8");
const lines = html.split("\n");
const stateIdx = lines.findIndex((l) => l.includes('"_state = {}"'));
if (stateIdx < 1) { console.error("could not locate glue block in page"); process.exit(1); }
const extracted = [];
for (let i = stateIdx - 1; i < lines.length; i++) {
  const line = lines[i].trim();
  if (line.startsWith("]")) break;
  const m = line.match(/^"(.*)",?$/);
  if (!m) { console.error("extractor stopped at: " + lines[i]); process.exit(1); }
  extracted.push(JSON.parse('"' + m[1] + '"'));
}
const glue = extracted.join("\n");
check("glue extracted from the page", glue.includes("def glue_estimate") && glue.includes("def glue_restore"),
  extracted.length + " python lines");

// ---- run it under the pinned pyodide --------------------------------
const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "scipy", "opencv-python", "scikit-image"]);
pyodide.FS.mkdir("deblur");
for (const name of readdirSync(join(ROOT, "deblur"))) {
  if (name.endsWith(".py")) {
    pyodide.FS.writeFile("deblur/" + name, readFileSync(join(ROOT, "deblur", name), "utf8"));
  }
}
let conftest = readFileSync(join(ROOT, "tests", "conftest.py"), "utf8");
conftest = conftest.replace(/^import pytest$/m, "").replace(/^@pytest\.fixture.*$/m, "");
pyodide.FS.writeFile("conftest.py", conftest);

pyodide.runPython(`
import json, cv2
import numpy as np
from deblur.browser import estimate_settings
from deblur.pipeline import enhance_image
from deblur.presets import Settings
from conftest import make_gt, to_bgr
from deblur.psf import apply_blur, motion_psf

blurred = apply_blur(make_gt(), motion_psf(15, 30))
ok, buf = cv2.imencode(".png", to_bgr(blurred))
assert ok
open("input.bin", "wb").write(buf.tobytes())
`);
pyodide.runPython(glue);

const est = JSON.parse(pyodide.runPython('glue_estimate("auto", "rl-tv", True)'));
console.log("estimate:", JSON.stringify(est));
check("estimate: family motion", est.blur_type === "motion", est.blur_type);
check("estimate: length 15", est.psf_params && est.psf_params.length === 15,
  JSON.stringify(est.psf_params));
check("estimate: angle 30", est.psf_params && est.psf_params.angle === 30);
check("estimate: input size 400x400",
  est.input_size && est.input_size[0] === 400 && est.input_size[1] === 400,
  JSON.stringify(est.input_size));

const rep = JSON.parse(pyodide.runPython("glue_restore(800)"));
console.log("report:", JSON.stringify(rep));
const out = pyodide.FS.readFile("output.png");
check("restore: output PNG written", out.length > 1000, out.length + " bytes");
check("restore: output long side is the 800 px target",
  Math.max(rep.output_size[0], rep.output_size[1]) === 800, JSON.stringify(rep.output_size));
check("restore: sharper than input", rep.output_score > rep.input_score,
  rep.input_score.toFixed(5) + " -> " + rep.output_score.toFixed(5) + " at output size");
check("restore: PNG magic bytes", out[0] === 0x89 && out[1] === 0x50 && out[2] === 0x4e && out[3] === 0x47);

if (failures) { console.error("\n" + failures + " glue check(s) FAILED"); process.exit(1); }
console.log("\nall glue checks passed");
