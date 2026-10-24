// Static checks for docs/live.html (the live clarify page) and
// docs/live-math.js:
//
//  * the page makes NO network calls at all: no fetch, no XHR, no
//    beacons, no WebSocket, no POST, no form. Its only script load
//    is the same-origin live-math.js, and its only absolute URL is
//    the project repository hyperlink, which loads nothing;
//  * the page does NOT load the heavy engine (no Pyodide): the live
//    chain is plain JavaScript math, which is the design;
//  * the required pieces are present: camera access, the measured
//    motion readout (phase correlation), the instant toggle with
//    its keyboard key, the capture-still handoff through on-device
//    storage, and the honest "live clarify" labeling;
//  * live-math.js exports the pure functions the behavior tests
//    exercise, so page and tests run the same code.
//
// Run:  node live_static_checks.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const html = readFileSync(join(ROOT, "docs", "live.html"), "utf8");
const math = readFileSync(join(ROOT, "docs", "live-math.js"), "utf8");

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? "PASS" : "FAIL") + "  " + label + (detail ? "  (" + detail + ")" : ""));
  if (!ok) failures += 1;
}

check("no fetch anywhere on the live page", !/fetch\s*\(/.test(html));
check("no XMLHttpRequest", !html.includes("XMLHttpRequest"));
check("no sendBeacon", !html.includes("sendBeacon"));
check("no WebSocket", !html.includes("WebSocket"));
check("no POST anywhere", !/["']POST["']/.test(html));
check("no form element (nothing to submit)", !/<form[\s>]/i.test(html));

const urls = [...new Set([...html.matchAll(/https?:\/\/[^\s"'<>]+/g)].map((m) => m[0]))];
check("the only absolute URL is the repository link",
  urls.length === 1 && urls[0] === "https://github.com/Frozen-In-Flames/frozen-in-flames-deblur",
  urls.join(", "));
check("same-origin live-math.js is the math source",
  html.includes('<script src="live-math.js"></script>'));
check("no heavy engine on the live page (no Pyodide)",
  !/pyodide/i.test(html));

check("camera access via getUserMedia", html.includes("getUserMedia"));
check("motion measured with LiveMath.phaseCorrelation",
  html.includes("LiveMath.phaseCorrelation"));
check("de-smear via LiveMath.directionalDesmear",
  html.includes("LiveMath.directionalDesmear"));
check("clarity auto-scales with zoom via LiveMath.clarityForZoom",
  html.includes("LiveMath.clarityForZoom"));
check("temporal blend gated via LiveMath.temporalAlpha",
  html.includes("LiveMath.temporalAlpha"));
check("big instant toggle present with aria-pressed",
  html.includes('id="toggle"') && html.includes("aria-pressed"));
check("keyboard toggle (E key) wired", html.includes("keydown") &&
  html.includes('"e"') && html.includes('"E"'));
check("capture still hands off through indexedDB to index.html#capture",
  html.includes("indexedDB") && html.includes("index.html#capture"));
check("honest labeling: live clarify, not the full engine",
  html.includes("Live clarify") && html.includes("not") &&
  html.includes("full deblurring engine"));
check("honest labeling: feed never leaves the device",
  html.includes("never leaves your device"));
check("no tracking scripts",
  !/google-analytics|gtag|plausible|mixpanel|segment\.com|cdn\.segment|analytics\.segment/i.test(html));

for (const name of ["phaseCorrelation", "directionalDesmear",
                    "clarityForZoom", "temporalAlpha", "edgeSteepness"]) {
  check("live-math.js provides " + name, math.includes(name));
}

if (failures) {
  console.error("\n" + failures + " live static check(s) FAILED");
  process.exit(1);
}
console.log("\nall live static checks passed");
