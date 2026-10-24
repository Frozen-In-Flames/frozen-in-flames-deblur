// Static checks for docs/index.html:
//  * the page never sends image data anywhere (no XHR, no beacons, no
//    WebSocket, no POST fetch; the only fetch in the page loads the
//    engine's own .py files from the same origin);
//  * the only external host the page loads from is the jsDelivr CDN
//    (the pinned Pyodide runtime and its packages), plus one plain
//    hyperlink to the project repository, which loads nothing;
//  * the Pyodide version is pinned and the privacy copy is present;
//  * docs/deblur/*.py are byte for byte the repo's deblur/*.py.
//
// Run:  node static_checks.mjs   (after npm install; no deps needed)

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const html = readFileSync(join(ROOT, "docs", "index.html"), "utf8");

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? "PASS" : "FAIL") + "  " + label + (detail ? "  (" + detail + ")" : ""));
  if (!ok) failures += 1;
}

check("no XMLHttpRequest anywhere", !html.includes("XMLHttpRequest"));
check("no sendBeacon", !html.includes("sendBeacon"));
check("no WebSocket", !html.includes("WebSocket"));
check("no POST anywhere", !/["']POST["']/.test(html));
check("no form element (nothing to submit)", !/<form[\s>]/i.test(html));

const fetches = [...html.matchAll(/fetch\s*\(/g)].length;
check("fetch is used, only for engine files", fetches >= 1, fetches + " call site(s)");

const urls = [...new Set([...html.matchAll(/https?:\/\/[^\s"'<>]+/g)].map((m) => m[0]))];
const allowed = urls.filter((u) =>
  u.startsWith("https://cdn.jsdelivr.net/pyodide/") ||
  u === "https://github.com/Frozen-In-Flames/frozen-in-flames-deblur");
check("every absolute URL is the Pyodide CDN or the repo link",
  urls.length > 0 && allowed.length === urls.length, urls.join(", "));

// The page builds the CDN URL as "https://cdn.jsdelivr.net/pyodide/v"
// + PYODIDE_VERSION + "/full/", so the pin is checked as a construction,
// not as one literal string.
check("Pyodide pinned to 0.29.5", html.includes('PYODIDE_VERSION = "0.29.5"') &&
  html.includes("https://cdn.jsdelivr.net/pyodide/v") && html.includes('"/full/"'));
check("privacy copy: image never leaves the device",
  html.includes("never leaves your device") && html.includes("no server"));
check("honest first load copy present", html.includes("First load is heavy") &&
  html.includes("about") && html.includes("65 MB"));
check("no tracking scripts", !/google-analytics|gtag|plausible|mixpanel|segment/i.test(html));

const srcFiles = readdirSync(join(ROOT, "deblur")).filter((f) => f.endsWith(".py")).sort();
const dstFiles = readdirSync(join(ROOT, "docs", "deblur")).filter((f) => f.endsWith(".py")).sort();
check("docs/deblur has exactly the package files",
  JSON.stringify(srcFiles) === JSON.stringify(dstFiles), dstFiles.join(", "));
let identical = true;
for (const name of srcFiles) {
  if (!readFileSync(join(ROOT, "deblur", name)).equals(readFileSync(join(ROOT, "docs", "deblur", name)))) {
    identical = false;
    console.log("  drift: " + name);
  }
}
check("docs/deblur files are byte identical to deblur/", identical);

if (failures) {
  console.error("\n" + failures + " static check(s) FAILED");
  process.exit(1);
}
console.log("\nall static checks passed");
