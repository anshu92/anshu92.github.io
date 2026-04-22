/**
 * Optional Satori cover: requires static/fonts/Inter-Latin.woff2 (or any woff2) at repo root.
 * Exits 1 if font missing — Python falls back to Pillow.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createElement } from "react";
import satori from "satori";
import { Resvg } from "@resvg/resvg-js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const [inPath, outPath] = args;
if (!inPath || !outPath) {
  console.error("usage: node render_cover.mjs <in.json> <out.png>");
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(inPath, "utf8"));
const title = String(data.title || "Post").slice(0, 200);
const takeaway = String(data.takeaway || data.title || "").slice(0, 280);
const fmt = String(data.format || "post");

const fontCandidates = [
  path.join(__dirname, "../../../../static/fonts/Inter-Latin.woff2"),
  path.join(__dirname, "../../../../static/fonts/Inter-VariableFont_opsz,wght.ttf"),
];

let fontData = null;
for (const p of fontCandidates) {
  if (fs.existsSync(p)) {
    fontData = fs.readFileSync(p);
    break;
  }
}
if (!fontData) {
  process.exit(1);
}

const el = createElement(
  "div",
  {
    style: {
      display: "flex",
      flexDirection: "column",
      width: "1200px",
      height: "630px",
      background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
      color: "#fff",
      padding: 48,
      fontFamily: "Inter",
    },
  },
  createElement(
    "div",
    { style: { fontSize: 18, opacity: 0.8, marginBottom: 16 } },
    `Format · ${fmt}`,
  ),
  createElement("div", { style: { fontSize: 42, fontWeight: 700, lineHeight: 1.2 } }, title),
  createElement("div", { style: { fontSize: 24, marginTop: 32, opacity: 0.95 } }, takeaway),
);

const fonts = [
  { name: "Inter", data: fontData, weight: 400, style: "normal" },
  { name: "Inter", data: fontData, weight: 700, style: "normal" },
];

const svg = await satori(el, { width: 1200, height: 630, fonts });
const resvg = new Resvg(svg, { fitTo: { mode: "width", value: 1200 } });
const png = resvg.render().asPng();
fs.writeFileSync(outPath, png);
