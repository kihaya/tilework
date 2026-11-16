// Rasterize every SVG under a directory to a 1024 x 1024 PNG next to it.
//
// Usage: node brand/icon/render_png.js [DIR]   (default: brand/icon)
// Requires Playwright with Chromium (`npm install playwright`).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

function svgFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return svgFiles(p);
    return e.name.endsWith('.svg') ? [p] : [];
  });
}

(async () => {
  const dir = process.argv[2] || __dirname;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1024, height: 1024 }, deviceScaleFactor: 1 });
  for (const file of svgFiles(dir)) {
    const svg = fs.readFileSync(file, 'utf8');
    await page.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
    const out = file.replace(/\.svg$/, '.png');
    await page.locator('svg').screenshot({ path: out, omitBackground: true });
    console.log(out);
  }
  await browser.close();
})();
