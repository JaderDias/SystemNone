// Renders the keynote page frame by frame with headless Chrome and encodes it with ffmpeg.
//   node render.js serve                     -> the page as a player at http://127.0.0.1:8123/<page>
//   node render.js stills 3.2 7.5 ...        -> out/stills/t-<time>.jpg
//   node render.js still 74 file.png [width] -> one frame, PNG, WebP or JPEG by extension, scaled to width (default 1920)
//   node render.js video [fps] [workers]     -> out/video.mp4 (no audio)
//   node render.js mux                       -> out/<page>-keynote.mp4 (video + out/music.wav from music.js)
//   node render.js web                       -> out/<page>-keynote-web.mp4, half the size, for player.html
//   node render.js poster [file]             -> out/poster.jpg, the opening still, for player.html
// PAGE picks the page (default system-none.html), OUT the output folder, CHROME the browser binary.
const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const puppeteer = require("puppeteer-core");
const timeline = require("./timeline");
const FFMPEG = require("ffmpeg-static");

const ROOT = __dirname;
const PAGE = process.env.PAGE || "system-none.html";
const OUT = process.env.OUT || "out";

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const fixed = {
    win32: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    darwin: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  }[process.platform];
  if (fixed) return fixed;
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"])
    for (const dir of (process.env.PATH || "").split(path.delimiter))
      if (fs.existsSync(path.join(dir, name))) return path.join(dir, name);
  throw new Error("Chrome not found; set CHROME=/path/to/chrome");
}
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".woff2": "font/woff2", ".css": "text/css", ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "video/mp4", ".jpg": "image/jpeg" };

function serve(port = 0) {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      const f = path.join(ROOT, decodeURIComponent(req.url.split("?")[0].split("#")[0]));
      if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
      // Range requests let the player seek the soundtrack.
      const type = TYPES[path.extname(f)] || "application/octet-stream", size = fs.statSync(f).size;
      const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
      if (!m) { res.writeHead(200, { "content-type": type, "content-length": size, "accept-ranges": "bytes" }); return fs.createReadStream(f).pipe(res); }
      const a = m[1] ? +m[1] : size - +m[2], b = m[1] && m[2] ? Math.min(+m[2], size - 1) : size - 1;
      res.writeHead(206, { "content-type": type, "content-length": b - a + 1, "content-range": `bytes ${a}-${b}/${size}`, "accept-ranges": "bytes" });
      fs.createReadStream(f, { start: a, end: b }).pipe(res);
    });
    srv.listen(port, "127.0.0.1", () => resolve(srv));
  });
}

async function openPage(port) {
  const browser = await puppeteer.launch({
    executablePath: findChrome(), headless: "shell",
    // GitHub's isolated Ubuntu runner cannot use Chrome's namespace sandbox.
    args: [...(process.env.CI ? ["--no-sandbox"] : []), "--hide-scrollbars", "--force-device-scale-factor=1", "--font-render-hinting=none", "--disable-lcd-text", "--force-color-profile=srgb"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:${port}/${PAGE}?render`, { waitUntil: "load" });
  await page.evaluate(() => window.ready);
  const cdp = await page.createCDPSession();
  const frame = async (t, format = "jpeg", scale = 1) => {
    await page.evaluate((t) => { window.render(t); return new Promise((r) => requestAnimationFrame(() => r())); }, t);
    // A clip with a scale makes Chrome draw the frame at that size, so text stays sharp instead of being resampled.
    const clip = scale === 1 ? undefined : { x: 0, y: 0, width: 1920, height: 1080, scale };
    const { data } = await cdp.send("Page.captureScreenshot", { format, quality: format === "png" ? undefined : 95, fromSurface: true, clip });
    return Buffer.from(data, "base64");
  };
  return { browser, page, frame };
}

async function stills(times) {
  const srv = await serve();
  const { browser, frame } = await openPage(srv.address().port);
  fs.mkdirSync(path.join(ROOT, OUT, "stills"), { recursive: true });
  for (const t of times) fs.writeFileSync(path.join(ROOT, OUT, "stills", `t-${t}.jpg`), await frame(parseFloat(t)));
  await browser.close(); srv.close();
}

async function still(t, file, width = 1920) {
  if (!t || !file) throw new Error("usage: node render.js still <time> <file.png|.webp|.jpg> [width]");
  const format = { ".png": "png", ".webp": "webp" }[path.extname(file).toLowerCase()] || "jpeg";
  const srv = await serve();
  const { browser, frame } = await openPage(srv.address().port);
  fs.writeFileSync(path.resolve(file), await frame(parseFloat(t), format, width / 1920));
  console.log(`wrote ${file} (${width} px wide, ${t} s)`);
  await browser.close(); srv.close();
}

async function segment(port, fps, f0, f1, out) {
  const { browser, frame } = await openPage(port);
  const ff = spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-c:v", "mjpeg", "-i", "-",
    "-vf", "scale=in_color_matrix=bt601:in_range=pc:out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
    "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-r", String(fps), out], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((res, rej) => ff.on("close", (c) => (c ? rej(new Error("ffmpeg " + c)) : res())));
  for (let f = f0; f < f1; f++) {
    const buf = await frame(f / fps);
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    if ((f - f0) % 300 === 0) console.log(`${path.basename(out)}: ${f - f0}/${f1 - f0}`);
  }
  ff.stdin.end();
  await done; await browser.close();
}

async function video(fps, workers) {
  const srv = await serve();
  const port = srv.address().port;
  const probe = await openPage(port);
  const duration = await probe.page.evaluate(() => window.DURATION);
  await probe.browser.close();
  const total = Math.round(duration * fps), per = Math.ceil(total / workers);
  const outDir = path.join(ROOT, OUT);
  fs.mkdirSync(outDir, { recursive: true });
  const t0 = Date.now();
  const parts = [];
  for (let i = 0; i < workers; i++) {
    const f0 = i * per, f1 = Math.min(total, f0 + per);
    if (f0 < f1) parts.push({ f0, f1, file: path.join(outDir, `part${String(i).padStart(2, "0")}.mp4`) });
  }
  await Promise.all(parts.map((p) => segment(port, fps, p.f0, p.f1, p.file)));
  fs.writeFileSync(path.join(outDir, "parts.txt"), parts.map((p) => `file '${p.file.replace(/\\/g, "/")}'`).join("\n"));
  await new Promise((res, rej) => spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", path.join(outDir, "parts.txt"), "-c", "copy", path.join(outDir, "video.mp4")], { stdio: "inherit" })
    .on("close", (c) => (c ? rej(new Error("concat " + c)) : res())));
  console.log(`rendered ${total} frames in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  srv.close();
}

function mux() {
  const out = path.join(ROOT, OUT, PAGE.replace(/\.html$/, "") + "-keynote.mp4");
  const soundtrack = path.join(ROOT, "out", "music.wav");
  return new Promise((res, rej) => spawn(FFMPEG, ["-y", "-loglevel", "error", "-i", path.join(ROOT, OUT, "video.mp4"), "-i", soundtrack,
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", out], { stdio: "inherit" })
    .on("close", (c) => (c ? rej(new Error("mux " + c)) : (console.log("wrote " + out), res()))));
}

// The MP4 for the web: CRF 20 instead of 14 halves it and looks the same (SSIM 0.999 against the
// render), with the page's AAC soundtrack copied in as it is.
function web() {
  const out = path.join(ROOT, OUT, PAGE.replace(/\.html$/, "") + "-keynote-web.mp4");
  return new Promise((res, rej) => spawn(FFMPEG, ["-y", "-loglevel", "error", "-i", path.join(ROOT, OUT, "video.mp4"), "-i", path.join(ROOT, "out", "music.m4a"),
    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-tune", "animation", "-pix_fmt", "yuv420p", "-profile:v", "high",
    "-c:a", "copy", "-movflags", "+faststart", "-shortest", out], { stdio: "inherit" })
    .on("close", (c) => (c ? rej(new Error("web " + c)) : (console.log("wrote " + out), res()))));
}

const [mode, ...rest] = process.argv.slice(2);
const run = {
  serve: () => serve(8123).then(() => console.log(`preview: http://127.0.0.1:8123/${PAGE}  (Ctrl+C to stop)`)),
  stills: () => stills(rest),
  still: () => still(rest[0], rest[1], parseInt(rest[2] || "1920")),
  video: () => video(parseInt(rest[0] || "60"), parseInt(rest[1] || "8")),
  mux,
  web,
  poster: () => still(String(timeline.playbackTime(timeline.posterTime)), rest[0] || path.join(ROOT, "out", "poster.jpg")),
}[mode];
module.exports = { serve, openPage };
if (require.main === module) {
  if (!run) { console.error("usage: node render.js serve | stills <t>... | still <t> <file> [width] | video [fps] [workers] | mux | web | poster [file]"); process.exit(2); }
  run().catch((e) => { console.error(e); process.exit(1); });
}
