const SCENARIO = window.SCENARIO;
const DURATION = KEYNOTE_TIMELINE.duration();
const $ = (id) => typeof id === "string" ? document.getElementById(id) : id;

// ---------- tiny deterministic animation engine ----------
const EASE = {
  lin: (p) => p,
  in: (p) => p * p * p,
  out: (p) => 1 - Math.pow(1 - p, 3),
  inOut: (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2),
  quint: (p) => 1 - Math.pow(1 - p, 5),
  expo: (p) => (p >= 1 ? 1 : 1 - Math.pow(2, -10 * p)),
  sine: (p) => -(Math.cos(Math.PI * p) - 1) / 2,
};
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const prog = (t, t0, t1, e = "lin") => EASE[e](clamp((t - t0) / (t1 - t0)));
function at(keys, t) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [t1, v1, e] = keys[i], [t0, v0] = keys[i - 1];
    if (t <= t1) return t1 === t0 ? v1 : v0 + (v1 - v0) * EASE[e || "lin"]((t - t0) / (t1 - t0));
  }
  return keys[keys.length - 1][1];
}
const TRACKS = [], HOOKS = [];
function anim(el, P) { TRACKS.push({ el: $(el), P }); return $(el); }
function hook(fn) { HOOKS.push(fn); }
const sortK = (a) => a.sort((x, y) => x[0] - y[0]);
function show(el, tin, tout, o = {}) {
  const d = o.dur ?? 1.0, od = o.outDur ?? 0.55, dy = o.dy ?? 40, b = o.blur ?? 14, s0 = o.s0 ?? 1, dx = o.dx ?? 0;
  const P = {
    o: [[tin, 0], [tin + d * 0.7, 1, "out"]],
    y: [[tin, dy], [tin + d, 0, "quint"]],
    x: [[tin, dx], [tin + d, 0, "quint"]],
    s: [[tin, s0], [tin + d, 1, "quint"]],
    b: [[tin, b], [tin + d * 0.75, 0, "out"]],
  };
  Object.assign(P, o.extra);
  for (const k in o.moves || {}) P[k] = sortK((P[k] || [[tin, k === "s" ? 1 : 0]]).concat(o.moves[k]));
  if (tout != null) {
    const last = (k) => P[k][P[k].length - 1][1];
    const oy = o.outDy ?? -20, os = o.outS ?? 1, ob = o.outBlur ?? 10;
    P.o.push([tout, 1], [tout + od, 0, "inOut"]);
    const ly = last("y"); P.y.push([tout, ly], [tout + od, ly + oy, "in"]);
    const ls = last("s"); P.s.push([tout, ls], [tout + od, ls * os, "in"]);
    P.b.push([tout, 0], [tout + od, ob, "in"]);
  }
  return anim(el, P);
}
function render(t) {
  t = KEYNOTE_TIMELINE.sourceTime(t); // every cue below is in source time
  for (const { el, P } of TRACKS) {
    const o = P.o ? at(P.o, t) : 1;
    if (o <= 0.001) { if (el.style.visibility !== "hidden") el.style.visibility = "hidden"; continue; }
    el.style.visibility = "";
    el.style.opacity = o.toFixed(4);
    const x = P.x ? at(P.x, t) : 0, y = P.y ? at(P.y, t) : 0, s = P.s ? at(P.s, t) : 1, rx = P.rx ? at(P.rx, t) : 0;
    el.style.transform = (rx ? `perspective(2200px) rotateX(${rx.toFixed(3)}deg) ` : "") + `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) scale(${s.toFixed(4)})`;
    const b = P.b ? at(P.b, t) : 0;
    el.style.filter = b > 0.05 ? `blur(${b.toFixed(2)}px)` : "none";
  }
  for (const h of HOOKS) h(t);
}
function rng(i) { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }
const fmt = (n) => Math.round(n).toLocaleString("en-US");

// ---------- auras ----------
anim("auraA", { o: [[0, 0], [45.0, 0], [45.8, 0.5, "out"], [50.4, 0.35], [51.2, 0, "inOut"], [199.0, 0], [200.0, 0.5, "out"], [203.6, 0.4], [205.0, 0],
  [213.4, 0], [214.4, 0.35, "out"], [224.0, 0.28], [224.8, 0, "inOut"]],
  y: [[45.0, 60], [51.2, -40], [199.0, 60], [205.6, -20], [213.4, 60], [224.8, -40]] });
anim("auraW", { o: [[11, 0], [14.2, 0], [15.2, 0.35, "out"], [20, 0.3], [20.8, 0, "inOut"]], y: [[14, 80], [21, 0]] });
anim("auraG", { o: [[0, 0], [88.8, 0], [89.8, 0.3, "out"], [110.2, 0.25], [111.0, 0, "inOut"], [228.0, 0], [229.0, 0.25, "out"], [239.6, 0.2], [240.4, 0, "inOut"]],
  y: [[88.6, 60], [110.6, -40], [228.0, 60], [240.4, -40]] });

// ================= S1 =================
show("s1card", 0.4, 10.2, { dy: 60, dur: 1.3, moves: { y: [[6.9, 0], [7.9, -170, "inOut"]], s: [[6.9, 1], [7.9, 0.84, "inOut"]] } });
const Q = SCENARIO.question;
hook((t) => {
  const n = Math.round(clamp((t - 1.0) / 1.8) * Q.length);
  const s = Q.slice(0, n);
  if ($("s1text").textContent !== s) $("s1text").textContent = s;
  $("s1caret").style.opacity = t < 3.3 && (t < 2.8 || Math.floor(t * 2.2) % 2 === 0) ? 1 : 0;
});
["s1c0", "s1c1", "s1c2"].forEach((id, i) => show(id, 2.95 + i * 0.14, null, { dy: 18, blur: 6, dur: 0.7,
  moves: i ? {} : { s: [[6.2, 1], [6.38, 1.14, "out"], [6.8, 1, "out"]] } }));
show("s1think", 3.5, null, { dy: 16, blur: 6, dur: 0.8 });
hook((t) => {
  $("s1c0").classList.toggle("on", t >= 6.2);
  $("s1spin").style.transform = `rotate(${(t * 420) % 360}deg)`;
  $("s1spin").style.opacity = t < 6.2 ? 1 : Math.max(0, 1 - (t - 6.2) * 4);
  const v = clamp((t - 3.8) / 2.4) * 2.4;
  $("s1timer").textContent = v.toFixed(1) + " s";
  $("s1timer").style.color = t >= 6.2 ? "#ff9f0a" : "#f5f5f7";
  $("s1status").textContent = t < 6.2 ? "Asking a frontier model…" : "Answered by a frontier model";
});
show("s1w0", 7.9, 10.2, { dy: 50 });
show("s1w1", 8.45, 10.2, { dy: 50 });
show("s1w2", 9.0, 10.2, { dy: 50 });

// ================= S2 =================
show("s2count", 10.9, 13.9, { s0: 0.9, outS: 1.06 });
show("s2lab", 11.1, 13.9);
hook((t) => {
  const p = clamp((t - 11.4) / 2.1);
  const v = p <= 0 ? 1 : Math.pow(10, 6 * EASE.inOut(p));
  $("s2count").textContent = fmt(v);
  $("s2lab").textContent = v < 1.5 ? "message" : "messages";
});
show("s2a", 14.3, 20.1, { dy: 60, s0: 0.94 });
show("s2b", 14.8, 20.1, { dy: 60, s0: 0.94 });
show("s2line", 16.6, 20.1);
show("s2foot", 14.8, 20.1, { dy: 0, blur: 0 });

// ================= S3 =================
show("s3q", 21.0, 24.1, { s0: 0.96 });

// The zoom: a datacenter hall of GPU racks (training), one rack of 72 GPUs (inference), one of its compute
// trays from above (the GPUs a System One model needs), then one CPU (the student model). Each level is
// drawn at its own size and placed in one shared world; the camera moves through that world and every
// level is transformed to match it, so each drawing stays sharp at any zoom.
// ZA is the screen point the camera centers on (above the captions); levels are drawn around it.
const ZA = [960, 430];
const f2 = (n) => +n.toFixed(2);
function paths() { // batches small shapes into one <path> per style, so a hall of racks stays light to draw
  const d = {}, add = (k, s) => (d[k] = d[k] || []).push(s);
  return {
    rect: (k, x, y, w, h) => add(k, `M${f2(x)} ${f2(y)}h${f2(w)}v${f2(h)}h${f2(-w)}z`),
    dot: (k, x, y, r) => add(k, `M${f2(x - r)} ${f2(y)}a${f2(r)} ${f2(r)} 0 1 0 ${f2(2 * r)} 0a${f2(r)} ${f2(r)} 0 1 0 ${f2(-2 * r)} 0z`),
    svg: (styles) => Object.keys(styles).filter((k) => d[k]).map((k) => `<path d="${d[k].join("")}" ${styles[k]}/>`).join(""),
  };
}
// A generic GPU rack from the front, laid out like a 72-GPU rack: a top panel, power shelves, 10 compute trays,
// 9 switch trays, 8 compute trays and power shelves. Each compute tray holds four GPUs, placed across it as on
// the board below (GPU_X and GPU_W are fractions of the tray's width). Small racks get less detail. Returns
// the trays' and GPUs' boxes so the rack level can light them.
const RACK = [["top", 2], ["psu", 3], ["gpu", 10], ["sw", 9], ["gpu", 8], ["psu", 3]]; // 35 units
const GPU_X = [0.049, 0.279, 0.509, 0.739], GPU_W = 0.212;
function rackFront(P, x, y, w, h, seed) {
  const pad = w * 0.07, u = (h - 2 * pad) / 35, rw = w - 2 * pad, rh = u * 0.78, fine = w > 50;
  const trays = [], gpus = [];
  let k = 0;
  for (const [kind, n] of RACK) for (let j = 0; j < (kind === "top" ? 1 : n); j++) {
    const rx = x + pad, ry = y + pad + k * u;
    k += kind === "top" ? 2 : 1;
    P.rect(kind === "psu" ? "psu" : "tray", rx, ry, rw, kind === "top" ? u + rh : rh);
    if (kind === "gpu") {
      trays.push([rx, ry, rw, rh]);
      GPU_X.forEach((gx, i) => {
        const cx = rx + rw * gx, cw = rw * GPU_W, cy = ry + rh * 0.18, ch = rh * 0.64;
        gpus.push([cx, cy, cw, ch]);
        if (fine) P.rect("gpu", cx, cy, cw, ch);
        if (!fine && i < 3) return; // far away, one status light per tray
        const on = 0.3 + 0.7 * rng(seed * 9001 + trays.length * 53 + i * 7), q = Math.min(3, Math.floor((on - 0.3) / 0.7 * 4));
        P.dot("halo" + q, cx + cw - rh * 0.4, ry + rh / 2, rh * (fine ? 0.5 : 1));
        P.dot("led" + q, cx + cw - rh * 0.4, ry + rh / 2, rh * (fine ? 0.18 : 0.34));
      });
    } else if (fine && kind === "sw") {
      for (let i = 0; i < 16; i++) P.rect("port", rx + rw * (0.06 + i * 0.056), ry + rh * 0.3, rw * 0.04, rh * 0.4);
    } else if (fine && kind === "psu") {
      for (let i = 0; i < 6; i++) P.rect("mod", rx + rw * (0.02 + i * 0.163), ry + rh * 0.2, rw * 0.15, rh * 0.6);
    }
  }
  return { trays, gpus };
}
const RACK_STYLE = { tray: 'fill="#1c1c1e"', psu: 'fill="#18181a"', mod: 'fill="#202022"', gpu: 'fill="#242427"', port: 'fill="#2c2c2e"' };
[0.39, 0.56, 0.74, 0.91].forEach((o, q) => {
  RACK_STYLE["halo" + q] = `fill="#ff9f0a" opacity="${(0.12 * o).toFixed(3)}"`;
  RACK_STYLE["led" + q] = `fill="#ffb340" opacity="${o}"`;
});
const DC_ROWS = [[150, 120, 32, 7, 50, 0.3], [300, 175, 47, 10, 34, 0.55], [500, 270, 72, 14, 23, 1]]; // y, height, rack width, gap, racks, opacity
function dcSvg() {
  let s = "", tgt;
  DC_ROWS.forEach(([y0, h, w, gap, n, a], r) => {
    const x0 = 960 - (n * (w + gap) - gap) / 2, P = paths();
    let frames = "";
    for (let i = 0; i < n; i++) {
      const x = x0 + i * (w + gap);
      frames += `<rect x="${f2(x)}" y="${y0}" width="${w}" height="${h}" rx="${f2(w * 0.05)}" fill="#111112" stroke="#2c2c2e"/>`;
      rackFront(P, x, y0, w, h, r * 100 + i);
      if (r === 2 && i === 12) tgt = { x, y: y0, w, h, seed: r * 100 + i }; // the rack the camera flies into
    }
    s += `<g opacity="${a}">${frames}${P.svg(RACK_STYLE)}</g>`;
  });
  return { tgt, svg: `<svg width="1920" height="1080"><defs>
    <linearGradient id="dcfade"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".16" stop-color="#fff"/><stop offset=".84" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <mask id="dcmask"><rect width="1920" height="1080" fill="url(#dcfade)"/></mask>
    <radialGradient id="dcglow"><stop offset="0" stop-color="#ff9f0a" stop-opacity=".16"/><stop offset="1" stop-color="#ff9f0a" stop-opacity="0"/></radialGradient></defs>
    <ellipse cx="960" cy="520" rx="980" ry="420" fill="url(#dcglow)"/><g mask="url(#dcmask)">${s}</g>
    <rect id="s3tgt" x="${f2(tgt.x)}" y="${tgt.y}" width="${tgt.w}" height="${tgt.h}" rx="${f2(tgt.w * 0.05)}" fill="none" stroke="#ff9f0a" stroke-width="1.5"
      opacity="0" style="filter: drop-shadow(0 0 6px rgba(255,159,10,.9))"/></svg>` };
}
const DCL = dcSvg();
// one rack, drawn at screen size around ZA, in the proportions of the hall's racks so the two line up
const RK = { h: 700, w: 700 * DCL.tgt.w / DCL.tgt.h };
function rackSvg() {
  const x = ZA[0] - RK.w / 2, y = ZA[1] - RK.h / 2, k = RK.h / DCL.tgt.h, P = paths();
  const { trays, gpus } = rackFront(P, x, y, RK.w, RK.h, DCL.tgt.seed);
  RK.tray = trays[9]; // the last compute tray above the switches: the camera goes into it
  const [tx, ty, tw, th] = RK.tray, top = trays[0][1], bot = trays[17][1] + trays[17][3], lx = x + RK.w + 26;
  return `<svg width="1920" height="1080"><defs>
    <radialGradient id="rkglow"><stop offset="0" stop-color="#ff9f0a" stop-opacity=".28"/><stop offset="1" stop-color="#ff9f0a" stop-opacity="0"/></radialGradient></defs>
    <ellipse id="s3rkglow" cx="${ZA[0]}" cy="${ZA[1]}" rx="${f2(RK.w * 1.7)}" ry="${f2(RK.h * 0.62)}" fill="url(#rkglow)" opacity="0"/>
    <rect x="${f2(x)}" y="${y}" width="${f2(RK.w)}" height="${RK.h}" rx="${f2(RK.w * 0.05)}" fill="#111112" stroke="#2c2c2e" stroke-width="${f2(k)}"/>
    ${P.svg(RACK_STYLE)}
    <g style="filter: drop-shadow(0 0 ${f2(3 * k)}px rgba(255,159,10,.8))">${gpus.map(([gx, gy, gw, gh], i) =>
      `<rect id="s3g${i}" x="${f2(gx)}" y="${f2(gy)}" width="${f2(gw)}" height="${f2(gh)}" rx="2" fill="#ff9f0a" opacity="0"/>`).join("")}</g>
    <rect id="s3rkhl" x="${f2(x)}" y="${y}" width="${f2(RK.w)}" height="${RK.h}" rx="${f2(RK.w * 0.05)}" fill="none" stroke="#ff9f0a" stroke-width="${f2(1.5 * k)}"
      opacity="0" style="filter: drop-shadow(0 0 ${f2(6 * k)}px rgba(255,159,10,.9))"/>
    <rect id="s3trayhl" x="${f2(tx - 3)}" y="${f2(ty - 3)}" width="${f2(tw + 6)}" height="${f2(th + 6)}" rx="4" fill="rgba(255,255,255,.1)" stroke="#f5f5f7" stroke-width="2"
      opacity="0" style="filter: drop-shadow(0 0 8px rgba(255,255,255,.8))"/>
    <g id="s3rklab" opacity="0"><path d="M${f2(lx)} ${f2(top)}h10V${f2(bot)}h-10" fill="none" stroke="#48484a" stroke-width="2"/>
      <text x="${f2(lx + 30)}" y="${f2((top + bot) / 2 + 9)}" font-family="Mono" font-size="26" font-weight="600" fill="#a1a1a6">72 × B300</text></g></svg>`;
}
const CPU_AT = [ZA[0] - 180, ZA[1] + 100]; // the CPU the camera zooms into, on the board
const GPU_ROW = ZA[1] - 165; // the middle of the board's GPUs, which sit on the tray's GPUs in the rack
const STUDENT_AT = CPU_AT;
function boardSvg() { // one compute tray from above: 1000 x 600 around ZA, the front at the bottom
  const x0 = ZA[0] - 500, y0 = ZA[1] - 300;
  let s = `<svg width="1920" height="1080"><rect x="${x0}" y="${y0}" width="1000" height="600" rx="18" fill="#0e0e0f" stroke="#3a3a3c" stroke-width="3"/>
    <rect x="${x0 + 30}" y="${GPU_ROW - 110}" width="940" height="220" rx="12" fill="#131314" stroke="#232325" stroke-width="2"/>`;
  GPU_X.forEach((gx) => { // four GPUs, still lit from the rack
    const x = x0 + 1000 * gx, y = GPU_ROW - 90;
    let fins = "";
    for (let f = x + 18; f < x + 200; f += 12) fins += `M${f} ${y + 44}V${y + 164}`;
    s += `<rect x="${x}" y="${y}" width="212" height="180" rx="10" fill="#1d1d1f" stroke="#3a3a3c" stroke-width="2"/>
      <path d="${fins}" stroke="#2a2a2d" stroke-width="4"/>
      <text x="${x + 16}" y="${y + 29}" font-family="Mono" font-size="20" font-weight="600" fill="#a1a1a6">B300</text>
      <rect x="${x}" y="${y}" width="212" height="180" rx="10" fill="rgba(255,159,10,.1)" stroke="#ff9f0a" stroke-width="3"
        style="filter: drop-shadow(0 0 10px rgba(255,159,10,.7))"/>`;
  });
  for (const cx of [CPU_AT[0], ZA[0] + 180]) { // two CPUs with their memory
    const cy = CPU_AT[1];
    for (let j = 0; j < 4; j++) for (const x of [cx - 74 - j * 15, cx + 66 + j * 15])
      s += `<rect x="${x}" y="${cy - 65}" width="8" height="130" rx="2" fill="#1d1d1f" stroke="#2c2c2e"/>`;
    // the lid and label line up with the close-up's (cpuSvg at 110 / 500 scale), so the crossfade doesn't ghost
    s += `<rect x="${cx - 55}" y="${cy - 55}" width="110" height="110" rx="10" fill="#1d1d1f" stroke="#48484a" stroke-width="2"/>
      <rect x="${cx - 42}" y="${cy - 42}" width="84" height="84" rx="4" fill="#2a2a2d"/>
      <text x="${cx}" y="${cy + 35.2}" text-anchor="middle" font-family="Mono" font-size="13.2" font-weight="600" letter-spacing="1.76" fill="#86868b">CPU</text>`;
  }
  s += `<rect id="s3cpuhl" x="${STUDENT_AT[0] - 42}" y="${STUDENT_AT[1] - 42}" width="84" height="84" rx="4" fill="rgba(48,209,88,.12)" stroke="#30d158" stroke-width="2"
    opacity="0" style="filter: drop-shadow(0 0 6px rgba(48,209,88,.8))"/>`;
  for (let i = 0; i < 7; i++) // fans
    s += `<circle cx="${x0 + 80 + i * 140}" cy="${y0 + 555}" r="26" fill="#121213" stroke="#2c2c2e" stroke-width="2"/><circle cx="${x0 + 80 + i * 140}" cy="${y0 + 555}" r="7" fill="#2c2c2e"/>`;
  return s + "</svg>";
}
function cpuSvg() { // the CPU package: 500 x 500 around ZA, with the student model on it
  const x0 = ZA[0] - 250, y0 = ZA[1] - 250;
  let s = `<svg width="1920" height="1080"><defs>
    <linearGradient id="ihs" x2="1" y2="1"><stop offset="0" stop-color="#48484a"/><stop offset=".5" stop-color="#2c2c2e"/><stop offset="1" stop-color="#1d1d1f"/></linearGradient>
    <radialGradient id="stu" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#b5f5c8"/><stop offset=".5" stop-color="#30d158"/><stop offset="1" stop-color="#1a9e45"/></radialGradient>
    <radialGradient id="stuglow"><stop offset="0" stop-color="#30d158" stop-opacity=".7"/><stop offset="1" stop-color="#30d158" stop-opacity="0"/></radialGradient></defs>
    <rect x="${x0}" y="${y0}" width="500" height="500" rx="22" fill="#161617" stroke="#3a3a3c" stroke-width="2"/>`;
  for (let i = 0; i < 18; i++) { // capacitors along the edges
    const d = 32 + i * 24.5;
    s += `<rect x="${x0 + d}" y="${y0 + 14}" width="12" height="8" rx="2" fill="#3a3a3c"/><rect x="${x0 + d}" y="${y0 + 478}" width="12" height="8" rx="2" fill="#3a3a3c"/>` +
      `<rect x="${x0 + 14}" y="${y0 + d}" width="8" height="12" rx="2" fill="#3a3a3c"/><rect x="${x0 + 478}" y="${y0 + d}" width="8" height="12" rx="2" fill="#3a3a3c"/>`;
  }
  return s + `<rect x="${x0 + 60}" y="${y0 + 60}" width="380" height="380" rx="18" fill="url(#ihs)" stroke="#55555a" stroke-width="2"/>
    <rect id="s3ihs" x="${x0 + 60}" y="${y0 + 60}" width="380" height="380" rx="18" fill="rgba(48,209,88,.06)" stroke="#30d158" stroke-width="4" opacity="0"
      style="filter: drop-shadow(0 0 16px rgba(48,209,88,.6))"/>
    <text x="${ZA[0]}" y="${y0 + 410}" text-anchor="middle" font-family="Mono" font-size="60" font-weight="600" letter-spacing="8" fill="#86868b">CPU</text>
    <circle id="s3stuglow" cx="${ZA[0]}" cy="${ZA[1] - 20}" r="130" fill="url(#stuglow)" opacity="0"/>
    <circle id="s3stu" cx="${ZA[0]}" cy="${ZA[1] - 20}" r="26" fill="url(#stu)" opacity="0" style="transform-origin: ${ZA[0]}px ${ZA[1] - 20}px"/></svg>`;
}
$("s3dc").innerHTML = DCL.svg; $("s3rack").innerHTML = rackSvg(); $("s3board").innerHTML = boardSvg(); $("s3cpu").innerHTML = cpuSvg();
// World placement, with s the world units per drawing pixel: the rack sits on its twin in the hall; the board's
// GPUs sit on the tray's GPUs, so the board is as wide as the tray; the CPU package sits on the board's CPU.
const inWorld = (c, s, p) => [0, 1].map((k) => c[k] + s * (p[k] - ZA[k])); // a level's drawing point p
const SR = DCL.tgt.h / RK.h, SB = SR * RK.tray[2] / 1000, SC = SB * 110 / 500;
const RC = [DCL.tgt.x + DCL.tgt.w / 2, DCL.tgt.y + DCL.tgt.h / 2];
const TW = inWorld(RC, SR, [RK.tray[0] + RK.tray[2] / 2, RK.tray[1] + RK.tray[3] / 2]);
const BC = [TW[0], TW[1] - SB * (GPU_ROW - ZA[1])];
const CPUW = inWorld(BC, SB, STUDENT_AT);
const LEVELS = [
  { el: "s3dc", c: ZA, s: 1, o: [[24.4, 0], [25.1, 1, "out"], [27.5, 1], [28.0, 0, "inOut"]], b: [[24.4, 16], [25.2, 0, "out"], [27.5, 0], [28.0, 6, "in"]] },
  { el: "s3rack", c: RC, s: SR, o: [[27.2, 0], [27.7, 1, "out"], [30.1, 1], [30.6, 0, "inOut"]], b: [[30.1, 0], [30.6, 10, "in"]] },
  { el: "s3board", c: BC, s: SB, o: [[30.0, 0], [30.5, 1, "out"], [34.3, 1], [34.8, 0, "inOut"]], b: [[30.0, 10], [30.5, 0, "out"]] },
  { el: "s3cpu", c: CPUW, s: SC, o: [[34.1, 0], [34.6, 1, "out"], [38.0, 1], [38.5, 0, "inOut"]], b: [[38.0, 0], [38.5, 10, "in"]] },
];
// camera keys: [time, {c: world point at ZA, z: screen pixels per world unit}, easing into it]
const CAM = [
  [24.4, { c: ZA, z: 0.93 }], [25.8, { c: ZA, z: 1 }, "quint"], [26.6, { c: ZA, z: 1.02 }],
  [28.0, { c: RC, z: 1 / SR }, "inOut"], [29.8, { c: RC, z: 1.03 / SR }],
  [30.7, { c: BC, z: 1 / SB }, "inOut"], [33.7, { c: BC, z: 1.04 / SB }],
  [34.9, { c: CPUW, z: 1 / SC }, "inOut"], [38.5, { c: CPUW, z: 1.06 / SC }],
];
function camAt(t) {
  if (t <= CAM[0][0]) return CAM[0][1];
  for (let i = 1; i < CAM.length; i++) {
    const [t1, b, e] = CAM[i], [t0, a] = CAM[i - 1];
    if (t > t1) continue;
    // the zoom runs in even log steps while the next subject slides to ZA at the same eased pace
    const p = EASE[e || "lin"]((t - t0) / (t1 - t0)), z = a.z * Math.pow(b.z / a.z, p);
    return { c: [0, 1].map((k) => b.c[k] - (b.c[k] - a.c[k]) * a.z * (1 - p) / z), z };
  }
  return CAM[CAM.length - 1][1];
}
const S3G = Array.from({ length: 72 }, (_, i) => $("s3g" + i));
hook((t) => {
  const cam = camAt(t);
  for (const L of LEVELS) {
    const el = $(L.el), o = at(L.o, t);
    if (o <= 0.001) { if (el.style.visibility !== "hidden") el.style.visibility = "hidden"; continue; }
    const s = L.s * cam.z, b = at(L.b, t) / s; // blur in screen pixels
    const x = (L.c[0] - L.s * ZA[0] - cam.c[0]) * cam.z + ZA[0], y = (L.c[1] - L.s * ZA[1] - cam.c[1]) * cam.z + ZA[1];
    el.style.visibility = "visible";
    el.style.opacity = o.toFixed(4);
    el.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) scale(${s.toFixed(5)})`;
    el.style.filter = b > 0.05 ? `blur(${b.toFixed(2)}px)` : "none";
  }
  const tg = prog(t, 26.0, 26.5, "out"); // the rack lights up in the hall, and its twin keeps the outline until its GPUs light
  $("s3tgt").style.opacity = tg; $("s3rkhl").style.opacity = tg * (1 - prog(t, 28.1, 28.6));
  S3G.forEach((el, i) => { el.style.opacity = 0.9 * prog(t, 28.1 + i * 0.011, 28.35 + i * 0.011, "out"); });
  $("s3rkglow").style.opacity = prog(t, 28.1, 29.1);
  $("s3rklab").style.opacity = prog(t, 28.3, 28.8, "out");
  $("s3trayhl").style.opacity = prog(t, 29.2, 29.6, "out");
  const hl = prog(t, 33.2, 33.6, "out"); // the CPU lights green on the board and stays lit in the close-up
  $("s3cpuhl").style.opacity = hl; $("s3ihs").style.opacity = hl;
  const g = prog(t, 35.0, 35.6, "out");
  $("s3stu").style.opacity = g; $("s3stu").style.transform = `scale(${0.4 + 0.6 * g})`;
  $("s3stuglow").style.opacity = g;
});
show("s3c1", 24.7, 26.6);
show("s3c2", 27.9, 29.8);
show("s3c3", 30.5, 32.8);
show("s3c4", 34.7, 38.0);
show("s3w0", 38.6, 40.8, { dy: 50 });
show("s3w1", 39.05, 40.8, { dy: 50 });
show("s3w2", 39.5, 40.8, { dy: 50 });
show("s3foot", 39.5, 40.8, { dy: 0, blur: 0 });
show("s3but", 41.1, 42.9);

// ================= S4 =================
show("s4intro", 43.3, 44.6, { dy: 20 });
show("s4icon", 45.2, 50.6, { s0: 0.55, dy: 30, dur: 1.2, blur: 24, moves: { y: [[46.5, 0], [47.4, -70, "inOut"]] } });
show("s4word", 45.32, 50.6, { s0: 1.14, dy: 0, dur: 1.4, blur: 30, moves: { y: [[46.5, 0], [47.4, -70, "inOut"]] } });
show("s4t1", 47.1, 50.6);
show("s4t2", 47.7, 50.6);
hook((t) => {
  const p = ((t - 45.2) * 0.08) % 2;
  $("s4grad").style.backgroundPosition = `${(p < 1 ? p : 2 - p) * 100}% 0`;
  const q = ((t - 199.4) * 0.08) % 2;
  $("s11grad").style.backgroundPosition = `${(q < 1 ? q : 2 - q) * 100}% 0`;
});

// ================= S5 =================
const S5_OUT = 75.9;
show("s5eyebrow", 51.4, S5_OUT, { dy: 16 });
show("s5t1", 51.6, 56.3);
show("s5t2", 56.55, 61.1);
show("s5t3", 61.35, 66.5);
show("s5t4", 66.75, S5_OUT); // stays while the student takes over; who answers when it is unsure comes later
show("s5app", 51.7, S5_OUT, { dy: 30, s0: 0.96 });
show("s5gw", 51.85, S5_OUT, { dy: 30, s0: 0.96 });
show("s5slot", 52.0, 67.6, { dy: 20, outDur: 0.3, outDy: 0, outS: 1.05 });
show("s5s1", 52.0, S5_OUT, { dy: 30, s0: 0.96 });
show("s5lines", 52.2, S5_OUT, { dy: 0, blur: 0 });
show("s5store", 56.7, S5_OUT, { dy: 30, s0: 0.96 });
show("s5storeline", 56.8, S5_OUT, { dy: 0, blur: 0 });
show("s5ready", 60.9, 66.0, { dy: 14, s0: 0.8, blur: 4, dur: 0.6 }); // until the student model is trained
show("s5work", 61.5, S5_OUT, { dy: 30, s0: 0.96 });
show("s5workline", 61.6, S5_OUT, { dy: 0, blur: 0 });
show("s5worklab", 61.7, S5_OUT, { dy: 14 });
show("s5share", 68.0, S5_OUT, { dy: 24 });
const STEP_T = [62.1, 63.1, 64.1, 65.1];
const FLY = { t0: 66.0, t1: 66.9, t2: 67.8, from: [1525, 700], ctl: [1330, 360], to: [940, 539] };
anim("s5stu", { o: [[FLY.t0, 0], [FLY.t0 + 0.4, 1, "out"], [S5_OUT, 1], [S5_OUT + 0.55, 0, "inOut"]] });
hook((t) => {
  $("s5cnt").textContent = fmt(10000 * prog(t, 57.0, 60.9, "inOut"));
  STEP_T.forEach((s, i) => {
    const el = $("s5st" + i), p = clamp((t - s) / 1.0);
    el.querySelector(".fill").style.width = (t >= s + 1.0 ? 0 : p * 100) + "%";
    const done = t >= s + 1.0, active = t >= s && !done;
    el.style.color = done ? "#f5f5f7" : active ? "#f5f5f7" : "#86868b";
    el.style.background = done ? "rgba(48,209,88,.16)" : "#232325";
    el.style.boxShadow = active ? "0 0 0 2px rgba(10,132,255,.8)" : "none";
  });
  // student chip flight
  let x, y, s = 1;
  if (t < FLY.t1) { [x, y] = FLY.from; s = 0.9 + 0.1 * prog(t, FLY.t0, FLY.t0 + 0.5, "out"); }
  else {
    const p = prog(t, FLY.t1, FLY.t2, "inOut"), q = 1 - p;
    x = q * q * FLY.from[0] + 2 * q * p * FLY.ctl[0] + p * p * FLY.to[0];
    y = q * q * FLY.from[1] + 2 * q * p * FLY.ctl[1] + p * p * FLY.to[1];
  }
  const stu = $("s5stu");
  const g = t >= FLY.t2 ? Math.exp(-(t - FLY.t2) * 2.5) : 0;
  stu.style.left = (x - 170) + "px"; stu.style.top = (y - 35) + "px";
  stu.style.transform = `scale(${s + 0.08 * g})`;
  stu.style.boxShadow = `0 0 ${30 + 90 * (g + S5glow)}px rgba(48,209,88,${0.35 + 0.5 * Math.min(1, g + S5glow)})`;
  $("s5pct").textContent = Math.round(89 * prog(t, 68.4, 72.8, "out"));
});

// particles: requests go right on the upper lane, answers come back on the lower lane
const L_UP = 447, L_DN = 483, DEPLOY = 67.9;
const P5 = [];
for (let k = 0; ; k++) {
  const ts = 52.3 + k * 0.34 + (rng(k) - 0.5) * 0.16;
  if (ts > 74.4) break;
  const arrive = ts + 0.55;
  P5.push({ ts, student: arrive >= DEPLOY && rng(k * 3 + 7) < 0.89, log: arrive >= 56.9 });
}
function seg(t, t0, t1, x0, y0, x1, y1) {
  if (t < t0 || t > t1) return null;
  const p = EASE.sine((t - t0) / (t1 - t0));
  return [x0 + (x1 - x0) * p, y0 + (y1 - y0) * p, (t - t0) / (t1 - t0)];
}
function dot(ctx, x, y, r, c, a, glow = 16) {
  ctx.globalAlpha = a; ctx.shadowColor = c; ctx.shadowBlur = glow; ctx.fillStyle = c;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
}
function edgeFade(p) { return Math.min(1, p * 6, (1 - p) * 6); }
let S5glow = 0;
const cv5 = $("s5cv"), cx5 = cv5.getContext("2d");
hook((t) => {
  cx5.clearRect(0, 0, 1920, 1080);
  S5glow = 0;
  if (t < 52.1 || t > S5_OUT + 0.6) return;
  const sceneA = Math.min(prog(t, 52.2, 52.8), 1 - prog(t, S5_OUT, S5_OUT + 0.5));
  let s1busy = 0;
  for (const P of P5) {
    const { ts } = P;
    let r;
    if ((r = seg(t, ts, ts + 0.55, 440, L_UP, 660, L_UP))) dot(cx5, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
    if (P.log && (r = seg(t, ts + 0.6, ts + 1.0, 940, 598, 940, 688))) dot(cx5, r[0], r[1], 4, "#64d2ff", 0.8 * sceneA * edgeFade(r[2]), 10);
    if (P.student) {
      const te = ts + 0.6;
      if (t >= te) S5glow += Math.exp(-(t - te) * 5) * 0.5;
      if ((r = seg(t, ts + 0.7, ts + 1.2, 660, L_DN, 440, L_DN))) dot(cx5, r[0], r[1], 6, "#30d158", sceneA * edgeFade(r[2]), 20);
    } else {
      if ((r = seg(t, ts + 0.62, ts + 1.17, 1220, L_UP, 1480, L_UP))) dot(cx5, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
      if (t > ts + 1.17 && t < ts + 2.2) s1busy++;
      if ((r = seg(t, ts + 2.2, ts + 2.75, 1480, L_DN, 1220, L_DN))) dot(cx5, r[0], r[1], 6, "#a1a1a6", sceneA * edgeFade(r[2]), 8);
      if ((r = seg(t, ts + 2.82, ts + 3.37, 660, L_DN, 440, L_DN))) dot(cx5, r[0], r[1], 6, "#a1a1a6", sceneA * edgeFade(r[2]), 8);
    }
  }
  cx5.globalAlpha = 1; cx5.shadowBlur = 0;
  S5glow = Math.min(1, S5glow);
  $("s5s1").style.borderColor = `rgba(255,255,255,${0.12 + 0.12 * Math.min(3, s1busy)})`;
});

// ================= S6 =================
show("s6title", 76.5, 88.1);
show("s6b", 77.8, 88.1, { dy: 50 });
show("s6target", 78.3, null, { dy: 16, s0: 0.9, dur: 0.7 });
show("s6labs", 80.2, null, { dy: 16 });
show("s6foot2", 80.2, 88.1, { dy: 0, blur: 0 });
hook((t) => { $("s6fill").style.width = 89 * prog(t, 78.7, 80.2, "inOut") + "%"; });

// ================= S7 =================
show("s7n1", 89.0, 94.4, { s0: 0.86, dy: 20, dur: 1.2, blur: 24 });
show("s7l1", 89.6, 94.4);
show("s7s1", 90.1, 94.4);
show("s7n2", 95.0, 100.4, { s0: 0.86, dy: 20, dur: 1.2, blur: 24 });
show("s7l2", 95.6, 100.4);
show("s7s2", 96.1, 100.4);
show("s7n3", 101.0, 106.2, { s0: 0.86, dy: 20, dur: 1.2, blur: 24 });
show("s7l3", 101.6, 106.2);
show("s7s3", 102.1, 106.2);
show("s7t1", 106.6, 110.6);
show("s7t2", 107.2, 110.6);
show("s7foot", 89.4, 110.6, { dy: 0, blur: 0, dur: 1.2 });
hook((t) => {
  $("s7v1").textContent = fmt(1 + 2499 * prog(t, 89.0, 90.5, "expo")) + "×";
  $("s7v3").textContent = (91.95 * prog(t, 101.0, 102.5, "expo")).toFixed(2) + "%";
});

// ================= why it's fast: what each request carries =================
// An LLM and a System One model get the prompt, the labels and the input; the student model gets only the input.
// A read head sweeps each request at the same speed, so the student model's ends first.
const RQ_OUT = 124.2;
const RQ = [["LLM", "255,159,10"], ["System One", "100,210,255"], ["Student model", "48,209,88"]];
const RQ_TOK = [["p", "What does the customer want?"], ["l", "refund"], ["l", "cancel"], ["l", "other"], ["i", Q]];
$("rqrows").innerHTML = RQ.map(([name, rgb], r) => {
  const tok = (k) => `<span id="rq${r}c${k}" class="tok ${RQ_TOK[k][0]}${r === 2 && k < 4 ? " ghost" : " a"}">${RQ_TOK[k][1]}</span>`;
  const lg = (k, w) => (r ? "" : `<span id="rqlg${k}" class="lg a">${w}</span>`);
  return `<div id="rqr${r}" class="rqrow a" style="top: ${400 + r * 140}px"><div id="rqn${r}" class="rqname a" style="color: rgb(${rgb})">${name}</div>
    <div class="rqchips"><div class="rqpl"><div class="grp">${lg(0, "prompt")}${tok(0)}</div><div class="grp">${lg(1, "labels")}${tok(1)}${tok(2)}${tok(3)}</div>
      ${r === 2 ? '<div id="rqlearn" class="learned a">Learned in training</div>' : ""}</div>
      <div class="grp">${lg(2, "input")}${tok(4)}</div>
      <svg id="rqck${r}" class="rqck" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" stroke="rgb(${rgb})" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div></div>`;
}).join("");
show("rqq", 111.2, 113.3, { s0: 0.96 });
show("rqt1", 113.6, 118.3);
show("rqt2", 118.6, RQ_OUT);
RQ.forEach((_, r) => {
  const t0 = 113.9 + r * 0.25;
  show("rqr" + r, t0, RQ_OUT, { dy: 0, blur: 0, dur: 0.2 });
  show("rqn" + r, t0, null, { dy: 30 });
  RQ_TOK.forEach((_, k) => { if (r < 2 || k === 4) show(`rq${r}c${k}`, t0 + 0.1 + k * 0.07, null, { dy: 18, blur: 6, dur: 0.7 }); });
});
[0, 1, 2].forEach((k) => show("rqlg" + k, 114.0 + [0, 0.07, 0.28][k], null, { dy: 12, blur: 4 }));
show("rqlearn", 118.9, null, { dy: 14, blur: 6, dur: 0.8 });
show("rqkick", 121.3, RQ_OUT);
const RQ_SW = 115.3, RQ_PER = 1.8, RQ_V = 1150; // read sweeps: first start, period, pixels per second
let RQX = null; // each row's visible chips as [el, x0, x1] in the row, measured on first use
const offX = (el, stop) => { let x = 0; for (; el && el !== stop; el = el.offsetParent) x += el.offsetLeft; return x; };
hook((t) => {
  if (t < 113.8 || t > RQ_OUT + 0.6) return;
  // In stage pixels, whether or not the browser reports the zoomed stage's offsets zoomed.
  const px = 1920 / document.querySelector("#stage > .center").offsetWidth;
  RQX = RQX || RQ.map((_, r) => RQ_TOK.map((_, k) => $(`rq${r}c${k}`)).filter((el) => !el.classList.contains("ghost"))
    .map((el) => { const x0 = offX(el, $("rqr" + r)) * px; return [el, x0, x0 + el.offsetWidth * px]; }));
  RQ.forEach(([, rgb], r) => {
    const chips = RQX[r], on = t >= RQ_SW && t < RQ_OUT;
    const hx = chips[0][1] + ((t - RQ_SW) % RQ_PER) * RQ_V; // the read head
    for (const [el, a0, a1] of chips) {
      const a = !on || hx < a0 ? 0 : hx <= a1 ? 1 : Math.exp(-(hx - a1) / (RQ_V * 0.3));
      el.style.boxShadow = a > 0.01 ? `0 0 ${(34 * a).toFixed(1)}px rgba(${rgb},${(0.5 * a).toFixed(3)})` : "none";
      el.style.borderColor = a > 0.01 ? `rgba(${rgb},${(0.9 * a).toFixed(3)})` : "";
    }
    const done = on ? (hx - chips[chips.length - 1][2]) / RQ_V : -1; // seconds since the head reached the end
    const ck = $("rqck" + r);
    ck.style.opacity = done < 0 ? 0 : (clamp(done / 0.1) * (1 - prog(done, 0.4, 0.6))).toFixed(3);
    ck.style.transform = `scale(${(0.6 + 0.4 * EASE.out(clamp(done / 0.25))).toFixed(3)})`;
  });
});

// ================= inside the models =================
// In the manner of 3Blue1Brown: each token becomes a column of numbers (a vector), the vectors run through a
// stack of layers, and the output scores what comes next. An LLM loops once per answer token; a System One model
// scores the labels in one pass; a student model reads only the input, with fewer layers and shorter vectors.
// Everything is drawn on one canvas from the times below; the sizes are GPT-3 175B's and Ettin 17M's.
const IM_OUT = 148.4, IM_Y = 600, IM_S1 = 134.3, IM_ST = 139.2; // out; the diagram's middle; System One and student stages
const IM_PASS = [128.6, 129.8, 131.0, 132.2]; // the LLM's passes, one per answer token, two beats apart
const IM_S1PASS = 135.2, IM_STPASS = 141.2;
const IM_BEATS = Array.from({ length: 9 }, (_, i) => 142.4 + i * 0.6); // the student model answering, on the beat
const IM_COL = { p: "161,161,166", l: "94,176,255", i: "245,245,247", a: "255,159,10" };
const IM_KIND = "ppppppllliiiiiiiiiiiaaaa"; // the prompt, the labels, the input, then the LLM's answer tokens
const IM_VOCAB = [ // what the LLM scores on each pass (illustrative)
  [["The", 0.58], ["Refund", 0.16], ["This", 0.09], ["It", 0.05], ["Customer", 0.03]],
  [["answer", 0.64], ["customer", 0.14], ["label", 0.1], ["request", 0.05], ["best", 0.02]],
  [["is", 0.86], [":", 0.07], ["would", 0.03], ["here", 0.01], ["seems", 0.01]],
  [["refund", 0.81], ["cancel", 0.08], ["other", 0.05], ["a", 0.02], ["the", 0.01]],
];
const IM_LABELS = ["refund", "cancel", "other"], IM_P1 = [0.92, 0.05, 0.03], IM_P2 = [0.94, 0.04, 0.02];
const IM_KEEP = Array.from({ length: SCENARIO.layers }, (_, i) => Math.round(i * 95 / (SCENARIO.layers - 1)));
const IM_LOOP = [[1560, 415], [1560, 365], [632, 365], [632, 420]]; // the answer token's way back in
const mixRGB = (a, b, p) => a.split(",").map((v, i) => Math.round(+v + (b.split(",")[i] - v) * p)).join(",");
function imStage(t) { // [LLM, System One, student] weights for labels and colors
  const a1 = prog(t, IM_S1, IM_S1 + 0.4), a2 = prog(t, IM_ST + 1.4, IM_ST + 1.8);
  return [1 - a1, prog(t, IM_S1 + 0.3, IM_S1 + 0.7) * (1 - prog(t, IM_ST, IM_ST + 0.4)), a2];
}
function loopAt(p) { // a point along IM_LOOP, p from 0 to 1 of its length
  const L = IM_LOOP.slice(1).map((q, i) => Math.hypot(q[0] - IM_LOOP[i][0], q[1] - IM_LOOP[i][1]));
  let d = p * L.reduce((a, b) => a + b), i = 0;
  while (i < L.length - 1 && d > L[i]) d -= L[i++];
  const f = clamp(d / L[i]), [a, b] = [IM_LOOP[i], IM_LOOP[i + 1]];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}
function rrect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(r, w / 2, h / 2)); ctx.fill(); }
function label(ctx, s, x, y, a, font, color, align = "center") {
  if (a <= 0.001 || !s) return;
  ctx.globalAlpha = a; ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(s, x, y);
}
function arrow(ctx, x0, x1, y, a) {
  if (a <= 0.001 || x1 - x0 < 12) return;
  ctx.globalAlpha = a; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = "#3a3a3c"; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  ctx.strokeStyle = "#9a9aa0"; ctx.beginPath(); ctx.moveTo(x1 - 8, y - 6); ctx.lineTo(x1, y); ctx.lineTo(x1 - 8, y + 6); ctx.stroke();
}
const cvIM = $("imcv"), cxIM = cvIM.getContext("2d");
anim("imcv", { o: [[125.0, 1], [IM_OUT, 1], [IM_OUT + 0.55, 0, "inOut"]], b: [[IM_OUT, 0], [IM_OUT + 0.55, 10, "in"]] });
hook((t) => {
  const ctx = cxIM;
  ctx.clearRect(0, 0, 1920, 1080);
  if (t < 125.5 || t > IM_OUT + 0.6) return;
  ctx.textBaseline = "middle"; ctx.letterSpacing = "0px";
  const [wL, w1, wS] = imStage(t);
  const accent = mixRGB(mixRGB("255,159,10", "100,210,255", prog(t, IM_S1, IM_S1 + 0.6)), "48,209,88", prog(t, IM_ST + 0.3, IM_ST + 0.9));
  // the passes: [start, sweep start, sweep end]
  const passes = IM_PASS.map((s) => [s, s + 0.05, s + 0.6]).concat([[IM_S1PASS, IM_S1PASS + 0.05, IM_S1PASS + 0.6], [IM_STPASS, IM_STPASS, IM_STPASS + 0.25]],
    IM_BEATS.map((s) => [s, s, s + 0.2]));
  const glowIn = passes.reduce((g, [s]) => Math.max(g, t >= s ? Math.exp(-(t - s) / 0.12) : 0), 0);

  // tokens as vectors
  const colH = 320 - 240 * prog(t, IM_ST + 0.4, IM_ST + 1.3, "inOut");
  const cols = [...IM_KIND].map((k, c) => {
    let m = 1, a;
    if (k === "a") { const s = IM_PASS[c - 20]; m = a = prog(t, s + 1.1, s + 1.3, "out") * (1 - prog(t, IM_S1, IM_S1 + 0.4, "inOut")); }
    else {
      a = prog(t, 125.7 + c * 0.035, 126.0 + c * 0.035, "out");
      if (k !== "i") m = 1 - prog(t, IM_ST + 0.2 + c * 0.04, IM_ST + 0.55 + c * 0.04, "inOut");
      a *= m;
    }
    return { k, m, a };
  });
  const colW = 22 * cols.reduce((s, c) => s + c.m, 0), X_IN = 390 + 170 * prog(t, IM_ST + 0.6, IM_ST + 1.5, "inOut");
  let x = X_IN - colW / 2;
  for (let c = 0; c < cols.length; c++) {
    const { k, m, a } = cols[c], cx = x + 11 * m;
    x += 22 * m;
    if (a <= 0.001) continue;
    ctx.globalAlpha = a;
    ctx.fillStyle = "#3a3a3c"; // brackets
    for (const sx of [-1, 1]) { ctx.fillRect(cx + sx * 8.5 - 0.75, IM_Y - colH / 2, 1.5, colH); ctx.fillRect(cx + sx * 8.5 - (sx > 0 ? 3 : 0), IM_Y - colH / 2, 3, 1.5); ctx.fillRect(cx + sx * 8.5 - (sx > 0 ? 3 : 0), IM_Y + colH / 2 - 1.5, 3, 1.5); }
    const J = Math.floor((colH / 2 - 6) / 11);
    for (let j = -J; j <= J; j++) {
      const v = 0.18 + 0.7 * rng(c * 131 + j * 17 + 5), edge = clamp((colH / 2 - 6 - Math.abs(j * 11)) / 11 + 0.5);
      ctx.globalAlpha = a * edge * Math.min(1, v + 0.3 * glowIn);
      ctx.fillStyle = `rgb(${IM_COL[k]})`;
      rrect(ctx, cx - 4.5, IM_Y + j * 11 - 4, 9, 8, 2);
    }
  }
  const inR = X_IN + colW / 2, inA = prog(t, 126.3, 126.8);
  // what the input side holds
  ctx.letterSpacing = "1.8px";
  const lgY = IM_Y + colH / 2 + 40, lgA = inA * (1 - prog(t, IM_ST + 0.2, IM_ST + 0.5));
  if (lgA > 0.001) {
    ctx.font = "600 20px Inter";
    const parts = [["PROMPT", IM_COL.p], [" · ", "134,134,139"], ["LABELS", IM_COL.l], [" · ", "134,134,139"], ["INPUT", IM_COL.i]];
    let lx = X_IN - parts.reduce((s, [w]) => s + ctx.measureText(w).width, 0) / 2;
    for (const [w, c] of parts) { label(ctx, w, lx, lgY, lgA, ctx.font, `rgb(${c})`, "left"); lx += ctx.measureText(w).width; }
  }
  label(ctx, "INPUT", X_IN, lgY, prog(t, IM_ST + 1.4, IM_ST + 1.8), "600 20px Inter", `rgb(${IM_COL.i})`);
  ctx.letterSpacing = "0px";
  label(ctx, "12,288 numbers per token", X_IN, lgY + 34, inA * wL, "500 22px Inter", "#86868b");
  label(ctx, "the whole request, every time", X_IN, lgY + 34, w1, "500 22px Inter", "#86868b");
  label(ctx, `${SCENARIO.hiddenSize} numbers per token`, X_IN, lgY + 34, wS, "500 22px Inter", "#86868b");

  // the layers
  const slH = 360 - 230 * prog(t, IM_ST + 0.5, IM_ST + 1.4, "inOut");
  const sl = Array.from({ length: 96 }, (_, j) => {
    const m = IM_KEEP.includes(j) ? 1 : 1 - prog(t, IM_ST + 0.5 + rng(j + 300) * 0.6, IM_ST + 0.85 + rng(j + 300) * 0.6, "inOut");
    return { m, a: m * prog(t, 126.5 + j * 0.007, 126.75 + j * 0.007, "out") };
  });
  const M = sl.reduce((s, q) => s + q.m, 0), pitch = clamp(480 / M, 5, 30), sw = Math.min(9, pitch * 0.55), X_L = 985;
  const slL = X_L - pitch * M / 2, slR = X_L + pitch * M / 2;
  x = slL;
  for (const q of sl) {
    const cx = x + pitch * q.m / 2;
    x += pitch * q.m;
    if (q.a <= 0.001) continue;
    let lit = 0;
    for (const [, s0, s1] of passes) if (t >= s0 && t <= s1 + 0.3) {
      const hx = slL + (slR - slL) * ((t - s0) / (s1 - s0)), sg = Math.max(14, pitch * 1.3);
      lit = Math.max(lit, Math.exp(-(((cx - hx) / sg) ** 2)));
    }
    ctx.globalAlpha = q.a;
    ctx.fillStyle = `rgba(${accent},${(0.26 + 0.74 * lit).toFixed(3)})`;
    rrect(ctx, pitch < 8 ? Math.round(cx - sw / 2) : cx - sw / 2, IM_Y - slH / 2, pitch < 8 ? Math.max(2, Math.floor(sw)) : sw, slH, 2);
  }
  const slA = prog(t, 127.0, 127.5), slY = IM_Y + slH / 2 + 40;
  label(ctx, "96 decoder layers", X_L, slY, slA * wL, "600 24px Inter", "#d1d1d6");
  label(ctx, "one pass", X_L, slY, w1, "600 24px Inter", "#d1d1d6");
  label(ctx, `${SCENARIO.layers} encoder layers`, X_L, slY, wS, "600 24px Inter", "#d1d1d6");
  label(ctx, "each: attention, then an MLP", X_L, slY + 32, prog(t, 127.3, 127.8) * wL + wS, "500 22px Inter", "#86868b");
  arrow(ctx, inR + 26, slL - 26, IM_Y, prog(t, 126.9, 127.3));
  arrow(ctx, slR + 26, 1330, IM_Y, prog(t, 127.4, 127.8));

  // the output: every token the LLM knows, or just the labels
  const outA = prog(t, 127.5, 128.0), vocA = outA * (1 - prog(t, IM_S1, IM_S1 + 0.4)), labA = prog(t, IM_S1 + 0.3, IM_S1 + 0.7);
  const bars = (rows, y0, a, hiRow, hi) => rows.forEach(([w, v], i) => {
    const y = y0 + i * 50, win = i === hiRow ? hi : 0;
    label(ctx, w, 1478, y, a, "600 26px Inter", win > 0.5 ? "#f5f5f7" : "#a1a1a6", "right");
    ctx.globalAlpha = a; ctx.fillStyle = `rgba(${accent},${(0.4 + 0.6 * win).toFixed(3)})`;
    if (v * 290 > 1) rrect(ctx, 1496, y - 13, v * 290, 26, 6);
  });
  label(ctx, "all 50,257 possible tokens", 1560, 445, vocA, "500 22px Inter", "#86868b");
  label(ctx, "only the 3 labels", 1560, 495, labA * (1 - prog(t, IM_ST + 0.3, IM_ST + 0.7)), "500 22px Inter", "#86868b");
  label(ctx, "one output per label", 1560, 495, prog(t, IM_ST + 1.4, IM_ST + 1.8), "500 22px Inter", "#86868b");
  let k = -1; IM_PASS.forEach((s, i) => { if (t >= s) k = i; });
  if (k >= 0 && vocA > 0.001) {
    const s = IM_PASS[k], grow = prog(t, s + 0.45, s + 0.65, "inOut");
    if (grow <= 0 && k > 0) bars(IM_VOCAB[k - 1].map(([w, v]) => [w, v * (1 - prog(t, s + 0.2, s + 0.42, "in"))]), IM_Y - 100, vocA * (1 - prog(t, s + 0.3, s + 0.45)), 0, 1);
    else bars(IM_VOCAB[k].map(([w, v]) => [w, v * grow]), IM_Y - 100, vocA * Math.min(1, grow * 3), 0, prog(t, s + 0.6, s + 0.7));
  }
  if (labA > 0.001) {
    const st = t >= IM_ST + 0.3, reset = prog(t, IM_ST + 0.3, IM_ST + 0.6);
    const g = st ? prog(t, IM_STPASS + 0.15, IM_STPASS + 0.35, "inOut") : prog(t, IM_S1PASS + 0.55, IM_S1PASS + 0.8, "inOut");
    const P = st && reset < 1 ? IM_P1.map((v) => v * (1 - reset)) : (st ? IM_P2 : IM_P1).map((v) => v * g);
    const hi = st ? (reset < 1 ? 1 - reset : prog(t, IM_STPASS + 0.3, IM_STPASS + 0.4)) : prog(t, IM_S1PASS + 0.75, IM_S1PASS + 0.85);
    bars(IM_LABELS.map((w, i) => [w, P[i]]), IM_Y - 50, labA, 0, hi);
    const flash = IM_BEATS.reduce((f, s) => Math.max(f, t >= s + 0.15 ? Math.exp(-(t - s - 0.15) / 0.15) : 0), 0);
    if (flash > 0.01) { ctx.globalAlpha = flash * 0.8; ctx.shadowColor = `rgb(${accent})`; ctx.shadowBlur = 24; ctx.fillStyle = `rgb(${accent})`; rrect(ctx, 1496, IM_Y - 63, IM_P2[0] * 290, 26, 6); ctx.shadowBlur = 0; }
  }
  // the LLM's answer so far, and the loop that feeds each new token back in
  const loopA = prog(t, 128.2, 128.6) * (1 - prog(t, IM_S1, IM_S1 + 0.4));
  if (loopA > 0.001) {
    const n = IM_PASS.filter((s) => t >= s + 1.15).length;
    ctx.font = "500 24px Inter";
    const head = "answer: ", body = ["The", "answer", "is", "refund"].slice(0, n).join(" "), hw = ctx.measureText(head).width;
    label(ctx, head, 1340, IM_Y + 162, loopA, ctx.font, "#86868b", "left");
    label(ctx, body, 1340 + hw, IM_Y + 162, loopA, "600 24px Inter", "#ffb340", "left");
    ctx.globalAlpha = loopA; ctx.strokeStyle = "#48484a"; ctx.lineWidth = 2; ctx.setLineDash([2, 8]); ctx.lineCap = "round";
    ctx.beginPath(); IM_LOOP.forEach(([lx, ly], i) => (i ? ctx.lineTo(lx, ly) : ctx.moveTo(lx, ly))); ctx.stroke(); ctx.setLineDash([]);
    const [ex, ey] = IM_LOOP[3];
    ctx.strokeStyle = "#9a9aa0"; ctx.beginPath(); ctx.moveTo(ex - 6, ey - 8); ctx.lineTo(ex, ey); ctx.lineTo(ex + 6, ey - 8); ctx.stroke();
    label(ctx, "each new token goes back in", 1096, 328, loopA, "500 22px Inter", "#86868b");
    IM_PASS.forEach((s, i) => { // the chosen token rides the loop into the input
      const p = prog(t, s + 0.8, s + 1.15, "inOut");
      if (t < s + 0.8 || t > s + 1.2) return;
      const [px, py] = loopAt(p), w = IM_VOCAB[i][0][0];
      ctx.font = "600 22px Inter";
      const cw = ctx.measureText(w).width + 24, fa = loopA * Math.min(1, p * 12) * (1 - prog(p, 0.85, 1));
      ctx.globalAlpha = fa; ctx.fillStyle = "#3a2508"; rrect(ctx, px - cw / 2, py - 17, cw, 34, 10);
      ctx.strokeStyle = "#ff9f0a"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect(px - cw / 2, py - 17, cw, 34, 10); ctx.stroke();
      label(ctx, w, px, py + 1, fa, ctx.font, "#ffb340");
    });
  }
  ctx.globalAlpha = 1; ctx.letterSpacing = "0px";
});
show("imeye", 125.2, IM_OUT, { dy: 16 });
show("imt1", 125.4, 128.0);
show("imt2", 128.25, 134.0);
show("imt3", IM_S1, 138.9);
show("imt4", IM_ST, IM_OUT);
show("imkick", 142.3, IM_OUT);
show("imfoot1", 125.6, 134.0, { dy: 0, blur: 0 });
show("imfoot2", IM_S1, 138.9, { dy: 0, blur: 0 });
show("imfoot3", IM_ST, IM_OUT, { dy: 0, blur: 0 });

// ================= S8 =================
const S8_OUT = 161.2;
show("s8q", 149.6, 151.9);
show("s8eye", 152.1, S8_OUT, { dy: 16 });
show("s8title", 152.2, S8_OUT);
show("s8n0", 152.5, S8_OUT, { dy: 30, s0: 0.94 });
show("s8n1", 152.65, S8_OUT, { dy: 30, s0: 0.94 });
show("s8n2", 152.8, S8_OUT, { dy: 30, s0: 0.94 });
show("s8lines", 152.9, S8_OUT, { dy: 0, blur: 0 });
show("s8al1", 153.1, S8_OUT, { dy: 12 });
show("s8al2", 153.25, S8_OUT, { dy: 12 });
show("s8cap", 153.6, S8_OUT);
const P8 = [];
for (let k = 0; ; k++) {
  const ts = 153.0 + k * 0.3 + (rng(k + 500) - 0.5) * 0.14;
  if (ts > 159.6) break;
  const r = rng(k * 5 + 900);
  P8.push({ ts, tier: r < 0.62 ? 0 : r < 0.86 ? 1 : 2 });
}
const cv8 = $("s8cv"), cx8 = cv8.getContext("2d");
const TIER_C = ["48,209,88", "100,210,255", "255,159,10"];
hook((t) => {
  cx8.clearRect(0, 0, 1920, 1080);
  const glow = [0, 0, 0];
  if (t > 152.8 && t < S8_OUT + 0.6) {
    const sceneA = Math.min(prog(t, 152.9, 153.4), 1 - prog(t, S8_OUT, S8_OUT + 0.5));
    for (const P of P8) {
      const { ts } = P;
      let r;
      if ((r = seg(t, ts, ts + 0.5, 70, 540, 250, 540))) dot(cx8, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
      const hits = [ts + 0.55, ts + 1.15, ts + 1.75];
      if (P.tier >= 1 && (r = seg(t, ts + 0.6, ts + 1.1, 600, 540, 790, 540))) dot(cx8, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
      if (P.tier >= 2 && (r = seg(t, ts + 1.2, ts + 1.7, 1140, 540, 1330, 540))) dot(cx8, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
      const te = hits[P.tier];
      if (t >= te) glow[P.tier] += Math.exp(-(t - te) * 4);
    }
    cx8.globalAlpha = 1; cx8.shadowBlur = 0;
  }
  for (let i = 0; i < 3; i++) {
    const g = Math.min(1, glow[i]);
    $("s8n" + i).style.borderColor = `rgba(${TIER_C[i]},${0.25 + 0.75 * g})`;
    $("s8n" + i).style.boxShadow = `0 0 ${20 + 70 * g}px rgba(${TIER_C[i]},${0.08 + 0.4 * g})`;
  }
});

// ================= human labels =================
const HL_OUT = 170.8;
const HL = [[81.6, 163.7, 164.7], [79.5, 164.3, 165.3], [91.5, 165.0, 166.4]]; // bar value, fill start, fill end
const HL_PASS = 165.0 + 1.4 * 0.7; // the people-taught bar passes the teacher's mark (inOut reaches 81.6 / 91.5 at p = 0.7)
show("hltitle", 161.9, HL_OUT);
show("hlsub", 162.4, HL_OUT);
[0, 1, 2].forEach((i) => show("hlr" + i, 163.1 + i * 0.15, HL_OUT, { dy: 30 }));
show("hlmark", 164.7, HL_OUT, { dy: 0, blur: 0, dur: 0.6 });
show("hlkick", 166.7, HL_OUT);
show("hlfoot", 163.1, HL_OUT, { dy: 0, blur: 0 });
hook((t) => {
  HL.forEach(([v, t0, t1], i) => {
    const p = prog(t, t0, t1, "inOut");
    $("hlf" + i).style.width = v * p + "%";
    $("hlv" + i).textContent = (v * p).toFixed(1) + "%";
  });
  const g = prog(t, HL_PASS, HL_PASS + 0.3, "out") * (0.6 + 0.4 * Math.exp(-Math.max(0, t - HL_PASS - 0.3) * 2.5));
  $("hlbar2").style.boxShadow = `0 0 ${60 * g}px rgba(48,209,88,${(0.6 * g).toFixed(3)})`;
});

// ================= S9 =================
show("s9title", 171.4, 180.1);
show("s9c1", 172.2, 180.1, { dy: 50 });
show("s9c2", 172.6, 180.1, { dy: 50 });

// ================= S10 =================
const ROWS = [
  ["What does the customer want?", ["refund", "cancel", "shipping", "other"], 842113, 31240, 0.94, "live"],
  ["Which personal data does this text contain?", ["id number", "contact", "address", "names"], 212450, 9812, 0.88, "live"],
  ["Is this message spam?", ["spam", "not spam"], 131902, 4406, 0.97, "live"],
  ["Which team should handle this ticket?", ["billing", "technical", "sales", "other"], 9210, 1033, 0, "train"],
  ["How does the reviewer feel?", ["positive", "neutral", "negative"], 4388, 612, 0, "coll"],
];
function spark(k, share, n = 14) {
  let s = `<svg width="170" height="34"><line x1="0" x2="170" y1="33.5" y2="33.5" stroke="#33332f"/>`;
  for (let i = 0; i < n; i++) {
    const h = 8 + 24 * (0.55 + 0.45 * rng(k * 31 + i)) * (0.4 + 0.6 * i / n);
    const sh = k < 3 ? Math.min(1, share * clamp((i - 2) / 4)) : 0;
    const hs = h * sh, bw = 170 / n;
    s += `<rect x="${i * bw + 1.5}" y="${33 - h}" width="${bw - 3}" height="${h - hs}" fill="#55544e"/><rect x="${i * bw + 1.5}" y="${33 - hs}" width="${bw - 3}" height="${hs}" fill="#4cc49a"/>`;
  }
  return s + "</svg>";
}
$("dtbl").innerHTML = `<div class="tr h"><span>#</span><span>Prompt</span><span class="num">Requests</span><span class="num">Today</span><span>Last 14 days</span><span>Student model share</span><span>Status</span></div>` +
  ROWS.map(([title, chips, req, today, share, st], i) => `<div class="tr"><span style="color:#a09f98">${i + 1}</span>
    <span><div class="ttl">${title}</div><div class="chips2">${chips.map((c) => `<span>${c}</span>`).join("")}</div></span>
    <span class="num">${fmt(req)}</span><span class="num">${fmt(today)}</span><span>${spark(i, share)}</span>
    <span class="share"><span class="sb"><i style="width:${share * 100}%"></i></span>${Math.round(share * 100)}%</span>
    <span>${st === "live" ? '<span class="st live">Student model live</span>' : st === "train" ? '<span class="st train">Training</span>' :
      `<span class="st coll">Collecting</span><div class="prog"><i style="width:44%"></i></div>`}</span></div>`).join("");
show("s10cap", 180.5, 186.6, { dy: 24 });
show("s10win", 180.7, 186.6, { dy: 160, s0: 0.92, dur: 2.0, blur: 10, moves: { s: [[182.7, 1], [186.6, 1.03, "lin"]] }, outS: 0.94, outDy: 20,
  extra: { rx: [[180.7, 26], [182.7, 0, "quint"]] } });

const WALL = [
  ["Rust gateway", 60, "#f5f5f7"], ["ONNX Runtime", 44, "#86868b"], ["Soft labels", 40, "#64d2ff"], ["DeBERTa-v3", 52, "#f5f5f7"],
  ["ModernBERT", 44, "#86868b"], ["Ettin", 58, "grad"], ["EmbeddingGemma heads", 46, "#f5f5f7"], ["No GPU needed", 40, "#30d158"],
  ["mmBERT", 40, "#86868b"], ["Language router", 44, "#f5f5f7"], ["A student per language", 38, "#86868b"], ["Calibrated thresholds", 52, "grad"],
  ["Drift monitoring", 44, "#f5f5f7"], ["Admin console", 40, "#86868b"], ["Usage dashboard", 44, "#f5f5f7"], ["Per-prompt overrides", 38, "#64d2ff"],
  ["S3 or a folder", 42, "#86868b"], ["OpenAI-compatible judges", 46, "#f5f5f7"], ["Train now", 40, "#ff9f0a"], ["Docker Compose", 52, "#f5f5f7"],
  ["Nix flake", 42, "#86868b"], ["Answer cache", 36, "#5eb0ff"], ["KL distillation", 44, "grad"], ["Python library & CLI", 42, "#f5f5f7"],
];
$("s10wall").innerHTML = WALL.map(([w, size, c], i) => c === "grad"
  ? `<span id="w${i}" class="a" style="font-size:${size}px"><span class="grad">${w}</span></span>`
  : `<span id="w${i}" class="a" style="font-size:${size}px;color:${c}">${w}</span>`).join("");
anim("s10wall", { o: [[187.0, 1], [193.0, 1], [193.6, 0, "inOut"]], s: [[187.0, 0.97], [193.6, 1.04]], b: [[193.0, 0], [193.6, 10, "in"]] });
WALL.forEach((_, i) => show("w" + i, 187.1 + rng(i + 77) * 1.9, null, { dy: 26, blur: 12, dur: 0.9 }));

// ================= S11 =================
show("s11term", 193.8, 198.8, { dy: 60, s0: 0.96 });
const CMD = "docker compose up -d";
hook((t) => {
  if (t < 193.6 || t > 199.4) return;
  const n = Math.round(clamp((t - 194.5) / 1.3) * CMD.length);
  let s = `<span style="color:#30d158">$</span> ${CMD.slice(0, n)}`;
  if (t < 196.0) s += (Math.floor(t * 2.2) % 2 === 0 || t < 195.8) ? '<span style="background:#e5e5ea">&nbsp;</span>' : "&nbsp;";
  if (t >= 196.1) s += `\n<span style="color:#30d158">✔</span> Container system-none-gateway-1  <span style="color:#86868b">Started</span>`;
  if (t >= 196.45) s += `\n<span style="color:#30d158">✔</span> Container system-none-worker-1   <span style="color:#86868b">Started</span>`;
  if (t >= 197.0) s += `\n\n<span style="color:#86868b">Open</span> <span style="color:#64d2ff">http://localhost:8080</span>`;
  if ($("s11pre").innerHTML !== s) $("s11pre").innerHTML = s;
});
// The video ends at 205.6 s, under the fade to black. The other-languages scenes come later in
// source time (210.8 s), so the close clears just after its end, and so does the black.
const CLOSE_END = 205.7;
show("s11icon", 199.4, CLOSE_END, { s0: 0.55, dy: 30, dur: 1.2, blur: 24 });
show("s11word", 199.52, CLOSE_END, { s0: 1.14, dy: 0, dur: 1.4, blur: 30 });
show("s11tag", 200.7, CLOSE_END);
show("s11link", 201.2, CLOSE_END);
anim("fade", { o: [[203.6, 0], [205.2, 1, "inOut"], [CLOSE_END + 0.6, 1], [CLOSE_END + 0.7, 0]] });

// ================= other languages: a bigger multilingual student =================
// Each model drawn to scale, a pixel for every 1,200 parameters: its layers above, its vocabulary
// table below (a vector for every token it knows).
const ML_OUT = 224.6;
show("mlq", 211.2, 213.0);
show("mlt1", 213.4, 219.4);
show("mlt2", 219.8, ML_OUT);
show("mlkick", 222.2, ML_OUT);
show("mlfoot", 213.8, ML_OUT, { dy: 0, blur: 0 });
const ML_MODELS = [
  { name: "Ettin 17M", sub: "English", vocab: 50368, dim: 256, layersP: 4.1e6, layers: 7, params: "17 million parameters",
    train: "minutes on a CPU", rgb: "48,209,88", cx: 560, t0: 213.9 },
  { name: "mmBERT-base", sub: "110 languages", vocab: 256000, dim: 768, layersP: 110e6, layers: 22, params: "307 million parameters",
    train: "longer, on a GPU", rgb: "100,210,255", cx: 1240, t0: 214.6 },
];
const ML_BASE = 760, ML_PX = 1200, ML_ASPECT = 1.5;
const mlBox = (m) => {
  const area = (m.vocab * m.dim + m.layersP) / ML_PX, w = Math.sqrt(area * ML_ASPECT);
  return { w, h: area / w, vocabShare: m.vocab * m.dim / (m.vocab * m.dim + m.layersP) };
};
const mlx = $("mlcv").getContext("2d");
hook((t) => {
  mlx.clearRect(0, 0, 1920, 1080);
  if (t < 213.8 || t > ML_OUT + 0.7) return;
  const out = 1 - prog(t, ML_OUT, ML_OUT + 0.55, "inOut");
  const small = mlBox(ML_MODELS[0]);
  ML_MODELS.forEach((m, i) => {
    const a = prog(t, m.t0, m.t0 + 0.6, "out") * out;
    if (a <= 0) return;
    const full = mlBox(m), g = i === 0 ? 1 : prog(t, m.t0, m.t0 + 1.6, "inOut");
    const w = small.w + (full.w - small.w) * g, h = small.h + (full.h - small.h) * g;
    const x = m.cx - w / 2, y = ML_BASE - h, vh = h * full.vocabShare;
    mlx.globalAlpha = a;
    mlx.fillStyle = `rgba(${m.rgb},0.14)`; rrect(mlx, x, y, w, h - vh, 8);           // the layers
    mlx.fillStyle = `rgba(${m.rgb},0.32)`; rrect(mlx, x, y + h - vh, w, vh, 8);      // the vocabulary table
    mlx.strokeStyle = `rgba(${m.rgb},0.85)`; mlx.lineWidth = 2;
    mlx.beginPath(); mlx.roundRect(x, y, w, h, 8); mlx.stroke();
    mlx.strokeStyle = `rgba(${m.rgb},0.22)`; mlx.lineWidth = 1;                     // one line per layer
    for (let k = 1; k < m.layers; k++) { const ly = y + (h - vh) * k / m.layers; mlx.beginPath(); mlx.moveTo(x + 6, ly); mlx.lineTo(x + w - 6, ly); mlx.stroke(); }
    mlx.globalAlpha = 1;
    label(mlx, m.name, m.cx, y - 46, a, "650 30px Inter", "#f5f5f7");
    label(mlx, m.sub, m.cx, y - 14, a, "500 22px Inter", `rgb(${m.rgb})`);
    label(mlx, m.params, m.cx, ML_BASE + 40, a, "600 26px Inter", "#f5f5f7");
    label(mlx, `${m.layers} layers · ${m.dim} numbers per token`, m.cx, ML_BASE + 74, a, "500 22px Inter", "#86868b");
    // Its vocabulary table, named once the model is full size.
    const v = prog(t, 216.4, 217.0, "out") * out;
    if (i === 1) {
      label(mlx, "Vocabulary: 256,000 tokens × 768 numbers", m.cx, y + h - vh / 2 - 8, v * g, "600 26px Inter", "#f5f5f7");
      label(mlx, "197 million parameters, two thirds of it", m.cx, y + h - vh / 2 + 26, v * g, "500 22px Inter", "#d1d1d6");
      label(mlx, `${m.layers} layers: 110 million`, m.cx, y + (h - vh) / 2 + 8, v * g, "500 22px Inter", "#d1d1d6");
    } else {
      label(mlx, "50,368 tokens × 256", m.cx - w / 2 - 18, ML_BASE - vh / 2 + 8, v, "500 22px Inter", "#86868b", "right");
    }
    // How long it takes to train, once the caption says so.
    const tr = prog(t, 220.3 + i * 0.25, 220.9 + i * 0.25, "out") * out;
    if (tr > 0) {
      mlx.globalAlpha = tr;
      mlx.font = "600 24px Inter";
      const pw = mlx.measureText(m.train).width + 44;
      mlx.fillStyle = `rgba(${m.rgb},0.18)`; rrect(mlx, m.cx - pw / 2, ML_BASE + 100, pw, 42, 21);
      mlx.globalAlpha = 1;
      label(mlx, m.train, m.cx, ML_BASE + 129, tr, "600 24px Inter", `rgb(${m.rgb})`);
    }
  });
  mlx.globalAlpha = 1;
});

// ================= other languages: a student per language =================
const RT_OUT = 240.2;
show("rtq", 225.6, 227.6);
show("rtt1", 228.0, RT_OUT);
["rtm0", "rtm1", "rtm2"].forEach((id, i) => show(id, 228.3 + i * 0.15, RT_OUT, { dx: -40, dy: 0 }));
show("rtrouter", 228.6, RT_OUT, { dy: 30, s0: 0.94 });
show("rtlines", 228.8, RT_OUT, { dy: 0, blur: 0 });
["rts0", "rts1", "rts2"].forEach((id, i) => show(id, 229.0 + i * 0.15, RT_OUT, { dy: 30, s0: 0.94 }));
show("rtkick", 236.8, RT_OUT);
show("rtfoot", 228.6, RT_OUT, { dy: 0, blur: 0 });
// [chip y, student y, color] for English, Portuguese and Hindi.
const RT_LANG = [[343, 335, "245,245,247"], [518, 485, "94,176,255"], [693, 635, "255,159,10"]];
const PRT = [];
for (let k = 0; ; k++) {
  const ts = 229.6 + k * 0.4 + (rng(k + 1300) - 0.5) * 0.12;
  if (ts > 238.6) break;
  const r = rng(k * 3 + 1700), lang = r < 0.45 ? 0 : r < 0.75 ? 1 : 2;
  PRT.push({ ts, lang });
}
const rtx = $("rtcv").getContext("2d");
hook((t) => {
  rtx.clearRect(0, 0, 1920, 1080);
  const glow = [0, 0, 0];
  let routerGlow = 0;
  if (t > 229.4 && t < RT_OUT + 0.6) {
    const sceneA = Math.min(prog(t, 229.4, 229.9), 1 - prog(t, RT_OUT, RT_OUT + 0.5));
    for (const P of PRT) {
      const [chipY, stuY, rgb] = RT_LANG[P.lang];
      let r;
      if ((r = seg(t, P.ts, P.ts + 0.4, 560, chipY, 690, 515))) dot(rtx, r[0], r[1], 6, `rgb(${rgb})`, sceneA * edgeFade(r[2]));
      if (t >= P.ts + 0.4) routerGlow += Math.exp(-(t - P.ts - 0.4) * 5);
      if ((r = seg(t, P.ts + 0.65, P.ts + 1.15, 1050, 515, 1250, stuY))) dot(rtx, r[0], r[1], 6, `rgb(${rgb})`, sceneA * edgeFade(r[2]));
      const at = P.ts + 1.15;
      if (t >= at) glow[P.lang] += Math.exp(-(t - at) * 4);
    }
    rtx.globalAlpha = 1; rtx.shadowBlur = 0;
  }
  const g = Math.min(1, routerGlow);
  $("rtrouter").style.borderColor = `rgba(255,255,255,${0.2 + 0.5 * g})`;
  for (let i = 0; i < 3; i++) {
    const e = Math.min(1, glow[i]);
    $("rts" + i).style.borderColor = `rgba(48,209,88,${0.35 + 0.65 * e})`;
    $("rts" + i).style.boxShadow = `0 0 ${16 + 60 * e}px rgba(48,209,88,${0.06 + 0.35 * e})`;
  }
});

// ================= other languages: the cache =================
// The cache can sit anywhere in the cascade, and saves the step after it. It shows the three
// places in turn; a message it has seen stops there, and the others go on.
const CA_OUT = 250.4;
show("caeye", 241.2, CA_OUT, { dy: 16 });
show("catitle", 241.3, CA_OUT);
show("casub", 241.9, CA_OUT);
["can0", "can1", "can2"].forEach((id, i) => show(id, 242.3 + i * 0.15, CA_OUT, { dy: 30, s0: 0.94 }));
show("calines", 242.6, CA_OUT, { dy: 0, blur: 0 });
show("cacard", 242.8, CA_OUT, { dy: 30, s0: 0.9 });
show("capos0", 242.9, 245.3, { dy: 14 });
show("capos1", 246.0, 247.5, { dy: 14 });
show("capos2", 248.2, CA_OUT, { dy: 14 });
show("cakick", 248.7, CA_OUT);
// Where the cache sits: before the student, between the student and System One, then between
// System One and the LLM. [move start, move end, slot]
const CA_SLOT_X = [160, 710, 1250], CA_MOVES = [[245.4, 246.0, 1], [247.6, 248.2, 2]];
const caSlotAt = (t) => {
  let x = CA_SLOT_X[0];
  for (const [a, b, slot] of CA_MOVES) x += (CA_SLOT_X[slot] - x) * prog(t, a, b, "inOut");
  return x;
};
const caSlot = (t) => CA_MOVES.reduce((s, [a, , slot]) => (t >= a ? slot : s), 0);
// [left, right] of the student, System One and the LLM, in the order a message meets them.
const CA_TIERS = [[290, 590], [830, 1130], [1370, 1670]];
const CA_C = ["48,209,88", "100,210,255", "255,159,10", "94,176,255"]; // the three tiers, then the cache
// Confident at the student, confident at System One, seen before by the cache. The first ones go
// to System One, the student and the LLM, then the cache answers one it has seen ⟨bell 244.95⟩.
const PCA = [
  { ts: 243.0, student: false, systemOne: true, seen: false },
  { ts: 243.45, student: true, systemOne: true, seen: false },
  { ts: 243.9, student: false, systemOne: false, seen: false },
  { ts: 244.914, student: false, systemOne: false, seen: true },
];
for (let k = 0; ; k++) {
  const ts = 246.05 + k * 0.3 + (rng(k + 2500) - 0.5) * 0.1;
  if (ts > 249.6) break;
  // None starts while the cache moves, or just before: it would pass where the cache is going.
  if (CA_MOVES.some(([a, b]) => ts > a - 0.7 && ts < b + 0.05)) continue;
  PCA.push({ ts, student: rng(k * 5 + 2900) < 0.45, systemOne: rng(k * 7 + 3100) < 0.7, seen: rng(k * 11 + 3300) < 0.4 });
}
const CA_SPEED = 1100; // pixels a second, between the cards
// The stations a message passes, with the cache at its slot, and where it stops: [left, right, who].
function caRoute(P) {
  const slot = caSlot(P.ts), cx = CA_SLOT_X[slot];
  const stations = CA_TIERS.map(([l, r], i) => [l, r, i]);
  stations.splice(slot, 0, [cx - 100, cx + 100, 3]);
  const stop = stations.findIndex(([, , who]) =>
    (who === 3 && P.seen) || (who === 0 && P.student) || (who === 1 && P.systemOne) || who === 2);
  return stations.slice(0, stop + 1);
}
const cax = $("cacv").getContext("2d");
hook((t) => {
  cax.clearRect(0, 0, 1920, 1080);
  const glow = [0, 0, 0, 0];
  if (t > 242.8 && t < CA_OUT + 0.6) {
    const sceneA = Math.min(prog(t, 242.9, 243.3), 1 - prog(t, CA_OUT, CA_OUT + 0.5));
    for (const P of PCA) {
      let x = 20, at = P.ts;
      const route = caRoute(P);
      route.forEach(([left, right, who], i) => {
        const arrive = at + (left - x) / CA_SPEED;
        const r = seg(t, at, arrive, x, 540, left, 540);
        if (r) dot(cax, r[0], r[1], 6, "#f5f5f7", sceneA * edgeFade(r[2]));
        if (i === route.length - 1 && t >= arrive) glow[who] += Math.exp(-(t - arrive) * 4);
        at = arrive + 0.08;
        x = right;
      });
    }
    cax.globalAlpha = 1; cax.shadowBlur = 0;
  }
  $("cacard").style.left = `${caSlotAt(t) - 100}px`;
  for (let i = 0; i < 4; i++) {
    const g = Math.min(1, glow[i]), el = $(i < 3 ? "can" + i : "cacard");
    el.style.borderColor = `rgba(${CA_C[i]},${(i < 3 ? 0.25 : 0.6) + (i < 3 ? 0.75 : 0.4) * g})`;
    el.style.boxShadow = `0 0 ${20 + 70 * g}px rgba(${CA_C[i]},${0.08 + 0.4 * g})`;
  }
});

// ---------- entry points ----------
window.DURATION = DURATION;
window.render = render;
window.ready = document.fonts.ready.then(() => { render(0); return true; });

// ---------- player ----------
// Opened in a browser, the page plays itself with controls and the soundtrack (out/music.m4a, from
// `node music.js`). render.js loads it with ?render and drives render(t) frame by frame instead.
if (!new URLSearchParams(location.search).has("render")) player();

function player() {
  const POSTER = KEYNOTE_TIMELINE.playbackTime(KEYNOTE_TIMELINE.posterTime);
  const stage = $("stage");
  // The stage fits the window: a phone held sideways fills its screen with it. Zoomed, it is laid out
  // at the size it shows. Scaled down from 1920 pixels instead, Safari draws each of its layers at
  // full size times the screen's density, which on an iPhone (3x) runs out of memory and reloads the
  // page. Fixed, it isn't part of the page's width, which a phone would zoom out to show. Browsers
  // without zoom scale it.
  const zoomed = CSS.supports("zoom", "0.5");
  Object.assign(stage.style, zoomed ? { position: "fixed", inset: "0", margin: "auto" } : { position: "fixed", left: "0", top: "0" });
  const fit = () => {
    const k = Math.min(innerWidth / 1920, innerHeight / 1080);
    if (zoomed) stage.style.zoom = k;
    else stage.style.transform = `translate(${(innerWidth - 1920 * k) / 2}px, ${(innerHeight - 1080 * k) / 2}px) scale(${k})`;
  };
  addEventListener("resize", fit); window.visualViewport?.addEventListener("resize", fit); fit();
  const touch = matchMedia("(pointer: coarse)").matches;

  const css = document.createElement("style");
  css.textContent = `
    html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; } /* iOS enlarges text when the phone turns */
    body { height: 100vh; height: 100dvh; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
    #pl-big { position: fixed; left: 50%; top: 50%; width: 112px; height: 112px; margin: -56px 0 0 -56px; border-radius: 50%; border: 0; z-index: 10;
      background: rgba(255,255,255,.14); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: opacity .3s, transform .2s, background .2s; }
    #pl-big:hover { transform: scale(1.06); background: rgba(255,255,255,.22); }
    #pl-big.off { opacity: 0; pointer-events: none; }
    #pl-bar { position: fixed; left: 0; right: 0; bottom: 0; z-index: 10; padding: 22px 24px 16px; display: flex; align-items: center; gap: 16px;
      font: 500 13px/1 Inter, system-ui, sans-serif; color: #a1a1a6; background: linear-gradient(transparent, rgba(0,0,0,.75)); transition: opacity .4s; }
    #pl-bar.off { opacity: 0; }
    #pl-track { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,.22); cursor: pointer; position: relative; touch-action: none; }
    #pl-track::before { content: ""; position: absolute; left: 0; right: 0; top: -10px; bottom: -10px; }
    #pl-fs { flex: none; display: flex; padding: 4px; margin: -4px; border: 0; background: none; cursor: pointer; }
    #pl-fs[hidden] { display: none; }
    #pl-turn { position: fixed; left: 0; right: 0; bottom: 84px; z-index: 10; display: none; align-items: center; justify-content: center; gap: 10px;
      font: 500 15px/1.3 Inter, system-ui, sans-serif; color: #a1a1a6; pointer-events: none; }
    @media (pointer: coarse) { #pl-track::before { top: -18px; bottom: -18px; } }
    @media (pointer: coarse) and (orientation: portrait) { #pl-turn { display: flex; } }
    #pl-fill { display: block; height: 100%; width: 0; border-radius: 2px; background: #f5f5f7; }
    #pl-time { color: #f5f5f7; font-variant-numeric: tabular-nums; }`;
  document.head.appendChild(css);
  const ICON_PLAY = '<svg width="44" height="44" viewBox="0 0 24 24"><path d="M8.5 5.5v13l10-6.5z" fill="#fff"/></svg>';
  const ICON_AGAIN = '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/></svg>';
  const icon = (d) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f5f5f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;
  const ICON_FS = icon("M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"), ICON_FS_EXIT = icon("M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5");
  const ICON_TURN = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a1a1a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="2.5" width="10" height="19" rx="2"/><path d="M20.5 9.5a8 8 0 0 1-3 8.5M20.5 14.5v-5h-5"/></svg>';
  const ui = document.createElement("div");
  ui.innerHTML = `<button id="pl-big" aria-label="Play">${ICON_PLAY}</button>
    <div id="pl-bar"><span id="pl-time">0:00 / 0:00</span><div id="pl-track"><i id="pl-fill"></i></div><span id="pl-hint"></span><button id="pl-fs" aria-label="Full screen">${ICON_FS}</button></div>
    <div id="pl-turn">${ICON_TURN}Turn your phone sideways</div>`;
  document.body.appendChild(ui);
  const big = $("pl-big"), bar = $("pl-bar"), track = $("pl-track"), fill = $("pl-fill"), time = $("pl-time"), hintEl = $("pl-hint"), fsButton = $("pl-fs");

  // Full screen, and sideways on a phone that lets a page turn it (Android); an iPhone has no full
  // screen for pages, so there the button is hidden and the page asks to turn the phone.
  const root = document.documentElement;
  const canFullscreen = !!(document.fullscreenEnabled || document.webkitFullscreenEnabled);
  const fullscreenNow = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
  const fullscreen = async (on) => {
    try {
      if (!on) return await (document.exitFullscreen ? document.exitFullscreen() : document.webkitExitFullscreen());
      await (root.requestFullscreen ? root.requestFullscreen() : root.webkitRequestFullscreen());
      await screen.orientation?.lock?.("landscape");
    } catch { /* not allowed here: the page still plays */ }
  };
  const fsIcon = () => {
    fsButton.innerHTML = fullscreenNow() ? ICON_FS_EXIT : ICON_FS;
    fsButton.setAttribute("aria-label", fullscreenNow() ? "Exit full screen" : "Full screen");
  };
  fsButton.hidden = !canFullscreen;
  document.addEventListener("fullscreenchange", fsIcon);
  document.addEventListener("webkitfullscreenchange", fsIcon);

  let playing = false, started = false, base = POSTER, since = 0, idle = 0, hasAudio = false, audioFailed = false;
  const audio = new Audio("out/music.m4a");
  audio.preload = "auto";
  // Phones have no keys to hint at.
  const hint = () => { hintEl.textContent = [touch ? "" : "Space play/pause · ← → 5 s · F full screen", audioFailed ? "no sound: run node music.js" : ""].filter(Boolean).join(" · "); };
  audio.addEventListener("loadeddata", () => { hasAudio = true; if (playing) syncAudio(now()); });
  audio.addEventListener("error", () => { hasAudio = false; audioFailed = true; hint(); });
  hint();

  const now = () => (playing ? base + (performance.now() - since) / 1000 : base);
  const syncAudio = (t) => {
    if (!hasAudio) return;
    audio.currentTime = Math.min(t, audio.duration || t);
    if (playing && t < DURATION) audio.play().catch(() => {}); else audio.pause();
  };
  const wake = () => { bar.classList.remove("off"); document.body.style.cursor = ""; idle = performance.now(); };
  const refresh = () => {
    big.classList.toggle("off", playing);
    stage.style.transition = "filter .3s";
    stage.style.filter = playing || !started ? "" : "brightness(0.55)";
    big.innerHTML = started && base >= DURATION - 0.05 ? ICON_AGAIN : ICON_PLAY;
    wake();
  };
  const play = () => {
    if (!started || base >= DURATION - 0.05) base = 0;
    // A phone held upright goes full screen and sideways on the first play, where it can.
    if (!started && touch && innerHeight > innerWidth && canFullscreen && !fullscreenNow()) fullscreen(true);
    started = true; playing = true; since = performance.now();
    // iOS loads the soundtrack only when a tap plays it: this starts it, and loadeddata syncs it.
    if (!hasAudio && !audioFailed) audio.play().catch(() => {});
    syncAudio(base); refresh();
  };
  const pause = () => { base = now(); playing = false; syncAudio(base); refresh(); };
  const toggle = () => (playing ? pause() : play());
  const seek = (t) => { started = true; base = clamp(t, 0, DURATION); since = performance.now(); syncAudio(base); refresh(); };

  big.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
  // A tap while the controls are hidden only shows them, as there is no mouse to move; the next pauses.
  let tapShowsControls = false;
  addEventListener("pointerdown", (e) => { tapShowsControls = e.pointerType !== "mouse" && playing && bar.classList.contains("off"); wake(); });
  stage.addEventListener("click", (e) => { if (!e.target.closest("a") && !tapShowsControls) toggle(); }); // the closing link opens, not pauses
  // Click or drag along the bar to seek.
  const seekAt = (e) => { const r = track.getBoundingClientRect(); seek(clamp((e.clientX - r.left) / r.width, 0, 1) * DURATION); };
  track.addEventListener("pointerdown", (e) => { track.setPointerCapture(e.pointerId); seekAt(e); });
  track.addEventListener("pointermove", (e) => { if (track.hasPointerCapture(e.pointerId)) seekAt(e); });
  fsButton.addEventListener("click", () => fullscreen(!fullscreenNow()));
  addEventListener("mousemove", wake);
  addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "k") toggle();
    else if (e.key === "ArrowLeft") seek(now() - 5);
    else if (e.key === "ArrowRight") seek(now() + 5);
    else if (e.key === "Home" || e.key === "0") seek(0);
    else if (e.key === "f") fullscreen(!fullscreenNow());
    else return;
    e.preventDefault();
  });

  const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const frame = () => {
    let t = now();
    if (playing && t >= DURATION) { base = t = DURATION; playing = false; audio.pause(); refresh(); }
    if (playing && hasAudio && !audio.paused && Math.abs(audio.currentTime - t) > 0.2) audio.currentTime = t;
    if (playing && performance.now() - idle > 2500) { bar.classList.add("off"); document.body.style.cursor = "none"; }
    render(t);
    const shown = started ? t : 0;
    fill.style.width = `${(shown / DURATION) * 100}%`;
    time.textContent = `${mmss(shown)} / ${mmss(DURATION)}`;
    requestAnimationFrame(frame);
  };
  refresh();
  window.ready.then(frame);
}
