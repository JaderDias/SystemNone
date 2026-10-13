// Synthesizes the soundtrack in source time, the time every scene's cues are in, then plays its
// passages in the video's order (timeline.js). Writes out/music.wav, lossless, for render.js to
// mux into the MP4, and out/music.m4a, AAC at 192 kb/s and an eighth of the size, for the page.
// 100 BPM; the beat grid starts at the reveal (45.2 s) so hits land on beats. The other-languages
// scenes are at 210.8 - 251.2 s in source time, after the close; the source runs on a little past
// them, so their last join has something to blend.
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const timeline = require("./timeline");
const { assemble } = require("./audio-edit");

const SR = 48000, DUR = 253.0, N = Math.ceil(SR * DUR);
const B0 = 45.2, BEAT = 0.6, BAR = 2.4, CHORD = 4.8;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mtof = (m) => 440 * Math.pow(2, (m - 69) / 12);
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const noise = () => rnd() * 2 - 1;
const buf = () => new Float32Array(N);
const S = (t) => Math.round(t * SR);
const inRanges = (t, rs) => rs.some(([a, b]) => t >= a - 1e-6 && t < b - 1e-6);
const grid = (a, b, step, off = 0) => { const out = []; for (let t = B0 + off + Math.ceil((a - B0 - off) / step - 1e-6) * step; t < b - 1e-6; t += step) out.push(+t.toFixed(4)); return out; };
const panLR = (p) => [Math.cos((p + 1) * Math.PI / 4), Math.sin((p + 1) * Math.PI / 4)];

// buses
const padL = buf(), padR = buf(), arpL = buf(), arpR = buf(), dryL = buf(), dryR = buf(), send = buf(), bassB = buf();

// ---------- harmony ----------
// [start, pad notes, bass root]
const chords = [
  [0.0, [50, 57, 64, 65], 38], [6.8, [46, 53, 57, 62], 34], [11.6, [43, 50, 57, 58], 31], [16.4, [45, 52, 55, 62], 33],
  [21.2, [50, 57, 64, 65], 38], [26.0, [46, 53, 60, 62], 34], [33.2, [43, 50, 57, 58], 31], [38.0, [48, 55, 60, 62, 67], 36],
];
const MAIN = [[[53, 57, 60, 64], 41], [[52, 55, 60, 62], 36], [[53, 57, 60, 62], 38], [[53, 57, 58, 62], 34]];
for (let k = 0; B0 + k * CHORD < 194.0 - 1e-6; k++) chords.push([B0 + k * CHORD, ...MAIN[k % 4]]);
chords.push([194.0, [53, 57, 58, 62], 34]);          // IV, held under the last riser
chords.push([199.4, [53, 57, 60, 64, 67], 41]);      // I, rings out
// The other languages: the main progression again, from a little before them so they lead in.
for (let k = 0; 206.0 + k * CHORD < DUR - 1e-6; k++) chords.push([206.0 + k * CHORD, ...MAIN[k % 4]]);
const chordAt = (t) => { let c = chords[0]; for (const x of chords) if (x[0] <= t + 1e-6) c = x; return c; };
const ARP = { 41: [65, 69, 72, 76], 36: [64, 67, 72, 74], 38: [62, 65, 69, 72], 34: [62, 65, 70, 74] };
const arpNotes = (c) => ARP[c[2]] && c[0] >= B0 - 1e-6 ? ARP[c[2]] : c[1].slice(-4).map((m) => m + 12);

// ---------- sections ----------
// The other languages: hats under the bigger multilingual student, the beat from the router on.
const KICK = [[51.8, 76.4], [89.0, 110.6], [152.6, 194.0], [225.2, DUR]];
const HATS = [[64.4, 76.4], [89.0, 110.6], [125.0, 149.0], [152.6, 194.0], [215.6, DUR]]; // hats alone under the models' passes
const CLAPS = [[89.0, 110.6], [174.2, 194.0], [237.2, DUR]];
function arpGain(t) {
  if (t >= 206.0) return t < 225.2 ? 0.7 : 1;
  if (t < 21.2) return 0;
  if (t < B0) return 0.25 + 0.35 * (t - 21.2) / (B0 - 21.2);
  if (t < 51.8) return 0.65; if (t < 76.4) return 0.8; if (t < 89.0) return 0.7; if (t < 110.6) return 1; if (t < 149.0) return 0.7;
  if (t < 152.6) return 0.5; if (t < 194.0) return 1; if (t < 199.4) return 0.5 + 0.5 * (t - 194.0) / 5.4; return 0;
}
function padCut(t) {
  if (t >= 206.0) return t < 213.6 ? 1700 : t < 225.2 ? 2100 : 2500;
  if (t < B0) return 450 + 1900 * Math.pow(t / B0, 2);
  if (t < 51.8) return 3200;
  if (t >= 76.4 && t < 89.0) return 1700;
  if (t >= 110.6 && t < 125.0) return 1700;
  if (t >= 125.0 && t < 149.0) return 2100;
  if (t >= 149.0 && t < 152.6) return 1300;
  if (t >= 194.0 && t < 199.4) return 1300 + 2400 * Math.pow((t - 194.0) / 5.4, 2);
  if (t >= 199.4) return 3400 - 2000 * clamp((t - 199.4) / 6);
  return 2500;
}

// ---------- oscillators ----------
const TS = 4096;
const SAW = new Float32Array(TS + 1);
for (let i = 0; i <= TS; i++) { let s = 0; for (let n = 1; n <= 36; n++) s += Math.sin(2 * Math.PI * n * i / TS) / n; SAW[i] = s * 0.55; }
const tab = (T, ph) => { const p = (ph - Math.floor(ph)) * TS, i = p | 0, f = p - i; return T[i] + (T[i + 1] - T[i]) * f; };

// ---------- pad ----------
chords.forEach((c, ci) => {
  const t0 = c[0], t1 = ci + 1 < chords.length ? chords[ci + 1][0] : DUR;
  const last = ci === chords.length - 1;
  const att = t0 < 0.1 ? 3.0 : 1.4, rel = last ? 0.1 : 2.2;
  const notes = c[1].concat([c[2] + 12]);
  notes.forEach((m, ni) => {
    const f = mtof(m), isBass = ni === notes.length - 1;
    [-8, 0, 8].forEach((cents, vi) => {
      const fr = f * Math.pow(2, cents / 1200), [gl, gr] = panLR(isBass ? 0 : [-0.7, 0, 0.7][vi]);
      let ph = rnd();
      const amp = (isBass ? 0.05 : 0.042) * (vi === 1 ? 1 : 0.8);
      for (let i = S(t0), end = Math.min(N, S(t1 + rel)); i < end; i++) {
        const t = i / SR;
        let env = t < t0 + att ? 0.5 - 0.5 * Math.cos(Math.PI * (t - t0) / att) : 1;
        if (t > t1) env *= 0.5 + 0.5 * Math.cos(Math.PI * clamp((t - t1) / rel));
        ph += fr / SR;
        const v = (isBass ? Math.sin(2 * Math.PI * ph) : tab(SAW, ph)) * env * amp;
        padL[i] += v * gl; padR[i] += v * gr;
      }
    });
  });
});

// ---------- filters ----------
function svf(x, cutAt, q, mode = "lp") {
  let ic1 = 0, ic2 = 0; const k = 1 / q;
  for (let n = 0; n < x.length; n++) {
    const g = Math.tan(Math.PI * Math.min(cutAt(n), SR * 0.45) / SR), a1 = 1 / (1 + g * (g + k)), a2 = g * a1, a3 = g * a2;
    const v3 = x[n] - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3;
    ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
    x[n] = mode === "lp" ? v2 : mode === "bp" ? v1 : x[n] - k * v1 - v2;
  }
  return x;
}
// smoothed cutoff curve for the pad
const cut = new Float32Array(N);
{ let c = padCut(0); const a = 1 - Math.exp(-1 / (SR * 0.35)); for (let n = 0; n < N; n++) { c += (padCut(n / SR) - c) * a; cut[n] = c; } }
svf(padL, (n) => cut[n], 0.8); svf(padR, (n) => cut[n], 0.8);

// ---------- drums & one-shots ----------
function place(t0, len, fn, outs) {
  const i0 = S(t0), n = Math.min(S(len), N - i0);
  for (let j = 0; j < n; j++) { const v = fn(j / SR); for (const [b, g] of outs) b[i0 + j] += v * g; }
}
const kicks = KICK.flatMap(([a, b]) => grid(a, b, BEAT));
kicks.forEach((t0) => {
  let ph = 0;
  place(t0, 0.6, (tt) => { ph += (44 + 100 * Math.exp(-tt / 0.03)) / SR; return Math.sin(2 * Math.PI * ph) * Math.exp(-tt / 0.2) * Math.min(1, tt / 0.0015) * 0.3 + noise() * Math.exp(-tt / 0.0025) * 0.07; },
    [[dryL, 1], [dryR, 1]]);
});
// offbeat bass, sidechained by nature (it sits between kicks)
KICK.flatMap(([a, b]) => grid(a, b, BEAT, BEAT / 2)).forEach((t0) => {
  const f = mtof(chordAt(t0)[2]); let ph = 0;
  place(t0, 0.32, (tt) => { ph += f / SR; const e = Math.min(1, tt / 0.004) * Math.exp(-tt / 0.14); const s = Math.sin(2 * Math.PI * ph) + 0.55 * Math.sin(4 * Math.PI * ph) + 0.2 * Math.sin(6 * Math.PI * ph); return Math.tanh(1.4 * s) * e * 0.13; },
    [[bassB, 1]]);
});
HATS.flatMap(([a, b]) => grid(a, b, BEAT, BEAT / 2)).forEach((t0, i) => {
  let lp = 0;
  place(t0, 0.09, (tt) => { const x = noise(); lp += (x - lp) * 0.35; return (x - lp) * Math.exp(-tt / 0.022) * (i % 2 ? 0.075 : 0.06); }, [[dryL, 0.8], [dryR, 1.1], [send, 0.2]]);
});
CLAPS.flatMap(([a, b]) => grid(a, b, BAR, BEAT)).concat(CLAPS.flatMap(([a, b]) => grid(a, b, BAR, 3 * BEAT))).forEach((t0) => {
  const x = new Float32Array(S(0.4));
  for (let j = 0; j < x.length; j++) { const tt = j / SR; const bursts = [0, 0.011, 0.022].reduce((s, o) => s + (tt >= o ? Math.exp(-(tt - o) / 0.006) : 0), 0) * 0.5 + Math.exp(-tt / 0.11) * 0.6; x[j] = noise() * bursts; }
  svf(x, () => 1300, 1.2, "bp");
  place(t0, 0.4, (tt) => x[Math.min(x.length - 1, S(tt))] * 0.34, [[dryL, 0.9], [dryR, 1], [send, 0.9]]);
});

// arpeggio: 16ths after the reveal, 8ths while the intro builds
const PAT = [0, 2, 1, 3, 2, 0, 3, 1, 0, 2, 1, 3, 2, 3, 1, 2];
function pluck(t0, m, vel) {
  const f = mtof(m); let ph = rnd();
  const [gl, gr] = panLR((m % 5) / 2.5 - 0.8);
  place(t0, 0.9, (tt) => { ph += f / SR; const b = Math.exp(-tt / 0.05); return (Math.sin(2 * Math.PI * ph) + 0.4 * b * Math.sin(4 * Math.PI * ph) + 0.2 * b * Math.sin(6 * Math.PI * ph)) * Math.min(1, tt / 0.003) * Math.exp(-tt / 0.17) * vel; },
    [[arpL, gl], [arpR, gr]]);
}
for (let t = 21.2; t < B0 - 1e-6; t += BEAT / 2) {
  const k = Math.round((t - 21.2) / (BEAT / 2)), notes = arpNotes(chordAt(t));
  pluck(t, notes[PAT[k % 16]], 0.06 * arpGain(t));
}
for (const [a, b] of [[B0, 199.4], [206.0, DUR]]) {
  for (let t = a; t < b - 1e-6; t += BEAT / 4) {
    const k = Math.round((t - B0) / (BEAT / 4)), notes = arpNotes(chordAt(t));
    const acc = k % 4 === 0 ? 1 : k % 2 === 0 ? 0.8 : 0.62;
    pluck(t, notes[PAT[k % 16]] + (k % 16 === 12 ? 12 : 0), 0.058 * acc * arpGain(t));
  }
}

// impacts, risers, bells, thuds, typing
function impact(t0, g) {
  let ph = 0, lp = 0;
  place(t0, 3.5, (tt) => { ph += (32 + 46 * Math.exp(-tt / 0.12)) / SR; return Math.sin(2 * Math.PI * ph) * Math.exp(-tt / 1.1) * 0.75 * g; }, [[dryL, 1], [dryR, 1]]);
  place(t0, 3.0, (tt) => { lp += (noise() - lp) * 0.06; return lp * Math.exp(-tt / 0.7) * 1.6 * g; }, [[dryL, 0.7], [dryR, 0.7], [send, 1.6]]);
}
function riser(t0, t1, g) {
  const x = new Float32Array(S(t1 - t0));
  for (let j = 0; j < x.length; j++) x[j] = noise();
  svf(x, (n) => 250 * Math.pow(26, n / x.length), 2.2, "bp");
  place(t0, t1 - t0, (tt) => { const p = tt / (t1 - t0); return x[Math.min(x.length - 1, S(tt))] * p * p * g * 0.55 * Math.min(1, (t1 - t0 - tt) / 0.02); },
    [[dryL, 0.9], [dryR, 1.1], [send, 0.8]]);
}
function bell(t0, m, g) {
  const f = mtof(m);
  place(t0, 3.2, (tt) => (Math.sin(2 * Math.PI * f * tt) * Math.exp(-tt / 1.5) + 0.4 * Math.sin(2 * Math.PI * f * 2.76 * tt) * Math.exp(-tt / 0.5) + 0.18 * Math.sin(2 * Math.PI * f * 5.4 * tt) * Math.exp(-tt / 0.25)) * Math.min(1, tt / 0.002) * g,
    [[dryL, 0.8], [dryR, 1], [send, 1.2]]);
}
function thud(t0, g) {
  let ph = 0;
  place(t0, 0.5, (tt) => { ph += (68 + 80 * Math.exp(-tt / 0.04)) / SR; return Math.sin(2 * Math.PI * ph) * Math.exp(-tt / 0.2) * g + noise() * Math.exp(-tt / 0.004) * 0.08 * g; }, [[dryL, 1], [dryR, 1], [send, 0.35]]);
}
function click(t0, g) {
  let lp = 0;
  place(t0, 0.012, (tt) => { const x = noise(); lp += (x - lp) * 0.5; return (x - lp) * Math.exp(-tt / 0.0015) * g; }, [[dryL, 0.9], [dryR, 1], [send, 0.15]]);
}
[[45.2, 1.0], [89.0, 0.75], [95.0, 0.5], [101.0, 0.5], [199.4, 1.0], [14.3, 0.4], [225.2, 0.5]].forEach(([t, g]) => impact(t, g));
[[41.6, 45.2, 1.0], [86.6, 89.0, 0.55], [195.8, 199.4, 1.0], [11.4, 13.9, 0.4], [222.8, 225.2, 0.5]].forEach(([a, b, g]) => riser(a, b, g));
bell(6.2, 81, 0.07);
bell(67.8, 84, 0.06); bell(67.8, 88, 0.045);
bell(141.45, 84, 0.045);                            // inside the models: the student model answers in one pass
bell(165.98, 84, 0.05); bell(165.98, 91, 0.035);   // human labels: the student passes its teacher
bell(244.95, 84, 0.045); bell(244.95, 88, 0.03);     // the cache, before the student, answers a message it has seen (244.914 + its trip)
thud(211.2, 0.32);                                  // "Other languages?"
[65, 69, 72, 76, 79].forEach((m, i) => bell(199.4 + i * 0.012, m + 12, 0.035));
[7.9, 8.45, 9.0, 38.6, 39.05, 39.5].forEach((t) => thud(t, 0.32));
const Q = 42; for (let i = 1; i <= Q; i++) click(1.0 + (i - 0.5) / Q * 1.8, 0.06 + 0.03 * rnd());
[129.75, 130.95, 132.15, 133.35].forEach((t) => click(t, 0.09));   // the LLM's answer tokens go back in
const C = 20; for (let i = 1; i <= C; i++) click(194.5 + (i - 0.5) / C * 1.3, 0.05 + 0.03 * rnd());

// ---------- effects ----------
// ping-pong dotted-eighth delay on the arpeggio
{
  const d = S(BEAT * 0.75), bl = new Float32Array(d), br = new Float32Array(d); let lpL = 0, lpR = 0, i = 0;
  for (let n = 0; n < N; n++) {
    const yl = bl[i], yr = br[i];
    lpL += (yr - lpL) * 0.3; lpR += (yl - lpR) * 0.3;
    bl[i] = arpL[n] + lpL * 0.38; br[i] = arpR[n] * 0.2 + lpR * 0.38;
    arpL[n] += yl * 0.32; arpR[n] += yr * 0.32;
    if (++i >= d) i = 0;
  }
}
// sidechain pump on pad and bass
const duck = new Float32Array(N).fill(1);
kicks.forEach((t0) => { for (let j = 0, i0 = S(t0); j < S(0.5) && i0 + j < N; j++) duck[i0 + j] = Math.min(duck[i0 + j], 1 - 0.42 * Math.exp(-j / SR / 0.13) * Math.min(1, j / (SR * 0.004)) - 0 ); });
for (let n = 0; n < N; n++) {
  const p = duck[n];
  dryL[n] += padL[n] * p + arpL[n] + bassB[n] * p; dryR[n] += padR[n] * p + arpR[n] + bassB[n] * p;
  send[n] += (padL[n] + padR[n]) * 0.5 * p * 0.9 + (arpL[n] + arpR[n]) * 0.5 * 0.7;
}

// Freeverb
function freeverb(inp, spread) {
  const k = SR / 44100, out = new Float32Array(N);
  const combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617].map((d) => ({ b: new Float32Array(Math.round((d + spread) * k)), i: 0, s: 0 }));
  const aps = [556, 441, 341, 225].map((d) => ({ b: new Float32Array(Math.round((d + spread) * k)), i: 0 }));
  const room = 0.87, damp = 0.32;
  for (let n = 0; n < N; n++) {
    const x = inp[n] * 0.015; let s = 0;
    for (const c of combs) { const y = c.b[c.i]; c.s = y * (1 - damp) + c.s * damp; c.b[c.i] = x + c.s * room; if (++c.i >= c.b.length) c.i = 0; s += y; }
    for (const a of aps) { const b = a.b[a.i]; const y = b - s; a.b[a.i] = s + b * 0.5; if (++a.i >= a.b.length) a.i = 0; s = y; }
    out[n] = s * 3;
  }
  return out;
}
const revL = freeverb(send, 0), revR = freeverb(send, 23);

// ---------- master ----------
// gain so the busiest section sits at about -15 dBFS RMS, then a soft clipper with a -0.5 dB ceiling
const WET = 0.42, CEIL = 0.944, TARGET = -15;
const L = new Float32Array(N), R = new Float32Array(N);
for (let n = 0; n < N; n++) { L[n] = dryL[n] + revL[n] * WET; R[n] = dryR[n] + revR[n] * WET; }
svf(L, () => 32, 0.7, "hp"); svf(R, () => 32, 0.7, "hp");
let acc = 0; for (let n = S(89.0); n < S(110.6); n++) acc += (L[n] * L[n] + R[n] * R[n]) / 2;
const gain = Math.pow(10, TARGET / 20) / Math.sqrt(acc / (S(110.6) - S(89.0)));
let peak = 0;
for (let n = 0; n < N; n++) {
  const t = n / SR;
  // In at the start; out at the close (203.6 - 205.5), then in again to lead into the other languages.
  const fade = Math.min(1, t / 0.4) * (t > 203.6 && t < 207.0 ? Math.max(0, 1 - (t - 203.6) / 1.9) : 1)
    * (t >= 207.0 && t < 208.2 ? (t - 207.0) / 1.2 : 1);
  L[n] = CEIL * Math.tanh(L[n] * gain / CEIL) * fade;
  R[n] = CEIL * Math.tanh(R[n] * gain / CEIL) * fade;
  peak = Math.max(peak, Math.abs(L[n]), Math.abs(R[n]));
}
const norm = 1;
// loudness report per section
const sections = [[0, 11], [11, 45.2], [45.2, 51.8], [51.8, 76.4], [76.4, 89.0], [89.0, 110.6], [110.6, 125.0], [125.0, 149.0], [149.0, 152.6], [152.6, 194.0], [194.0, 199.4], [199.4, 203.6], [210.8, 225.2], [225.2, 251.2]];
for (const [a, b] of sections) {
  let s = 0; for (let n = S(a); n < S(b); n++) s += (L[n] * L[n] + R[n] * R[n]) / 2;
  console.log(`${a.toFixed(1).padStart(6)}-${b.toFixed(1).padEnd(6)} RMS ${(10 * Math.log10(s / (S(b) - S(a)) * norm * norm)).toFixed(1)} dBFS`);
}
// 16-bit stereo WAV
const out = Buffer.alloc(44 + N * 4);
out.write("RIFF", 0); out.writeUInt32LE(36 + N * 4, 4); out.write("WAVE", 8); out.write("fmt ", 12);
out.writeUInt32LE(16, 16); out.writeUInt16LE(1, 20); out.writeUInt16LE(2, 22); out.writeUInt32LE(SR, 24);
out.writeUInt32LE(SR * 4, 28); out.writeUInt16LE(4, 32); out.writeUInt16LE(16, 34); out.write("data", 36); out.writeUInt32LE(N * 4, 40);
for (let n = 0; n < N; n++) {
  out.writeInt16LE(Math.round(clamp(L[n] * norm, -1, 1) * 32767), 44 + n * 4);
  out.writeInt16LE(Math.round(clamp(R[n] * norm, -1, 1) * 32767), 46 + n * 4);
}
fs.mkdirSync(path.join(__dirname, "out"), { recursive: true });
const wav = path.join(__dirname, "out", "music.wav"), m4a = path.join(__dirname, "out", "music.m4a");
fs.writeFileSync(wav, assemble(out, timeline.segments));
const aac = spawnSync(require("ffmpeg-static"), ["-y", "-loglevel", "error", "-i", wav,
  "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", m4a], { stdio: "inherit" });
if (aac.status !== 0) throw new Error(`ffmpeg ${aac.status}: out/music.m4a`);
console.log(`gain ${(20 * Math.log10(gain)).toFixed(1)} dB, peak ${peak.toFixed(3)}; wrote out/music.wav and out/music.m4a (${timeline.duration()} s)`);
