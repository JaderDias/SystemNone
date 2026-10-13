const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const { assemble } = require("./audio-edit");
const timeline = require("./timeline");

const rate = 1000, B0 = 45.2, BEAT = 0.6;
const end = Math.max(...timeline.segments.map(([, b]) => b)) + 2;
const samples = Math.round(end * rate);
const wav = Buffer.alloc(44 + samples * 4);
wav.writeUInt32LE(rate, 24);
wav.writeUInt32LE(samples * 4, 40);
const fill = (f) => {
  for (let i = 0; i < samples; i++) {
    const [l, r] = f(i);
    wav.writeInt16LE(l, 44 + i * 4);
    wav.writeInt16LE(r, 46 + i * 4);
  }
};
// A source as continuous as the real one, whose level rises across the whole track: the two sides
// of every join differ, so a hard cut, a silence dip or a stereo mix-up shows.
const level = (i) => Math.round(3000 + 6000 * i / samples);
fill((i) => [level(i), -level(i)]);
const edited = assemble(wav, timeline.segments);
const length = Math.round(timeline.duration() * rate);
assert.equal(edited.length, 44 + length * 4);
assert.equal(edited.readUInt32LE(4), edited.length - 8);
assert.equal(edited.readUInt32LE(40), edited.length - 44);
const joins = [];
let at = 0;
for (const [a, b] of timeline.segments) {
  // Away from the joins, each passage is the source as it was.
  const keep = (b - a) * rate - 2 * 1200 - 2;
  if (keep > 0) {
    const from = Math.round((at + (at > 0 ? 1.201 : 0)) * rate);
    const src = Math.round((a + (at > 0 ? 1.201 : 0)) * rate);
    assert.deepEqual(edited.subarray(44 + from * 4, 44 + (from + keep) * 4), wav.subarray(44 + src * 4, 44 + (src + keep) * 4));
  }
  at += b - a;
  joins.push(Math.round(at * rate));
}
joins.pop();
for (const join of joins) {
  for (let i = join - 1200; i <= join + 1200; i++) {
    for (const channel of [0, 2]) {
      const offset = 44 + i * 4 + channel;
      assert.ok(Math.abs(edited.readInt16LE(offset)) >= 3000, "A join must not dip into silence");
      assert.ok(Math.abs(edited.readInt16LE(offset) - edited.readInt16LE(offset - 4)) <= 10, "A join must be continuous");
    }
  }
}
// A beat on the 100 BPM grid keeps its phase and level across every join whose two passages are
// on the same phase of it. The one join that isn't goes from a beat into a passage without one.
const phase = (t) => Math.round((((t - B0) % BEAT) + BEAT) % BEAT * 1000) / 1000 % BEAT;
const off = timeline.segments.slice(0, -1).map(([, b], k) => [b, timeline.segments[k + 1][0]])
  .filter(([out, inc]) => phase(out) !== phase(inc));
assert.deepEqual(off, [[171.4, 111.2]], "Only the join into 'Why is the student model so fast?' may change the beat's phase");
fill((i) => { const s = Math.round(Math.sin(2 * Math.PI * (i / rate - B0) / BEAT) * 12000); return [s, -s]; });
const rhythmic = assemble(wav, timeline.segments);
at = 0;
timeline.segments.forEach(([a, b], k) => {
  at += b - a;
  if (k + 1 === timeline.segments.length || phase(b) !== phase(timeline.segments[k + 1][0])) return;
  const join = Math.round(at * rate);
  for (let i = join - 1200; i < join + 1200; i++) {
    const src = Math.round(b * rate) + (i - join);
    assert.ok(Math.abs(rhythmic.readInt16LE(44 + i * 4) - wav.readInt16LE(44 + src * 4)) <= 1, `beat phase at the join at ${at} s`);
  }
});
// The soundtrack, if it has been made, plays as long as the video.
const music = path.join(__dirname, "out", "music.wav");
if (fs.existsSync(music)) {
  const made = fs.readFileSync(music);
  assert.equal(made.readUInt32LE(40) / 4, Math.round(timeline.duration() * made.readUInt32LE(24)), "out/music.wav: run npm run music again");
  // The page plays the compressed copy, made from this one.
  const page = path.join(__dirname, "out", "music.m4a");
  assert.ok(fs.existsSync(page) && fs.statSync(page).mtimeMs >= fs.statSync(music).mtimeMs, "out/music.m4a: run npm run music again");
}
console.log("Soundtrack assembly keeps the duration, stereo, beat phase and continuous joins.");
