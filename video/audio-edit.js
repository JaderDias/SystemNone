// Assemble the soundtrack from the synthesized 16-bit stereo WAV, in the order the video plays its
// source passages (timeline.js), so every visual cue keeps its sound.
//
// At each join, the outgoing passage, as it would have gone on, blends into the incoming one, as
// it would have led in, over `fadeSeconds` centered on the join. Complementary cosine weights
// keep the level of shared sustained notes and have zero slope at both ends; unlike fades to
// silence, no gap is added. Joins with a beat on both sides keep its phase (timeline.js).
function assemble(wav, segments, fadeSeconds = 2.4) {
  const rate = wav.readUInt32LE(24), total = wav.readUInt32LE(40) / 4;
  const read = (i, channel) => (i >= 0 && i < total ? wav.readInt16LE(44 + i * 4 + channel) : 0);
  const spans = segments.map(([a, b]) => [Math.round(a * rate), Math.round(b * rate)]);
  const samples = spans.reduce((n, [a, b]) => n + b - a, 0);
  const out = Buffer.alloc(44 + samples * 4);
  wav.copy(out, 0, 0, 44);
  out.writeUInt32LE(36 + samples * 4, 4);
  out.writeUInt32LE(samples * 4, 40);
  let p = 0;
  for (const [a, b] of spans) {
    wav.copy(out, 44 + p * 4, 44 + a * 4, 44 + b * 4);
    p += b - a;
  }
  let join = 0;
  for (let k = 0; k + 1 < spans.length; k++) {
    const [a0, b0] = spans[k], [a1, b1] = spans[k + 1];
    join += b0 - a0;
    const half = Math.min(Math.round(fadeSeconds * rate / 2), b0 - a0, b1 - a1);
    const width = half * 2;
    for (let j = 0; j < width; j++) {
      const incoming = (1 - Math.cos(Math.PI * j / (width - 1))) / 2;
      for (const channel of [0, 2]) {
        const before = read(b0 - half + j, channel), after = read(a1 - half + j, channel);
        out.writeInt16LE(Math.round(before * (1 - incoming) + after * incoming), 44 + (join - half + j) * 4 + channel);
      }
    }
  }
  return out;
}

module.exports = { assemble };
