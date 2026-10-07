// The video plays these source passages in order. Each scene keeps its cues in source time, so a
// scene moves by moving its passage, not its cues; the soundtrack follows the same edit. Every
// join falls where the screen is blank, and the beat keeps its phase across joins with a beat on
// both sides (music.js), so the 100 BPM grid survives the edit.
(function (root) {
  const SEGMENTS = [
    [0.0, 111.2], //    the question, the bill, the insight, the reveal, how it works, calibration, the numbers
    [149.0, 171.4], //  escalation, human labels
    [111.2, 149.0], //  why it's fast: what each request carries, inside the models
    [210.8, 251.2], //  other languages: a bigger multilingual student, a student per language, the cache
    [171.4, 205.6], //  careful with your data, the product, the close
  ];
  const starts = [];
  let total = 0;
  for (const [a, b] of SEGMENTS) {
    starts.push(total);
    total += b - a;
  }
  const duration = Math.round(total * 1000) / 1000;
  const timeline = {
    segments: SEGMENTS,
    posterTime: 48.8, // The opening still is this exact frame of the Introducing reveal (source time).
    duration: () => duration,
    // The source time shown at playback time t.
    sourceTime: (t) => {
      for (let i = SEGMENTS.length - 1; i >= 0; i--) {
        if (t >= starts[i] - 1e-9) return SEGMENTS[i][0] + Math.min(t - starts[i], SEGMENTS[i][1] - SEGMENTS[i][0]);
      }
      return SEGMENTS[0][0];
    },
    // The playback time of source time s (the first passage that shows it).
    playbackTime: (s) => {
      for (let i = 0; i < SEGMENTS.length; i++) {
        const [a, b] = SEGMENTS[i];
        if (s >= a - 1e-9 && s < b - 1e-9) return starts[i] + (s - a);
      }
      return s >= SEGMENTS[SEGMENTS.length - 1][1] - 1e-9 ? duration : 0;
    },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = timeline;
  else root.KEYNOTE_TIMELINE = timeline;
})(globalThis);
