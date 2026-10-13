# HTML video workflow

The deliverable is `system-none.html`, using shared CSS and JavaScript. Keep it a directly
playable HTML entry point.

Every scene's cues are in source time; `timeline.js` lists the source passages the video plays,
in order, and the soundtrack follows the same list. To move a scene, move its passage; to add one,
give it source times after the last scene's (the other-languages scenes start at 210.8 s) and a
passage in `timeline.js`. Keep every join on a blank screen (the HTML checks test it), and keep the
beat's phase across joins with a beat on both sides (the audio checks test it).

For routine copy, slide and timing edits, run `npm run check` and inspect the HTML player. Generate
audio with `npm run music` when soundtrack timing changes. MP4 export is optional: use
`npm run export:mp4` only when the user requests an MP4. The exporter has already been tested end
to end; do not regenerate MP4s merely to validate content changes. If the encoding pipeline itself
changes, a short export smoke test is enough unless the user requests a complete export.

The opening still must stay pixel-identical to the reveal after “Introducing”.
Use the shared render function at `timeline.js`'s `posterTime`; do not add separate
poster text, layout or dimming. Keep the poster/reveal comparison in the HTML checks.
