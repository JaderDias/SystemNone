// Exercise the video in the actual renderer, including arbitrary-time seeking.
const assert = require("node:assert/strict");
const { serve, openPage } = require("./render");
const timeline = require("./timeline");

function captureStage() {
  const stage = document.getElementById("stage");
  const properties = ["display", "visibility", "opacity", "transform", "filter",
    "color", "background", "background-position", "font-family", "font-size",
    "font-weight", "line-height", "letter-spacing", "text-align", "box-shadow",
    "border", "border-radius", "padding", "margin", "position", "overflow"];
  return {
    filter: getComputedStyle(stage).filter,
    nodes: Array.from(stage.querySelectorAll("*")).map((el) => {
      const style = getComputedStyle(el), rect = el.getBoundingClientRect();
      return { id: el.id, text: el.textContent,
        bounds: [rect.x, rect.y, rect.width, rect.height],
        styles: properties.map((property) => style.getPropertyValue(property)) };
    }),
  };
}

// The text a viewer can see at playback time t.
function visibleText(t) {
  window.render(t);
  const walker = document.createTreeWalker(document.getElementById("stage"), NodeFilter.SHOW_TEXT);
  const parts = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    let el = node.parentElement, shown = true;
    while (el) {
      const s = getComputedStyle(el);
      if (s.display === "none" || s.visibility === "hidden" || +s.opacity < 0.01) { shown = false; break; }
      el = el.parentElement;
    }
    if (shown && node.textContent.trim()) parts.push(node.textContent.trim());
  }
  return parts.join(" ");
}

async function check() {
  const server = await serve();
  let browser;
  try {
    const renderer = await openPage(server.address().port);
    browser = renderer.browser;
    const page = renderer.page;
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.evaluate(() => window.ready);
    await page.evaluate(`window.visibleText = ${visibleText.toString()}`);
    const duration = await page.evaluate(() => window.DURATION);
    assert.equal(duration, 246.0);
    const at = (source) => timeline.playbackTime(source);
    // The story: the English student first, then why it's fast, then other languages, then the close.
    const order = [["the numbers", 101], ["escalation", 152.2], ["human labels", 166.7], ["why it's fast", 113.6],
      ["inside the models", 139.2], ["a bigger multilingual student", 213.4], ["a student per language", 228.0],
      ["the cache", 241.3], ["careful with your data", 171.4], ["the close", 199.4]];
    order.slice(1).forEach(([name, s], i) => assert.ok(at(s) > at(order[i][1]), `${name} comes after ${order[i][0]}`));
    // Every second, and the important beats, seek and render cleanly.
    const beats = [33.2, 37, 74, 88.8, 101, 108.6, 143, 167, 216.8, 221.0, 231.0, 236.0, 245.0, 248.0].map(at);
    for (const t of [...Array.from({ length: Math.ceil(duration) }, (_, i) => i), ...beats]) {
      const visible = await page.evaluate((t) => window.visibleText(t), t);
      assert.doesNotMatch(visible, /Need other languages|prompt's student|Not sure of the language/, `a line the video no longer has at ${t}s`);
    }
    // Every join falls where the screen shows no text, on both sides.
    let join = 0;
    for (const [a, b] of timeline.segments.slice(0, -1)) {
      join += b - a;
      for (const t of [join - 0.02, join]) assert.equal(await page.evaluate((t) => window.visibleText(t), t), "", `text at the join at ${t}s`);
    }
    // The other-languages scenes say what they are about.
    const says = async (source, text) => assert.match(await page.evaluate((t) => window.visibleText(t), at(source)), text, `at source ${source}s`);
    await says(212.0, /Other languages\?/);
    await says(217.0, /A multilingual student must be bigger/);
    await says(221.0, /It takes longer to train/);
    await says(223.4, /Still faster than System One/);
    await says(231.0, /A language router reads each message first.*Language router.*Student.*English.*Português/);
    await says(236.0, /A language router reads each message first.*Hindi|हिन्दी/);
    await says(245.0, /Seen it before\?.*Student model.*System One.*LLM.*Cache.*Before the student model/);
    await says(247.0, /Between the student model and System One/);
    await says(249.5, /Between System One and the LLM.*every step after it/);
    const state = await page.evaluate(() => {
      window.render(KEYNOTE_TIMELINE.playbackTime(108.6));
      return {
        training: document.getElementById("s7t1").textContent,
        hardware: document.getElementById("s7t2").textContent,
        layers: IM_KEEP.length,
        hiddenSize: SCENARIO.hiddenSize,
        device: document.getElementById("s3cpu").textContent.trim(),
      };
    });
    assert.deepEqual(state, { training: "Trained in 10 minutes.", hardware: "On a CPU.", layers: 7, hiddenSize: 256, device: "CPU" });
    // Compare the actual player still with the reveal render, including every
    // element's text, geometry, typography, colors, opacity and effects.
    // A fresh page each, as some hooks leave what earlier frames drew.
    const renderUrl = page.url();
    await page.goto(renderUrl);
    await page.evaluate(() => window.ready);
    await page.evaluate(async () => {
      window.render(KEYNOTE_TIMELINE.playbackTime(KEYNOTE_TIMELINE.posterTime));
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    });
    const reveal = await page.evaluate(captureStage);
    await page.goto(renderUrl.replace("?render", ""));
    await page.evaluate(() => window.ready);
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const poster = await page.evaluate(captureStage);
    assert.equal(JSON.stringify(poster), JSON.stringify(reveal), "The poster must match the reveal frame");
    // On a phone held sideways the stage fills the screen's height; upright, it fits the width and
    // asks to turn the phone. A tap during playback shows the hidden controls; the next one pauses.
    const phone = await browser.newPage();
    phone.on("pageerror", (error) => errors.push(error.message));
    const onPhone = async (width, height) => {
      await phone.setViewport({ width, height, isMobile: true, hasTouch: true, isLandscape: width > height, deviceScaleFactor: 2 });
      await phone.goto(renderUrl.replace("?render", ""));
      await phone.evaluate(() => window.ready);
      return phone.evaluate(() => {
        const r = document.getElementById("stage").getBoundingClientRect();
        return { window: [innerWidth, innerHeight], stage: [Math.round(r.width), Math.round(r.height)],
          turn: getComputedStyle(document.getElementById("pl-turn")).display !== "none",
          hint: document.getElementById("pl-hint").textContent };
      });
    };
    assert.deepEqual(await onPhone(390, 844), { window: [390, 844], stage: [390, 219], turn: true, hint: "" });
    assert.deepEqual(await onPhone(844, 390), { window: [844, 390], stage: [693, 390], turn: false, hint: "" });
    const shown = () => phone.evaluate(() => ({
      playing: document.getElementById("pl-big").classList.contains("off"),
      controls: !document.getElementById("pl-bar").classList.contains("off"),
    }));
    await phone.tap("#pl-big");
    assert.equal((await shown()).playing, true, "the play button plays");
    await phone.waitForFunction(() => document.getElementById("pl-bar").classList.contains("off"), { timeout: 5000 });
    await phone.touchscreen.tap(422, 195);
    assert.deepEqual(await shown(), { playing: true, controls: true }, "a tap shows the controls");
    await phone.touchscreen.tap(422, 195);
    assert.deepEqual(await shown(), { playing: false, controls: true }, "the next tap pauses");
    await phone.close();
    assert.deepEqual(errors, []);
    console.log("The video renders and seeks cleanly, its scenes play in order, every join is blank, the English measurements hold, and it plays on a phone.");
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}
check().catch((error) => { console.error(error); process.exitCode = 1; });
