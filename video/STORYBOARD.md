# SystemNone keynote video: storyboard and script

Optional export: `out/system-none-keynote.mp4` (built by `npm run export:mp4`, not committed), 4:06 (246.0 s), 1920×1080, 60 fps, stereo music, no voiceover.
`player.html` plays its half-size copy for the web, `out/system-none-keynote-web.mp4`, with `out/poster.jpg` as its
opening still, in the browser's own video player.
The HTML player, `system-none.html`, takes its visible markup from `scenes.js`, its styles from
`keynote.css` and its animation from `keynote.js` (search for the element id).

To watch it without rendering, run `npm ci` and `node music.js` once, then open `system-none.html` in a browser.
It plays like a video: click or press Space to play, ← → to skip 5 s, F for full screen. On a phone, turn it
sideways; a tap shows the controls.

Style: black background, Inter Display type, one idea per beat. Text fades up from a slight blur.
Accent colors: green for the student model, cyan for System One, orange for the LLM, and a
mint→blue→violet gradient for the SystemNone wordmark.

## Order

The video follows an English student model to why it is fast, then turns to other languages, and
closes. Its scenes keep their cues in source time (the times below and in `keynote.js`), and
`timeline.js` plays these passages of source time in order, the soundtrack too:

| Plays at | Source | Scenes |
|---|---|---|
| 0:00 | 0.0 – 111.2 | 1–7: the question, the bill, the insight, the reveal, how it works, calibration, the numbers |
| 1:51 | 149.0 – 171.4 | 10–11: escalation, human labels |
| 2:13 | 111.2 – 149.0 | 8–9: why it's fast, inside the models |
| 2:51 | 210.8 – 251.2 | 15–17: other languages: a bigger multilingual student, a student per language, the cache |
| 3:31 | 171.4 – 205.6 | 12–14: careful with your data, the product, the close |

Every join falls on a blank screen, and the beat keeps its phase across the joins with a beat on
both sides; the one from human labels into scene 8 goes into a passage without one.

## Scenes

Times are source times, in seconds. ⟨music⟩ marks a sound cue in `music.js`.

### Poster (before the first play)
The opening still is the exact reveal frame at `timeline.js`'s `posterTime`
(48.8 seconds), after “Introducing”. The HTML player calls the same render
function for this still and for playback: there is no separate poster copy,
layout or dimming. Only the player controls are overlaid. Changes to the reveal
therefore update the opening still automatically. CI compares their text, geometry and rendered styles.

### 1. The question (0.0 – 10.8)
- 0.4: a card slides up, labeled **YOUR CUSTOMER ASKS**.
- 1.0 – 2.8: typed in: **"I was charged twice. I want my money back."** ⟨typing clicks⟩
- 2.95: label chips appear: **refund · cancel · other**.
- 3.5: status **"Asking a frontier model…"**, a spinner, and a timer counting 0.0 → 2.4 s.
- 6.2: **refund** lights up blue, the status becomes **"Answered by a frontier model"**, and the timer freezes at **2.4 s** in orange. ⟨bell⟩
- 6.9: the card rises and shrinks.
- 7.9 / 8.45 / 9.0: **"Slow."  "Expensive."  "Every single time."** (the last one in a warm gradient) ⟨soft thud on each⟩

### 2. The bill (10.9 – 20.7)
- 10.9: a big counter runs **1 → 1,000,000 messages**. ⟨riser 11.4 – 13.9⟩
- 14.3: **$8,120** *per million messages*. 14.8: **2.4 s** *for every answer*. Warm glow. ⟨impact⟩
- 16.6: **"Just to pick one label."**
- Footnote: *Claude Opus 5.5 on banking77, a 77-label customer-support benchmark.*

### 3. The insight (21.0 – 43.3)
- 21.0: **"What if a *tiny* model could answer instead?"** ("tiny" in green)
- A zoom from a datacenter down to one CPU, in one camera move: training, then inference, then System One, then the student model.
  - 24.4: a datacenter hall, three rows of GPU racks with amber status lights, fading out at both sides.
    24.7: **"A *frontier model* trains on a whole datacenter."** ("frontier model" in orange)
  - 26.0: one rack in the front row lights up orange. 26.6 – 28.0: the camera zooms into it while the rest
    of the hall fades away, leaving one rack: power shelves, 18 compute trays of four GPUs each and 9 switch trays.
  - 27.9: **"It runs on multiple 8 GPU servers."** From 28.1 its 72 GPUs light up orange, one after another,
    and a bracket labels them *72 × B300*.
  - 29.2: one compute tray lights up white. 29.8 – 30.7: the camera zooms into it. Inside, seen from above:
    four **B300** GPUs, still lit orange, two CPUs with their memory, and a row of fans.
  - 30.5: **"A *System One model* requires multiple GPUs."** ("System One model" in cyan), while the camera holds on the tray.
  - 33.2: one CPU lights up green. 33.7 – 34.9: the camera zooms into it until the CPU fills the frame.
  - 34.7: **"A *student model* trains and runs on one CPU."** ("student model" in green) /
    *"As small as 17 million parameters, answering in under a millisecond."* 35.0: a glowing green dot appears on the CPU.
- 38.6 / 39.05 / 39.5: **"Faster."  "Cheaper."  "Just as accurate."** ⟨thuds⟩
  - Footnote: *banking77 test set: Claude Opus 5.5 92%, a 17M-parameter Ettin student model 91.95%.*
- 41.1: **"But few teams know how to fine-tune one."** ⟨riser 41.6 – 45.2⟩

Under the zoom the chords go D minor (21.2), B♭ (26.0), G minor (33.2) and C (38.0), resolving to F at the reveal.

### 4. Reveal (43.3 – 51.2)
- 43.3: *Introducing*, as large as the line before it
- 45.2: the ∅ app icon and the **SystemNone** wordmark resolve from a blur. ⟨impact; the beat grid starts here⟩
- 47.1: **"Zero-click fine-tuning, performance and cost optimization."** 47.7: *"Behind the API you already call."*

### 5. How it works (51.4 – 76.5)
The diagram:
- **Your app** (`POST /v1/systemone`) → **SystemNone gateway** (*Rust*) → **System One** (*model or LLM*)
- Below the gateway, **Store** (*n / 10,000 inputs*) and **Worker · Python** (Judge → Fine-tune → Calibrate → ONNX)

Dots: requests go out on the upper lane (white). Answers come back on the lower lane: green ones
return fast from the student model, grey ones return from System One after about a second's wait.

| Time | Caption at the top | What moves |
|---|---|---|
| 51.6 | **Day one: your System One model answers.** | all requests go to System One ⟨beat starts 51.8⟩ |
| 56.55 | **SystemNone logs the inputs.** | blue copies drop into the Store, and its counter runs 0 → 10,000; **Ready to train** from 60.9 until the student model is trained (66.0) |
| 61.35 | **At 10,000 requests, a student model is trained for you.** | worker steps light up at 62.1, 63.1, 64.1 and 65.1 ⟨hi-hats from 64.4⟩ |
| 66.75 | **The gateway loads it. Your app changes nothing.** (until the scene ends: who answers when the student model is unsure comes in scenes 6, 10 and 17) | chip **✓ Student model** appears at 66.0, flies into the gateway 66.9 – 67.8 ⟨bell 67.8⟩ |
| 71.05 | | 89% of requests turn green; *Answered by the student model* counts to **89%** |

### 6. Calibration (76.5 – 88.7)
The beat drops out.
- 76.5: **"Confidence you can trust."**
- 77.8: **"You choose the accuracy."** *"SystemNone picks the threshold."*
  - The pill **Target: 95% accuracy**, then a bar that fills to 89% green.
  - Labels: **The student model answers 89%** / *System One answers the rest*.
  - Footnote: *banking77 test set, Ettin 17M student model, 95% accuracy target.* ⟨riser 86.6 – 89.0⟩

### 7. The numbers (89.0 – 111.2)
Giant green gradient numbers over a soft green glow; the beat comes back with claps.
- 89.0: **2,500×** (counts up) / **faster** / *0.96 ms instead of 2.4 s* ⟨impact⟩
- 95.0: **≈ $0** / **per million messages** / *instead of $8,120* ⟨impact⟩
- 101.0: **91.95%** (counts up) / **accurate** / *A frontier LLM scored 92%.* ⟨impact⟩
- 106.6: **"Trained in 10 minutes."** 107.2: **"On a CPU."** (gradient)
- Footnote throughout: *banking77, 77 labels. Ettin 17M trained by SystemNone on banking77's human labels, scored on 3,080 test messages, ONNX on CPU. Claude Opus 5.5 scored on 200 messages (±2 points).*

### 8. Why it's fast: what each request carries (111.2 – 124.8)
The beat drops out; the pad and the arpeggio carry on.
- 111.2: **"Why is the student model so fast?"**
- 113.6: **"An LLM or a System One model gets the prompt and the labels"** / *"with every request, on top of the input."*
- 113.9: three requests build up, chip by chip, under the labels PROMPT, LABELS and INPUT:
  - **LLM** (orange) and **System One** (cyan): *What does the customer want?* · *refund* · *cancel* · *other* · *I was charged twice. I want my money back.*
  - **Student model** (green): only *I was charged twice. I want my money back.*, lined up under the other inputs.
- From 115.3, every 1.8 s, a read head sweeps each request at the same speed, lighting each chip in its model's color, and a check mark appears when it reaches the end. The student model's request is the shortest, so its check comes first.
- 118.6: **"A student model learned them in training."** / *"It only needs the input."* 118.9: a dashed green box, *Learned in training*, fills the student model's empty prompt and labels.
- 121.3: **"Less to read, every time: *one reason it's faster.*"** (green gradient)

### 9. Inside the models (125.2 – 149.0)
In the manner of 3Blue1Brown: tokens become columns of numbers (vectors), the vectors run through a stack of layers, and the output scores what comes next. Everything but the captions is drawn on one canvas (`imcv`). Hi-hats come in on the offbeats (125.0 – 149.0).
- 125.2: eyebrow *Inside the models*.
- **The LLM** (orange):
  - 125.4: **"An LLM turns each token into a vector"** / *"and runs the vectors through its layers."*
  - 125.7: 20 bracketed columns of numbers build up: the prompt's tokens (grey), the labels' (blue) and the input's (white), captioned PROMPT · LABELS · INPUT / *12,288 numbers per token*.
  - 126.5: 96 thin layers build up to the right: *96 decoder layers* / *each: attention, then an MLP*. 127.5: the output, *all 50,257 possible tokens*.
  - 128.25: **"It writes its answer one token at a time,"** / *"running every layer again for each one."*
  - Four passes, two beats apart (128.6, 129.8, 131.0, 132.2): a glow sweeps through the layers, five tokens get scored, the top one lights up and rides a dotted loop (*each new token goes back in*) into the input as a new orange column. *answer:* builds up to **The answer is refund**. ⟨a click as each token goes back in: 129.75, 130.95, 132.15, 133.35⟩
  - Footnote: *GPT-3 175B's published sizes (Brown et al., 2020); frontier models don't publish theirs. The tokens and scores are illustrative.*
- **System One** (cyan), 134.3: the loop and the answer tokens go, and the output becomes *only the 3 labels*.
  - **"A System One model skips the loop."** / *"One pass through its layers scores the labels."* The input is still PROMPT · LABELS · INPUT, *the whole request, every time*; the layers read *one pass*.
  - 135.2: one sweep, and **refund** fills to 92%.
  - Footnote: *As Decider-2B does: it reads every option's probability from one forward pass, without generating a token.*
- **The student model** (green), 139.2: **"A student model is simpler still:"** / *"fewer layers, smaller vectors, only the input."*
  - The prompt and label columns drop out, the columns shrink to *256 numbers per token* (INPUT), and 89 of the 96 layers fade, leaving *7 encoder layers* / *each: attention, then an MLP*. The output is *one output per label*.
  - 141.2: one quick sweep, and **refund** fills to 94%. ⟨bell 141.45⟩ From 142.4 it sweeps again on every beat.
  - 142.3: **"One small pass, *under a millisecond.*"** (green gradient)
  - Footnote: *Ettin 17M's published sizes: 7 layers, 256 numbers per token. 0.96 ms per answer on banking77, ONNX on a CPU.*

### 10. Escalation (149.6 – 161.8)
- 149.6: **"And when System One isn't sure?"** (the pad closes in and the hats stop, 149.0 – 152.6)
- 152.1: eyebrow *New: escalation* (orange). 152.2: **"It asks a stronger model."** ⟨the beat comes back at 152.6⟩
- The cascade: **Student model** (*fastest*) —*unsure*→ **System One** (*model*) —*still unsure*→ **LLM** (*strongest*). Each card glows as requests land on it: about 62% stop at the student model, 24% at System One and 14% at the LLM. The shares are illustrative.
- 153.6: *"Classifier, System One, LLM:"* **"a three-step cascade."**

### 11. Human labels (161.8 – 171.4)
- 161.9: **"It learns from people, too."**
- 162.4: *"You can upload datasets with human labels, which are superior to those from LLM judges."*
- 163.1: three bars fill in turn:
  - **The teacher: an LLM judge**, 81.6% (orange)
  - *Student model, taught by the judge*, 79.5%
  - **Student model, taught by people**, 91.5% (green)
- 164.7: a dashed orange line marks the teacher's score. At 165.98 the last bar passes it and glows. ⟨bell 165.98⟩
- 166.7: **"The student model can *surpass its teacher.*"** (green gradient)
- Footnote: *banking77, Ettin 17M student model. With Kimi K3 as the judge (81.6%), it scored 79.5% on the judge's labels and 91.5% on human labels.*

### 12. Careful with your data (171.4 – 180.7)
**"Careful with your data."** and two columns:
1. 📄 **License-aware.** Judges whose terms forbid distillation are refused.
2. 🛡 **Private training.** Training runs on your servers. With a self-hosted judge, no closed-model provider sees personal data.

### 13. Product (180.5 – 193.6)
- 180.5: **"Every prompt, and who answered it."** A usage dashboard window tilts up into view. Its data is illustrative:
  - tiles: Requests, Today, *Answered by student models* 91%, *Student models live* 3
  - five prompts, with student model share and status
- 187.1: a feature wall fades in:
  - Rust gateway, ONNX Runtime, Soft labels, DeBERTa-v3, ModernBERT, Ettin
  - EmbeddingGemma heads, No GPU needed, mDeBERTa-v3, XLM-R, Routes English to the best student model
  - Calibrated thresholds, Drift monitoring, Admin console, Usage dashboard, Per-prompt overrides
  - S3 or a folder, OpenAI-compatible judges, Train now, Docker Compose, Nix flake, cargo audit
  - KL distillation, Python library & CLI

### 14. Close (193.8 – 205.6)
- 193.8: a terminal, with the beat off. ⟨typing clicks⟩ ⟨riser 195.8 – 199.4⟩

  ```
  $ docker compose up -d
  ✔ Container system-none-gateway-1  Started
  ✔ Container system-none-worker-1   Started

  Open http://localhost:8080
  ```
- 199.4: the ∅ icon and the **SystemNone** wordmark. ⟨final impact and a shimmer chord⟩
- 200.7: *"Faster answers."* **"Same API."**
- 201.2: the link **github.com/JaderDias/SystemNone** (cyan; in the player it opens https://github.com/JaderDias/SystemNone).
- 203.6 – 205.2: fade to black.

### 15. Other languages: a bigger multilingual student (210.8 – 225.2)
Hats only under the pad, as under the models' passes. ⟨thud 211.2⟩
- 211.2: **"Other languages?"**
- 213.4: **"A multilingual student must be bigger:"** / *"its vocabulary and its vectors hold every language."*
- Two models drawn to scale (a pixel for every 1,200 parameters), each with its layers above and its
  vocabulary table (a vector for every token it knows) below:
  - 213.9: **Ettin 17M** / *English*: *17 million parameters*, *7 layers · 256 numbers per token*; its table, *50,368 tokens × 256*.
  - 214.6 – 216.2: **mmBERT-base** / *110 languages* grows out of it: *307 million parameters*,
    *22 layers · 768 numbers per token*. 216.4: *Vocabulary: 256,000 tokens × 768 numbers* / *197 million
    parameters, two thirds of it*; *22 layers: 110 million*.
- 219.8: **"It takes longer to train,"** / *"and it wants a GPU."* 220.3: under each model, *minutes on a CPU* and *longer, on a GPU*.
- 222.2: **"Still faster than System One, *and cheaper to run.*"** ⟨riser 222.8 – 225.2⟩
- Footnote: *Published sizes: Ettin 17M, 50,368 tokens × 256 numbers; mmBERT-base, 256,000 tokens × 768 numbers, 307M parameters.*

### 16. Other languages: a student per language (225.2 – 240.8)
The beat comes back. ⟨impact 225.2⟩
- 225.6: **"Or: a *tiny* student for each language."**
- 228.0: **"A language router reads each message first"** / *"and sends it to its language's own student."*
- 228.3: three messages, *I was charged twice.*, *Cobraram-me duas vezes.* and *मुझसे दो बार पैसे लिए गए।*; 228.6: the
  **Language router** (*which language?*); 229.0: three green **Student** cards, *English*, *Português* and *हिन्दी*.
- From 229.6, every 0.4 s, a message goes into the router and comes out to its language's student, in the
  language's color, and the student glows. The router glows as each message reaches it.
- 236.8: **"Each student stays tiny: *fast on a CPU.*"**
- Footnote: *The router: a small trigram model in the gateway. A language gets its own student once the router is sure of enough of its requests (300 by default).*

### 17. Other languages: the cache (240.8 – 251.2)
- 241.2: eyebrow *New: the cache* (blue). 241.3: **"Seen it before?"** 241.9: *"The cache answers with what System One or the LLM said last time."*
- 242.3: the cascade from scene 10, **Student model** (*fastest*) → **System One** (*model*) → **LLM** (*strongest*);
  242.8: a blue **Cache** card (*seen before?*) at the front of it, captioned *Before the student model*.
- 243.0: a new message passes the cache and goes to System One; 243.45, one stops at the student model; 243.9, one
  goes on to the LLM; 244.9, one the cache has seen stops there ⟨bell 244.95⟩.
- 245.4: the cache slides between the student model and System One (*Between the student model and System One*);
  247.6: between System One and the LLM (*Between System One and the LLM*). A message it has seen stops at it; the
  others go on, each to the first step sure of it. None starts while it moves. The shares are illustrative.
- 248.7: **"Anywhere you put it, *it saves every step after it.*"**

## Where the numbers come from
All figures are from the project README unless noted otherwise.

| On screen | Source |
|---|---|
| $8,120 / 2.4 s / 92% | README comparison table (Maxime Rivest), Claude Opus 5.5 on banking77, 200 test messages |
| 91.95%, 0.96 ms, ≈ $0, 10 minutes | README banking77 benchmark: Ettin 17M on human labels, 3,080 test messages, 586 s of CPU fine-tuning, ONNX p50 |
| 2,500× faster | 2.4 s ÷ 0.96 ms |
| trains and runs on one CPU (zoom) | README banking77 benchmark: Ettin 17M fine-tuned in 586 s on a CPU, 0.96 ms per answer on a CPU |
| 89% answered by the student model at a 95% target | README banking77 table ("89%, of which 96.4% right") |
| 10,000 requests | gateway default `default_threshold: 10_000` in `gateway/src/db.rs`, configurable per prompt |
| 81.6% / 79.5% / 91.5% | README "Human labels": tiny-classifiers on banking77, Kimi K3 itself, then Ettin 17M on Kimi K3's labels and on human labels |
| 96 decoder layers, 12,288 numbers per token, 50,257 possible tokens | GPT-3 175B as published: Brown et al., "Language Models are Few-Shot Learners" (2020), Table 2.1 (96 layers, d_model 12,288) and its 50,257-token vocabulary |
| 7 encoder layers, 256 numbers per token | Ettin 17M's `config.json` on Hugging Face (`jhu-clsp/ettin-encoder-17m`: `num_hidden_layers` 7, `hidden_size` 256) |
| System One scores the labels in one pass | Decider-2B's model card (`Mapika/decider-2b`): one forward pass gives every option's probability, and the option letters are never generated |
| mmBERT-base: 256,000 × 768, 22 layers, 307M, 197M of it the vocabulary table, 110M the layers, 110 languages | `jhu-clsp/mmBERT-base`'s model card ("307M" total, "110M" non-embedding, the Gemma 2 tokenizer with "256,000" tokens) and the mmBERT announcement (huggingface.co/blog/mmbert: 2.3T tokens in 60 languages, then 600B in 110; the 1,833 of its last 100B tokens are too few to show as languages it knows) and `config.json` (`vocab_size` 256000, `hidden_size` 768, `num_hidden_layers` 22) |
| Ettin 17M: 50,368 × 256 | `jhu-clsp/ettin-encoder-17m`'s `config.json` (`vocab_size` 50368, `hidden_size` 256) |
| 300 requests for a language's student | the gateway's `per_language.min_examples` default (`PerLanguageSettings` in `gateway/src/db.rs`) |

Claims to keep true if the product changes:
- The video never says that System One answers go unstored, and it presents the student model's size and speed only as the banking77 example (Ettin 17M). The opening question asks about a *tiny* model, and the zoom answers it with that example ("As small as 17 million parameters"). Student models can be bigger; keep the rest of the wording size-neutral.
- The zoom is a picture of scale, not a measurement: a frontier model trains on a datacenter of GPUs and serves from multiple 8 GPU servers (the rack drawing illustrates a larger deployment, using the layout of a GB300 NVL72: 18 compute trays of four GPUs and two CPUs, 9 switch trays), a System One model needs several GPUs (the tray's four), while a student model both trains and runs on one CPU. Only the last is measured (Ettin 17M on banking77: 586 s of fine-tuning and 0.96 ms per answer, both on a CPU); System One stands for whatever model or API your app calls today, and the video doesn't measure its hardware. Without a GPU, for prompts with up to 10 labels, the worker's automatic choice is an embedding student (a frozen encoder with a small trained head, `choose_base_model` in `src/system_none/worker.py`), which also trains on a CPU. The racks, tray and CPU are generic drawings with a "B300" text label, not any vendor's hardware or artwork.
- **A multilingual student must be bigger** rests on mmBERT-base's published sizes, drawn to scale: a vocabulary for every language and wider vectors make its vocabulary table alone 197M parameters, against Ettin 17M's whole 17M. *It takes longer to train, and it wants a GPU* and *minutes on a CPU* / *longer, on a GPU* compare kinds of hardware, not measurements: only Ettin's minutes are measured (banking77, 586 s on a CPU). *Still faster than System One, and cheaper to run* rests on what an encoder does (one pass over the input, as in "Inside the models"), not on a measurement of mmBERT; the base model is a setting, the admin console's student base model.
- **A student per language** follows the gateway (`gateway/src/language.rs` and README "Students per language"): its router, whatlang, tags each logged request with its language and how sure it is; a language the router is at least `min_confidence` (0.7) sure of in `min_examples` (300) requests gets its own student, of `base_model` (Ettin 17M by default), trained on those requests alone; and a request goes to its language's student when the router is that sure. It is a per-prompt option, off by default. *Each student stays tiny* is Ettin 17M's size; Ettin is an English model, so for other languages a multilingual base model may do better, which the README says.
- **The cache** follows `gateway/src/cache.rs` and README "Cache": it keeps answers in the gateway's memory, per prompt, model and text, and answers a text it has seen without a call. Its `position` is one of the three places drawn: `before_student`, `after_student` (the default) and `before_escalation`. Before or after the student it keeps System One's answer, or the LLM's when System One escalated; between System One and the LLM, System One answers every time and the cache keeps only the LLM's answers. *Saves every step after it* means a cached answer skips the calls after it. It is off by default.
- **Private training** holds only with a self-hosted judge. The judge can be any OpenAI-compatible endpoint, and it receives the logged inputs.
- **Just as accurate** rests on the banking77 comparison, where the student model learned from human labels. In production, student models learn from a judge, which caps their accuracy.
- **It learns from people** holds when some human labels train: by default (`train.human_holdout: 1.0`) they are all held out to score the judge and the students, and only a lower `human_holdout` lets the rest train with the human label in place of the judge's. Hence "can surpass its teacher". **Upload datasets with human labels** means `system-none log --label-field` in the Python library, or files under `human_labels/{prompt_id}/` in the gateway's store (README "Human labels"). **Superior to those from LLM judges** rests on the same banking77 run: Kimi K3, the judge, agreed with the human labels 81.6% of the time, and the student taught by people scored 91.5% against 79.5% for the one taught by the judge. The three bars are tiny-classifiers' measurements, not a SystemNone run.
- **What each request carries** follows the gateway's code (`gateway/src/upstream.rs` and `prompt.rs`): an `openai_chat` model gets the prompt's instructions, every label (with its description) and the text in each request, a `systemone` model gets the text as `state` and the instructions and labels as a `choice` question, while a student gets only the text (`Student::predict_proba` in `student.rs`); its labels are its output layer. Embedding students put their encoder's fixed classification prefix in front of the text, not the prompt or the labels. **One reason it's faster** stays qualitative: the video doesn't measure how much of the gap comes from the shorter input.
- **Inside the models** is a picture of how these models work, not a measurement. The LLM's sizes are GPT-3 175B's, a well-known published example, because frontier models such as Claude Opus 5.5 don't publish theirs; the footnote says so. It writes its answer token by token, as a chat LLM does; SystemNone's own LLM calls ask for a single option letter when there are up to 26 labels (or the label's name above that), so there the loop can be short, but each pass still runs every layer and scores the whole vocabulary. **Skips the loop** rests on Decider-2B, the System One model `make` runs: its model card says it reads every option's probability from one forward pass and never generates a letter. Jev's architecture isn't public, so the video gives System One no size. The student model's sizes are Ettin 17M's, the banking77 example, and its *under a millisecond* is that example's 0.96 ms; other student models are bigger. The tokens, scores and the LLM's answer are illustrative.
- The dashboard data, the dot animations, the cascade shares and the read sweeps are illustrations, not measurements.

## Editing the HTML player
The entry point is `system-none.html`, with `keynote.css`, `scenes.js`, `keynote.js`, `scenario.js`
and `timeline.js`. Routine edits require HTML checks, not MP4 rendering. The exporter has already
been tested end to end. Use `npm run export:mp4` only when an MP4 is wanted.

Requirements: Node 22+ and Chrome or Chromium. `render.js` looks in the usual place on Windows and macOS and for `google-chrome` or `chromium` on the PATH elsewhere; set `CHROME=/path/to/chrome` to override.

```bash
cd video
npm ci                                   # once: puppeteer-core, ffmpeg, fonts
npm run music                            # optional soundtrack for the HTML player
npm run preview                          # the same player, served at http://127.0.0.1:8123/system-none.html
node render.js stills 33.2 51.6 71.1 108.6   # check single frames in out/stills/
npm run readme-image                     # the README's still: 74 s (scene 5) -> how-it-works.png, 1280 px
npm run build                            # HTML/audio checks only; no MP4 rendering
```

- **Change text:** edit shared markup in `scenes.js`, then check the frame with `node render.js stills <time>`.
- **Change timing:** edit `keynote.js`; each element has `show(id, fadeInAt, fadeOutAt, …)`, in source time. The dot flows, counters and typing are `hook(...)` functions keyed to the same times.
- **Change the order, or add a scene:** `timeline.js` lists the passages of source time the video plays. Move a scene by moving its passage; add one at source times after the last scene's (the other-languages scenes start at 210.8 s) with a passage of its own. Every join must fall on a blank screen (`npm run check` tests it, on both sides), and a join with a beat on both sides must keep its phase: the two source times must be on the same phase of the 0.6 s grid from 45.2 s (`check-audio.js` tests it).
- **Change the models scene:** `IM_PASS`, `IM_S1`, `IM_ST` and the other `IM_` constants in "inside the models" set its times, and one `hook` draws the whole diagram on `#imcv` from them: the vectors (`IM_KIND`), the 96 layers (`IM_KEEP` are the 7 the student model keeps), the output rows and the loop (`IM_LOOP`).
- **Change the zoom:** the four drawings are built by `dcSvg`, `rackSvg`, `boardSvg` and `cpuSvg` in S3; the hall and the single rack share `rackFront`, so the rack lines up with its twin in the hall. `LEVELS` says where each sits in the world and when it fades; `CAM` says where the camera is and when. Each drawing stays at its own size and the camera hook transforms it, so every level is sharp at any zoom.
- **Change the poster:** edit the reveal slide; the poster uses the same frame automatically. `timeline.js` holds `posterTime`, which must stay on the fully visible reveal after “Introducing”.
- **Keep the README image in sync:** `how-it-works.png` is the frame at 74 s, the end of scene 5, and the README's "How it works"
  section shows it. It is the one rendered file that is committed. If you change scene 5, its wording or its timing, run
  `npm run readme-image` again, and if the scene moves, change the time in that script in `package.json`.
- **Keep music in sync:** `music.js` synthesizes the soundtrack in source time, then `audio-edit.js` plays its passages in the video's order, blending each join over 2.4 s. The beat grid starts at 45.2 s (100 BPM, one beat every 0.6 s), and the section lists (`KICK`, `HATS`, `CLAPS`), impacts, risers, bells and typing clicks are plain times near the end of the file. If you move a scene's cues, move its sound cues too; moving its passage moves both. The music is generated deterministically, so the same file always gives the same audio.
- **Replace the music:**
  ```
  ffmpeg -i out/video.mp4 -i track.mp3 -map 0:v -map 1:a -c:v copy -c:a aac -shortest out.mp4
  ```
  ffmpeg is at `node_modules/ffmpeg-static/ffmpeg.exe`.
