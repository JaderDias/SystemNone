# SystemNone

SystemNone is a gateway for System One models that speeds up inference by fine-tuning classifiers
automatically. The classifiers seamlessly replace the larger System One model without
clients redirecting their calls: the calls keep going to the gateway, which answers with a
classifier once one is ready. When a faster model isn't confident about an answer, SystemNone
falls back to a better one.

If Jev is a System One model, classifiers are System Zero: faster, cheaper and specialized. What
has held them back is that few teams have people who know how to fine-tune them. SystemNone
removes that barrier. You can use classifiers from day one: until a prompt has enough traffic,
your System One model or an LLM answers behind the same API. Once your application reaches a
critical volume, SystemNone automates the fine-tuning and serves the classifier, reducing
latency and costs.

A classifier's confidence is calibrated on held-out data, so it means how often the classifier
agrees with the judge, and each prompt's serving threshold is chosen to meet the accuracy you
set. Below that threshold, the System One model answers.

## Small classifiers vs. big models

Comparing fine-tuned classifiers with large models on banking77, which sorts bank customer
messages into 77 categories:

| Model | Right answers | Cost per million messages | Time per message |
|---|---|---|---|
| Claude Opus 5.5 | 92% | $8,120 | 2.4 s |
| Kimi K3 | 82% | $1,540 | 0.5 s |
| Jev, TypeSafe's classification API | 79.4% | $4 | 0.12 s |
| ModernBERT (150M), fine-tuned in 77 s | 93.2% | ≈ $0 | 14 ms |
| **Ettin (17M) trained by `system-none`, 586 s on a CPU** | **91.95%** | ≈ $0 | 0.96 ms (ONNX, CPU) |

The fine-tuned models learned from banking77's human labels and were tested on its 3,076 test
questions; Opus and Kimi were tested on 200 of them (±2 points). Training took a few minutes of
one gaming GPU. SystemNone's students learn from a judge model's labels instead of human ones,
so the judge's accuracy is their ceiling. The last row is SystemNone given the human labels, on
all 3,080 test messages: see [Benchmarking on banking77](#benchmarking-on-banking77).

## How it works

The [explainer video](video/STORYBOARD.md) plays in an HTML player, with instructions to
preview it; MP4 export is optional. It follows an English student model through to why it is
fast, then turns to other languages: why a multilingual student is bigger and slower to train,
a small student for each language behind a language router, and at last the cache.

It comes in two parts:

- **`gateway/`: a realtime gateway in Rust.** It serves a System One-style classification API,
  logs every input to a local folder or S3, answers with a distilled student when the student
  is confident, and otherwise calls the configured System One model or LLM. It includes a usage
  dashboard and an admin console.
- **`src/system_none/`: a Python library, CLI and training worker.** It judges logged
  inputs, fine-tunes students (DeBERTa-v3, mDeBERTa-v3, XLM-R, ModernBERT, Ettin, …),
  calibrates them and exports them to ONNX. It also works on its own, without the gateway.

Training stays in Python because Rust has no mature equivalent of PyTorch + transformers for
fine-tuning and exporting DeBERTa. Serving doesn't need Python: the `ort` and `tokenizers`
crates run the exported student in the gateway, with the same probabilities as the Python
runtime.

![A frame from the SystemNone keynote video. Your app sends POST /v1/systemone to the SystemNone gateway, which logs the inputs to a store. At 10,000 inputs, a Python worker judges, fine-tunes, calibrates and exports a student model to ONNX, and the gateway loads it: your app changes nothing. The student model now answers 89% of requests.](video/how-it-works.png)

The same flow in more detail:

```
clients ──► POST /v1/classify ──► gateway (Rust) ──► confident student? ──yes──► answer
                                     │                      │ no
                                     │                      └──► System One model / LLM ──► answer
                                     └── inputs only ──► store (folder or S3)
                                                             │  prompt reaches its request threshold
                                                             ▼
                        system-none worker (Python): judge ─► fine-tune ─► calibrate ─► ONNX ─► store
                                                                                               │
                                                        gateway loads the new student ◄────────┘
```

## Running everything with `make`

On Linux, macOS or WSL, `make` runs the whole system on your machine: the gateway, the training
worker and the [mock judge](#the-mock-judge), which answers from labelled data instead of running
a model. The mock judge answers as the gateway's System One model and is the judge the worker
trains students from. Given a TypeSafe API key, TypeSafe's API answers as the System One model
instead (see [With a TypeSafe API key](#with-a-typesafe-api-key)). Local models such as
Decision-2.0-Kai-0.6B and Decider-2B never run unless you ask for them
([With a local model as the judge](#with-a-local-model-as-the-judge)).

Before the first `make`, install [uv](https://docs.astral.sh/uv/), which also gets the right
Python, and [Rust](https://rustup.rs), for `cargo`, unless you already have them:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                  # uv, in ~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # rustup, cargo and Rust, in ~/.cargo/bin
```

Both installers add their folder to your PATH for new shells, so open a new terminal (or run
`. "$HOME/.local/bin/env"` and `. "$HOME/.cargo/env"` in this one). CI builds the gateway with the
latest stable Rust, which `rustup update` gets. Cargo also needs a C compiler to link
(`sudo apt install build-essential` on Debian and Ubuntu, `xcode-select --install` on macOS).
`make` needs curl, and Python 3.10+ without uv. Then:

```bash
make                  # start what isn't running, wait until it is healthy, configure the gateway
make stack-status     # what is up
make stack-logs       # follow the logs
make stack-down       # stop what make started
make help             # every target and setting
```

`make` starts the gateway on `127.0.0.1:8080` (its data in `gateway-data/`), the worker, and the
mock judge at `127.0.0.1:8001`, then makes the mock judge the gateway's default System One model
and judge. What already answers at those addresses is used as it is, so running `make` again only
applies the configuration again (`make stack-configure` does just that). The first run writes the
admin and worker tokens to `.env`: sign in to `/admin` with its `SN_ADMIN_TOKEN`. Pids and logs are
in `runs/stack/` (`RUN_DIR`); a restart appends to the logs, so how the last run ended stays
there. It needs a few GB of disk, mostly for PyTorch in `.venv`, which the worker trains with.
`GATEWAY_HOST`, `GATEWAY_PORT`, `MOCK_JUDGE_HOST` and `MOCK_JUDGE_PORT` change the addresses.

Students are trained once a prompt has received 10,000 requests; lower the threshold in `/admin`
to train sooner. `make traffic` sends test requests (see [Sending test traffic](#sending-test-traffic)).

### The mock judge

The mock judge (`system-none mock-judge`, in `src/system_none/mock_judge.py`) runs no model: it
answers each text with the label its labelled data gives it, so the system works end to end in
seconds, and students learn the datasets' own answers. It loads the labels of banking77 (both
splits, downloaded when it starts) and of the ai4privacy slice (through the key in
`tests/data/ai4privacy`, which only judges read), so `make traffic` with either dataset gets
answers that agree with the dataset, and the worker trains on them. It serves `POST /v1/systemone`
as Decider does, so the gateway calls it as a `systemone` System One model named `mock` and the
worker as a judge in mode `systemone`; its answer puts probability `MOCK_JUDGE_CONFIDENCE` (0.9) on
the label and spreads the rest over the other options. A text it has no label for, or whose label
isn't one of the prompt's, gets a `422` that says so: it never guesses, so such a request fails as
a System One error, and the worker counts it as a failed judgment.

`MOCK_JUDGE_DATASETS` picks the datasets (any of `system-none traffic --list` with labels; Fast
Decisions needs the `bench` extra), and `MOCK_JUDGE_LABELS` adds JSONL files of
`{"text": ..., "label": ...}` rows, such as your own labelled data:

```bash
make MOCK_JUDGE_DATASETS=banking77 MOCK_JUDGE_LABELS=my-labels.jsonl
```

A mock judge that is already running keeps its data: `make stack-down` first. Its answers are the
datasets' own labels, so training on them is as their licenses allow: banking77 is CC BY 4.0, and
ai4privacy CC BY 4.0 and the Llama Community License (see `tests/data/ai4privacy/README.md`);
for your own files, that is yours to check.

### With a TypeSafe API key

The first `make` in a terminal asks once for a TypeSafe API key and keeps the answer in `.env` as
`TYPESAFE_API_KEY`, empty for none (delete that line to be asked again). Without a terminal, as in
CI, it never asks; `TYPESAFE_API_KEY=… make` passes a key instead. With a key, TypeSafe's API is
the gateway's default System One model (a `systemone` model named `typesafe`), and the mock judge
(or Decider, with `JUDGE=decider`) stays the judge. `TYPESAFE_URL` (`https://api.typesafe.ai`, the
root under which it serves `POST /v1/systemone`) and `TYPESAFE_MODEL` (`jev-latest`;
`GET /v1/models` lists the others) change the API and the model, on the command line or in
`local.mk`. A gateway that is already running reads a new key only once it restarts.

TypeSafe's terms forbid training on its outputs (see
[Licensing of training data](#licensing-of-training-data)), so it can't be the judge; `/admin`
takes another judge.

### With a local model as the judge

`make` never sets up, downloads or starts a local model unless you ask for one with
`JUDGE=decider`:

```bash
make JUDGE=decider                                      # Decision-2.0-Kai-0.6B
make JUDGE=decider DECIDER_MODEL=Mapika/decider-2b      # the previous Decider-2B
```

`make JUDGE=decider` serves `DECIDER_MODEL` at `127.0.0.1:8000` and makes it the judge, and the
gateway's default System One model without a TypeSafe API key. Pass `JUDGE=decider` to
`make stack-configure`, `stack-status` and the like too, or put `JUDGE = decider` in `local.mk`;
`make` alone goes back to the mock judge (`make stack-down` stops a Decider that `make` started).
`make stack-decider` only starts it. The default Decision 2.0 model is served through
`system_none.decision_server`, using Transformers 5.17+ in the separate `.venv-decider`
environment; for a local Decision 2.0 package, also set
`DECIDER_APP=system_none.decision_server:app`. The legacy Decider-2B server needs Python 3.11 or
3.12 (`make PYTHON=python3.12` picks one, without uv). This needs several more GB of disk, for
PyTorch in `.venv-decider` and the model's weights. When switching between Decider-2B and Kai,
run `make stack-down` before restarting so the new model is loaded.

`DECIDER_HOST`, `DECIDER_PORT`, `DECIDER_MODEL` (a Hub id or a local folder) and `DECIDER_DEVICE`
(`cpu`, `cuda`, `mps` or `auto`) change the defaults, e.g. `make JUDGE=decider DECIDER_DEVICE=cuda`
(rebuild `.venv-decider` first if it has CPU-only PyTorch).
Kai defaults to `auto`: it selects CUDA only when PyTorch can initialize it and run BF16
matrix operations, otherwise logs the reason and uses CPU. Its official runtime requires
CUDA BF16 support; a visible GPU alone is insufficient, especially on older cards such as
the GTX 1060. `DECIDER_DEVICE=cpu` forces CPU; `cuda` forces CUDA and reports errors rather
than falling back. Automatic virtualenv setup chooses CPU PyTorch when no NVIDIA GPU is
detected. `DECIDER_TORCH_INDEX` can select a CUDA build compatible with your driver.
`GET http://127.0.0.1:8000/health` reports the model and selected device; `make stack-status`
checks readiness. Explicit legacy Decider model overrides retain the CPU default.

On CPU, Kai handles two requests concurrently using one shared model. Its per-inference
thread budget defaults to PyTorch's initial thread count divided by that concurrency.
`DECIDER_CPU_CONCURRENCY` and `DECIDER_THREADS` override those values (threads `0` means
automatic). For example, `make JUDGE=decider DECIDER_CPU_CONCURRENCY=2 DECIDER_THREADS=4` uses two
inferences with four CPU threads each. Restart the local model server to apply changes.
CUDA inference remains serialized because its runtime reuses graph buffers. The worker
already sends multiple judging requests concurrently; increasing client concurrency alone
does not remove a server bottleneck.
For two independent CPU model processes behind one HTTP server, run
`make JUDGE=decider DECIDER_DEVICE=cpu DECIDER_WORKERS=2 DECIDER_CPU_CONCURRENCY=1 DECIDER_THREADS=4`.
A shared process queue sends each request to an available model process, so persistent HTTP
connections cannot pin the client's workload to one model. Each process loads its own model,
so this needs additional RAM; keep `DECIDER_WORKERS=1` for GPU serving to avoid duplicating
the model in VRAM. Automatic thread budgets also divide by the process count. Each model
process executes one inference at a time. Settings only take effect after the existing server is stopped.

Decision-2.0-Kai-0.6B is Apache-2.0, like its base model Qwen3-0.6B-Base, and neither its model card nor its
repository restricts using its outputs, so `make` acknowledges its license for the judge (see
[Licensing of training data](#licensing-of-training-data)). For another `DECIDER_MODEL`, `make`
configures no judge until you have checked that model's license and run
`make JUDGE=decider DECIDER_LICENSE_ACK=yes`.

### With Fastino's GLiDE

[Fastino's GLiDE](https://docs.fastino.ai/inference/systemone) serves the same
`POST /v1/systemone`, so it is a `systemone` System One model too; `make` doesn't set it up yet.
Put its key in `.env` (or the gateway's environment) as `FASTINO_API_KEY`, and add the model in
`/admin`, or with the admin API:

```bash
curl -X PUT http://localhost:8080/admin/api/system-one-models/glide \
  -H "Authorization: Bearer $SN_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"kind": "systemone", "base_url": "https://api.fastino.ai", "model": "fastino/glide",
       "api_key_env": "FASTINO_API_KEY"}'
```

The key goes as a bearer token, which Fastino accepts as well as its own `X-API-Key` header.
GLiDE's `confidence` is the margin between its two most likely labels; the gateway's confidence
is the probability of the answer, as with Decider, so `escalate_below` means the same for both.
A cold model answers `425` for about a minute while it warms up, which the gateway reports like
any other failure.

GLiDE can't be the judge: Fastino's Terms of Service (§4.A) prohibit using its inference
responses to train, fine-tune or distill models that compete with Fastino, as students trained to
answer in its place would. They also let Fastino train its own models on the inputs you send
(§6), unless your plan lets you opt out.

## Running the gateway

The gateway and the worker run as two containers. CI builds their images with Nix and publishes
them to GitHub's container registry, so Docker Compose only pulls:

```bash
printf 'SN_ADMIN_TOKEN=%s\nSN_WORKER_TOKEN=%s\n' "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" > .env
docker compose up -d
```

`docker-compose.yaml` pulls the latest build of `main`; `SN_IMAGE_TAG=sha-<commit>` in `.env`
pins the build of one commit. While the repository is private, so are its packages: log in with
`docker login ghcr.io` and a token that can read packages.

The images are reproducible: `flake.lock` pins every input, and building a commit again gives the
same bytes, which CI checks before publishing. With Nix (on Linux, with flakes enabled),
`nix run .#compose` builds them locally instead, loads them under the names Compose uses and
writes `.env` with new tokens if there is none; `docker compose up -d` then starts them.

`.env` holds the admin and worker tokens. Both containers read it, so it is also where API keys
for System One models and judges go, `HF_TOKEN` for gated checkpoints,
and `SN_STORE=s3://bucket/prefix` with the usual `AWS_*` variables for an S3 store. Otherwise
the store is a folder in the shared `system-none_data` volume, next to the gateway's SQLite
database, the downloaded students and the Hugging Face checkpoints. The worker trains on CPU.
System One models and judges name the variable that holds their API key; names starting with
`SN_` or `AWS_` are refused, because those hold the gateway's own tokens and store credentials.

Both containers run as `nobody` (uid 65534) with no Linux capabilities and `no-new-privileges`;
only `/data` and `/tmp` are writable. A volume created by an older, root-run image needs its
owner changed once: `docker run --rm -v system-none_data:/data alpine chown -R 65534:65534 /data`.

Without containers, `nix run .#gateway -- --listen 0.0.0.0:8080` and
`nix run .#system-none -- worker` run the same builds, with the tokens in the environment.

Then open `http://localhost:8080/admin`, sign in with the `SN_ADMIN_TOKEN` from `.env`, and set up:

- **System One models:** the models that answer while there is no confident student.
  - `openai_chat` works with any OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter, a hosted LLM).
    It asks for the option's letter, or for the label's name when a prompt has more than 26 labels.
  - `decision_http` POSTs `{model, instructions, labels, input}` and expects `{label, confidence}`.
    Put a small adapter in front of a proprietary System One API.
  - `systemone` speaks TypeSafe's `POST /v1/systemone`, which a
    [Decider](https://huggingface.co/Mapika/decider-2b) server and Fastino's GLiDE serve (see
    [With Fastino's GLiDE](#with-fastinos-glide)): its endpoint URL is the server's root. The
    prompt is one Choice question whose options are the labels, with their descriptions, and the
    confidence is the probability of the answer.

  Each model has a price, for the cost estimate on the usage page (see
  [Cost estimate](#cost-estimate)): `pricing: {usd_per_call, usd_per_million_input_tokens,
  usd_per_million_output_tokens}`, TypeSafe Jev's list price unless set.
- **Judges:** the models that label logged inputs for training. Each one needs a license
  acknowledgement, and the denylist refuses services that forbid distillation. A judge is an
  OpenAI-compatible endpoint (mode `logprobs` or `sample`), or a `POST /v1/systemone` server
  such as Decider (mode `systemone`), whose answer probabilities become the soft labels.
- **Defaults:** how many requests a prompt must receive before its student is trained, the base
  model, the judge, the target accuracy, advanced training settings, how failed trainings
  are retried, and the [cache](#cache) of System One's answers.
- **Per-prompt overrides:** a different System One model, judge, base model, threshold or target,
  [a student per language](#students-per-language),
  for one prompt, plus "Fine-tune selected model" and "Stop using student". Ettin
  (`jhu-clsp/ettin-encoder-17m`) is the default encoder; older `auto` settings also choose Ettin.
  Select a larger encoder and press the fine-tune button to save the prompt's choices and queue training.
  Judgments are preserved in the shared store under `judgments/{judge_context_hash}/`
  as each label arrives, including when judging or training fails. A new job reuses them and judges only new inputs;
  changing the prompt or the judge's endpoint, model, mode or sample count starts a separate cache,
  and so does a new [judge version](#judge-versions).
- **Training jobs:** each job's status and the worker's last message. While the worker judges or
  trains, a progress bar shows how far that stage has got (inputs judged, training steps, or
  texts embedded for a frozen embedding model). The table refreshes itself every few seconds
  while a job is queued or running. When the worker reports that a job failed, the gateway
  queues a retry for 1 minute later, then 2, 4 and 8 (the wait doubles, up to an hour), for 5
  attempts in all by default; only then does the prompt show "failed". The retry waits in
  SQLite, so it survives a restart, and "Train now" runs it at once. Cancel stops a queued or
  running job and ends its retries: a cancelled job, and a student the gateway can't load, are not
  retried automatically. Retry, on a job an admin cancelled, queues its prompt again as "Train
  now" would, at once and from attempt 1, unless the prompt has another job queued or running.

`http://localhost:8080/` shows the most used prompts: total and daily requests, the share
answered by students, and progress towards training. Expand a prompt to see the request count,
nearest-rank p50 and maximum whole-request latency for each ordered model route over the last
seven days. Routes distinguish Ettin, other students, System One models, and cascades or
escalations between them. These latency histograms persist in SQLite across gateway restarts;
requests made before this feature was installed have no route samples. They are counted in
memory and written every second, with the usage counters, rather than once per request.

A request's latency is the whole request in the gateway, not the student's model speed (which
the worker measures after training, one text at a time, and the student's details show). Each
student answers one request at a time, so with more requests at once than it can take, they
queue, and their latency grows with the load: on a 16-core machine, an Ettin 17M student that
takes 1.3 ms a text reached 17 ms at the median with 8 requests in flight, at about 400 a second.

Each prompt's details on the usage page say in one sentence where its student stands and
what happens next, then show its pipeline (collect, train, test, serve), its audit, the
student's numbers, who answered its traffic and how fast, and a collapsible training history
with stage progress and attempt times, without exposing worker messages or private training
settings. `/jobs` shows every prompt's training attempts in one list. Usage requires
`SN_VIEWER_TOKEN` when configured and also accepts the admin token.

### Cost estimate

The usage page estimates what System One costs, over the last seven days and for each prompt,
and what the students saved. Every answered System One call costs its model's price per call,
plus its input and output tokens at the model's prices per million. The tokens are the ones the
model reports (`systemone` and `openai_decisions` answers, and OpenAI-compatible `usage`), or,
when it reports none, an estimate of one token per four characters of the text, instructions and
labels it was sent, which leaves out what a provider adds around them. An escalation's calls
both count. The student's savings are its answers at the prompt's average cost of a System One
answer over the same seven days. Failed calls and judge calls are not counted.

Every System One model starts at TypeSafe Jev's list price, $42 per billion input tokens with
output tokens free (typesafe.ai, October 2026): about $0.0000126 for a call of 300 input tokens.
That makes the estimate what Jev would cost even for a local model or the mock judge; set the
model's price to 0 in `/admin` (or `"pricing"` in its `PUT /admin/api/system-one-models/{name}`)
for one that costs nothing per call, or to your provider's prices for another. Costs are counted
from the time this was installed, at the price in force when each call was made.

### Students per language

A small encoder can't hold every language well: a student that serves many languages has to be
bigger, slower to train, and slower to answer. A prompt can instead train, besides its student for
every language, a small student for each language its traffic is in: tick "A student per language"
in its settings in `/admin` (`"per_language": true` in `PUT /admin/api/prompts/{id}`).

The gateway's language router ([whatlang](https://github.com/greyblake/whatlang-rs), a small
trigram model compiled into the gateway) tags every logged request with its language as an ISO 639-3
code and how sure it is, from 0 to 1 (`lang` and `lang_confidence` in the store). At the next
training, each language the router is at least `min_confidence` sure of in at least
`min_examples` requests gets a student of its own, of `base_model`, trained on those requests
alone; the prompt's student still trains on every request. Then a request the router is that
sure of goes to its language's student, with that student's own threshold, and any other request
(another language, or one the router isn't sure of) to the prompt's student. A reply from a
language's student says `"language"`, and names its model as `{model}@{code}`; the usage page
lists each language's student and times their requests apart.

The settings are in Defaults → Students per language, or `PUT /admin/api/settings`:

```json
{"per_language": {"min_confidence": 0.7, "min_examples": 300, "base_model": "jhu-clsp/ettin-encoder-17m"}}
```

The router errs on short texts: on banking77, only three in four short English messages come out
as English, about half with a confidence under 0.5. From 0.7, the ones it is sure of are right:
every English one, and Hindi and Telugu ones in the ai4privacy slice as often as the slice's own
language tags agree with them (92-95%), which a higher bar doesn't raise; it only routes fewer.
Requests from before this was installed have no language, so they train the prompt's student only.
Ettin is an English model, so for other languages a multilingual `base_model` may do better.

### Cache

The gateway can keep System One's answers in memory, so a text System One answered before gets
the same answer again, without a call and at no cost. It is off by default; turn it on under
Defaults → Cache in `/admin`, or with `PUT /admin/api/settings`:

```json
{"cache": {"enabled": true, "position": "after_student", "policy": "lru", "capacity": 10000, "ttl_secs": 0}}
```

- `position`: where in the cascade it sits; it saves the call after it.
  `before_student` makes it the first layer: an answer in the cache wins over the student's.
  `after_student` (the default), between the student and System One, makes it an alternative to
  the student: it answers what no confident student does, and prompts without a student.
  `before_escalation`, between System One and the stronger model it escalates to (its
  `escalate_to`): System One answers every time, and when it is unsure, the stronger model's
  earlier answer to the text comes from the cache, which keeps only those.
- `policy`: when it holds `capacity` answers, a new one replaces the least recently used (`lru`)
  or the least frequently used, the least recently used among those (`lfu`).
- `ttl_secs`: answers older than this are asked for again; 0 keeps them until they are replaced.

An answer is kept per prompt, System One model and text (ignoring surrounding whitespace), with the
confidence and probabilities System One gave, and only System One's answers are kept: a student's
cost nothing to compute again. Each gateway has its own cache, emptied when it restarts, when a
System One model is changed or deleted, when the policy, capacity or expiry changes, or with
`DELETE /admin/api/cache`; `GET /admin/api/cache` gives its size, hits, misses and evictions.
A request the cache answers says `"source": "cache"` and names the model that answered it first.
Its input is still logged for training, and the usage page and `make traffic` count it apart
from the student's and System One's answers, with what it saved at the cost of a System One answer.

Admin's calibration controls select empirical agreement or a one-sided Clopper–Pearson
lower bound, with a statistical confidence level (95% by default) distinct from the target
judge agreement. The exact option adjusts for threshold and candidate-model search using
Bonferroni and assumes IID held-out observations. These settings apply to new training jobs;
existing students retain their saved thresholds. Python configuration uses
`cascade.method: clopper_pearson` and `cascade.confidence_level: 0.95`; the gateway stores
these under `calibration`. Exact calculations use SciPy's inverse incomplete beta function.

Audits compare student predictions with known answers or with the training judge, never with
System One (see below). Audit answers are never saved as training labels.

### API

```bash
curl -s localhost:8080/v1/classify -H 'content-type: application/json' -d '{
  "prompt": {"instructions": "What does the customer want?",
             "labels": ["refund", "cancel", {"name": "other", "description": "anything else"}]},
  "input": "I was charged twice, I want my money back"}'
# {"prompt_id":"3f0c…","label":"refund","confidence":0.97,"source":"student","model":"…","latency_ms":1.8,"probs":{…}}
```

A prompt is its instructions plus its ordered labels (2 to 255), so the same prompt always gets
the same id. After the first call you can send `{"prompt_id": "…", "input": "…"}` instead.
`"model"` forces a particular System One model. `source` says who answered: `student` or `system_one`. If System
One fails and a less confident student exists, the student answers with `"degraded": true`.

| Endpoint | Who | What |
|---|---|---|
| `POST /v1/classify` | clients | classify one input |
| `POST /v1/decisions` | clients | typed choice, score and yes/no questions over one input |
| `GET /api/summary`, `/api/prompts`, `/api/prompts/{id}` | usage page (optional `SN_VIEWER_TOKEN`) | usage and estimated System One cost per prompt and day |
| `GET /api/jobs` | usage page and its training view (optional `SN_VIEWER_TOKEN`) | the 50 newest jobs' progress and prompts; filter with `?prompt_id=…` |
| `/admin/api/…` | admin (`SN_ADMIN_TOKEN`) | settings, System One models, judges, prompts, jobs, reference labels (`POST /admin/api/labels`), the cache (`GET` and `DELETE /admin/api/cache`) |
| `POST /internal/jobs/claim`, `…/{id}/progress`, `…/{id}/complete` | worker (`SN_WORKER_TOKEN`) | training jobs |

The worker reports each stage to `…/{id}/progress` as `{"message": "…"}`, with `"done"` and
`"total"` while it counts the stage: at most one report every 2 seconds, and the stage's last
count. A message without counts clears the last ones, so the admin page's bar doesn't outlive
its stage.

### Decisions API

Decisions support assumes OpenAI's API uses
[SGLang's Decisions schema](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/protocol.py).
Until an official OpenAI specification is available, this is a compatibility assumption.

To use a Decisions server as an upstream, add a System One model with kind `openai_decisions`.
The URL includes `/v1`, like `openai_chat` (unlike `systemone`, whose URL is the server root):

```bash
curl -X PUT http://localhost:8080/admin/api/system-one-models/decisions \
  -H "Authorization: Bearer $SN_ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"kind":"openai_decisions","base_url":"http://localhost:30000/v1",
       "model":"my-model","api_key_env":"DECISIONS_API_KEY"}'
```

Select it as the default in `/admin`, or send `"model":"decisions"` to force it. The gateway
sends one choice question with ordered options and their descriptions to `{base_url}/decisions`.
The probability of the chosen option drives escalation; `label_mass` is a separate statistic.
An upstream must return a valid distribution over exactly the requested labels and a choice
that has the highest probability. Missing or malformed probabilities fail the call.

Clients can send the same typed request format to the gateway:

```bash
curl http://localhost:8080/v1/decisions -H 'Content-Type: application/json' -d '{
  "input":"I was charged twice",
  "questions":[
    {"id":"team","type":"choice","question":"Who should handle this?",
     "options":[{"name":"billing","description":"Payments and refunds"},{"name":"support"}]},
    {"id":"duplicate","type":"yes_no","question":"Does this report a duplicate charge?"},
    {"id":"urgency","type":"score","question":"How urgent is this?","levels":["normal","urgent"]}
  ]}'
```

The response has `object: "decisions"`, `model`, `prompt_format_version: 1` and `answers` keyed
by question ID. Each answer includes its type and option probabilities; choice adds `choice`,
and score adds the probability-weighted level index (zero-based). Yes/no probabilities use
`yes` and `no`. Positive `temperature` rescales the probabilities without changing label mass.
Text fields accept strings, JSON objects or arrays; structured values become compact JSON.
There are 1–32 questions per request, 2–26 choice options and 2–10 score levels. IDs must be
unique and non-blank. Omitted model, or `"default"`, uses normal gateway routing.

Each question goes through the existing student, upstream and escalation pipeline, and has its
own prompt ID, input log, usage counters and training threshold. A choice with the same question
and ordered labels as `/v1/classify` shares its student. Score and yes/no become classification
prompts with numbered levels or yes/no labels. Probabilities retain their precision. A backend
that only returns a label cannot answer this endpoint: it returns 502 rather than inventing a
distribution. Existing chat backends can supply probabilities through their option logprobs.

`label_mass` is passed through when an upstream supplies it; students use 1 for their closed
label distribution. It is omitted when unavailable. `usage` sums the answering upstreams'
reported prompt and completion token counts across questions; local students consume zero
upstream tokens. It is omitted if any answering backend supplies no token accounting. Counts
exclude discarded escalation attempts. These optional omissions allow non-Decisions backends
to serve the gateway without inventing vocabulary probabilities or token counts.
Images, non-empty `chat_template_kwargs` and `return_prompt_token_ids: true` are rejected;
empty/default values are accepted. Only prompt format version 1 is supported. This gateway
version identifies its adapter, and does not promise SGLang's exact model prompt rendering.

Python judges and incumbents use `mode: openai_decisions` with a URL ending in `/v1` too.
For a judge, the existing license acknowledgement and provider denylist still apply; the worker
passes this mode through and trains on the returned probabilities. Configure a judge only after
checking permission to train on the selected model's outputs.

The worker's log (`runs/stack/worker.log` with `make`, the container's log with Docker Compose)
says more, each line with its time: when each stage of a job starts and ends (collecting inputs,
judging, loading the base model, training, calibrating, the ONNX export, scoring, publishing to
the store, reporting to the gateway), how many inputs the judge labeled and how many failed, with
the most common reasons, each epoch, and the counts of a long stage about once a minute. A job
that fails leaves its traceback there, and so does a crash in native code. When the gateway
doesn't take a progress report, the log says why, since the admin page then stops moving while
the job goes on. If the log stops mid-stage and `make stack-status` shows the worker down,
something outside killed it, such as the kernel when memory ran out (`dmesg` says so). The log
never holds tokens, API keys or the inputs' texts.

The worker checkpoints each completed judgment in the shared store under `judgments/`, keyed
by the task (including label order) and judge settings. Retries reuse these labels even with a
new job ID or on another worker, and judge only new or previously failed inputs. Changes to the
task or judge use a separate cache; changing the student or training settings keeps the labels.
Fine-tuning loads weights in float32 (including half-precision checkpoints) and fails immediately
on nonfinite losses or gradients instead of spending hours training a student that cannot export.

The gateway monitors students by default. A completed student first runs in **shadow**: it
predicts while System One answers. Each audit compares one of its predictions with a reference
answer:

- a **known answer**, when the request's text has a reference label the student was kept from
  in training (see [Human labels](#human-labels)). Known answers cost nothing, so every such
  request is checked;
- otherwise the prompt's **judge**, the model the student learned from, called in the
  background for a random share of requests.

The serving fallback is never the reference: a student trained on one model's answers can't be
expected to match a different model's (with a TypeSafe key, Jev answers but may not be the judge).
Texts whose known answer the student trained on aren't audited at all, since agreeing with an
answer it learned proves nothing.

Only predictions confident enough to be served count. After at least 100 of them, the student
goes live once its agreement reaches the prompt's target accuracy minus 0.03. It fails its
shadow audit only when agreement is *clearly* below that bar: when even the upper end of a
one-sided 99% Wilson interval is below it, or when the window of 500 checks fills without the
student reaching it. A student that fails in shadow stops there, and System One keeps answering:
retraining it on the same data would fail the same way, so an admin decides what to change (add
labeled examples, choose another judge or base model, lower the target) and presses Train now,
or Re-run test.

**Re-run test** in `/admin` reloads the retained student and starts fresh shadow evidence with
the current target agreement and tolerance. It cancels a queued retraining job (a running one
must be cancelled first), keeps the saved serving confidence threshold, and System One keeps
answering until the test passes.

Once live, the student is checked on fresh traffic in a rolling window of up to 500 confident
answers. When its agreement is clearly below the bar (the same Wilson test), it stops serving,
requests switch to System One, and one retraining job is queued with the latest traffic. A
replacement student goes through shadow again. Judge errors provide no agreement evidence and
cannot promote or demote a student. If there is neither a known answer nor an enabled, permitted
judge, the candidate stays in shadow and System One serves.

The default judge-audit share is 2%. It rises to 10% while fewer than 100 confident audits are
available or agreement is within 0.02 above the bar, and returns to 2% when evidence clears that
margin. The share uses prior audit evidence; selection never depends on the current request's
confidence or route. A separate rolling window includes every successful audit, including
low-confidence predictions routed to System One.

The Defaults section of `/admin` controls `monitor.enabled`, `sample_rate`,
`uncertain_sample_rate`, `inconclusive_margin`, `window`, `min_samples`, and `tolerance`;
these settings are also accepted by `PUT /admin/api/settings`. Disabling monitoring restores
immediate student serving. Re-enabling it, or changing the configured judge, starts fresh shadow
evidence. The usage page and `/api/prompts` show the audit phase, how many checks used known
answers and how many the judge, confident and all-traffic agreement, the verdict so far
(`collecting`, `passing`, `inconclusive` or `failing`), the effective judge share, errors and
skipped audits, and, for a stopped student, the agreement, bar and disagreement count it
stopped on. The model table groups requests by the answering model and excludes background
checks. State persists in SQLite
across restarts; late results for removed or replaced students or changed judges are discarded.
At most 32 judge calls run at once, each bounded to 30 seconds; saturation skips audits without
holding up responses. These checks detect observed agreement changes; they are not a novelty
detector.

#### Judge versions

A judge's model name can be an alias, such as `-latest`, that its provider points at a new
model from one day to the next: the training target changes, though no setting did. The model
the judge answers as (the `model` its answers name, as OpenAI-compatible, `systemone` and
Decisions servers do) is its version. Each training job asks the judge once which version it
is, keeps the labels of each version in a cache of their own (so a new version judges every
input again: a new lineage), records the version in each label, and reports it with the student
(`judge_version` in its metrics; the usage page shows it next to the judge). When an audit's
judge answers as another version than the student's evidence is against, the student is tested
again in shadow against the new version, as "Re-run test" would: System One answers until it
passes, and the audit panel says when the judge changed, from which version to which. A judge
that names no version, or names its configured model, keeps the cache and evidence it had.

The gateway doesn't start without `SN_ADMIN_TOKEN`. Without `SN_WORKER_TOKEN`, no worker can
claim training jobs, and the gateway's log says so; it never shows a token. The worker token is
never the admin token, so a worker machine can't change settings.

### Sending test traffic

`make traffic` (or `system-none traffic`) sends texts from a public dataset to `POST /v1/classify`,
so you can watch the gateway log, route and train before real clients arrive. A menu lists the
datasets: banking77, the ai4privacy texts in `tests/data/`, and each
[Fast Decisions](#benchmarking-on-fast-decisions) domain (these need `pip install -e ".[bench]"`).
It then asks how many requests to send, how many at a time and how many per second at most. At
the end it reports how many the student and System One answered, latency percentiles, how often
the answer matched the dataset's own label, and the errors.

```bash
make traffic                                     # the menu; GATEWAY_URL=http://host:8080 for another gateway
make traffic ARGS="--dataset banking77 --n 500"  # no menu: flags answer its questions
make traffic ARGS="--dataset fast-decisions/email_triage/category --n 200 --concurrency 8 --rate 50"
make traffic ARGS=--list                         # the dataset names
```

`make traffic` runs the CLI from `.venv`, which it sets up first; `SYSTEM_NONE=system-none` runs the
one on your PATH instead.

Each dataset is one prompt: its question and labels, such as banking77's 77 intents. A prompt takes
at most 255 labels, so a dataset with more would keep its 255 commonest labels and their texts.
Requests fail until a System One model is set up in `/admin` (plain `make` sets up the mock judge, or
TypeSafe's API). The mock judge answers only the datasets it loads (banking77 and ai4privacy unless
`MOCK_JUDGE_DATASETS` says otherwise), so other texts fail with it. Students answer once a prompt has received enough requests to train one (10,000
by default; lower it in `/admin` to train sooner).

### What is stored

Only the inputs are logged. The answers of System One models are never written anywhere, so
models whose terms forbid training on their outputs can still serve traffic. Settings, prompts,
daily usage and jobs are kept in SQLite (`--db`). Inputs and students go to the store, which the
worker reads and writes:

```
prompts/{prompt_id}.json
inputs/{prompt_id}/{YYYY-MM-DD}/{ms}-{rand}.jsonl     {"id", "text", "ts"}
models/{prompt_id}/{job_id}/model.onnx, tokenizer.json, manifest.json
human_labels/{prompt_id}/*.jsonl                      {"text", "label", "split"?}, from you or make labels (see Human labels)
models/{prompt_id}/{job_id}/reference_labels.jsonl    {"text", "label", "use": "train" | "audit"}, the known answers to audit against
```

## The five steps

| Step | What happens | Module |
|---|---|---|
| 1. Shadow | Copy **inputs only** from live traffic into `shadow.jsonl` (sampling, de-dup, optional redaction). Today's model keeps serving. | `shadow.py` |
| 2. Judge | Any OpenAI-compatible endpoint (vLLM, Ollama, TGI, OpenRouter…) labels each input with a **soft label**: the probability of each option, read from the logprobs of the answer letter (or vote shares in `sample` mode). With more than 26 labels, it answers with the label's name instead of a letter. A `POST /v1/systemone` server such as Decider can judge too (`mode: systemone`): its answer probabilities are the soft label. The judge must be one whose license lets you train on its outputs; you confirm this in the config, and a denylist blocks services that forbid distillation. | `judge.py` |
| 3. Student | Fine-tune candidate encoders (DeBERTa-v3, ModernBERT, Ettin; mDeBERTa-v3, mmBERT, XLM-R for other languages) by minimizing **KL(judge ‖ student)**. Small training sets get more epochs, so training takes at least `min_steps` optimizer steps ([measured](#benchmarking-on-banking77)). Epochs compete on validation KL after temperature scaling, and the best one is kept (raw KL would favour early, less confident epochs over later, more accurate ones). | `train.py`, `pipeline.py` |
| 4. Calibrate | Fit one temperature `T` on held-out data so confidence means agreement; report ECE before/after. | `calibrate.py` |
| 5. Cascade | Export to ONNX (verified against PyTorch), pick the confidence threshold that meets your target accuracy on validation data, route by language, fall back to today's model below the threshold. | `export.py`, `runtime.py` |
| Monitor | A sample of live student decisions goes to a review queue; `system-none monitor` has the judge score them, adds them to the training data, and exits with code 2 when agreement drifts. | `monitor.py` |
| Human labels | Inputs whose answers a person checked score the judge, every candidate and the cascade against people rather than against the judge. A share of them can train instead, with the human label replacing the judge's. | `human.py` |

Candidates are compared on test accuracy (agreement with the judge), and among those within
`accuracy_tolerance` of the best, the fastest p50 latency wins. When both groups are trained,
English is routed to whichever student scores better on held-out English text (the English
specialist keeps it unless it trails by more than `accuracy_tolerance`). Results land in
`leaderboard.json`; the chosen models, thresholds and routes in `deploy.json`.

## Quickstart

```bash
pip install -e ".[train,serve]"
system-none init                       # writes system-none.yaml: edit the task, judge and students
```

1. **Log inputs** from your service, either with the decorator…

   ```python
   from system_none import ShadowLogger, shadow

   log = ShadowLogger("runs/support-intent/shadow.jsonl", sample_rate=0.2)

   @shadow(log)                    # logs the text argument, never the result
   def classify(text): ...         # your existing LLM / System One call, unchanged
   ```

   …or import a file of past inputs: `system-none log traffic.jsonl --field text` (add
   `--label-field label` for answers a person checked: see [Human labels](#human-labels)).

2. **Judge**: `system-none judge` (resumable; run it again as more inputs arrive).
3. **Train**: `system-none train` fine-tunes, calibrates, exports, benchmarks and picks. Training
   runs on CUDA or Apple Silicon when available, otherwise on CPU.
4. **Use it**:

   ```python
   from system_none.runtime import load_deployment
   classify = load_deployment("runs/support-intent", fallback=my_current_classifier)
   classify("Where is my parcel?")   # Decision(label='shipping', confidence=0.97, source='student', ...)
   ```

   or `system-none serve` (POST `/classify` on localhost; `--host 0.0.0.0` for other machines), or
   `system-none predict "some text"`. The runtime needs
   only `onnxruntime`, `tokenizers` and `numpy`.
5. **Monitor**: schedule `system-none monitor`; retrain with `system-none train` when it exits with 2.

`system-none bench` reports accuracy, macro-F1, ECE, KL to the judge, p50/p99 latency (batch 1)
and batch throughput on the test split.

## Human labels

Students are scored by how often they agree with the judge, which can't show how good the judge
is. A few hundred inputs whose answers a person checked can. Import them with your other inputs;
lines without a label are logged as plain inputs:

```bash
system-none log checked.jsonl --field text --label-field label   # {"text": "…", "label": "refund"} per line
system-none judge    # also prints how often the judge agrees with people: "judge_vs_humans": {"n": 300, "accuracy": 0.91, …}
system-none train
```

`system-none train` keeps them out of training, validation and test, and scores every candidate
against them: `leaderboard.json` and `deploy.json` get a `human` entry with the student's accuracy
and macro-F1, the judge's accuracy on the same inputs, the share the student answers alone at its
threshold, and how often it is right then. `system-none bench` reports the same.

Once you have more than you need to measure, `train.human_holdout` (default 1.0) sets the share
held out, and the rest train with the human label instead of the judge's. A student that learns
from people is no longer capped by the judge's mistakes: on banking77,
[tiny-classifiers](https://github.com/MaximeRivest/tiny-classifiers) measured Ettin 17M at 91.5%
with human labels and 79.5% with those of Kimi K3, which itself scores 81.6%.

With the gateway, put the files in the store under `human_labels/{prompt_id}/`, one
`{"text": …, "label": …}` per line; with Docker Compose, `docker compose cp human_labels gateway:/data/store/`
copies a local `human_labels/{prompt_id}/checked.jsonl`. The worker reads them at every job,
has the judge label them too, and reports the scores in the job's message and in the deployment's
metrics; the usage page shows them next to the test accuracy. `human_holdout` goes in the
training settings, like `epochs`.

A line can also say how to use it, overriding `human_holdout`: `"split": "train"` trains on it,
and `"split": "audit"` keeps it out of training. The worker publishes the held-out answers with
the student, and the gateway [audits](#running-the-gateway) the student against them: every
request whose text has one is checked against that known answer instead of the judge. Texts
with a known answer skip the judge unless an earlier job judged them already, which saves hours
with a slow judge.

A labeled dataset is the quickest way to get both. `make labels` uploads one through
`POST /admin/api/labels` (admin token; the prompt is registered if it is new): its official
training split trains the student and its test split audits it, or a stable 20% of its texts is
held out when it has no test split.

```bash
make labels DATASET=banking77                # then Train now in /admin
make labels DATASET=banking77 ARGS=--train   # or queue the training job at once
make traffic ARGS="--dataset banking77 --n 1500"   # traffic for the shadow test
```

On banking77, the Decision-2.0-Kai-0.6B judge is right on 76% of the texts, so a student trained on
its labels can't do better, and wouldn't match TypeSafe Jev's answers either. Trained on the
dataset's labels instead, Ettin 17M reaches about 92% (see
[Benchmarking on banking77](#benchmarking-on-banking77)), and its test checks those answers
against the 3,080 test texts it never saw.

## Licensing of training data

Some services forbid using their outputs to train models. TypeSafe's Master Customer Agreement
(§2.3(b)), for example, prohibits using Jev outputs for model distillation, and Fastino's Terms of
Service (§4.A, effective August 6, 2026) prohibit using its inference responses to train,
fine-tune or distill models that compete with Fastino, so this tool refuses to use either as a
judge. Such a model can still be the `incumbent` (the cascade's fallback), because its answers are
never stored. For any judge, check the model license and your provider's terms,
then set `judge.license.allows_training_on_outputs: true`. This is not legal advice.

## Benchmarking on banking77

```bash
system-none banking77 --model jhu-clsp/ettin-encoder-17m     # or: nix run .#system-none -- banking77
```

This downloads [banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI,
CC BY 4.0) and trains a student on the human labels of its training split with the usual
pipeline: fine-tune, calibrate, export to ONNX and pick the serving threshold. It then scores the
ONNX student on the 3,080 test messages and writes `runs/banking77/banking77.json`. The human
labels replace the judge, so the scores compare directly with published ones. (A judge can label
banking77 too: with 77 labels, it answers with the label's name.)

Ettin 17M on an AMD Ryzen 7 9700X, CPU only, trained on 8,817 of the 10,003 training messages
(1,000 pick the epoch, the temperature and the threshold):

| Settings | Test accuracy | Macro-F1 | ECE | Fine-tuning | Latency (p50) | Student answers at a 95% target |
|---|---|---|---|---|---|---|
| 10 epochs, lr 2e-4, batch 32 (the defaults) | 91.95% | 0.919 | 0.016 | 586 s | 0.96 ms | 89%, of which 96.4% right |
| 5 epochs, lr 1e-4, batch 32 | 91.1% | 0.911 | 0.018 | 246 s | 1.01 ms | 89%, of which 96.0% right |

Choosing epochs on raw validation KL instead of scaled KL kept epoch 2 of the 5-epoch run, which
scored 87.1%.

With less data, a fixed number of epochs is too little training. Like
[tiny-classifiers](https://github.com/MaximeRivest/tiny-classifiers), which keeps the number of
optimizer steps about the same whatever the data size, SystemNone raises the epochs until training
takes at least `train.min_steps` steps (default 1,000; at most 100 epochs; 0 turns it off). The
best epoch is still picked on validation data. On 2,000 of the training messages (1,765 to train
on, 193 to validate), tested on all 3,080 test messages:

| Settings | Epochs | Steps | Test accuracy | Student answers at a 95% target |
|---|---|---|---|---|
| lr 3e-5, batch 16 (the default epochs, lr and batch size), `min_steps: 0` | 3 | 333 | 36.7% | never |
| the same, `min_steps: 1000` (the default) | 10 | 1,110 | 66.1% | 41%, of which 94.1% right |
| lr 2e-4, batch 32, 10 epochs, `min_steps: 0` | 10 | 560 | 79.0% | 62%, of which 94.8% right |
| the same, `min_steps: 1000` | 18 | 1,008 | 80.2% | 65%, of which 95.5% right |
| the same, 32 epochs | 32 | 1,792 | 80.9% | 67%, of which 95.1% right |

tiny-classifiers reports 81.0% for Ettin 17M on 2,000 human-labeled banking77 messages, with
about 1,780 steps. Training time grows with the number of steps.

## Benchmarking on Fast Decisions

Fastino's [Fast Decisions](https://huggingface.co/datasets/fastino/fast-decisions) suite (Apache-2.0)
publishes a development split of 100 rows for each of 17 domains:

```bash
pip install -e ".[bench]"
system-none fast-decisions configs --out fd_configs        # a starter config per single-label head
system-none fast-decisions eval --head support_intent/intent --model runs/.../onnx
```

Report these as dev-split scores; Fastino's published numbers use a held-out test split.

## Tests and CI

End-to-end tests come in three sizes:

| Size | What runs | Where | Time |
|---|---|---|---|
| tiny | Rust unit and integration tests; Python unit tests; gateway + worker on 400 synthetic requests with a tiny BERT; a tiny DeBERTa-v2 export check | CI, on every push and pull request; `make test test-tiny` | a few minutes |
| medium | 3,928 real English, Hindi and Telugu texts, tiny models trained from scratch | `make test-medium`, on your machine | about 5 minutes |
| big | the same texts with pretrained `microsoft/deberta-v3-small` and `microsoft/mdeberta-v3-base` | `make test-big`, on your machine's GPU if it has one | about an hour on CPU |

The medium and big runs are not in CI. The Makefile runs every size on a Linux or WSL machine:

```bash
make help             # lists the targets and their settings
make lint test        # ruff, ty, bandit, cargo fmt and pedantic clippy; Python unit tests and Rust tests
make fmt              # formats the Python code and sorts its imports (ruff)
make test-tiny        # tiny end to end (builds the gateway)
make test-medium      # medium
make test-big         # big (downloads the checkpoints)
make verify           # TLA+ models with TLC
```

The Python targets first set up `.venv` with `system-none[dev]` (delete it to rebuild it), and the
Rust ones need `cargo`: [Running everything with `make`](#running-everything-with-make) shows how
to install uv and Rust. PyTorch is the CPU-only build unless `nvidia-smi` sees a GPU; then it is
the default CUDA build, the big run trains on the GPU, and `make test-big` stops before training if
that build can't run on it. `TORCH_INDEX` picks another build, such as
`https://download.pytorch.org/whl/cu126` for a GPU too old for the default one, and `PYTEST_ARGS`
passes extra pytest arguments, e.g. `make test PYTEST_ARGS="-k router"`. Settings and targets for
your own machine can go in an untracked `local.mk`, which the Makefile includes.

`uv.lock` pins every Python dependency, for every platform and extra. With uv, `.venv` gets those
versions, PyTorch's too, in whichever build `TORCH_INDEX` picks; CI installs them the same way.
After changing the dependencies in `pyproject.toml`, run `uv lock`: until then, make and CI stop
with an out-of-date lock. Dependabot proposes the new versions once a month.

CI runs what `make lint test test-tiny verify` runs on every push and pull request (bandit runs in
the Security workflow), with the linter versions the `lint` extra pins. `pyproject.toml` configures
ruff, and clippy fails on any warning, its pedantic lints included (`gateway/Cargo.toml`'s
`[lints]` turns them on).

`make verify` model-checks the TLA+ specs of the gateway's and the worker's concurrency in
[`verification/`](verification/README.md) with TLC (about eight minutes in CI): every property holds
for the code as it is, and every bug the models found shows up again in the model of the code
before its fix. It needs `tlc`, `TLA2TOOLS=/path/to/tla2tools.jar` with Java 11 or later, or Nix.

`nix build .#gateway` runs the Rust tests too, against the ONNX Runtime from nixpkgs. The Images
workflow builds both images with Nix and checks that rebuilding them gives the same bytes. It then
starts them with Docker Compose and checks the tokens, the users and that neither container
restarts; on `main` it publishes them.

The Security workflow runs on every push and pull request and every Monday: `cargo audit` and
`pip-audit` check the locked Rust and Python dependencies for known vulnerabilities, `bandit`
checks the Python code (the package and its tests), `zizmor` checks the workflows, and `gitleaks`
looks for secrets in every commit, redacting what it finds (`.gitleaksignore` lists the findings
that aren't secrets). To stop a secret before it is even committed, run `uvx pre-commit install`
once in your clone: gitleaks then checks what each commit stages.

The medium and big runs use 3,928 real texts in English (2,592), Hindi (747) and Telugu (589)
from [ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
(CC BY 4.0 and the Llama Community License; built with Llama; see `tests/data/ai4privacy/`).
The task is the most sensitive kind of personal data in each text: an ID number, contact
details, an address, or names only. The inputs carry no labels. The labels sit in a separate key
that only a hardcoded judge reads, so they reach training the same way they would in production.

| Held-out (763 texts), medium | Accuracy |
|---|---|
| Always the commonest label | 0.40 |
| Multilingual student, all languages | 0.77 |
| … English / Hindi / Telugu | 0.76 / 0.77 / 0.79 |
| Cascade (student when confident, else today's model) | 0.90, with the student answering 64% |

The medium numbers come from 2-layer models trained from scratch, so they show that the pipeline
works, not what a real student reaches. The big run checks for at least 0.85 overall and 0.75
in each language. Tokenizers and initial weights are deterministic, so the medium numbers repeat
from run to run on the same machine.

## Roadmap

### Models and APIs

- **Escalation.** When the System One model is unsure (Jev below a confidence you set), the
  gateway asks a stronger model, such as an LLM, before answering. With the student in front,
  that makes a three-step cascade: classifier, System One model, LLM.
- **More LLM providers.** `openai_chat` reaches OpenAI and any OpenAI-compatible server, and
  every System One model and judge sends its key as a bearer token. Native kinds will add
  Anthropic's Messages API, Amazon Bedrock, Google Vertex AI and Azure OpenAI, in the gateway
  and the worker, each with the credentials it expects: an `x-api-key` header, SigV4 with an AWS
  role, a Google service account, Entra ID. Where an API returns no logprobs, as Anthropic's
  doesn't, judges use `mode: sample`. Each provider's terms still decide whether it can judge
  (see [Licensing of training data](#licensing-of-training-data)).
- **More student bases.**
  - [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m), for prompts with few
    labels and little compute: the encoder stays frozen and only a small head is trained, so no
    GPU is needed.
  - [GLiNER2.5](https://huggingface.co/collections/fastino/gliner25-models) (Apache-2.0), which
    classifies text from the label names alone, so it could answer before a prompt has any
    training data.
- **Multi-label tasks.** Every task is single-label: each input gets one of its labels, so
  `system-none fast-decisions` and `system-none traffic` skip Fast Decisions' multi-label heads.
  The library and the gateway will also take multi-label tasks, where one input can get several
  labels.
- **Drop-in APIs.** The gateway will also serve TypeSafe's `POST /v1/systemone` and SGLang's
  `/v1/decisions` and `/v1/score`, so teams already on those APIs switch by changing a base URL.
  Students answer what they are confident about, and the rest passes through to Jev or SGLang.
  SGLang's `/v1/decisions` also makes a good judge: it reads the answer probabilities on the
  server, and its `label_mass` flags answers that fall outside the labels.
- **Monitoring refinements.** The gateway audits live student predictions, requires shadow
  promotion, and falls back and retrains on agreement drift. Statistical confidence bounds,
  resolved teacher versions and unfamiliar-input checks can strengthen those decisions.

### Enterprise deployments

Today SystemNone runs as one gateway process, with its state in SQLite, a store in a folder or
S3, and three static tokens. Running it as shared infrastructure, for many teams and on any
cloud, takes:

- **An external database.** Settings, prompts, daily usage, jobs and deployments will move from
  SQLite to Postgres (`--db postgres://…`), so they outlive the gateway's machine and replicas
  can share them: Amazon RDS or Aurora, Google Cloud SQL, Azure Database for PostgreSQL, or your
  own. Besides passwords, the gateway will sign in with the cloud's identity: RDS IAM
  authentication (a short-lived token signed with SigV4), Cloud SQL IAM database authentication,
  Entra ID on Azure. Backups and point-in-time recovery then come with the managed database.
  SQLite stays the default on one machine.
- **Stores on every cloud.** The `s3://` store already signs its requests with SigV4, in the
  gateway (Rust's `object_store`) and the worker (boto3), with keys in `AWS_*` variables or,
  without keys, the role of an EKS pod (IAM roles for service accounts or Pod Identity), an ECS
  task or an EC2 instance. `gs://` for Google Cloud Storage (a service account or Workload
  Identity on GKE) and `az://` for Azure Blob Storage (a managed identity) will join it, and
  S3-compatible stores such as MinIO will be tested with a custom endpoint.
- **Horizontal scaling on Kubernetes.** Several workers can already share one gateway: each job
  is claimed by one of them. The gateway runs as one replica, because besides SQLite each
  process holds the admin configuration, request counts, daily usage and loaded students in
  memory. On Postgres, replicas behind one Service will see each other's configuration changes
  and deployments, and count requests in the database so a prompt still gets one training job.
  A Helm chart will run them with an autoscaler and readiness probes that check the database and
  the store (`/health` only says the process is up), and workers with a CUDA image on GPU nodes.
  Jobs will get leases, so a job whose worker dies goes back to the queue instead of waiting for
  an admin to cancel and retry it (see [what the TLA+ models leave open](verification/README.md#not-covered)).
  The images already run as a non-root user with no capabilities, as a restricted pod security
  context requires.
- **A model server for students.** Students run inside the gateway process, on CPU, and each
  handles one request at a time; ONNX Runtime uses several threads per request (`--threads`).
  Every deployed student stays loaded in the gateway's memory. Planned: a separate model server,
  for students that need a GPU or more students than fit in one machine's memory, and batching.
- **Sign-in, roles and audit.** `/admin` takes the one admin token, which can change everything;
  the usage page is open unless `SN_VIEWER_TOKEN` is set; `POST /v1/classify` takes no token, so
  it belongs on a private network or behind an API gateway, which is also where TLS ends: the
  gateway speaks plain HTTP. Planned: single sign-on for `/admin` and `/` with OIDC or SAML
  (Okta, Entra ID, Google Workspace), roles (admin, operator, viewer) instead of shared tokens,
  API keys with rate limits for each client of `/v1/classify`, and an append-only log of who
  changed which setting, model, judge, prompt or job, for a SIEM to collect.
- **Teams.** A gateway has one set of System One models, judges and defaults, and the same prompt
  gets the same id whoever sends it. Tenants will get their own prompts, models, judges, store
  prefix, usage and admins, chosen by their API keys, so one deployment serves many teams and
  reports each one's usage.
- **Metrics and traces.** The gateway logs to stdout (`RUST_LOG` sets the level), and the usage
  page counts requests, the student's share, errors and latency per prompt and day. A Prometheus
  `/metrics` endpoint and OpenTelemetry traces will export these and more: latency percentiles,
  errors per System One model, queued jobs and their durations. Its persisted student audits can
  also feed alerts when agreement drifts.
- **Secrets.** Tokens and API keys are environment variables, which Kubernetes Secrets can fill
  as they are; each System One model and judge names the variable that holds its key, and a
  running gateway reads a new key only once it restarts. The gateway and the worker will also
  read them from AWS Secrets Manager, Google Secret Manager, Azure Key Vault or HashiCorp Vault,
  and pick up rotated keys while running.
- **Retention and personal data.** Logged inputs stay in the store until someone deletes them
  (meanwhile, an S3 lifecycle rule on the store's `inputs/` prefix can expire them by age),
  and the gateway logs them as they arrive: the Python `ShadowLogger` can redact text, the
  gateway can't. Planned: a retention period per prompt, redaction before an input is logged,
  and finding and deleting given texts on request.

## Limits

These come with how SystemNone works. What it doesn't do yet is on the [roadmap](#roadmap).

- A gateway prompt has 2 to 255 labels. Up to 26, `openai_chat` System One models and judges
  answer with the option's letter, and the probabilities come from that token's logprobs. With
  more, they answer with the label's name (`judge.answer: auto` in the CLI, or `name` /
  `letter`). `systemone` models and judges (Decider) take up to 255 options as they are. The
  gateway's confidence is then the probability of the tokens that named the label. A judge's soft
  labels come from vote shares (`mode: sample`), or in `logprobs` mode from the probability of its
  answer, the rest spread evenly over the other labels. An answer that names no label, or several,
  is an error in the gateway and stays unlabeled in the judge.
- Accuracy is measured as agreement with the judge, so the judge sets the ceiling.
  [Human labels](#human-labels) measure the judge itself, and let students learn past it.
- The judge prompt answers with one letter (or one label name); reasoning models need thinking
  disabled (`extra_body`) or `mode: sample`.
