"""`system-none worker`: train students for the Rust gateway.

The gateway queues a job when a prompt reaches its request threshold (or an admin presses
"Train now"). The worker claims it, reads the prompt's logged inputs from the shared store,
labels them with the judge the admin chose, fine-tunes the chosen base model, publishes the
ONNX student to the store and tells the gateway, which starts serving it.

Its log (with `make`, runs/stack/worker.log) says when each stage of a job starts and ends, counts
the long ones about once a minute, and holds the traceback of anything that fails, so the log
alone shows how far a job got. It never holds tokens, API keys or the inputs' texts.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import httpx

from .config import TrainSpec, config_from_dict
from .shadow import read_log
from .stores import MODEL_FILES, open_store

MIN_EXAMPLES = 20


def log_line(message: str) -> None:
    """The worker's log: each line starts with the time, and goes out at once, even to a file."""
    message = message.lstrip("\n")  # train_all puts a blank line before each group, for the CLI
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


# Older gateway settings may still use "auto"; these now choose the default encoder.
FINE_TUNED_MODEL = "jhu-clsp/ettin-encoder-17m"


def choose_base_model(base_model: str, num_labels: int, log=print) -> str:
    return FINE_TUNED_MODEL if base_model == "auto" else base_model


# Counts go to the gateway at most this often (seconds), plus each stage's last one: judging
# 3,928 inputs takes a few dozen requests, not 3,928.
PROGRESS_INTERVAL = 2.0
# A report holds up the judge or the training loop until it is sent, so it gives a slow gateway
# only this long (seconds).
PROGRESS_TIMEOUT = 5.0
# The log gets the counts at most this often (seconds), plus each stage's last one: enough to see
# that a stage moves, and where it stopped.
LOG_INTERVAL = 60.0


class GatewayClient:
    def __init__(self, url: str, token: str, timeout: float = 120):
        self.http = httpx.Client(
            base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=timeout
        )

    def claim(self) -> dict | None:
        r = self.http.post("/internal/jobs/claim")
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def progress(self, job_id: str, message: str, done: int | None = None, total: int | None = None) -> str | None:
        """The job's stage, with (done, total) when it is counted; without them, the gateway
        clears the last counts. Progress is best effort: this returns why the gateway didn't take
        the report (it is down, or answered with an error), or None when it did."""
        body: dict = {"message": message}
        if done is not None and total is not None:
            body |= {"done": done, "total": total}
        try:
            r = self.http.post(f"/internal/jobs/{job_id}/progress", json=body, timeout=PROGRESS_TIMEOUT)
        except httpx.HTTPError as e:
            return f"{type(e).__name__}: {e}"
        return None if r.is_success else f"{r.status_code} {r.text[:200]}".rstrip()

    def complete(self, job_id: str, result: dict) -> dict:
        r = self.http.post(f"/internal/jobs/{job_id}/complete", json=result)
        if r.status_code >= 400:
            return {"deployed": False, "error": r.text[:500]}
        return r.json()


class JobCancelled(Exception):
    """The gateway says the job no longer runs: an admin cancelled it."""


class JobProgress:
    """What the admin page shows of a running job: its stage, and how far the stage has got.

    `stage` reports a new message, which clears the last stage's counts. `count` is the judge's
    and the training loop's `on_progress`, called for every input or step; it reports at most
    every `interval` seconds, and always the stage's last count (done == total).

    The log gets each stage, its counts at most every `log_interval` seconds and its last count,
    and a line whenever the gateway stops or starts taking the reports again: the admin page
    stops moving when the gateway doesn't take them, although the job goes on.
    """

    def __init__(
        self,
        gw: GatewayClient,
        job_id: str,
        interval: float = PROGRESS_INTERVAL,
        clock=time.monotonic,
        log=print,
        log_interval: float = LOG_INTERVAL,
    ):
        self.gw, self.job_id, self.interval, self.clock = gw, job_id, interval, clock
        self.log, self.log_interval = log, log_interval
        self.message, self.reported, self.logged, self.refused = "", clock(), clock(), None

    def stage(self, message: str) -> None:
        self.message = message
        self.log(f"job {self.job_id}: {message}")
        self.logged = self.clock()
        self._send()

    def count(self, done: int, total: int) -> None:
        if done >= total or self.clock() - self.logged >= self.log_interval:
            self.log(f"job {self.job_id}: {self.message}: {done}/{total}")
            self.logged = self.clock()
        if done < total and self.clock() - self.reported < self.interval:
            return
        self._send(done, total)

    def _send(self, done: int | None = None, total: int | None = None) -> None:
        refused = self.gw.progress(self.job_id, self.message, done, total)
        if refused and refused.startswith("409"):  # an admin cancelled it: stop working on it
            raise JobCancelled(refused)
        if refused and refused != self.refused:  # once, not at every report while the gateway is down
            self.log(f"job {self.job_id}: the gateway didn't take a progress report ({refused})")
        elif self.refused and not refused:
            self.log(f"job {self.job_id}: progress reaches the gateway again")
        self.refused = refused
        self.reported = self.clock()


def collect_inputs(store, prefix: str, out: Path) -> int:
    """Merge the gateway's JSONL objects into one de-duplicated shadow log, with the language its
    router saw in each input (`lang`, `lang_confidence`), when it saw one."""
    seen, n = set(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for key in store.list(prefix):
            if not key.endswith(".jsonl"):
                continue
            for line in store.get(key).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["id"] in seen or not rec.get("text", "").strip():
                    continue
                seen.add(rec["id"])
                row = {"id": rec["id"], "text": rec["text"], "ts": rec.get("ts")}
                row.update({k: rec[k] for k in ("lang", "lang_confidence") if k in rec})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def build_config(spec: dict, workdir: Path, base_model: str):
    judge = spec["judge"]
    train = {k: v for k, v in (spec.get("train") or {}).items() if k in {f.name for f in fields(TrainSpec)}}
    per_language = spec.get("per_language") or {}
    if per_language.get("enabled"):
        train["per_language_min_examples"] = int(per_language.get("min_examples", 300))
    return config_from_dict(
        {
            "project": spec["prompt_id"],
            "workdir": str(workdir),
            "task": {
                "name": spec["prompt_id"],
                "question": spec["prompt"]["instructions"],
                "labels": {label["name"]: label.get("description", "") for label in spec["prompt"]["labels"]},
            },
            "judge": {
                "base_url": judge["base_url"],
                "model": judge["model"],
                "api_key_env": judge.get("api_key_env"),
                "mode": judge.get("mode", "logprobs"),
                "samples": judge.get("samples", 7),
                "license": {"allows_training_on_outputs": bool(judge.get("license_ack"))},
            },
            # The prompt's student trains on every language the prompt receives; with students per
            # language, each language the router is sure of in enough inputs gets one more.
            "students": {
                "english": [base_model],
                "multilingual": [],
                "per_language": [per_language["base_model"]] if per_language.get("enabled") else [],
            },
            "train": train,
            "cascade": {"target_accuracy": spec.get("target_accuracy", 0.95), **spec.get("calibration", {})},
        }
    )


def judgments_prefix(cfg, version: str | None = None) -> str:
    """Reuse labels across job IDs, but never across changed tasks, judge settings or judge
    versions. `version` is the model the judge answers as: when an alias such as `-latest`
    resolves to another one, the training target changed, and its labels start a cache of their
    own. A judge that names no version, or its configured model, keeps the cache it had before
    versions were recorded."""
    judge = asdict(cfg.judge)
    if version and version != cfg.judge.model:
        judge["version"] = version
    identity = json.dumps(
        {"task": asdict(cfg.task), "labels": cfg.task.label_names, "judge": judge},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"judgments/{digest}/"


def judged_as(labels_path: Path) -> str | None:
    """The version most of the labels name (`judge_version`), for a judge that didn't answer the
    first call."""
    versions = Counter(r.get("judge_version") for r in read_log(labels_path) if r.get("judge_version"))
    return versions.most_common(1)[0][0] if versions else None


REFERENCE_FILE = "reference_labels.jsonl"
LANGUAGES_FILE = "languages.json"


def write_routes(cfg, min_confidence: float) -> Counter:
    """`routes.json`: the language of each logged input the gateway's router was at least
    `min_confidence` sure of, which the students per language train on. Returns the counts."""
    from .pipeline import ROUTES_FILE

    routes = {
        r["id"]: r["lang"]
        for r in read_log(cfg.shadow_log)
        if r.get("lang") and float(r.get("lang_confidence") or 0) >= min_confidence
    }
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / ROUTES_FILE).write_text(json.dumps(routes), encoding="utf-8")
    return Counter(routes.values())


def skip_known_answers(cfg, saved: set[str]) -> int:
    """Leave texts whose answer a train or audit reference label gives out of judging, unless the
    judge labeled them already (those still score the judge). The judge can take seconds a text."""
    from .human import read_human_labels

    known = {i for i, h in read_human_labels(cfg.human_labels_path, cfg.task.label_names).items() if h["split"]}
    skip = known - saved
    if not skip:
        return 0
    rows = [r for r in read_log(cfg.shadow_log) if r["id"] not in skip]
    with cfg.shadow_log.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(skip)


def write_reference_labels(cfg, path: Path) -> dict:
    """The human labels the student trained on (`train`) and was kept from (`audit`), for the
    gateway: it audits the student against the held-out answers, and not at all on the others."""
    from .human import is_held_out, read_human_labels

    counts = {"train": 0, "audit": 0}
    human = read_human_labels(cfg.human_labels_path, cfg.task.label_names)
    with path.open("w", encoding="utf-8") as f:
        for i, h in human.items():
            use = "audit" if is_held_out(i, cfg.train.human_holdout, cfg.train.seed, h["split"]) else "train"
            counts[use] += 1
            f.write(json.dumps({"text": h["text"], "label": h["label"], "use": use}, ensure_ascii=False) + "\n")
    return counts


def process(spec: dict, root: Path, gw: GatewayClient, log=print) -> dict:
    from .human import add_store_labels, describe
    from .judge import Judge, label_log
    from .pipeline import train_all

    job = spec["job_id"]

    def say(message: str) -> None:
        log(f"job {job}: {message}")

    store = open_store(spec["store"])
    base_model = choose_base_model(spec["base_model"], len(spec["prompt"]["labels"]), log=log)
    if base_model != spec["base_model"]:
        say(f"base model {spec['base_model']}: {base_model}")
    cfg = build_config(spec, root / job, base_model)
    n = collect_inputs(store, spec["inputs_prefix"], cfg.shadow_log)
    say(f"{n} distinct inputs logged under {spec['inputs_prefix']}")
    if n < MIN_EXAMPLES:
        return {"ok": False, "message": f"only {n} distinct inputs logged; need at least {MIN_EXAMPLES}"}
    per_language = spec.get("per_language") or {}
    min_confidence = float(per_language.get("min_confidence", 0.7))
    if per_language.get("enabled"):
        counts = write_routes(cfg, min_confidence)
        enough = {code: k for code, k in counts.most_common() if k >= cfg.train.per_language_min_examples}
        say(
            f"students per language: the router is sure of {sum(counts.values())} inputs' language; "
            + (
                "a student each for " + ", ".join(f"{code} ({k})" for code, k in enough.items())
                if enough
                else "no language"
            )
            + f", with at least {cfg.train.per_language_min_examples}"
        )
    # Human labels an admin put in the store; their texts are judged too, to score the judge.
    human = add_store_labels(
        store,
        spec.get("human_labels_prefix", f"human_labels/{spec['prompt_id']}/"),
        cfg.human_labels_path,
        cfg.task.label_names,
        shadow_log=cfg.shadow_log,
    )
    if human["total"] or human["unknown_labels"]:
        skipped = sum(human["unknown_labels"].values())
        say(
            f"{human['total']} human labels ({human['added']} new)"
            + (f"; skipped {skipped} whose label the prompt doesn't have" if skipped else "")
        )
    # The model the judge answers as now (an alias resolved), from one logged input: a new one
    # means new labels, and the gateway tests the student again when audits see another one.
    judge = Judge(cfg.judge, cfg.task)
    version = judge.answered_as(next(iter(read_log(cfg.shadow_log)))["text"])
    if version and version != cfg.judge.model:
        say(f"the judge {cfg.judge.model} answers as {version}")
    labels_prefix = judgments_prefix(cfg, version)
    report = JobProgress(gw, job, log=log)
    # The shared store survives retries on another worker, as well as a local restart.
    saved = {r["id"] for r in read_log(cfg.labels_path)}
    with cfg.labels_path.open("a", encoding="utf-8") as f:
        for key in store.list(labels_prefix):
            if not key.endswith(".jsonl"):
                continue
            rec = json.loads(store.get(key))
            if rec["id"] not in saved:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved.add(rec["id"])
    if saved:
        say(f"reusing {len(saved)} saved judgments")
    skipped = skip_known_answers(cfg, saved)
    if skipped:
        say(f"{skipped} texts with a train or audit reference label skip the judge: their label is known")

    def save_label(rec: dict) -> None:
        key = hashlib.sha256(rec["id"].encode("utf-8")).hexdigest()
        store.put(f"{labels_prefix}{key}.jsonl", (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    to_judge = len(read_log(cfg.shadow_log))  # the logged inputs, and the human-labeled texts added above
    report.stage(f"judging {to_judge} inputs with {spec['judge']['model']}")
    t0 = time.monotonic()
    stats = label_log(
        judge,
        cfg.shadow_log,
        cfg.labels_path,
        progress=False,
        on_progress=report.count,
        on_label=save_label,
    )
    failures = "; ".join(f"{error} ({times})" for error, times in stats.get("errors", {}).items())
    say(
        f"judging done in {time.monotonic() - t0:.0f}s: {stats['labeled']} labeled, {stats['failed']} failed, "
        f"{stats['total']} labeled in all" + (f"; most common failures: {failures}" if failures else "")
    )
    # Counts from here on are training steps, or texts embedded for a frozen embedding model.
    report.stage(f"training a student on {base_model} with {stats['total']} judged inputs")
    t0 = time.monotonic()
    try:
        deploy = train_all(cfg, keep_hf=False, log=log, on_progress=report.count, on_stage=report.stage)
    except SystemExit as e:  # train_all's way out for the CLI when too few inputs were labeled: the job fails
        return {"ok": False, "message": f"not enough labeled data to train: {e}"}
    say(f"training done in {time.monotonic() - t0:.0f}s")
    group = deploy["groups"].get("english")  # the prompt's student, for every language
    if group is None:
        return {"ok": False, "message": "not enough labeled data to train"}
    model_dir = cfg.workdir / group["model_dir"]
    prefix = spec["models_prefix"].rstrip("/")
    say(f"publishing the student ({group['model']}) to {prefix}/ in the store")
    references = write_reference_labels(cfg, model_dir / REFERENCE_FILE)
    files = [*MODEL_FILES, *([REFERENCE_FILE] if references["train"] or references["audit"] else [])]
    size = 0
    for name in files:
        data = (model_dir / name).read_bytes()
        store.put(f"{prefix}/{name}", data)
        size += len(data)
    say(f"published {len(files)} files, {size / 1e6:.1f} MB")
    if references["audit"] or references["train"]:
        say(
            f"reference labels: {references['train']} trained on, {references['audit']} held out "
            "for the gateway to audit the student against"
        )
    languages = publish_languages(cfg, deploy, store, prefix, min_confidence, say)
    test, cascade = group["test"], group["cascade"]
    result = {
        "ok": True,
        "threshold": group["threshold"],
        "message": f"test agreement with {'its training labels' if references['train'] else 'the judge'} "
        f"{test['accuracy']:.3f} on {test['n']} held-out inputs; "
        f"answers alone for {cascade['coverage']:.0%} of validation traffic" + describe(group.get("human")),
        "metrics": {
            "test": test,
            "cascade": cascade,
            "inputs": n,
            "labeled": stats["total"],
            "base_model": base_model,
            "judge": spec["judge"]["model"],
            "judge_version": version or judged_as(cfg.labels_path),
            "reference_labels": references,
        },
    }
    if languages:
        result["metrics"]["languages"] = languages
        result["message"] += "; students for " + ", ".join(
            f"{code} ({info['test']['accuracy']:.3f})" for code, info in languages.items()
        )
    if "human" in group:  # scores against the human labels held out of training
        result["metrics"]["human"] = group["human"]
    return result


def publish_languages(cfg, deploy: dict, store, prefix: str, min_confidence: float, say) -> dict:
    """The students per language, each to `{prefix}/lang/{code}/`, and `languages.json`, which
    lists them for the gateway: their thresholds, and the router's confidence they need. Returns
    each one's numbers, for the job's metrics."""
    from .pipeline import ROUTES_FILE

    groups = {g[len("lang-") :]: info for g, info in deploy["groups"].items() if g.startswith("lang-")}
    if not groups:
        return {}
    routes = json.loads((cfg.workdir / ROUTES_FILE).read_text(encoding="utf-8"))
    counts = Counter(routes.values())
    languages = {}
    for code, info in groups.items():
        model_dir = cfg.workdir / info["model_dir"]
        for name in MODEL_FILES:
            store.put(f"{prefix}/lang/{code}/{name}", (model_dir / name).read_bytes())
        languages[code] = {
            "model": info["model"],
            "threshold": info["threshold"],
            "examples": counts[code],
            "test": info["test"],
            "cascade": info["cascade"],
        }
        say(f"published the {code} student ({info['model']}): test agreement {info['test']['accuracy']:.3f}")
    manifest = {
        "min_confidence": min_confidence,
        "languages": {
            code: {"threshold": v["threshold"], "model": v["model"], "examples": v["examples"]}
            for code, v in languages.items()
        },
    }
    store.put(f"{prefix}/{LANGUAGES_FILE}", json.dumps(manifest).encode("utf-8"))
    return languages


def run_worker(
    gateway: str, token: str, workdir: str | Path = "runs/worker", once: bool = False, poll: float = 15, log=log_line
) -> int:
    """Process jobs until interrupted (or until the queue is empty, with `once`). Returns jobs processed."""
    gw, root, done = GatewayClient(gateway, token), Path(workdir), 0
    log(f"worker: waiting for jobs from {gateway}, working in {root}")
    while True:
        try:
            spec = gw.claim()
        except httpx.TransportError as e:  # the gateway is starting or restarting: wait for it
            if once:
                raise
            log(f"gateway unreachable ({e}); retrying in {poll:g}s")
            time.sleep(poll)
            continue
        if spec is None:
            if once:
                return done
            time.sleep(poll)
            continue
        job, t0 = spec["job_id"], time.monotonic()
        log(f"job {job}: prompt {spec['prompt_id']}, base model {spec['base_model']}, judge {spec['judge']['model']}")
        try:
            result = process(spec, root, gw, log=log)
        except JobCancelled:
            log(f"job {job}: cancelled by an admin after {time.monotonic() - t0:.0f}s; stopped")
            done += 1
            continue
        except Exception as e:  # report the failure instead of leaving the job hanging
            log(f"job {job}: failed after {time.monotonic() - t0:.0f}s\n{traceback.format_exc().rstrip()}")
            result = {"ok": False, "message": f"{type(e).__name__}: {e}"[:500]}
        except BaseException as e:  # Ctrl-C: nothing tells the gateway, so the job shows as running
            log(f"job {job}: stopped by {type(e).__name__}; it shows as running until an admin cancels it")
            raise
        outcome = "finished" if result.get("ok") else "failed"
        log(f"job {job}: {outcome} in {time.monotonic() - t0:.0f}s: {result.get('message')}")
        reply = None
        while reply is None:  # a result that never reaches the gateway leaves the job running for good
            try:
                reply = gw.complete(job, result)
            except httpx.TransportError as e:
                if once:
                    raise
                log(f"gateway unreachable ({e}); retrying in {poll:g}s")
                time.sleep(poll)
        retry = reply.get("retry") or {}
        log(
            f"job {job}: reported to the gateway: deployed={reply.get('deployed')}"
            + (f", error {reply['error']}" if reply.get("error") else "")
            + (f", retry {retry.get('attempt')} not before {retry.get('not_before')}" if retry else "")
        )
        done += 1
