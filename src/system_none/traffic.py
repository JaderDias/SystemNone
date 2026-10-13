"""Test traffic for the gateway: texts from a public dataset, sent to `POST /v1/classify`.

`system-none traffic` (and `make traffic`) asks which dataset to use, how many requests to send, how
many at a time and how fast. It then sends each text with the dataset's prompt, in the gateway's
request shape (`ClassifyReq` in gateway/src/api.rs):

    {"prompt": {"instructions": "...", "labels": ["...", {"name": "...", "description": "..."}]},
     "input": "..."}

It shows progress, and ends with who answered (student, the gateway's cache or System One), latency percentiles,
agreement with the dataset's own labels, and the errors. Flags answer the menu's questions, so it
also runs unattended: `system-none traffic --dataset banking77 --n 500`.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import importlib.util
import json
import math
import random
import sys
import threading
import time
import typing
from pathlib import Path

DEFAULT_GATEWAY = "http://127.0.0.1:8080"
DEFAULTS = {"n": 100, "concurrency": 4, "rate": 0}
# The gateway's limit per prompt (MAX_LABELS in gateway/src/prompt.rs). Up to 26 labels, openai_chat
# System One models answer with a letter; above, with the label's name.
MAX_LABELS = 255
FAST_DECISIONS = "fast-decisions"

# The ai4privacy slice of the multilingual tests, with the question and labels of tests/hardcoded_judge.py.
# Its labels live in a key only judges read (that module and the mock judge), so no agreement is reported for it.
AI4PRIVACY_QUESTION = "Which kind of personal data is the most sensitive item in this text?"
AI4PRIVACY_LABELS = {
    "id_number": "the most sensitive item is an identifier: passport, licence, tax, account, card or other ID number",
    "contact": "the most sensitive item is contact data: an email address or phone number",
    "address": "the most sensitive item is a location: street address, postcode, city or area",
    "identity_only": "only names or demographic details (age, gender, title), no number, contact or place",
}


class TrafficError(Exception):
    """A problem to report to the caller in one line (unknown dataset, no gateway, ...)."""


@dataclasses.dataclass
class Dataset:
    """One prompt (instructions and labels) and the texts to send with it."""

    name: str
    instructions: str
    labels: list[str]
    texts: list[str]
    gold: list[str] | None = None  # each text's label in the dataset, when it is public
    descriptions: dict[str, str] = dataclasses.field(default_factory=dict)
    splits: list[str] | None = None  # each text's official split ("train" or "test"), when there is one

    def prompt(self) -> dict:
        labels = [
            {"name": label, "description": self.descriptions[label]} if self.descriptions.get(label) else label
            for label in self.labels
        ]
        return {"instructions": self.instructions, "labels": labels}


def request_body(ds: Dataset, text: str, model: str | None = None) -> dict:
    """The JSON body of one `POST /v1/classify`; `model` forces a System One model."""
    body = {"prompt": ds.prompt(), "input": text}
    if model:
        body["model"] = model
    return body


def fit_labels(ds: Dataset, limit: int = MAX_LABELS) -> Dataset:
    """Drop empty texts; past `limit` labels, keep the commonest ones (in their order) and their texts."""
    keep = [i for i, t in enumerate(ds.texts) if t.strip()]
    labels = ds.labels
    if len(labels) > limit:
        if ds.gold is None:
            raise TrafficError(f"{ds.name} has {len(labels)} labels and the gateway takes at most {limit}")
        counts = collections.Counter(ds.gold[i] for i in keep)
        top = set(sorted(labels, key=lambda label: -counts[label])[:limit])  # stable sort: ties keep the label order
        labels = [label for label in labels if label in top]
        keep = [i for i in keep if ds.gold[i] in top]
    return dataclasses.replace(
        ds,
        labels=labels,
        texts=[ds.texts[i] for i in keep],
        gold=None if ds.gold is None else [ds.gold[i] for i in keep],
        splits=None if ds.splits is None else [ds.splits[i] for i in keep],
    )


# ---- datasets ---------------------------------------------------------------------------------


def ai4privacy_dir() -> Path | None:
    """tests/data/ai4privacy in the current directory or in the source checkout (not in installed packages)."""
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        d = base / "tests" / "data" / "ai4privacy"
        if (d / "inputs.jsonl").is_file():
            return d
    return None


def catalog() -> list[tuple[str, str]]:
    """(name, description) of every dataset the menu offers."""
    from .fastdecisions import DOMAINS

    rows = [("banking77", "bank customer messages, 77 intents (downloaded)")]
    if ai4privacy_dir():
        rows.append(("ai4privacy", "3,928 English, Hindi and Telugu texts: the kind of personal data in them"))
    bench = "" if importlib.util.find_spec("datasets") else ', needs pip install -e ".[bench]"'
    rows += [
        (f"{FAST_DECISIONS}/{d}", f"Fastino Fast Decisions dev split, 100 texts (downloaded{bench})") for d in DOMAINS
    ]
    return rows


def load_banking77() -> Dataset:
    import httpx

    from . import banking77

    try:
        train, test = banking77.load("train"), banking77.load("test")
    except httpx.HTTPError as e:
        raise TrafficError(f"could not download banking77: {e}") from e
    rows = train + test
    return Dataset(
        "banking77",
        banking77.QUESTION,
        sorted({intent for _, intent in rows}),
        [text for text, _ in rows],
        [intent for _, intent in rows],
        splits=["train"] * len(train) + ["test"] * len(test),
    )


def load_ai4privacy() -> Dataset:
    d = ai4privacy_dir()
    if d is None:
        raise TrafficError("tests/data/ai4privacy not found: run this from a checkout of the repository")
    rows = [json.loads(line) for line in (d / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset(
        "ai4privacy",
        AI4PRIVACY_QUESTION,
        list(AI4PRIVACY_LABELS),
        [r["text"] for r in rows],
        descriptions=dict(AI4PRIVACY_LABELS),
    )


def load_fast_decisions(domain: str, head: str | None = None, pick=None) -> Dataset:
    """One single-label head of a Fast Decisions domain; `pick(title, options)` chooses among several."""
    from .fastdecisions import DOMAINS, load_heads

    if domain not in DOMAINS:
        raise TrafficError(f"unknown Fast Decisions domain '{domain}'; one of: {', '.join(DOMAINS)}")
    try:
        heads = {key.split("/", 1)[1]: h for key, h in load_heads([domain]).items()}
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] != "datasets":
            raise
        raise TrafficError('Fast Decisions needs the datasets package: pip install -e ".[bench]"') from e
    if head is None:
        if len(heads) == 1:
            head = next(iter(heads))
        elif pick is not None:
            head = pick(
                f"{domain} has {len(heads)} questions:", [(n, f"{len(h['labels'])} labels") for n, h in heads.items()]
            )
        else:
            raise TrafficError(
                f"{domain} has {len(heads)} questions; name one: "
                + ", ".join(f"{FAST_DECISIONS}/{domain}/{n}" for n in heads)
            )
    if head not in heads:
        raise TrafficError(f"{domain} has no single-label question '{head}'; one of: {', '.join(heads)}")
    h = heads[head]
    instructions = f"Pick the {head.replace('_', ' ')} label that fits this {domain.replace('_', ' ')} text."
    return Dataset(
        f"{FAST_DECISIONS}/{domain}/{head}", instructions, h["labels"], h["texts"], [h["labels"][g] for g in h["gold"]]
    )


def load(name: str, pick=None) -> Dataset:
    """The dataset called `name` (see `catalog`; Fast Decisions also takes `<domain>/<head>`), fitted to the gateway."""
    if name == "banking77":
        ds = load_banking77()
    elif name == "ai4privacy":
        ds = load_ai4privacy()
    elif name.startswith(FAST_DECISIONS + "/"):
        domain, _, head = name[len(FAST_DECISIONS) + 1 :].partition("/")
        ds = load_fast_decisions(domain, head or None, pick)
    else:
        raise TrafficError(f"unknown dataset '{name}'; `system-none traffic --list` shows them")
    ds = fit_labels(ds)
    if not ds.texts:
        raise TrafficError(f"{ds.name} has no texts")
    return ds


# ---- reference labels -------------------------------------------------------------------------


def reference_labels(ds: Dataset, audit_share: float = 0.2) -> list[dict]:
    """The dataset's own answers, for the gateway to train students on and audit them against.

    Texts of an official test split are `audit`: kept out of training, they are the answers the
    student is checked against. Every other text is `train`. Without official splits, a stable
    `audit_share` of the texts (by a hash of the text) is held out instead.
    """
    if ds.gold is None:
        raise TrafficError(f"{ds.name} has no public labels to upload")
    import hashlib

    def split(i: int) -> str:
        if ds.splits is not None:
            return "audit" if ds.splits[i] == "test" else "train"
        digest = hashlib.sha256(ds.texts[i].encode("utf-8")).digest()
        return "audit" if int.from_bytes(digest[:8], "big") / 2**64 < audit_share else "train"

    return [{"text": t, "label": ds.gold[i], "split": split(i)} for i, t in enumerate(ds.texts)]


def upload_labels(gateway: str, token: str, ds: Dataset, rows: list[dict], client=None) -> dict:
    """`POST /admin/api/labels`: the gateway stores them for the prompt's next training job."""
    import httpx

    client = client or httpx.Client(timeout=120)
    name = ds.name.replace("/", "-")[:64]
    try:
        r = client.post(
            f"{gateway}/admin/api/labels",
            json={"prompt": ds.prompt(), "name": name, "labels": rows},
            headers={"Authorization": f"Bearer {token}"},
        )
    except httpx.TransportError as e:
        raise TrafficError(f"no gateway at {gateway} ({e}); start it with make") from e
    if r.status_code == 401:
        raise TrafficError("the gateway refused the admin token: set SN_ADMIN_TOKEN (make reads it from .env)")
    if r.status_code >= 400:
        raise TrafficError(f"the gateway refused the labels ({r.status_code}): {r.text[:300]}")
    return r.json()


def train_now(gateway: str, token: str, prompt_id: str, client=None) -> dict | None:
    """Queue a training job for the prompt; None if one is queued or running already."""
    import httpx

    client = client or httpx.Client(timeout=30)
    r = client.post(f"{gateway}/admin/api/prompts/{prompt_id}/train", headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 409:
        return None
    if r.status_code >= 400:
        raise TrafficError(f"the gateway could not queue training ({r.status_code}): {r.text[:300]}")
    return r.json()


def labels_main(args, out=print, client=None) -> dict:
    """`system-none labels`: upload a dataset's answers as reference labels, then optionally train."""
    import os

    token = os.environ.get(args.token_env, "")
    if not token:
        raise TrafficError(f"set {args.token_env} to the gateway's admin token (make reads it from .env)")
    ds = load(args.dataset)
    rows = reference_labels(ds, args.audit_share)
    result = upload_labels(args.gateway, token, ds, rows, client=client)
    out(
        f"{ds.name}: uploaded {result['train']:,} training and {result['audit']:,} audit answers "
        f"for prompt {result['prompt_id']}"
    )
    out(
        "The next training job trains on the training answers instead of the judge's, and the student "
        "is audited against the audit answers, which it never sees in training."
    )
    if args.train:
        job = train_now(args.gateway, token, result["prompt_id"], client=client)
        out(f"queued training job {job['id']}" if job else "a training job is already queued or running")
    else:
        out(f"Press Train now in {args.gateway}/admin, or run this again with --train.")
    return result


# ---- menu -------------------------------------------------------------------------------------


def parse_choice(answer: str, names: list[str]) -> str | None:
    """A menu answer: an option's number, its name, or the start of exactly one name."""
    answer = answer.strip()
    if answer.isdigit():
        return names[int(answer) - 1] if 1 <= int(answer) <= len(names) else None
    if answer in names:
        return answer
    starts = [n for n in names if answer and n.startswith(answer)]
    return starts[0] if len(starts) == 1 else None


def choose(title: str, options: list[tuple[str, str]], ask=input, out=print) -> str:
    """A numbered menu; asks again until the answer names an option."""
    names = [n for n, _ in options]
    out(title)
    width = max(len(n) for n in names)
    for i, (name, about) in enumerate(options, 1):
        out(f"  {i:>2}) {name:<{width}}  {about}")
    while True:
        answer = ask(f"Pick 1-{len(names)}: ")
        name = parse_choice(answer, names)
        if name is not None:
            return name
        out(f"  '{answer.strip()}' is not one of the options")


def parse_number(answer: str, default, cast=int, minimum=0):
    """A number answer: the default when empty, None when not a number of at least `minimum`."""
    answer = answer.strip()
    if not answer:
        return default
    try:
        value = cast(answer)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= minimum else None


def ask_number(question: str, default, cast=int, minimum=0, ask=input, out=print):
    while True:
        value = parse_number(ask(f"{question} [{default}]: "), default, cast, minimum)
        if value is not None:
            return value
        out(f"  enter {'a whole number' if cast is int else 'a number'} of at least {minimum}")


# ---- sending ----------------------------------------------------------------------------------


@dataclasses.dataclass
class Result:
    ms: float  # the round trip, as the caller sees it
    source: str | None = None  # "student", "cache" or "system_one"; None on error
    label: str | None = None
    degraded: bool = False  # a student answered because System One failed
    prompt_id: str | None = None
    error: str | None = None
    gold: str | None = None  # the dataset's label for the text


def send(client, url: str, body: dict, gold: str | None = None) -> Result:
    import httpx

    t0 = time.perf_counter()
    try:
        r = client.post(url, json=body)
    except httpx.HTTPError as e:
        return Result((time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}"[:200], gold=gold)
    ms = (time.perf_counter() - t0) * 1000
    try:
        answer = r.json()
    except ValueError:
        answer = {}
    if r.status_code != 200 or not isinstance(answer, dict) or "label" not in answer:
        message = answer.get("error") if isinstance(answer, dict) and answer.get("error") else r.text.strip()
        return Result(ms, error=f"HTTP {r.status_code}: {message}"[:200], gold=gold)
    return Result(
        ms, answer.get("source"), answer["label"], bool(answer.get("degraded")), answer.get("prompt_id"), gold=gold
    )


def order(size: int, n: int, seed: int = 0) -> list[int]:
    """Which text each request sends: a seeded shuffle, repeated when `n` exceeds the dataset."""
    idx = list(range(size))
    random.Random(seed).shuffle(idx)
    return [idx[i % size] for i in range(n)]


class Schedule:
    """Hands out request numbers, `1 / rate` seconds apart when a rate is set, until `n` or `stop()`."""

    def __init__(self, n: int, rate: float = 0.0, clock=time.perf_counter, sleep=None):
        self.halt = threading.Event()
        self.n, self.interval, self.clock = n, 1 / rate if rate > 0 else 0.0, clock
        self.sleep = sleep or self.halt.wait  # stop() wakes the waiting workers
        self.next_i, self.start = 0, None
        self.lock = threading.Lock()

    @property
    def stopped(self) -> bool:
        return self.halt.is_set()

    def next(self) -> int | None:
        with self.lock:
            if self.stopped or self.next_i >= self.n:
                return None
            i, self.next_i = self.next_i, self.next_i + 1
            if self.start is None:
                self.start = self.clock()
        delay = self.start + i * self.interval - self.clock()
        if delay > 0:
            self.sleep(delay)
        return None if self.stopped else i

    def stop(self):
        self.halt.set()


class Progress:
    """One status line: redrawn in place on a terminal, printed at every 10% otherwise."""

    def __init__(self, total: int, out=None):
        self.total, self.out = total, out if out is not None else sys.stderr
        self.tty = bool(getattr(self.out, "isatty", lambda: False)())
        self.counts, self.done = collections.Counter(), 0
        self.t0 = self.last = time.perf_counter()
        self.lock = threading.Lock()

    def line(self) -> str:
        c, rate = self.counts, self.done / max(time.perf_counter() - self.t0, 1e-9)
        cache = f"{c['cache']} cache, " if c["cache"] else ""
        return (
            f"{self.done}/{self.total} sent: {c['student']} student, {cache}{c['system_one']} System One, "
            f"{c['error']} errors, {rate:.1f}/s"
        )

    def add(self, r: Result):
        with self.lock:
            self.done += 1
            self.counts["error" if r.error else r.source] += 1
            now = time.perf_counter()
            if self.tty and (now - self.last >= 0.2 or self.done == self.total):
                self.last = now
                self.out.write("\r" + self.line())
                self.out.flush()
            elif not self.tty and self.done * 10 // self.total > (self.done - 1) * 10 // self.total:
                self.out.write(self.line() + "\n")
                self.out.flush()

    def close(self):
        if self.tty and self.done:
            self.out.write("\n")
            self.out.flush()


def check_gateway(gateway: str, client=None):
    import httpx

    try:
        r = (client or httpx).get(f"{gateway}/health", timeout=10)
    except httpx.HTTPError as e:
        raise TrafficError(
            f"no gateway answers at {gateway} ({type(e).__name__}); start one (see the README, "
            "Running the gateway) or pass --gateway"
        ) from e
    if r.status_code != 200:
        raise TrafficError(f"{gateway}/health answered HTTP {r.status_code}; is this a SystemNone gateway?")


def run(
    ds: Dataset,
    gateway: str = DEFAULT_GATEWAY,
    n: int = 100,
    concurrency: int = 4,
    rate: float = 0.0,
    model: str | None = None,
    seed: int = 0,
    timeout: float = 60.0,
    client=None,
    out=None,
) -> dict:
    """Send `n` texts of `ds` to the gateway, `concurrency` at a time, at most `rate` per second; returns `summarize`.

    Ctrl-C stops sending new requests; the summary then covers the ones already answered.
    """
    import httpx

    out = out if out is not None else sys.stderr
    gateway = gateway.rstrip("/")
    url = f"{gateway}/v1/classify"
    own = client is None
    if own:
        client = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=concurrency))
    try:
        check_gateway(gateway, client)
        picks = order(len(ds.texts), n, seed)
        results: list[Result | None] = [None] * n
        schedule, progress = Schedule(n, rate), Progress(n, out)

        def worker():
            while (i := schedule.next()) is not None:
                t = picks[i]
                result = send(client, url, request_body(ds, ds.texts[t], model), ds.gold[t] if ds.gold else None)
                results[i] = result
                progress.add(result)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(concurrency, n)))]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(0.1)
        except KeyboardInterrupt:
            schedule.stop()
            out.write("\nstopping: waiting for the requests in flight\n")
            for t in threads:
                t.join()
        progress.close()
        summary = summarize(results, time.perf_counter() - t0)
        summary["stopped_early"] = schedule.stopped
        return summary
    finally:
        if own:
            client.close()


# ---- summary ----------------------------------------------------------------------------------


def percentile(ascending: list[float], p: float) -> float | None:
    """Nearest-rank percentile of an ascending list."""
    if not ascending:
        return None
    return ascending[max(0, math.ceil(p / 100 * len(ascending)) - 1)]


def summarize(results: list[Result | None], seconds: float) -> dict:
    """Counts, latency percentiles (ms) and agreement with the dataset, per source and overall."""
    done = [r for r in results if r is not None]
    answered = [r for r in done if r.error is None]

    def group(rs: list[Result]) -> dict:
        ms = sorted(r.ms for r in rs)
        scored = [r for r in rs if r.gold is not None]
        return {
            "answers": len(rs),
            "latency_ms": {  # percentile() is None only for an empty list, and ms has values here
                k: round(typing.cast(float, percentile(ms, p)), 2)
                for k, p in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))
            }
            if ms
            else None,
            "agreement": round(sum(r.label == r.gold for r in scored) / len(scored), 4) if scored else None,
        }

    return {
        "requests": len(done),
        "planned": len(results),
        "seconds": round(seconds, 2),
        "per_second": round(len(done) / seconds, 1) if seconds > 0 else None,
        "student": group([r for r in answered if r.source == "student"]),
        "cache": group([r for r in answered if r.source == "cache"]),
        "system_one": group([r for r in answered if r.source == "system_one"]),
        "all": group(answered),
        "degraded": sum(r.degraded for r in answered),
        "errors": len(done) - len(answered),
        "error_kinds": collections.Counter(r.error for r in done if r.error).most_common(5),
        "prompt_ids": sorted({r.prompt_id for r in answered if r.prompt_id}),
    }


def format_summary(s: dict, name: str, gateway: str) -> str:
    total = s["requests"]

    def share(k: int) -> str:
        return f"{k / total:.1%}" if total else "-"

    head = f"{name}: {total} requests to {gateway.rstrip('/')}/v1/classify in {s['seconds']} s"
    lines = [head + (f" ({s['per_second']}/s)" if s["per_second"] is not None else "")]
    if s.get("stopped_early"):
        lines.append(f"stopped early: {total} of {s['planned']} sent")
    cols = ["answers", "share", "p50 ms", "p90 ms", "p99 ms", "max ms", "agree"]
    lines.append(f"  {'':<12}" + "".join(f"{c:>9}" for c in cols))
    rows = [("student", s["student"]), ("cache", s.get("cache")), ("System One", s["system_one"]), ("all", s["all"])]
    for label, g in [(label, g) for label, g in rows if g and (g["answers"] or label != "cache")]:
        lat = g["latency_ms"] or {}
        cells = [g["answers"], share(g["answers"])] + [
            f"{lat[k]:.1f}" if lat else "-" for k in ("p50", "p90", "p99", "max")
        ]
        cells.append(f"{g['agreement']:.1%}" if g["agreement"] is not None else "-")
        lines.append(f"  {label:<12}" + "".join(f"{c:>9}" for c in cells))
    lines.append(f"  {'errors':<12}{s['errors']:>9}{share(s['errors']):>9}")
    for message, k in s["error_kinds"]:
        lines.append(f"    {k} x {message}")
    if s["degraded"]:
        lines.append(
            f"{s['degraded']} student answers were degraded: System One failed and a less confident student answered."
        )
    if s["all"]["agreement"] is not None:
        lines.append("agree: how often the answer is the dataset's own label.")
    for pid in s["prompt_ids"]:
        lines.append(f"prompt {pid}: usage at {gateway.rstrip('/')}/")
    return "\n".join(lines)


# ---- command ----------------------------------------------------------------------------------


def main(args, ask=input, out=print, err=None, client=None) -> dict | None:
    """`system-none traffic`. Without --dataset, a menu asks for the dataset and the settings the flags leave open."""
    if args.list:
        for name, about in catalog():
            out(f"{name:<34} {about}")
        return None
    for key, minimum in (("n", 1), ("concurrency", 1), ("rate", 0)):
        given = getattr(args, key)
        if given is not None and not (math.isfinite(given) and given >= minimum):
            raise TrafficError(f"--{key} must be at least {minimum}")
    check_gateway(args.gateway.rstrip("/"), client)  # before the menu and the download
    interactive = args.dataset is None
    menu = functools.partial(choose, ask=ask, out=out)
    name = args.dataset or menu("Datasets:", catalog())
    out(f"Loading {name}...")
    ds = load(name, pick=menu if interactive else None)
    out(f"{ds.name}: {len(ds.texts)} texts, {len(ds.labels)} labels")

    def setting(key: str, question: str, cast, minimum):
        given = getattr(args, key)
        if given is not None:
            return given
        return ask_number(question, DEFAULTS[key], cast, minimum, ask, out) if interactive else DEFAULTS[key]

    n = setting("n", "How many requests?", int, 1)
    concurrency = setting("concurrency", "How many at a time?", int, 1)
    rate = setting("rate", "At most how many per second (0: no limit)?", float, 0)
    if n > len(ds.texts):
        out(f"{ds.name} has {len(ds.texts)} texts, so they repeat.")
    out(
        f"Sending {n} requests to {args.gateway.rstrip('/')}/v1/classify, {concurrency} at a time"
        + (f", at most {rate:g}/s" if rate else "")
        + "; Ctrl-C stops early."
    )
    summary = run(
        ds,
        args.gateway,
        n,
        concurrency,
        rate,
        model=args.model,
        seed=args.seed,
        timeout=args.timeout,
        client=client,
        out=err,
    )
    out(format_summary(summary, ds.name, args.gateway))
    return summary
