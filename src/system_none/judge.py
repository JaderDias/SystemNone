"""The judge: label logged inputs with soft labels from a model you are allowed to train on.

Works with any OpenAI-compatible chat-completions endpoint. In "logprobs" mode one
call per text returns the probability of each option letter; in "sample" mode the
judge is sampled several times and vote shares become the soft label.

Letters stop at 26 options. With more labels the judge answers with the label's name instead:
vote shares still work, and in "logprobs" mode the probability the judge gave its answer goes
to that label, the rest spread evenly over the others.

Every result names the model the judge answered as (`model`), when its endpoint says: what an
alias such as `-latest` resolved to. A new version is a new training target (see the worker).

In "systemone" mode the judge is a server speaking TypeSafe's POST /v1/systemone, such as
Decider (huggingface.co/Mapika/decider-2b): the task is one Choice question, and the
probabilities it answers with are the soft label. Fastino's GLiDE speaks it too, but its terms
keep it serve-only (the incumbent).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

from .config import JudgeSpec, ModelEndpoint, TaskSpec
from .shadow import read_log, text_id

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The model the endpoint answered as in the call being labeled: each `alabel` sets its own, and
# `_post` fills it, so the many paths through the modes needn't pass it along.
_ANSWERED_AS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("answered_as", default=None)

# Services whose terms forbid using their outputs to train or distill models.
# They may serve traffic (as the incumbent) but can never act as the judge.
DENYLIST = {
    "typesafe": "TypeSafe's Master Customer Agreement §2.3(b) prohibits using Jev outputs for model distillation.",
    "jev": "TypeSafe's Master Customer Agreement §2.3(b) prohibits using Jev outputs for model distillation.",
    "fastino": "Fastino's Terms of Service §4.A prohibit using its inference responses to train, fine-tune or "
    "distill models that compete with Fastino.",
}


class LicenseError(RuntimeError):
    pass


def check_license(spec: JudgeSpec) -> None:
    """Refuse to label training data with a model whose outputs you may not train on."""
    ident = f"{spec.model} {spec.base_url}".lower()
    for key, reason in DENYLIST.items():
        if key in ident:
            raise LicenseError(f"'{spec.model}' can't be used as a judge: {reason} Keep it serve-only (incumbent).")
    if not spec.license.allows_training_on_outputs:
        raise LicenseError(
            "Set judge.license.allows_training_on_outputs: true after checking that the judge model's "
            "license and your provider's terms allow training models on its outputs."
        )


def build_messages(task: TaskSpec, text: str, answer: str = "letter") -> list[dict]:
    if answer == "name":
        options = "\n".join(f"- {name}" + (f": {desc}" if desc else "") for name, desc in task.labels.items())
        return [
            {
                "role": "system",
                "content": "You are a careful, calibrated classifier. Answer with the name of one option.",
            },
            {
                "role": "user",
                "content": f'{task.question}\n\nOptions:\n{options}\n\nText:\n"""\n{text}\n"""\n\n'
                "Answer with the name of the best option, exactly as written above, and nothing else.",
            },
        ]
    options = "\n".join(
        f"{LETTERS[i]}) {name}" + (f": {desc}" if desc else "") for i, (name, desc) in enumerate(task.labels.items())
    )
    return [
        {"role": "system", "content": "You are a careful, calibrated classifier. Answer with a single capital letter."},
        {
            "role": "user",
            "content": f'{task.question}\n\nOptions:\n{options}\n\nText:\n"""\n{text}\n"""\n\n'
            f"Answer with the letter of the best option ({LETTERS[0]}-{LETTERS[task.num_labels - 1]}) and nothing else.",
        },
    ]


def probs_from_top_logprobs(top: list[dict], k: int) -> tuple[list[float] | None, float]:
    """Turn the first token's top_logprobs into a distribution over the k option letters."""
    mass = [0.0] * k
    for item in top:
        tok = str(item.get("token", "")).strip().strip("().:").upper()
        if len(tok) == 1 and tok in LETTERS[:k]:
            mass[LETTERS.index(tok)] += math.exp(float(item["logprob"]))
    total = sum(mass)
    if total <= 0:
        return None, 0.0
    return [m / total for m in mass], total


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def letter_from_text(content: str, k: int) -> int | None:
    content = (content or "").strip()
    if content[:1].upper() in LETTERS[:k] and (len(content) == 1 or not content[1:2].isalpha()):
        return LETTERS.index(content[0].upper())
    for m in _LETTER_RE.finditer(content.upper()):
        if m.group(1) in LETTERS[:k]:
            return LETTERS.index(m.group(1))
    return None


def _norm(s: str) -> str:
    """Case, `_` / `-` versus spaces and repeated whitespace don't tell labels apart."""
    return " ".join(re.sub(r"[_\-]+", " ", s).split()).casefold()


# What a model may wrap its answer in, curly quotes included (RUF001 takes those for typos).
_QUOTES = "`*\"'“”‘’«» \t"  # noqa: RUF001
_ANSWER_PREFIX = re.compile(r"^(?:the\s+)?(?:answer|label|option|category|class)\s*(?:is)?\s*[:=-]?\s*", re.I)


def label_from_text(content: str, names: list[str]) -> int | None:
    """The label a free-text answer names, or None when it names none or several.

    An exact match comes first (as is, then ignoring case, quotes, markdown, a trailing full stop
    and an "Answer:" prefix); otherwise the first line must name exactly one label as whole words.
    Labels that differ only in case or `_` / `-` versus spaces need their exact name. The gateway
    reads `openai_chat` answers the same way (`label_from_name` in gateway/src/prompt.rs).
    """
    exact = {n: i for i, n in enumerate(names)}
    index: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        if _norm(n):
            index.setdefault(_norm(n), []).append(i)

    def only(ids: list[int]) -> int | None:
        return ids[0] if len(ids) == 1 else None

    lines = (content or "").strip().splitlines()
    first = lines[0] if lines else ""
    for cand in ((content or "").strip(), first):
        for c in (cand, _ANSWER_PREFIX.sub("", cand.strip(_QUOTES)).strip(_QUOTES).rstrip(".!;,").strip(_QUOTES)):
            if c in exact:
                return exact[c]
            if _norm(c) in index:
                return only(index[_norm(c)])
    said = _norm(first)
    hits = {n for n in index if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", said)}
    hits = {n for n in hits if not any(n != m and n in m for m in hits)}  # "card arrival", not also "card"
    return only(index[hits.pop()]) if len(hits) == 1 else None


def answer_probability(tokens: list[dict], name: str) -> float:
    """exp(sum of logprobs) of the generated tokens, up to the end of the label's name."""
    target, text, total = _norm(name), "", 0.0
    for t in tokens:
        text += str(t.get("token", ""))
        total += float(t.get("logprob", 0.0))
        if target in _norm(text):
            break
    return math.exp(total)


def probs_from_answer(k: int, idx: int, p: float) -> list[float]:
    """Probability p on the answer, the rest spread evenly; the answer always keeps the most."""
    p = min(1.0, max(p, 2 / (k + 1)))
    rest = (1 - p) / (k - 1)
    return [p if i == idx else rest for i in range(k)]


class Judge:
    """Soft-label a text with an OpenAI-compatible model."""

    def __init__(
        self,
        spec: ModelEndpoint,
        task: TaskSpec,
        transport: httpx.AsyncBaseTransport | None = None,
        enforce_license: bool = True,
    ):
        if enforce_license:
            if not isinstance(spec, JudgeSpec):
                raise LicenseError("A judge needs a JudgeSpec with a license acknowledgement.")
            check_license(spec)
        answer = getattr(spec, "answer", "auto")
        self.answer = answer if answer != "auto" else ("letter" if task.num_labels <= len(LETTERS) else "name")
        if self.answer == "letter" and task.num_labels > len(LETTERS):
            raise ValueError(
                f"A judge answering with a letter handles up to {len(LETTERS)} labels; "
                f"'{task.name}' has {task.num_labels}. Set answer: name (or auto)."
            )
        # Room for the longest label name, whatever the tokenizer (about one token per character at worst).
        self._name_tokens = min(128, 8 + max(len(n) for n in task.label_names))
        self.spec = spec
        self.task = task
        self._transport = transport
        if spec.mode == "openai_decisions" and task.num_labels > 26:
            raise ValueError("openai_decisions supports at most 26 choice options.")

    def _client(self) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key:
            headers["Authorization"] = f"Bearer {self.spec.api_key}"
        return httpx.AsyncClient(
            base_url=self.spec.base_url.rstrip("/"),
            headers=headers,
            timeout=self.spec.timeout,
            transport=self._transport,
        )

    async def _post(self, client: httpx.AsyncClient, body: dict, path: str = "/chat/completions") -> dict:
        """POST with retries on network errors, 425 (a model still warming up, as Fastino's do), 429
        and 5xx (other errors fail fast)."""
        delay = 1.0
        for attempt in range(self.spec.max_retries + 1):
            try:
                r = await client.post(path, json={**body, **self.spec.extra_body})
            except httpx.TransportError:
                if attempt >= self.spec.max_retries:
                    raise
            else:
                if r.status_code not in (425, 429, 500, 502, 503, 504) or attempt >= self.spec.max_retries:
                    r.raise_for_status()
                    data = r.json()
                    seen = _ANSWERED_AS.get()
                    model = data.get("model") if isinstance(data, dict) else None
                    if seen is not None and "model" not in seen and isinstance(model, str) and model:
                        seen["model"] = model
                    return data
            await asyncio.sleep(delay + random.random() * 0.5)
            delay *= 2
        raise RuntimeError("unreachable")

    async def alabel(self, client: httpx.AsyncClient, text: str) -> dict:
        """The soft label of `text`, with the model the endpoint answered as (`model`, or None)."""
        seen: dict = {}
        token = _ANSWERED_AS.set(seen)
        try:
            res = await self._alabel(client, text)
        finally:
            _ANSWERED_AS.reset(token)
        return {**res, "model": seen.get("model")}

    async def _alabel(self, client: httpx.AsyncClient, text: str) -> dict:
        if self.spec.mode == "openai_decisions":
            return await self._alabel_decisions(client, text)
        if self.spec.mode == "systemone":
            return await self._alabel_systemone(client, text)
        k = self.task.num_labels
        msgs = build_messages(self.task, text, self.answer)
        if self.answer == "name":
            return await self._alabel_by_name(client, msgs)
        if self.spec.mode == "logprobs":
            data = await self._post(
                client,
                {
                    "model": self.spec.model,
                    "messages": msgs,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": 20,
                },
            )
            try:
                top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    "The endpoint returned no logprobs. Use judge.mode: sample for this server."
                ) from None
            probs, mass = probs_from_top_logprobs(top, k)
            if probs is None:
                return {"probs": None, "mode": "logprobs", "letter_mass": 0.0}
            return {"probs": probs, "mode": "logprobs", "letter_mass": round(mass, 4)}

        # "sample" mode: vote shares over n samples
        votes: Counter = Counter()
        n = self.spec.samples
        data = await self._post(
            client,
            {"model": self.spec.model, "messages": msgs, "max_tokens": 4, "temperature": self.spec.temperature, "n": n},
        )
        contents = [c.get("message", {}).get("content", "") for c in data.get("choices", [])]
        while len(contents) < n:  # servers that ignore `n`
            more = await self._post(
                client,
                {"model": self.spec.model, "messages": msgs, "max_tokens": 4, "temperature": self.spec.temperature},
            )
            contents += [c.get("message", {}).get("content", "") for c in more.get("choices", [])]
        for c in contents[:n]:
            idx = letter_from_text(c, k)
            if idx is not None:
                votes[idx] += 1
        total = sum(votes.values())
        if total == 0:
            return {"probs": None, "mode": "sample", "letter_mass": 0.0}
        return {"probs": [votes[i] / total for i in range(k)], "mode": "sample", "letter_mass": total / n}

    async def _alabel_by_name(self, client: httpx.AsyncClient, msgs: list[dict]) -> dict:
        """The judge answers with a label's name. `letter_mass` is then the share of answers naming a label."""
        k, names = self.task.num_labels, self.task.label_names
        body = {"model": self.spec.model, "messages": msgs, "max_tokens": self._name_tokens}
        if self.spec.mode == "logprobs":
            data = await self._post(client, {**body, "temperature": 0, "logprobs": True})
            try:
                choice = data["choices"][0]
                tokens = choice["logprobs"]["content"]
                content = choice["message"]["content"]
                if not tokens:
                    raise TypeError
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    "The endpoint returned no logprobs. Use judge.mode: sample for this server."
                ) from None
            idx = label_from_text(content, names)
            if idx is None:
                return {"probs": None, "mode": "logprobs", "answer": "name", "letter_mass": 0.0}
            p = answer_probability(tokens, names[idx])
            return {
                "probs": probs_from_answer(k, idx, p),
                "mode": "logprobs",
                "answer": "name",
                "letter_mass": round(p, 4),
            }

        n = self.spec.samples
        body = {**body, "temperature": self.spec.temperature}
        data = await self._post(client, {**body, "n": n})
        contents = [c.get("message", {}).get("content", "") for c in data.get("choices", [])]
        while len(contents) < n:  # servers that ignore `n`
            more = await self._post(client, body)
            contents += [c.get("message", {}).get("content", "") for c in more.get("choices", [])]
        votes = Counter(i for i in (label_from_text(c, names) for c in contents[:n]) if i is not None)
        total = sum(votes.values())
        if total == 0:
            return {"probs": None, "mode": "sample", "answer": "name", "letter_mass": 0.0}
        return {
            "probs": [votes[i] / total for i in range(k)],
            "mode": "sample",
            "answer": "name",
            "letter_mass": total / n,
        }

    async def _alabel_decisions(self, client: httpx.AsyncClient, text: str) -> dict:
        """SGLang's Decisions schema, assumed to match OpenAI pending a public specification.

        A complete probability distribution is required; never invent training labels from
        a choice or an unrelated confidence/mass statistic.
        """
        body = {
            "model": self.spec.model or "default",
            "input": text,
            "questions": [
                {
                    "id": "label",
                    "type": "choice",
                    "question": self.task.question,
                    "options": [{"name": n, "description": d} for n, d in self.task.labels.items()],
                }
            ],
        }
        data = await self._post(client, body, "/decisions")
        answer = (data.get("answers") or {}).get("label") or {}
        probs = answer.get("probabilities")
        names = self.task.label_names
        if answer.get("type") != "choice" or answer.get("choice") not in names:
            raise RuntimeError("Invalid Decisions choice answer.")
        if not isinstance(probs, dict) or set(probs) != set(names):
            raise RuntimeError("Decisions probabilities must match the task labels exactly.")
        p = [probs[n] for n in names]
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in p
        ):
            raise RuntimeError("Invalid Decisions probabilities.")
        total = sum(p)
        if abs(total - 1) > 1e-4 or p[names.index(answer["choice"])] + 1e-8 < max(p):
            raise RuntimeError("Decisions probabilities must sum to one and agree with the choice.")
        mass = answer.get("label_mass", 1.0)
        if (
            isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or not 0 <= mass <= 1
        ):
            raise RuntimeError("Invalid Decisions label_mass.")
        return {"probs": [x / total for x in p], "mode": "openai_decisions", "letter_mass": mass}

    async def _alabel_systemone(self, client: httpx.AsyncClient, text: str) -> dict:
        """POST {base_url}/v1/systemone: the text is the state, the task one Choice question whose
        options are the labels, with their descriptions ("" for none: Fastino's GLiDE documents only
        strings there, and Decider shows the name alone for either). The answer's probabilities are
        the soft label (`letter_mass` is the share of them on the task's labels); an answer without
        them puts its confidence on its choice."""
        names = self.task.label_names
        question = {
            "type": "choice",
            "instructions": self.task.question,
            "criteria": {name: desc or "" for name, desc in self.task.labels.items()},
        }
        body = {"state": text, "questions": {"label": question}}
        if self.spec.model:
            body["model"] = self.spec.model
        data = await self._post(client, body, "/v1/systemone")
        answer = (data.get("answers") or {}).get("label") or {}
        probs = answer.get("probabilities")
        if isinstance(probs, dict):
            p = [float(probs.get(n) or 0.0) for n in names]
            total = sum(p)
            if total > 0:
                return {"probs": [x / total for x in p], "mode": "systemone", "letter_mass": round(total, 4)}
        elif answer.get("choice") in names and answer.get("confidence") is not None:
            conf = float(answer["confidence"])
            return {
                "probs": probs_from_answer(len(names), names.index(answer["choice"]), conf),
                "mode": "systemone",
                "letter_mass": round(conf, 4),
            }
        return {"probs": None, "mode": "systemone", "letter_mass": 0.0}

    async def alabel_many(
        self,
        texts: list[str],
        progress: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        """Labels in the order of `texts`. `progress` prints a line every 50 texts; `on_progress`
        is called with (done, total) after each one, failed or not."""
        sem = asyncio.Semaphore(self.spec.concurrency)
        done = 0
        async with self._client() as client:

            async def one(i: int, t: str) -> dict:
                nonlocal done
                async with sem:
                    try:
                        res = await self.alabel(client, t)
                    except Exception as e:  # keep going; failures are reported, not fatal
                        res = {"probs": None, "error": f"{type(e).__name__}: {e}"}
                    if on_result is not None:
                        on_result(i, res)
                    done += 1
                    if progress and done % 50 == 0:
                        print(f"  judged {done}/{len(texts)}", flush=True)
                    if on_progress is not None:
                        on_progress(done, len(texts))
                    return res

            return await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))  # in the order of texts

    def label_many(
        self,
        texts: list[str],
        progress: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        return asyncio.run(self.alabel_many(texts, progress=progress, on_progress=on_progress, on_result=on_result))

    def answered_as(self, text: str) -> str | None:
        """The model the judge answers as now (an alias such as `-latest` resolved), from labeling
        `text`; None when its endpoint doesn't say, or the call fails."""
        return self.label_many([text])[0].get("model")

    def classify(self, text: str) -> tuple[str, list[float] | None]:
        """Argmax label (used when an endpoint acts as the incumbent / fallback)."""
        res = self.label_many([text])[0]
        probs = res.get("probs")
        if not probs:
            return self.task.label_names[-1], None
        return self.task.label_names[max(range(len(probs)), key=probs.__getitem__)], probs


def label_log(
    judge: Judge,
    shadow_log: Path,
    labels_path: Path,
    limit: int | None = None,
    progress: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_label: Callable[[dict], None] | None = None,
) -> dict:
    """Label every logged input that isn't labeled yet. Resumable: results are appended.

    `on_progress(done, total)` counts the inputs judged in this call, as `Judge.alabel_many` does.
    Each successful result is flushed immediately and passed to `on_label`, when set, before
    reporting progress. An interrupted run keeps completed judgments.
    With failures, `errors` in the counts has their three most common reasons, and how often each
    came up: whether the judge was unreachable, or answered without naming a label.
    """
    done_ids = {r["id"] for r in read_log(labels_path)}
    todo = [r for r in read_log(shadow_log) if r["id"] not in done_ids]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return {"labeled": 0, "failed": 0, "total": len(done_ids)}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    errors: Counter = Counter()
    with open(labels_path, "a", encoding="utf-8") as f:

        def save(i: int, res: dict) -> None:
            nonlocal ok, failed
            rec = todo[i]
            if not res.get("probs"):
                failed += 1
                errors[" ".join(str(res.get("error") or "an answer naming no label").split())[:200]] += 1
                return
            labeled = {
                "id": rec["id"],
                "text": rec["text"],
                "probs": [round(p, 6) for p in res["probs"]],
                "judge": judge.spec.model,
                "judge_version": res.get("model"),
                "mode": res.get("mode"),
                "letter_mass": res.get("letter_mass"),
            }
            f.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            f.flush()
            if on_label is not None:
                on_label(labeled)
            ok += 1

        judge.label_many([r["text"] for r in todo], progress=progress, on_progress=on_progress, on_result=save)
    stats: dict = {"labeled": ok, "failed": failed, "total": len(done_ids) + ok}
    if errors:
        stats["errors"] = dict(errors.most_common(3))
    return stats


def label_texts(judge: Judge, texts: Iterable[str]) -> list[dict]:
    texts = list(texts)
    res = judge.label_many(texts, progress=False)
    return [{"id": text_id(t), "text": t, **r} for t, r in zip(texts, res, strict=True)]
