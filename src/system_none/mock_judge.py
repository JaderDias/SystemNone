"""`system-none mock-judge`: a judge that answers from labelled data instead of running a model.

`make` starts it by default, as the judge the worker trains students from and, without a TypeSafe
API key, as the gateway's System One model, so the whole system runs without downloading or
running a model. Local models such as Decision-2.0-Kai-0.6B and Decider-2B only run when asked for
(`make JUDGE=decider`).

It serves TypeSafe's `POST /v1/systemone`, as Decider does, so the gateway calls it as a `systemone`
System One model and the worker as a judge in mode `systemone`. For each choice question, it answers
with the label its data gives the text: probability `confidence` on that label, the rest spread
evenly over the other options (as `probs_from_answer` does for judges that name one answer).

Its data: the datasets of `system-none traffic` whose labels are known (by default banking77, both
splits, and the ai4privacy slice through the key in tests/data/ai4privacy, which only judges read),
and JSONL files of {"text": ..., "label": ...} rows, such as the ones `POST /admin/api/labels` takes.
A text it has no label for among the question's options gets a 422 that says so: it never guesses.
"""

import collections
import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .judge import _norm, probs_from_answer

MODEL = "mock-judge"
DEFAULT_DATASETS = ("banking77", "ai4privacy")
DEFAULT_CONFIDENCE = 0.9


def text_key(text: str) -> str:
    """sha256 of the stripped text, as the ai4privacy key has it."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class LabelledData:
    """Labels by text hash, counted when a text has several (a dataset with repeated texts, or two
    datasets sharing a text)."""

    def __init__(self) -> None:
        self.labels: dict[str, collections.Counter] = {}
        self.sources: dict[str, int] = {}

    def add(self, source: str, key: str, label: str) -> None:
        self.labels.setdefault(key, collections.Counter())[label] += 1
        self.sources[source] = self.sources.get(source, 0) + 1

    def add_texts(self, source: str, texts: Iterable[str], labels: Iterable[str]) -> None:
        for text, label in zip(texts, labels, strict=True):
            self.add(source, text_key(text), label)

    def __len__(self) -> int:
        return len(self.labels)

    def answer(self, text: str, options: list[str]) -> str | None:
        """The option the data names for `text`, the commonest when it names several; None when it
        names none of them. Case, `_` / `-` versus spaces and repeated whitespace don't tell a label
        from an option, unless that matches several options."""
        known = self.labels.get(text_key(text))
        if not known:
            return None
        votes: collections.Counter = collections.Counter()
        for label, n in known.items():
            same = [label] if label in options else [o for o in options if _norm(o) == _norm(label)]
            if len(same) == 1:
                votes[same[0]] += n
        return votes.most_common(1)[0][0] if votes else None


def load_labels_file(path: str | Path, data: LabelledData) -> int:
    """Rows of {"text": ..., "label": ...}; rows without either are skipped. Returns the rows added."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("text"), str) and row["text"].strip() and isinstance(row.get("label"), str):
                data.add(Path(path).name, text_key(row["text"]), row["label"])
                n += 1
    return n


def load_dataset(name: str, data: LabelledData) -> int:
    """A dataset of `system-none traffic --list` with labels. Returns the labels added."""
    from .traffic import TrafficError, ai4privacy_dir, load

    before = data.sources.get(name, 0)
    if name == "ai4privacy":  # its labels are in the key only judges read, by text hash
        d = ai4privacy_dir()
        if d is None:
            raise TrafficError("tests/data/ai4privacy not found: run this from a checkout of the repository")
        for line in (d / "judge_key.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                data.add(name, row["sha256"], row["label"])
    else:
        ds = load(name)
        if ds.gold is None:
            raise TrafficError(f"{name} has no public labels")
        data.add_texts(name, ds.texts, ds.gold)
    return data.sources.get(name, 0) - before


def load_labelled_data(
    datasets: Iterable[str] = DEFAULT_DATASETS, files: Iterable[str | Path] = (), log: Callable[[str], None] = print
) -> LabelledData:
    """The datasets and files' labels. A dataset that can't be loaded (no network, not a checkout) is
    left out with a warning; a file that can't be read is an error, and so is no data at all."""
    from .traffic import TrafficError

    data = LabelledData()
    for name in datasets:
        try:
            log(f"mock judge: {name}: {load_dataset(name, data):,} labels")
        except (TrafficError, OSError) as e:
            log(f"mock judge: {name} left out: {e}")
    for path in files:
        log(f"mock judge: {path}: {load_labels_file(path, data):,} labels")
    if not data.labels:
        raise SystemExit("mock judge: no labelled data loaded, so it could answer nothing")
    log(f"mock judge: answers {len(data):,} distinct texts")
    return data


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=64)
    model: str | None = None


def create_app(data: LabelledData, confidence: float = DEFAULT_CONFIDENCE, model: str = MODEL) -> FastAPI:
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    app = FastAPI(title="SystemNone mock judge")
    sources = ", ".join(data.sources)

    @app.get("/health")
    def health():
        return {"ok": True, "model": model, "texts": len(data), "sources": data.sources, "confidence": confidence}

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest):
        if not isinstance(request.state, str) or not request.state.strip():
            raise HTTPException(status_code=422, detail="the mock judge answers text states only")
        answers = {}
        for qid, question in request.questions.items():
            criteria = question.get("criteria")
            options = list(criteria) if isinstance(criteria, (dict, list)) else []
            if question.get("type") != "choice" or len(options) < 2 or not all(isinstance(o, str) for o in options):
                raise HTTPException(
                    status_code=422, detail=f"question '{qid}': the mock judge answers choice questions only"
                )
            label = data.answer(request.state, options)
            if label is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"question '{qid}': the mock judge has no label for this text among the question's "
                    f"options; it knows only the texts of its labelled data ({sources})",
                )
            probs = probs_from_answer(len(options), options.index(label), confidence)
            answers[qid] = {
                "type": "choice",
                "choice": label,
                "confidence": probs[options.index(label)],
                "probabilities": dict(zip(options, probs, strict=True)),
            }
        return {"model": model, "answers": answers}

    return app
