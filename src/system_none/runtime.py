"""Serving: an ONNX student behind a confidence cascade, routed by language.

Needs only onnxruntime, tokenizers and numpy (no PyTorch).
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from .calibrate import softmax
from .data import detect_language
from .shadow import text_id


@dataclass
class Decision:
    label: str
    confidence: float
    source: str  # "student" or "fallback"
    probs: dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Classifier:
    """A calibrated student exported by `system-none train`."""

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.labels: list[str] = self.manifest["labels"]
        self.temperature: float = float(self.manifest.get("temperature", 1.0))
        self.inputs: list[str] = self.manifest["inputs"]
        self.pad_id = int(self.manifest.get("pad_token_id", 0))
        self.prefix: str = self.manifest.get("prefix", "")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=int(self.manifest.get("max_length", 256)))
        self.name = self.manifest.get("source_model", self.path.name)

    def logits(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch([self.prefix + t for t in texts])
        n = max(len(e.ids) for e in encs)
        feed = {k: np.zeros((len(encs), n), dtype=np.int64) for k in self.inputs}
        if "input_ids" in feed:
            feed["input_ids"][:] = self.pad_id
        for i, e in enumerate(encs):
            L = len(e.ids)
            feed["input_ids"][i, :L] = e.ids
            feed["attention_mask"][i, :L] = e.attention_mask
            if "token_type_ids" in feed:
                feed["token_type_ids"][i, :L] = e.type_ids
        # run() is typed for any ONNX output (sparse tensors, sequences, maps); a student's logits are dense.
        return cast(np.ndarray, self.session.run(None, feed)[0])

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(softmax(self.logits(texts[i : i + batch_size]), self.temperature))
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, text: str) -> Decision:
        t0 = time.perf_counter()
        p = self.predict_proba([text])[0]
        k = int(p.argmax())
        return Decision(
            label=self.labels[k],
            confidence=float(p[k]),
            source="student",
            probs={label: round(float(v), 4) for label, v in zip(self.labels, p, strict=True)},
            model=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def choose_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_accuracy: float,
    min_served: int = 20,
    *,
    method: str = "empirical",
    confidence_level: float = 0.95,
    candidate_models: int = 1,
) -> dict:
    """Choose maximum coverage meeting the empirical or exact lower-bound agreement target.

    Evaluate complete confidence ties, since inference serves every prediction >= threshold.
    Exact mode uses a one-sided Clopper-Pearson bound, with a Bonferroni adjustment over
    at most n thresholds per candidate model. This relies on IID held-out observations.
    """
    if method not in {"empirical", "clopper_pearson"}:
        raise ValueError("Unknown calibration method")
    if not 0.0 < confidence_level < 1.0 or candidate_models < 1 or min_served < 1:
        raise ValueError("Invalid calibration confidence or counts")
    if not 0.5 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0.5 and 1")
    metadata = {"target_accuracy": target_accuracy, "method": method, "confidence_level": confidence_level}
    empty = {"threshold": 1.01, "coverage": 0.0, "accuracy": None, **metadata}
    if len(probs) == 0:
        return empty
    conf = probs.max(-1)
    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    correct = (probs.argmax(-1) == labels)[order]
    # Last index of each tie is the actual serving set for that threshold.
    ends = np.flatnonzero(np.r_[sorted_conf[:-1] != sorted_conf[1:], True])
    counts = ends + 1
    successes = np.cumsum(correct)[ends]
    accuracy = successes / counts
    evidence = accuracy
    if method == "clopper_pearson":
        from scipy.special import betaincinv

        alpha = (1.0 - confidence_level) / (len(probs) * candidate_models)
        evidence = np.zeros(len(counts))
        positive = successes > 0
        evidence[positive] = betaincinv(successes[positive], counts[positive] - successes[positive] + 1, alpha)
        metadata.update(bound="one_sided_lower", correction="bonferroni", candidate_models=candidate_models)
    ok = np.flatnonzero((evidence >= target_accuracy) & (counts >= min_served))
    if len(ok) == 0:
        return {**empty, **metadata}
    j = ok[-1]
    return {
        "threshold": float(sorted_conf[ends[j]]),
        "coverage": round(float(counts[j] / len(conf)), 4),
        "accuracy": round(float(accuracy[j]), 4),
        **metadata,
        **({"agreement_lower_bound": float(evidence[j])} if method == "clopper_pearson" else {}),
    }


class ReviewSampler:
    """Send a small sample of live decisions to a queue that the judge scores later (drift monitoring)."""

    def __init__(self, path: str | Path, sample_rate: float = 0.01, seed: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def maybe_add(self, text: str, decision: Decision) -> None:
        if decision.source != "student" or self._rng.random() >= self.sample_rate:
            return
        rec = {
            "id": text_id(text),
            "text": text,
            "label": decision.label,
            "confidence": round(decision.confidence, 4),
            "model": decision.model,
            "ts": time.time(),
        }
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Cascade:
    """Answer locally when the student is confident enough; otherwise call today's model."""

    def __init__(
        self,
        student: Classifier,
        threshold: float,
        fallback: Callable[[str], str] | None = None,
        review: ReviewSampler | None = None,
    ):
        self.student = student
        self.threshold = threshold
        self.fallback = fallback
        self.review = review

    def __call__(self, text: str) -> Decision:
        d = self.student.predict(text)
        if d.confidence >= self.threshold or self.fallback is None:
            if self.review:
                self.review.maybe_add(text, d)
            return d
        t0 = time.perf_counter()
        label = self.fallback(text)
        return Decision(
            label=label,
            confidence=float("nan"),
            source="fallback",
            model="fallback",
            latency_ms=d.latency_ms + (time.perf_counter() - t0) * 1000,
        )


class LanguageRouter:
    """Route by detected language: `routes` maps a language code to a model, `default` takes the rest."""

    def __init__(self, routes: dict[str, Callable[[str], Decision]], default: Callable[[str], Decision]):
        self.routes = routes
        self.default = default

    def __call__(self, text: str) -> Decision:
        return self.routes.get(detect_language(text), self.default)(text)


def load_deployment(
    workdir: str | Path,
    fallback: Callable[[str], str] | None = None,
    review: ReviewSampler | None = None,
    threads: int | None = None,
) -> Callable[[str], Decision]:
    """Build the serving pipeline from `<workdir>/deploy.json` written by `system-none train`."""
    workdir = Path(workdir)
    deploy = json.loads((workdir / "deploy.json").read_text())
    cascades = {}
    for group, info in deploy["groups"].items():
        clf = Classifier(workdir / info["model_dir"], threads=threads)
        cascades[group] = Cascade(clf, info["threshold"], fallback=fallback, review=review)
    if "english" in cascades and "multilingual" in cascades:
        routes = deploy.get("routes", {"en": "english"})  # set by `system-none train` from held-out English scores
        return LanguageRouter({lang: cascades[g] for lang, g in routes.items()}, default=cascades["multilingual"])
    return next(iter(cascades.values()))
