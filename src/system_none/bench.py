"""Measure a student: agreement with the judge, calibration, latency."""

from __future__ import annotations

import time

import numpy as np

from .calibrate import expected_calibration_error


def macro_f1(pred: np.ndarray, gold: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def latency(classifier, texts: list[str], n: int = 200, warmup: int = 10) -> dict:
    """Single-request latency (batch 1), the number that matters for online serving."""
    texts = [texts[i % len(texts)] for i in range(n + warmup)]
    for t in texts[:warmup]:
        classifier.predict_proba([t])
    ms = []
    for t in texts[warmup:]:
        t0 = time.perf_counter()
        classifier.predict_proba([t])
        ms.append((time.perf_counter() - t0) * 1000)
    ms = np.array(ms)
    t0 = time.perf_counter()
    classifier.predict_proba(texts[:n], batch_size=32)
    thr = n / (time.perf_counter() - t0)
    return {
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p99_ms": round(float(np.percentile(ms, 99)), 2),
        "throughput_per_s_batch32": round(thr, 1),
    }


def evaluate(classifier, texts: list[str], targets: np.ndarray, measure_latency: bool = True) -> dict:
    """`targets`: soft labels (n, k) or integer labels (n,)."""
    probs = classifier.predict_proba(texts)
    k = probs.shape[1]
    gold = targets.argmax(-1) if targets.ndim == 2 else targets.astype(int)
    pred = probs.argmax(-1)
    res = {
        "n": len(texts),
        "accuracy": round(float((pred == gold).mean()), 4),
        "macro_f1": round(macro_f1(pred, gold, k), 4),
        "ece": round(expected_calibration_error(probs, gold), 4),
    }
    if targets.ndim == 2:
        p = np.clip(targets, 1e-12, 1)
        res["kl_to_judge"] = round(float((targets * (np.log(p) - np.log(np.clip(probs, 1e-12, 1)))).sum(-1).mean()), 4)
    if measure_latency and texts:
        res.update(latency(classifier, texts))
    return res


def score_against_humans(classifier, examples, threshold: float | None = None) -> dict:
    """The student, the cascade and the judge against human labels, on examples held out of training.

    `examples` carry the human label as `probs` and the judge's soft label, if any, in
    `meta["judge_probs"]`. With a threshold, also how often the student answers alone, and how
    often it is right then.
    """
    gold = np.array([e.label for e in examples])
    probs = classifier.predict_proba([e.text for e in examples])
    right = probs.argmax(-1) == gold
    res = {
        "n": len(examples),
        "accuracy": round(float(right.mean()), 4),
        "macro_f1": round(macro_f1(probs.argmax(-1), gold, probs.shape[1]), 4),
    }
    if threshold is not None:
        answers = probs.max(-1) >= threshold
        res["student_answers"] = round(float(answers.mean()), 4)
        res["accuracy_when_student_answers"] = round(float(right[answers].mean()), 4) if answers.any() else None
    judged = [
        (int(np.argmax(e.meta["judge_probs"])), e.label) for e in examples if e.meta.get("judge_probs") is not None
    ]
    if judged:
        res["judge_accuracy"] = round(sum(a == b for a, b in judged) / len(judged), 4)
        res["judge_n"] = len(judged)
    return res
