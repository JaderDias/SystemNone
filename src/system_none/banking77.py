"""Benchmark: banking77 (PolyAI, CC BY 4.0), bank customer messages sorted into 77 intents.

`system-none banking77` trains a student on the human labels of the official train split with the
usual pipeline (split, fine-tune, calibrate, export to ONNX, pick the serving threshold), then
scores the ONNX student on the official test split. The human labels stand in for the judge:
with them the student learns the task itself, not a judge's view of it, so scores compare
directly with published ones.
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
QUESTION = "Which banking intent does this customer message express?"


def load(split: str) -> list[tuple[str, str]]:
    """(text, intent) rows of the official "train" or "test" split."""
    import httpx

    r = httpx.get(URL.format(split=split), timeout=60, follow_redirects=True)
    r.raise_for_status()
    return [(row["text"], row["category"]) for row in csv.DictReader(io.StringIO(r.text))]


def run(model: str, workdir: str | Path, train: dict | None = None, log=print) -> dict:
    """Train on the train split, test on the test split; writes and returns `<workdir>/banking77.json`."""
    from .bench import evaluate
    from .config import config_from_dict
    from .pipeline import train_all
    from .runtime import Classifier
    from .shadow import text_id

    workdir = Path(workdir)
    rows, test = load("train"), load("test")
    labels = sorted({intent for _, intent in rows})
    onehot = np.eye(len(labels))
    cfg = config_from_dict(
        {
            "project": "banking77",
            "workdir": str(workdir),
            "task": {"name": "banking77", "question": QUESTION, "labels": labels},
            "students": {"english": [model], "multilingual": []},
            # The official test split is the test set, so only a sliver of train is held out as one.
            "train": {"val_fraction": 0.1, "test_fraction": 0.02, "max_length": 128, **(train or {})},
        }
    )
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    with open(cfg.labels_path, "w", encoding="utf-8") as f:
        for text, intent in rows:
            f.write(
                json.dumps(
                    {"id": text_id(text), "text": text, "probs": onehot[labels.index(intent)].tolist(), "lang": "en"}
                )
                + "\n"
            )
    t0 = time.time()
    deploy = train_all(cfg, keep_hf=False, log=log)
    seconds = time.time() - t0
    group = deploy["groups"]["english"]
    candidate = json.loads((cfg.workdir / "leaderboard.json").read_text())["english"]["candidates"][0]

    clf = Classifier(cfg.workdir / group["model_dir"])
    texts = [text for text, _ in test]
    gold = np.array([labels.index(intent) for _, intent in test])
    metrics = evaluate(clf, texts, gold)
    probs = clf.predict_proba(texts)
    right = probs.argmax(-1) == gold
    answers = probs.max(-1) >= group["threshold"]
    result = {
        "model": model,
        "train_rows": len(rows),
        "test": metrics,
        "fine_tune_seconds": candidate["train_seconds"],
        "pipeline_seconds": round(seconds, 1),
        "train": {
            k: getattr(cfg.train, k) for k in ("epochs", "min_steps", "lr", "batch_size", "max_length", "device")
        },
        "cascade": {
            "threshold": round(group["threshold"], 4),
            "student_answers": round(float(answers.mean()), 4),
            "accuracy_when_student_answers": round(float(right[answers].mean()), 4) if answers.any() else None,
        },
        "model_dir": str(cfg.workdir / group["model_dir"]),
    }
    (cfg.workdir / "banking77.json").write_text(json.dumps(result, indent=1))
    return result
