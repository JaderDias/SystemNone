"""255 labels, the most a gateway prompt takes: judged by name, trained, exported and served."""

import json
import math
import random
import re
import time

import numpy as np
import pytest
from fastapi import Request  # module level: annotations are resolved from module globals

from system_none import worker
from system_none.runtime import Classifier
from system_none.shadow import read_log, text_id
from system_none.stores import MODEL_FILES, LocalStore

from conftest import serve

LABELS = [f"intent_{i:03d}" for i in range(255)]


@pytest.fixture(scope="module")
def name_judge():
    """An OpenAI-compatible judge that answers with the intent the text names, and its logprobs."""
    from fastapi import FastAPI

    app = FastAPI()
    seen = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        seen.append(body)
        quoted = re.search(r'"""\n(.*)\n"""', body["messages"][-1]["content"], re.S)
        assert quoted, body
        named = re.search(r"intent_\d{3}", quoted.group(1))
        assert named, body
        name = named.group(0)
        tokens = [
            {"token": "intent", "logprob": math.log(0.9)},
            {"token": "_", "logprob": 0.0},
            {"token": name[-3:], "logprob": 0.0},
        ]
        return {"choices": [{"message": {"content": name}, "logprobs": {"content": tokens}}]}

    with serve(app) as url:
        yield {"url": url, "seen": seen}


class Gateway:
    def __init__(self):
        self.reports = []

    def progress(self, job, message, done=None, total=None):
        self.reports.append((message, done, total))


def test_worker_trains_a_student_with_255_labels(tmp_path, tiny_model, name_judge, monkeypatch):
    store = LocalStore(tmp_path / "store")
    monkeypatch.setattr(worker, "open_store", lambda url: store)
    rng = random.Random(5)
    texts = [f"customer message {i} about intent_{rng.randrange(len(LABELS)):03d}" for i in range(300)]
    store.put(
        "inputs/p/2026-09-27/1-a.jsonl",
        "\n".join(json.dumps({"id": text_id(t), "text": t, "ts": 0}) for t in texts).encode(),
    )
    spec = {
        "job_id": "j",
        "prompt_id": "p",
        "base_model": tiny_model,
        "target_accuracy": 0.9,
        "prompt": {"instructions": "Which intent?", "labels": [{"name": label} for label in LABELS]},
        "judge": {"base_url": name_judge["url"], "model": "fake-judge", "mode": "logprobs", "license_ack": True},
        "train": {
            "epochs": 2,
            "min_steps": 0,
            "lr": 2e-3,
            "batch_size": 16,
            "max_length": 32,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "device": "cpu",
        },
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/j",
    }

    # Gateway stands in for GatewayClient, with the one method process() calls.
    gw, t0 = Gateway(), time.monotonic()
    result = worker.process(spec, tmp_path / "work", gw, log=lambda *a: None)  # ty: ignore[invalid-argument-type]
    seconds = time.monotonic() - t0

    assert result["ok"], result
    assert result["metrics"]["inputs"] == result["metrics"]["labeled"] == 300
    # Progress: each stage's message, then its counts, up to its last (every input judged, every
    # training step), at most one every PROGRESS_INTERVAL seconds in between: not one per input.
    stages = [message for message, done, _ in gw.reports if done is None]
    judging = stages[0]
    assert judging == "judging 300 inputs with fake-judge" and stages[1].startswith("training a student on ")
    # Then each candidate's own stages: its training steps are counted under the first.
    assert [s.split()[0] for s in stages[2:]] == ["training", "calibrating", "exporting", "validating"], stages
    training = stages[2]
    assert [r for r in gw.reports if r[0] == judging and r[1] is not None][-1] == (judging, 300, 300)
    steps = [r for r in gw.reports if r[0] == training and r[1] is not None]
    assert steps[-1][1] == steps[-1][2] > 0
    assert len(gw.reports) <= 8 + seconds / worker.PROGRESS_INTERVAL, (seconds, gw.reports)
    # The judge (answer: auto) was asked for a name, with room for the longest one.
    body = name_judge["seen"][-1]
    user = body["messages"][-1]["content"]
    assert "- intent_000\n" in user and "- intent_254\n" in user and "A) " not in user
    assert body["max_tokens"] >= len("intent_254") and body["logprobs"] is True
    # Soft labels: the answer's probability on the named label, the rest spread over the other 254.
    for rec in read_log(tmp_path / "work" / "j" / "labels.jsonl"):
        p = np.array(rec["probs"])
        assert len(p) == 255 and LABELS[int(p.argmax())] in rec["text"]
        assert p.max() == pytest.approx(0.9, abs=1e-4) and p.sum() == pytest.approx(1, abs=1e-4)
    # The published student has a 255-way head, and the runtime (like the gateway) reads it.
    model_dir = tmp_path / "published"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(store.get(f"models/p/j/{name}"))
    clf = Classifier(model_dir)
    assert clf.labels == LABELS
    probs = clf.predict_proba(texts[:3])
    assert probs.shape == (3, 255) and np.allclose(probs.sum(-1), 1, atol=1e-5)
    assert clf.predict(texts[0]).label in LABELS
