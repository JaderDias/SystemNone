"""Completed judgments survive interruption, failed training and a retry on another worker."""

import asyncio
import json

import pytest

from system_none import judge, pipeline, worker
from system_none.config import JudgeSpec, LicenseAck, TaskSpec
from system_none.shadow import ShadowLogger, read_log, text_id
from system_none.stores import LocalStore

from conftest import LABELS, synthetic_texts

TASK = TaskSpec(name="intent", question="What does the customer want?", labels=dict.fromkeys(LABELS, ""))


def test_judgments_are_saved_before_progress(tmp_path, mock_transport):
    shadow, labels = tmp_path / "shadow.jsonl", tmp_path / "labels.jsonl"
    for text in synthetic_texts(10):
        ShadowLogger(shadow).log(text)
    saved = []

    def interrupt(done, total):
        assert read_log(labels) and saved
        raise RuntimeError("worker interrupted")

    spec = JudgeSpec(base_url="http://fake/v1", model="fake-judge", license=LicenseAck(allows_training_on_outputs=True))
    j = judge.Judge(spec, TASK, transport=mock_transport)
    with pytest.raises(RuntimeError, match="worker interrupted"):
        judge.label_log(j, shadow, labels, on_label=saved.append, on_progress=interrupt)
    assert len(read_log(labels)) == len(saved)
    stats = judge.label_log(j, shadow, labels, progress=False)
    assert stats["total"] == 10 and stats["labeled"] < 10


@pytest.mark.parametrize("interrupt_judging", [False, True])
def test_retry_reuses_store_judgments(tmp_path, monkeypatch, mock_transport, interrupt_judging):
    store = LocalStore(tmp_path / "store")
    texts = synthetic_texts(30)
    store.put("inputs/p/a.jsonl", "\n".join(json.dumps({"id": text_id(t), "text": t}) for t in texts).encode())
    spec = {
        "job_id": "first",
        "prompt_id": "p",
        "base_model": "some/model",
        "prompt": {"instructions": TASK.question, "labels": [{"name": n} for n in LABELS]},
        "judge": {"base_url": "http://judge/v1", "model": "q", "license_ack": True},
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/first",
    }
    calls = []
    original = judge.Judge

    class CountingJudge(original):
        def answered_as(self, text):  # the judge names no version: no call to learn it
            return None

        async def alabel(self, client, text):
            await asyncio.sleep(0.01)  # leave unfinished requests when the worker is interrupted
            calls.append(text)
            return await super().alabel(client, text)

    monkeypatch.setattr(worker, "open_store", lambda url: store)
    monkeypatch.setattr(judge, "Judge", lambda spec, task: CountingJudge(spec, task, transport=mock_transport))
    progress = worker.JobProgress
    monkeypatch.setattr(worker, "JobProgress", lambda gw, job, log: progress(gw, job, log=log, interval=0))

    def fail_training(cfg, **kwargs):
        assert len(read_log(cfg.labels_path)) == 30
        raise RuntimeError("export failed")

    monkeypatch.setattr(pipeline, "train_all", fail_training)

    class Gateway:
        def progress(self, job, message, done=None, total=None):
            if interrupt_judging and job == "first" and done is not None:
                raise RuntimeError("judging interrupted")
            return None

    for job, root in [("first", "worker-one"), ("retry", "worker-two")]:
        spec["job_id"] = job
        error = "judging interrupted" if interrupt_judging and job == "first" else "export failed"
        with pytest.raises(RuntimeError, match=error):
            worker.process(spec, tmp_path / root, Gateway(), log=lambda s: None)  # ty: ignore[invalid-argument-type]
        if interrupt_judging and job == "first":
            assert 0 < len(store.list("judgments/")) < 30
        spec["base_model"] = "another/student"  # student settings don't invalidate judgments
    assert len(calls) == 30
    assert len(store.list("judgments/")) == 30
    # Changing the judge or the ordered labels cannot reuse incompatible probability vectors.
    cfg = worker.build_config(spec, tmp_path, "some/model")
    before = worker.judgments_prefix(cfg)
    cfg.judge.model = "another-judge"
    assert worker.judgments_prefix(cfg) != before
    cfg.judge.model = "q"
    cfg.task.labels = dict(reversed(list(cfg.task.labels.items())))
    assert worker.judgments_prefix(cfg) != before
