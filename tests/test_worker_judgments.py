import copy
import json

from system_none import judge, pipeline, worker
from system_none.shadow import read_log, text_id
from system_none.stores import LocalStore


def test_judgments_survive_failed_training_and_reuse_on_another_worker(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    texts = [f"input {i} 日本語 🐈" for i in range(24)]
    store.put("inputs/p/batch.jsonl", "\n".join(json.dumps({"id": text_id(t), "text": t}) for t in texts).encode())
    spec: dict = {
        "job_id": "first",
        "prompt_id": "p",
        "base_model": worker.FINE_TUNED_MODEL,
        "prompt": {"instructions": "Choose", "labels": [{"name": "yes"}, {"name": "no"}]},
        "judge": {"base_url": "http://judge/v1", "model": "judge", "license_ack": True},
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/first",
    }
    calls = []

    def label_many(self, inputs, **kwargs):
        calls.extend(inputs)
        results = [{"probs": [0.8, 0.2]} for _ in inputs]
        for i, result in enumerate(results):
            kwargs["on_result"](i, result)
        return results

    def fail_training(cfg, **kwargs):
        assert len(read_log(cfg.labels_path)) == 24
        raise SystemExit("training failed")

    class Gateway(worker.GatewayClient):
        def __init__(self):
            pass

        def progress(self, *args, **kwargs):
            return None

    monkeypatch.setattr(worker, "open_store", lambda url: store)
    monkeypatch.setattr(judge.Judge, "label_many", label_many)
    monkeypatch.setattr(judge.Judge, "answered_as", lambda self, text: None)  # the judge names no version
    monkeypatch.setattr(pipeline, "train_all", fail_training)
    assert not worker.process(spec, tmp_path / "worker1", Gateway())["ok"]
    assert len(calls) == 24
    assert store.list(worker.judgments_prefix(worker.build_config(spec, tmp_path / "cfg", spec["base_model"])))
    spec.update(job_id="second", base_model="jhu-clsp/ettin-encoder-150m")
    assert not worker.process(spec, tmp_path / "worker2", Gateway())["ok"]
    assert len(calls) == 24
    store.put("inputs/p/new.jsonl", json.dumps({"id": text_id("new"), "text": "new"}).encode())
    monkeypatch.setattr(pipeline, "train_all", lambda *args, **kwargs: {"groups": {}})
    worker.process({**spec, "job_id": "third"}, tmp_path / "worker3", Gateway())
    assert calls[-1] == "new" and len(calls) == 25


def test_a_new_judge_version_starts_a_new_cache_of_judgments(tmp_path, monkeypatch):
    """Issue #87: an alias such as `-latest` that resolves to another model is a new training target.
    Its labels are a cache of their own, the labels say which version gave them, and the student's
    metrics name it, for the gateway to test the student again when audits see another one."""
    store = LocalStore(tmp_path / "store")
    texts = [f"input {i}" for i in range(20)]
    store.put("inputs/p/batch.jsonl", "\n".join(json.dumps({"id": text_id(t), "text": t}) for t in texts).encode())
    spec: dict = {
        "job_id": "first",
        "prompt_id": "p",
        "base_model": worker.FINE_TUNED_MODEL,
        "prompt": {"instructions": "Choose", "labels": [{"name": "yes"}, {"name": "no"}]},
        "judge": {"base_url": "http://judge/v1", "model": "teacher-latest", "license_ack": True},
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/first",
    }
    version, calls, labeled = ["teacher-2026-09"], [], []

    def label_many(self, inputs, **kwargs):
        calls.extend(inputs)
        results = [{"probs": [0.8, 0.2], "model": version[0]} for _ in inputs]
        for i, result in enumerate(results):
            if kwargs.get("on_result"):
                kwargs["on_result"](i, result)
        return results

    def train_all(cfg, **kwargs):
        labeled[:] = read_log(cfg.labels_path)
        return {"groups": {}}

    class Gateway(worker.GatewayClient):
        def __init__(self):
            pass

        def progress(self, *args, **kwargs):
            return None

    monkeypatch.setattr(worker, "open_store", lambda url: store)
    monkeypatch.setattr(judge.Judge, "label_many", label_many)
    monkeypatch.setattr(pipeline, "train_all", train_all)
    lines = []
    worker.process(spec, tmp_path / "w1", Gateway(), log=lines.append)
    assert len(calls) == 21  # one call to learn the version, then the 20 inputs
    assert {r["judge_version"] for r in labeled} == {"teacher-2026-09"}
    assert "job first: the judge teacher-latest answers as teacher-2026-09" in lines
    cfg = worker.build_config(spec, tmp_path / "cfg", spec["base_model"])
    assert len(store.list(worker.judgments_prefix(cfg, "teacher-2026-09"))) == 20
    # Same version: its labels are reused.
    worker.process({**spec, "job_id": "second"}, tmp_path / "w2", Gateway(), log=lines.append)
    assert len(calls) == 22
    # The alias moves: every input is judged again, by the new version.
    version[0] = "teacher-2026-10"
    worker.process({**spec, "job_id": "third"}, tmp_path / "w3", Gateway(), log=lines.append)
    assert len(calls) == 43
    assert {r["judge_version"] for r in labeled} == {"teacher-2026-10"}
    # A version that is the configured model name keeps the cache from before versions were known.
    assert worker.judgments_prefix(cfg, "teacher-latest") == worker.judgments_prefix(cfg)
    assert worker.judgments_prefix(cfg, None) == worker.judgments_prefix(cfg)


def test_judgment_context_changes_with_judge_and_prompt_but_not_student(tmp_path):
    spec: dict = {
        "prompt_id": "p",
        "prompt": {"instructions": "Choose", "labels": [{"name": "a"}, {"name": "b"}]},
        "judge": {"model": "judge", "base_url": "http://judge/v1", "license_ack": True},
        "base_model": "small",
    }

    def prefix(value):
        return worker.judgments_prefix(worker.build_config(value, tmp_path / "cfg", value["base_model"]))

    original = prefix(spec)
    spec["base_model"] = "large"
    assert prefix(spec) == original
    for key, value in (("model", "other"), ("mode", "sample"), ("samples", 3), ("base_url", "http://other")):
        changed = copy.deepcopy(spec)
        changed["judge"][key] = value
        assert prefix(changed) != original
    spec["prompt"]["labels"].reverse()
    assert prefix(spec) != original
