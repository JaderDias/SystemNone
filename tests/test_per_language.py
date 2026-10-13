"""Students per language (issue #96): the gateway's language router tags each logged input, and a
prompt that asks for it gets, besides its student for every language, a small student for each
language the router is sure of in enough inputs."""

import json

import numpy as np

from system_none import pipeline, worker
from system_none.config import config_from_dict
from system_none.data import Example
from system_none.shadow import read_log, text_id
from system_none.stores import MODEL_FILES, LocalStore


def config(tmp_path, per_language=("jhu-clsp/ettin-encoder-17m",), min_examples=3):
    return config_from_dict(
        {
            "workdir": str(tmp_path / "work"),
            "task": {"name": "t", "question": "Which?", "labels": ["a", "b"]},
            "students": {"english": ["base"], "multilingual": [], "per_language": list(per_language)},
            "train": {"per_language_min_examples": min_examples},
        }
    )


def examples(routes):
    return [Example(id=str(i), text=f"text {i}", probs=np.array([1.0, 0.0]), route=r) for i, r in enumerate(routes)]


def test_each_language_with_enough_routed_examples_gets_a_group(tmp_path):
    exs = examples(["por"] * 3 + ["hin"] * 2 + [None] * 4)
    groups = pipeline.groups_for(config(tmp_path), exs)
    assert list(groups) == ["english", "lang-por"]
    assert len(groups["english"][1]) == 9, "the prompt's student trains on every example"
    assert groups["lang-por"] == (["jhu-clsp/ettin-encoder-17m"], exs[:3])
    # Without students per language, only the prompt's student.
    assert list(pipeline.groups_for(config(tmp_path, per_language=()), exs)) == ["english"]


def test_examples_take_their_route_from_routes_json(tmp_path):
    cfg = config(tmp_path)
    cfg.workdir.mkdir(parents=True)
    with open(cfg.labels_path, "w", encoding="utf-8") as f:
        for t in ("bom dia", "good morning"):
            f.write(json.dumps({"id": text_id(t), "text": t, "probs": [0.9, 0.1]}) + "\n")
    (cfg.workdir / pipeline.ROUTES_FILE).write_text(json.dumps({text_id("bom dia"): "por"}))
    exs, _ = pipeline.load_examples(cfg, log=lambda *a: None)
    assert {e.text: e.route for e in exs} == {"bom dia": "por", "good morning": None}


def test_inputs_keep_the_routers_language_and_routes_need_its_confidence(tmp_path):
    store = LocalStore(tmp_path / "store")
    rows = [
        {"id": "1", "text": "bom dia", "ts": 1, "lang": "por", "lang_confidence": 0.95},
        {"id": "2", "text": "boa noite", "ts": 2, "lang": "por", "lang_confidence": 0.5},
        {"id": "3", "text": "12345", "ts": 3},
    ]
    store.put("inputs/p/a.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
    cfg = config(tmp_path)
    assert worker.collect_inputs(store, "inputs/p/", cfg.shadow_log) == 3
    assert read_log(cfg.shadow_log)[0] == rows[0]
    assert worker.write_routes(cfg, 0.7) == {"por": 1}
    assert json.loads((cfg.workdir / pipeline.ROUTES_FILE).read_text()) == {"1": "por"}


def test_the_worker_publishes_a_student_per_language(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    rows = [
        {"id": text_id(f"bom dia {i}"), "text": f"bom dia {i}", "lang": "por", "lang_confidence": 0.9}
        for i in range(25)
    ] + [{"id": text_id(f"hello {i}"), "text": f"hello {i}", "lang": "eng", "lang_confidence": 0.4} for i in range(5)]
    store.put("inputs/p/a.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
    spec = {
        "job_id": "j",
        "prompt_id": "p",
        "base_model": "base",
        "prompt": {"instructions": "Which?", "labels": [{"name": "a"}, {"name": "b"}]},
        "judge": {"base_url": "http://judge/v1", "model": "judge", "license_ack": True},
        "per_language": {"enabled": True, "min_confidence": 0.7, "min_examples": 20, "base_model": "tiny"},
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/j",
    }
    seen = {}

    def label_log(judge, shadow_log, labels_path, **kw):
        with open(labels_path, "w", encoding="utf-8") as f:
            for r in read_log(shadow_log):
                f.write(json.dumps({"id": r["id"], "text": r["text"], "probs": [0.9, 0.1]}) + "\n")
        return {"labeled": 30, "failed": 0, "total": 30}

    def train_all(cfg, **kw):
        exs, _ = pipeline.load_examples(cfg, log=lambda *a: None)
        seen["groups"] = {g: (models, len(e)) for g, (models, e) in pipeline.groups_for(cfg, exs).items()}
        deploy = {"groups": {}}
        for group in seen["groups"]:
            d = cfg.workdir / "models" / group
            d.mkdir(parents=True)
            for name in MODEL_FILES:
                (d / name).write_text(f"{group} {name}")
            metrics = {"accuracy": 0.9, "n": 5}
            deploy["groups"][group] = {
                "model": seen["groups"][group][0][0],
                "model_dir": f"models/{group}",
                "threshold": 0.6 if group == "english" else 0.7,
                "test": metrics,
                "cascade": {"coverage": 0.8},
            }
        return deploy

    class Gateway:
        def progress(self, *a, **kw):
            return None

    monkeypatch.setattr(worker, "open_store", lambda url: store)
    monkeypatch.setattr("system_none.judge.Judge.answered_as", lambda self, text: None)
    monkeypatch.setattr("system_none.judge.label_log", label_log)
    monkeypatch.setattr(pipeline, "train_all", train_all)
    lines = []
    result = worker.process(spec, tmp_path / "w", Gateway(), log=lines.append)  # ty: ignore[invalid-argument-type]
    assert result["ok"], result
    # The English inputs the router wasn't sure of train only the prompt's student.
    assert seen["groups"] == {"english": (["base"], 30), "lang-por": (["tiny"], 25)}
    assert any(
        "router is sure of 25 inputs' language; a student each for por (25), with at least 20" in line for line in lines
    )
    assert store.get("models/p/j/lang/por/model.onnx") == b"lang-por model.onnx"
    assert store.get("models/p/j/model.onnx") == b"english model.onnx"
    manifest = json.loads(store.get("models/p/j/languages.json"))
    assert manifest == {
        "min_confidence": 0.7,
        "languages": {"por": {"threshold": 0.7, "model": "tiny", "examples": 25}},
    }
    assert result["metrics"]["languages"]["por"]["test"] == {"accuracy": 0.9, "n": 5}
    assert result["message"].endswith("; students for por (0.900)")
