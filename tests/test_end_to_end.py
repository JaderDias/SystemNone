"""Full flow through the CLI: log -> judge -> train (fine-tune, calibrate, export, pick) -> predict -> cascade -> monitor."""

import json

import numpy as np
import pytest

from system_none.cli import main
from system_none.config import load_config
from system_none.runtime import Cascade, Classifier, LanguageRouter, ReviewSampler, load_deployment
from system_none.shadow import read_log

from conftest import LABELS, synthetic_texts, write_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_model, judge_server):
    d = tmp_path_factory.mktemp("proj")
    cfg_path = write_config(d / "system-none.yaml", d / "run", tiny_model, judge_url=judge_server["url"])
    (d / "traffic.txt").write_text("\n".join(synthetic_texts(600, seed=7)))
    main(["log", "-c", str(cfg_path), str(d / "traffic.txt")])
    main(["judge", "-c", str(cfg_path)])
    main(["train", "-c", str(cfg_path)])
    return load_config(cfg_path), cfg_path


def test_training_outputs(trained):
    cfg, _ = trained
    deploy = json.loads((cfg.workdir / "deploy.json").read_text())
    g = deploy["groups"]["english"]
    assert g["test"]["accuracy"] >= 0.85, g  # ~15% of texts are genuine ties, so the ceiling is ~0.92
    assert g["test"]["ece"] < 0.1
    assert g["test"]["p50_ms"] > 0 and g["test"]["p99_ms"] >= g["test"]["p50_ms"]
    assert 0 < g["threshold"] <= 1 and g["cascade"]["accuracy"] >= 0.9
    board = json.loads((cfg.workdir / "leaderboard.json").read_text())
    assert board["english"]["candidates"][0]["temperature"] > 0
    assert (cfg.workdir / g["model_dir"] / "model.onnx").exists()


def test_onnx_matches_pytorch(trained):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg, _ = trained
    g = json.loads((cfg.workdir / "deploy.json").read_text())["groups"]["english"]
    onnx_dir = cfg.workdir / g["model_dir"]
    hf_dir = onnx_dir.parent / "hf"
    texts = synthetic_texts(20, seed=123)
    clf = Classifier(onnx_dir)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForSequenceClassification.from_pretrained(hf_dir).eval()
    # AutoTokenizer.from_pretrained is typed as possibly None (see system_none/train.py).
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")  # ty: ignore[call-non-callable]
    with torch.no_grad():
        ref = model(**enc).logits.numpy()
    assert np.abs(clf.logits(texts) - ref).max() < 1e-2


def test_cascade_routes_uncertain_to_fallback(trained):
    cfg, _ = trained
    calls = []

    def fallback(text):
        calls.append(text)
        return "other"

    review = ReviewSampler(cfg.workdir / "q.jsonl", sample_rate=1.0)
    pipe = load_deployment(cfg.workdir, fallback=fallback, review=review)
    confident = pipe("please refund refund money back charged twice")
    assert confident.source == "student" and confident.label == "refund"
    assert isinstance(pipe, Cascade)  # one group, so no LanguageRouter
    strict = Cascade(pipe.student, threshold=1.01, fallback=fallback)  # nothing clears the bar
    assert strict("where is my package").source == "fallback" and calls
    assert read_log(cfg.workdir / "q.jsonl")  # student decisions were sampled for review


def test_language_router():
    class Fake:
        def __init__(self, name):
            self.name = name

        def __call__(self, text):
            return self.name

    router = LanguageRouter({"en": Fake("english")}, default=Fake("multilingual"))
    assert router("Where is my package? It was supposed to arrive yesterday.") == "english"
    assert router("¿Dónde está mi paquete? Debía llegar ayer por la mañana.") == "multilingual"


def test_predict_and_bench_cli(trained, capsys):
    _, cfg_path = trained
    main(["predict", "-c", str(cfg_path), "I am locked out, my password stopped working"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["label"] == "login" and out["source"] == "student"
    main(["bench", "-c", str(cfg_path)])
    res = json.loads(capsys.readouterr().out)
    assert res["accuracy"] >= 0.85 and "p99_ms" in res


def test_monitor_detects_drift(trained, judge_server):
    cfg, cfg_path = trained
    pipe = load_deployment(cfg.workdir, review=ReviewSampler(cfg.review_queue, sample_rate=1.0))
    for t in synthetic_texts(60, seed=5):
        pipe(t)
    with pytest.raises(SystemExit) as ok:
        main(["monitor", "-c", str(cfg_path)])
    assert ok.value.code == 0  # the judge still agrees

    judge_server["state"]["flip"] = True  # the world shifts
    try:
        for t in synthetic_texts(300, seed=6):
            pipe(t)
        with pytest.raises(SystemExit) as drift:
            main(["monitor", "-c", str(cfg_path)])
        assert drift.value.code == 2
    finally:
        judge_server["state"]["flip"] = False
    n_labels = len(read_log(cfg.labels_path))
    assert n_labels > 600  # judged live traffic is folded into the training set


def test_judge_refuses_without_license(tmp_path, tiny_model):
    cfg_path = write_config(tmp_path / "c.yaml", tmp_path / "run", tiny_model, ack=False)
    from system_none.judge import LicenseError

    with pytest.raises(LicenseError):
        main(["judge", "-c", str(cfg_path)])


def test_http_server(trained):
    from fastapi.testclient import TestClient

    from system_none.server import create_app

    cfg, _ = trained
    client = TestClient(create_app(cfg))
    assert client.get("/health").json()["labels"] == LABELS
    r = client.post("/classify", json={"text": "where is my package, no tracking number yet"}).json()
    assert r["label"] == "shipping" and r["source"] == "student"
    logged = [x["text"] for x in read_log(cfg.shadow_log)]
    assert "where is my package, no tracking number yet" in logged  # input logged, answer not
