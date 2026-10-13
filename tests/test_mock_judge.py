"""The mock judge: answers from labelled data over TypeSafe's POST /v1/systemone, as the gateway's
System One model and the worker's judge call it, and never guesses."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from system_none import cli
from system_none.config import JudgeSpec, LicenseAck, TaskSpec
from system_none.judge import Judge
from system_none.mock_judge import LabelledData, create_app, load_dataset, load_labelled_data, text_key
from system_none.traffic import Dataset, TrafficError

from hardcoded_judge import KEY, PII_LABELS, PII_QUESTION, HardcodedJudge, load_inputs

TASK = TaskSpec("intent", "What does the customer want?", {"refund": "wants money back", "cancel": "", "other": ""})


def data(*rows):
    d = LabelledData()
    for text, label in rows:
        d.add("rows", text_key(text), label)
    return d


def ask(client, text, labels=("refund", "cancel", "other"), **question):
    body = {
        "state": text,
        "questions": {
            "label": {"type": "choice", "instructions": "?", "criteria": dict.fromkeys(labels, ""), **question}
        },
    }
    return client.post("/v1/systemone", json=body)


def test_answers_the_label_with_its_confidence():
    client = TestClient(create_app(data(("refund my order", "refund")), confidence=0.8))
    r = ask(client, "  refund my order\n")  # the text is stripped, as the ai4privacy key has it
    assert r.status_code == 200
    answer = r.json()["answers"]["label"]
    assert answer["choice"] == "refund"
    assert answer["confidence"] == pytest.approx(0.8)
    assert answer["probabilities"] == pytest.approx({"refund": 0.8, "cancel": 0.1, "other": 0.1})
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["texts"] == 1


def test_never_guesses():
    client = TestClient(create_app(data(("refund my order", "refund"))))
    unknown = ask(client, "something else")
    assert unknown.status_code == 422
    assert "no label for this text" in unknown.json()["detail"]
    # A known text whose label isn't one of the options is unknown too.
    assert ask(client, "refund my order", labels=("yes", "no")).status_code == 422
    assert ask(client, "refund my order", type="score").status_code == 422


def test_label_matching_and_conflicts():
    d = data(("a", "Card Arrival"), ("b", "x"), ("b", "y"), ("b", "y"))
    assert d.answer("a", ["card_arrival", "other"]) == "card_arrival"  # case and _ versus spaces
    assert d.answer("a", ["card arrival", "Card_Arrival"]) is None  # ambiguous: matches both
    assert d.answer("b", ["x", "y"]) == "y"  # the commonest
    assert d.answer("b", ["x", "z"]) == "x"  # the commonest among the options


def test_the_worker_judge_trains_on_its_answers():
    """Judge in mode systemone, as the worker builds it from the gateway's job, gets soft labels."""
    client = TestClient(create_app(data(("refund my order", "refund"), ("cancel it", "cancel"))))

    def forward(request):
        response = client.post(request.url.path, content=request.content, headers={"Content-Type": "application/json"})
        return httpx.Response(response.status_code, json=response.json())

    spec = JudgeSpec(
        base_url="http://mock-judge:8001",
        model="mock-judge",
        mode="systemone",
        license=LicenseAck(allows_training_on_outputs=True),
    )
    results = Judge(spec, TASK, transport=httpx.MockTransport(forward)).label_many(["cancel it", "unknown"])
    assert results[0]["probs"] == pytest.approx([0.05, 0.9, 0.05])
    assert results[1]["probs"] is None and "422" in results[1]["error"]


def test_ai4privacy_answers_match_the_hardcoded_judge():
    d = LabelledData()
    assert load_dataset("ai4privacy", d) == len(KEY.read_text(encoding="utf-8").splitlines())
    client = TestClient(create_app(d))
    hardcoded = HardcodedJudge()
    for row in load_inputs()[:50]:
        r = ask(client, row["text"], labels=tuple(PII_LABELS), instructions=PII_QUESTION)
        probs = hardcoded.probs(row["text"])
        assert r.json()["answers"]["label"]["choice"] == list(PII_LABELS)[probs.index(max(probs))]


def test_loads_datasets_and_files(tmp_path, monkeypatch):
    rows = tmp_path / "labels.jsonl"
    rows.write_text(
        json.dumps({"text": "hi", "label": "other"}) + "\n\n" + json.dumps({"text": "no label"}) + "\n",
        encoding="utf-8",
    )

    def fake_load(name):
        if name == "offline":
            raise TrafficError("could not download offline")
        return Dataset(name, "?", ["refund", "other"], ["refund me", "hello"], ["refund", "other"])

    monkeypatch.setattr("system_none.traffic.load", fake_load)
    logged = []
    d = load_labelled_data(["toy", "offline"], [rows], log=logged.append)
    assert d.sources == {"toy": 2, "labels.jsonl": 1}
    assert d.answer("refund me", ["refund", "other"]) == "refund"
    assert d.answer("hi", ["refund", "other"]) == "other"
    assert any("offline left out" in line for line in logged)
    with pytest.raises(SystemExit, match="no labelled data"):
        load_labelled_data(["offline"], [], log=logged.append)


def test_cli_defaults_and_no_datasets(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "system_none.mock_judge.load_labelled_data", lambda datasets, files: seen.update(d=datasets, f=files)
    )
    monkeypatch.setattr("system_none.mock_judge.create_app", lambda data, confidence: seen.update(c=confidence))
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: seen.update(port=port))
    cli.main(["mock-judge"])
    assert seen == {"d": ["banking77", "ai4privacy"], "f": [], "c": 0.9, "port": 8001}
    cli.main(["mock-judge", "--datasets", "--labels", "a.jsonl", "--confidence", "1"])
    assert seen["d"] == [] and seen["f"] == ["a.jsonl"] and seen["c"] == 1.0
