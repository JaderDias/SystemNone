"""A judge that speaks TypeSafe's POST /v1/systemone, as a Decider server (Mapika/decider-2b) does,
and Fastino's GLiDE, which serves the same API but may only be the incumbent."""

import json

import httpx
import numpy as np
import pytest

from system_none import judge as judge_module
from system_none.config import JudgeSpec, LicenseAck, ModelEndpoint, TaskSpec
from system_none.judge import Judge, LicenseError, label_log
from system_none.shadow import ShadowLogger, read_log

TASK = TaskSpec(
    "intent",
    "What does the customer want?",
    {"refund": "wants money back", "cancel": "", "shipping": "asks where an order is", "other": ""},
)


def spec(**kw):
    return JudgeSpec(
        base_url=kw.pop("base_url", "http://decider:8000/"),
        model="Mapika/decider-2b",
        mode="systemone",
        license=LicenseAck(allows_training_on_outputs=kw.pop("ack", True)),
        **kw,
    )


def decider(answer=None):
    """A Decider stand-in: 0.7 on the label the text names, the rest spread over the others, as its
    `probabilities`. Its `confidence` is TypeSafe's rescaled one, which the judge must not use."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        names = list(body["questions"]["label"]["criteria"])
        choice = next((n for n in names if n in body["state"]), "other")
        probs = {n: 0.7 if n == choice else 0.3 / (len(names) - 1) for n in names}
        a = (
            answer
            if answer is not None
            else {"type": "choice", "choice": choice, "confidence": 0.6, "x_p_max": 0.7, "probabilities": probs}
        )
        return httpx.Response(200, json={"model": "decider-2b-v11", "answers": {"label": a}})

    return httpx.MockTransport(handler), seen


def test_soft_labels_are_the_answer_probabilities(tmp_path):
    transport, seen = decider()
    log = ShadowLogger(tmp_path / "s.jsonl")
    texts = ["refund my order", "cancel it", "where is my shipping", "hello"]
    for t in texts:
        log.log(t)
    stats = label_log(
        Judge(spec(), TASK, transport=transport), tmp_path / "s.jsonl", tmp_path / "l.jsonl", progress=False
    )
    assert stats == {"labeled": 4, "failed": 0, "total": 4}
    by_text = {r["text"]: r for r in read_log(tmp_path / "l.jsonl")}
    for text, gold in zip(texts, ["refund", "cancel", "shipping", "other"], strict=True):
        rec = by_text[text]
        expected = [0.7 if n == gold else 0.1 for n in TASK.label_names]
        assert np.allclose(rec["probs"], expected) and rec["mode"] == "systemone" and rec["letter_mass"] == 1.0
    # One Choice question per text, its options the labels in order, with their descriptions.
    path, body = seen[0]
    assert path == "/v1/systemone"
    assert body["state"] in texts and body["model"] == "Mapika/decider-2b"
    assert body["questions"]["label"]["type"] == "choice"
    assert body["questions"]["label"]["instructions"] == TASK.question
    assert list(body["questions"]["label"]["criteria"].items()) == [
        ("refund", "wants money back"),
        ("cancel", ""),
        ("shipping", "asks where an order is"),
        ("other", ""),
    ]


def test_more_than_26_labels_need_no_letters(tmp_path):
    names = [f"intent_{i:02d}" for i in range(40)]
    transport, _ = decider()
    res = Judge(spec(), TaskSpec("many", "Which intent?", dict.fromkeys(names, "")), transport=transport).label_many(
        ["about intent_33"]
    )[0]
    assert len(res["probs"]) == 40 and names[int(np.argmax(res["probs"]))] == "intent_33"
    assert res["probs"][33] == pytest.approx(0.7)


def test_an_answer_without_probabilities_puts_its_confidence_on_its_choice():
    transport, _ = decider({"type": "choice", "choice": "cancel", "confidence": 0.8})
    res = Judge(spec(), TASK, transport=transport).label_many(["stop it"])[0]
    assert res["probs"] == pytest.approx([0.2 / 3, 0.8, 0.2 / 3, 0.2 / 3])


@pytest.mark.parametrize("answer", [{"type": "choice", "choice": "unknown", "confidence": 0.9}, {}])
def test_an_answer_naming_no_label_stays_unlabeled(tmp_path, answer):
    transport, _ = decider(answer)
    log = ShadowLogger(tmp_path / "s.jsonl")
    log.log("hello")
    stats = label_log(
        Judge(spec(), TASK, transport=transport), tmp_path / "s.jsonl", tmp_path / "l.jsonl", progress=False
    )
    assert stats == {"labeled": 0, "failed": 1, "total": 0, "errors": {"an answer naming no label": 1}}


def test_the_license_gate_still_applies():
    with pytest.raises(LicenseError):
        Judge(spec(ack=False), TASK)


def test_the_worker_passes_the_gateway_judge_mode_through(tmp_path):
    from system_none.worker import build_config

    job = {
        "prompt_id": "p1",
        "prompt": {"instructions": "Which?", "labels": [{"name": "a"}, {"name": "b"}]},
        "judge": {
            "base_url": "http://127.0.0.1:8000",
            "model": "Mapika/decider-2b",
            "mode": "systemone",
            "samples": 7,
            "license_ack": True,
        },
    }
    cfg = build_config(job, tmp_path, "microsoft/mdeberta-v3-base")
    assert (cfg.judge.mode, cfg.judge.base_url, cfg.judge.license.allows_training_on_outputs) == (
        "systemone",
        "http://127.0.0.1:8000",
        True,
    )


def fastino(statuses=()):
    """A stand-in for Fastino's GLiDE (POST https://api.fastino.ai/v1/systemone), answering in the Choice
    shape docs.fastino.ai documents: 0.8 on the label the text names, and as `confidence` the margin over
    the runner-up. As Fastino documents, it answers 401 without its key as a bearer token, and 422 to a
    top-level field other than model, state and questions, or to a criterion that isn't a string.
    `statuses` are answered first, one per request (425: the model is warming up)."""
    seen = []
    pending = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("authorization"), body))
        if pending:
            return httpx.Response(pending.pop(0), json={"detail": "model_warming"})
        if request.headers.get("authorization") != "Bearer fast_sk_test":
            return httpx.Response(401, json={"detail": "Invalid API key"})
        question = body["questions"]["label"]
        criteria = question.get("criteria")
        if (
            set(body) - {"model", "state", "questions"}
            or not body.get("model")
            or question.get("type") != "choice"
            or not question.get("instructions")
            or not isinstance(criteria, dict)
            or not 2 <= len(criteria) <= 255
            or not all(isinstance(d, str) for d in criteria.values())
        ):
            return httpx.Response(422, json={"detail": "validation error"})
        names = list(criteria)
        choice = next((n for n in names if n in body["state"]), "other")
        rest = 0.2 / (len(names) - 1)
        probs = {n: 0.8 if n == choice else rest for n in names}
        answer = {"type": "choice", "choice": choice, "confidence": 0.8 - rest, "probabilities": probs}
        usage = {"input_tokens": 89, "output_tokens": 1}
        return httpx.Response(
            200, json={"model": "glide", "answers": {"label": answer}, "usage": usage, "token_usage": 90}
        )

    return httpx.MockTransport(handler), seen


def glide() -> ModelEndpoint:
    return ModelEndpoint(
        base_url="https://api.fastino.ai", model="fastino/glide", api_key_env="FASTINO_API_KEY", mode="systemone"
    )


def test_fastino_glide_answers_as_the_incumbent(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    transport, seen = fastino()
    incumbent = Judge(glide(), TASK, transport=transport, enforce_license=False)
    res = incumbent.label_many(["where is my shipping"])[0]
    # The answer's probabilities, not its margin.
    assert res["probs"] == pytest.approx([0.2 / 3, 0.2 / 3, 0.8, 0.2 / 3]) and res["mode"] == "systemone"
    assert incumbent.classify("refund my order")[0] == "refund"
    path, auth, body = seen[0]
    assert (path, auth, body["model"], body["state"]) == (
        "/v1/systemone",
        "Bearer fast_sk_test",
        "fastino/glide",
        "where is my shipping",
    )


def test_fastino_without_its_key_answers_401(monkeypatch):
    monkeypatch.delenv("FASTINO_API_KEY", raising=False)
    transport, seen = fastino()
    res = Judge(glide(), TASK, transport=transport, enforce_license=False).label_many(["hello"])[0]
    assert res["probs"] is None and "401" in res["error"]
    assert len(seen) == 1  # not retried


def test_a_warming_fastino_model_is_retried(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(judge_module.asyncio, "sleep", no_wait)
    transport, seen = fastino(statuses=[425, 425])
    res = Judge(glide(), TASK, transport=transport, enforce_license=False).label_many(["cancel it"])[0]
    assert res["probs"] == pytest.approx([0.2 / 3, 0.8, 0.2 / 3, 0.2 / 3]) and len(seen) == 3


def test_fastino_can_never_judge():
    for base_url, model in [("https://api.fastino.ai", "glide"), ("http://proxy:8000", "fastino/glide")]:
        ack = LicenseAck(allows_training_on_outputs=True)
        with pytest.raises(LicenseError, match=r"§4\.A"):
            Judge(JudgeSpec(base_url=base_url, model=model, mode="systemone", license=ack), TASK)
