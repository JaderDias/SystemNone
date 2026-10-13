"""Decisions soft labels use the assumed SGLang/OpenAI schema, preserving the license gate."""

import json

import httpx
import pytest

from system_none.config import JudgeSpec, LicenseAck, ModelEndpoint, TaskSpec
from system_none.judge import Judge, LicenseError

TASK = TaskSpec("intent", "Which intent?", {"refund": "money back", "other": ""})


def spec(**kw):
    return JudgeSpec(
        base_url="http://decisions/v1/",
        model="decision-model",
        mode="openai_decisions",
        license=LicenseAck(allows_training_on_outputs=kw.pop("ack", True)),
        **kw,
    )


def test_decisions_soft_labels_and_auth(monkeypatch):
    monkeypatch.setenv("DECISIONS_TEST_KEY", "test-token")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "decision-model-2026-10",
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": "refund",
                        "probabilities": {"other": 0.2, "refund": 0.8},
                        "label_mass": 0.6,
                    }
                },
            },
        )

    judge = Judge(spec(api_key_env="DECISIONS_TEST_KEY"), TASK, transport=httpx.MockTransport(handler))
    result = judge.label_many(["refund me"])[0]
    # The model it answered as is the version its labels come from (issue #87).
    assert result == {
        "probs": [0.8, 0.2],
        "mode": "openai_decisions",
        "letter_mass": 0.6,
        "model": "decision-model-2026-10",
    }
    assert judge.answered_as("refund me") == "decision-model-2026-10"
    assert seen[0].url.path == "/v1/decisions"
    assert seen[0].headers["authorization"] == "Bearer test-token"
    assert json.loads(seen[0].content) == {
        "model": "decision-model",
        "input": "refund me",
        "questions": [
            {
                "id": "label",
                "type": "choice",
                "question": TASK.question,
                "options": [{"name": n, "description": d} for n, d in TASK.labels.items()],
            }
        ],
    }


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"type": "choice", "choice": "refund"},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": -0.1, "other": 1.1}},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": True, "other": 0}},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.1, "other": 0.9}},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "other": 0.1}},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "other": 0.1, "unknown": 0.1}},
        {"type": "choice", "choice": "unknown", "probabilities": {"refund": 0.8, "other": 0.2}},
        {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "other": 0.2}, "label_mass": -1},
    ],
)
def test_bad_answers_never_become_training_labels(answer):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": {"label": answer}}))
    result = Judge(spec(), TASK, transport=transport).label_many(["x"])[0]
    assert result["probs"] is None and "error" in result


def test_label_limit_and_license_gate():
    with pytest.raises(ValueError, match="26"):
        Judge(spec(), TaskSpec("many", "q", dict.fromkeys([str(i) for i in range(27)], "")))
    with pytest.raises(LicenseError):
        Judge(spec(ack=False), TASK)
    # Serving with an incumbent still does not require a training license.
    Judge(ModelEndpoint(base_url="http://decisions/v1", mode="openai_decisions"), TASK, enforce_license=False)


def test_worker_passes_decisions_judge_mode(tmp_path):
    from system_none.worker import build_config

    cfg = build_config(
        {
            "prompt_id": "p",
            "prompt": {"instructions": "q", "labels": [{"name": "a"}, {"name": "b"}]},
            "judge": {"base_url": "http://decisions/v1", "model": "m", "mode": "openai_decisions", "license_ack": True},
        },
        tmp_path,
        "base-model",
    )
    assert cfg.judge.mode == "openai_decisions"
    assert cfg.judge.base_url == "http://decisions/v1"


def test_decisions_reuses_transient_retry_policy(monkeypatch):
    import system_none.judge as module

    async def no_wait(_):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_wait)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": "refund",
                        "probabilities": {"refund": 0.8, "other": 0.2},
                    }
                }
            },
        )

    result = Judge(spec(max_retries=1), TASK, transport=httpx.MockTransport(handler)).label_many(["x"])[0]
    assert result["probs"] == [0.8, 0.2] and len(calls) == 2
