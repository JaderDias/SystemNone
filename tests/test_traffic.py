"""`system-none traffic`: request building, the menu, and runs against a fake gateway."""

import argparse
import asyncio
import io
import socket
import threading
import time
from typing import Any

import pytest
from fastapi import Request  # module level: annotations are resolved from module globals
from fastapi.responses import JSONResponse

from system_none import traffic
from system_none.cli import main as cli
from system_none.traffic import Dataset, TrafficError

from conftest import serve

# refund and cancel get right answers from the "student", the slow one a wrong one from "System One",
# and the last one fails.
TOY = Dataset(
    "toy",
    "What does the customer want?",
    ["refund", "cancel", "shipping", "other"],
    ["refund: charged twice", "cancel my plan", "where is it (slow)", "fail: boom"],
    ["refund", "cancel", "shipping", "refund"],
    descriptions={"other": "anything else"},
)


def check_body(body: dict) -> str | None:
    """The gateway's rules for a request (ClassifyReq and PromptSpec::validate); None when it is valid."""
    if set(body) - {"prompt", "prompt_id", "input", "model"}:
        return f"unexpected fields {sorted(set(body) - {'prompt', 'prompt_id', 'input', 'model'})}"
    prompt = body.get("prompt")
    if not isinstance(prompt, dict) or set(prompt) != {"instructions", "labels"}:
        return "prompt must be {instructions, labels}"
    if not isinstance(prompt["instructions"], str) or not prompt["instructions"].strip():
        return "prompt.instructions is empty"
    names = [entry if isinstance(entry, str) else entry.get("name") for entry in prompt["labels"]]
    if not 2 <= len(names) <= 255 or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        return "a prompt needs 2 to 255 unique, non-empty labels"
    if any(isinstance(entry, dict) and set(entry) - {"name", "description"} for entry in prompt["labels"]):
        return "labels are names or {name, description}"
    if not isinstance(body.get("input"), str) or not body["input"].strip():
        return "'input' is empty"
    return None


@pytest.fixture(scope="module")
def fake_gateway():
    """Answers POST /v1/classify like the gateway: the first label named in the input, else the last one."""
    from fastapi import FastAPI

    app = FastAPI()
    state: dict[str, Any] = {"bodies": [], "inflight": 0, "max_inflight": 0, "delay": 0.0}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/classify")
    async def classify(request: Request):
        body = await request.json()
        state["bodies"].append(body)
        problem = check_body(body)
        if problem:
            return JSONResponse({"error": problem}, status_code=400)
        state["inflight"] += 1
        state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        try:
            await asyncio.sleep(state["delay"])
        finally:
            state["inflight"] -= 1
        text = body["input"]
        if "fail" in text:
            return JSONResponse({"error": "System One model 'fake' failed; see the gateway log"}, status_code=502)
        names = [entry if isinstance(entry, str) else entry["name"] for entry in body["prompt"]["labels"]]
        label = next((n for n in names if n in text), names[-1])
        return {
            "prompt_id": "p1",
            "label": label,
            "confidence": 0.9,
            "model": "fake",
            "latency_ms": 1.0,
            "source": "system_one" if "slow" in text else "cache" if "again" in text else "student",
        }

    with serve(app) as url:
        yield {"url": url.removesuffix("/v1"), "state": state}


@pytest.fixture
def gw(fake_gateway):
    fake_gateway["state"].update(bodies=[], max_inflight=0, delay=0.0)
    return fake_gateway


def args(**kw):
    base = dict(
        dataset=None,
        n=None,
        concurrency=None,
        rate=None,
        gateway=traffic.DEFAULT_GATEWAY,
        model=None,
        seed=0,
        timeout=10.0,
        list=False,
    )
    return argparse.Namespace(**{**base, **kw})


class Scripted:
    """An `input()` stand-in that answers in order and records the questions."""

    def __init__(self, *answers: str):
        self.answers = iter(answers)
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        return next(self.answers)


# ---- request building -----------------------------------------------------------------------------


def test_request_body_has_the_gateway_shape():
    body = traffic.request_body(TOY, "I want my money back")
    assert body == {
        "prompt": {
            "instructions": "What does the customer want?",
            "labels": ["refund", "cancel", "shipping", {"name": "other", "description": "anything else"}],
        },
        "input": "I want my money back",
    }
    assert check_body(body) is None
    assert traffic.request_body(TOY, "hi", model="jev")["model"] == "jev"


def test_fit_labels_keeps_the_commonest_labels_that_the_gateway_takes():
    assert traffic.MAX_LABELS == 255
    labels = [f"l{i:02}" for i in range(30)]
    # l23 is the rarest, then l24, ...
    gold = [label for i, label in enumerate(labels) for _ in range(1 + (i + 7) % 30)]
    ds = Dataset("many", "q", labels, [f"text {i}" for i in range(len(gold))] + ["  "], [*gold, "l00"])
    assert traffic.fit_labels(ds).labels == labels  # 30 labels fit
    fit = traffic.fit_labels(ds, limit=26)
    assert len(fit.labels) == 26
    assert fit.labels == [label for label in labels if label not in {"l23", "l24", "l25", "l26"}]  # in their order
    assert fit.gold is not None and set(fit.gold) == set(fit.labels) and len(fit.texts) == len(fit.gold)
    assert all(t.strip() for t in fit.texts)
    assert traffic.fit_labels(TOY) == TOY
    with pytest.raises(TrafficError, match="at most 26"):
        traffic.fit_labels(Dataset("no gold", "q", labels, ["a"]), limit=26)
    many = [f"l{i:03}" for i in range(260)]
    fit = traffic.fit_labels(Dataset("260", "q", many, [f"t{i}" for i in range(260)], many))
    assert len(fit.labels) == 255 and check_body(traffic.request_body(fit, fit.texts[0])) is None


def test_datasets_become_valid_prompts(monkeypatch):
    rows = [(f"message {i}", f"intent_{i % 30:02}") for i in range(300)] + [("extra", "intent_00")]
    monkeypatch.setattr("system_none.banking77.load", lambda split: rows if split == "train" else rows[:30])
    b77 = traffic.load("banking77")
    assert len(b77.labels) == 30 and "intent_00" in b77.labels  # every intent: 77 in the real dataset
    assert check_body(traffic.request_body(b77, b77.texts[0])) is None

    pii = traffic.load("ai4privacy")  # the fixture under tests/data
    assert len(pii.texts) == 3928 and pii.gold is None
    assert check_body(traffic.request_body(pii, pii.texts[0])) is None
    assert {label["name"] for label in pii.prompt()["labels"]} == {"id_number", "contact", "address", "identity_only"}

    with pytest.raises(TrafficError, match="unknown dataset"):
        traffic.load("nope")


def fake_heads(domains):
    assert domains == ["email_triage"]
    texts = ["please reply to this", "win a prize now", "invoice attached", "meeting at noon"]
    return {
        "email_triage/needs_reply": {"labels": ["yes", "no"], "texts": texts, "gold": [0, 1, 1, 0]},
        "email_triage/is_phishing": {"labels": ["true", "false"], "texts": texts, "gold": [1, 0, 1, 1]},
    }


def test_fast_decisions_questions(monkeypatch):
    monkeypatch.setattr("system_none.fastdecisions.load_heads", fake_heads)
    ds = traffic.load("fast-decisions/email_triage/is_phishing")
    assert ds.name == "fast-decisions/email_triage/is_phishing"
    assert ds.labels == ["true", "false"] and ds.gold == ["false", "true", "false", "false"]
    assert "is phishing" in ds.instructions and check_body(traffic.request_body(ds, ds.texts[0])) is None
    with pytest.raises(TrafficError, match="fast-decisions/email_triage/needs_reply"):
        traffic.load("fast-decisions/email_triage")  # several questions and no menu: name one
    with pytest.raises(TrafficError, match="unknown Fast Decisions domain"):
        traffic.load("fast-decisions/nope")


# ---- menu -------------------------------------------------------------------------------------------


def test_parse_choice():
    names = ["banking77", "ai4privacy", "fast-decisions/support_intent", "fast-decisions/support_topic"]
    assert traffic.parse_choice("1", names) == "banking77"
    assert traffic.parse_choice(" 4 ", names) == "fast-decisions/support_topic"
    assert traffic.parse_choice("ai4privacy", names) == "ai4privacy"
    assert traffic.parse_choice("bank", names) == "banking77"  # the start of exactly one name
    assert traffic.parse_choice("fast-decisions/support_i", names) == "fast-decisions/support_intent"
    for bad in ("", "0", "5", "-1", "fast", "zzz"):
        assert traffic.parse_choice(bad, names) is None, bad


def test_choose_asks_again_until_the_answer_is_an_option():
    shown = []
    ask = Scripted("", "7", "b")
    assert traffic.choose("Pick:", [("alpha", "first"), ("beta", "second")], ask=ask, out=shown.append) == "beta"
    assert len(ask.asked) == 3 and ask.asked[0] == "Pick 1-2: "
    assert shown[1].split() == ["1)", "alpha", "first"]
    assert sum("is not one of the options" in s for s in shown) == 2


def test_numbers():
    assert traffic.parse_number("", 100) == 100
    assert traffic.parse_number(" 250 ", 100, int, 1) == 250
    assert traffic.parse_number("0", 100, int, 1) is None
    assert traffic.parse_number("2.5", 100, int, 1) is None
    assert traffic.parse_number("2.5", 0.0, float, 0) == 2.5
    assert traffic.parse_number("nan", 0.0, float, 0) is None
    shown = []
    assert traffic.ask_number("How many?", 100, int, 1, ask=Scripted("lots", "-3", "12"), out=shown.append) == 12
    assert len(shown) == 2


# ---- sending ----------------------------------------------------------------------------------------


def test_schedule_spaces_requests_by_the_rate():
    slept = []
    s = traffic.Schedule(4, rate=10, clock=lambda: 0.0, sleep=slept.append)
    assert [s.next() for _ in range(5)] == [0, 1, 2, 3, None]
    assert slept == pytest.approx([0.1, 0.2, 0.3])
    unlimited = traffic.Schedule(3, clock=lambda: 0.0, sleep=slept.append)
    assert unlimited.next() == 0
    unlimited.stop()
    assert unlimited.next() is None and len(slept) == 3
    slow = traffic.Schedule(2, rate=0.01)  # 100 s apart, but stop() wakes the waiting worker
    assert slow.next() == 0
    threading.Timer(0.05, slow.stop).start()
    t0 = time.monotonic()
    assert slow.next() is None and time.monotonic() - t0 < 5


def test_order_and_percentiles():
    picks = traffic.order(3, 7, seed=1)
    assert sorted(picks[:3]) == [0, 1, 2] and picks[3:6] == picks[:3] and picks == traffic.order(3, 7, seed=1)
    ms = [float(i) for i in range(1, 101)]
    assert [traffic.percentile(ms, p) for p in (50, 90, 99, 100)] == [50, 90, 99, 100]
    assert traffic.percentile([], 50) is None


def test_run_reports_sources_latency_agreement_and_errors(gw):
    out = io.StringIO()
    s = traffic.run(TOY, gw["url"], n=4, concurrency=2, model="fake", out=out)
    bodies = gw["state"]["bodies"]
    assert sorted(b["input"] for b in bodies) == sorted(TOY.texts)
    assert all(check_body(b) is None and b["model"] == "fake" for b in bodies)
    assert s["requests"] == 4 and s["errors"] == 1
    assert (s["student"]["answers"], s["system_one"]["answers"], s["all"]["answers"]) == (2, 1, 3)
    assert (s["student"]["agreement"], s["system_one"]["agreement"], s["all"]["agreement"]) == (1.0, 0.0, 0.6667)
    assert s["error_kinds"] == [("HTTP 502: System One model 'fake' failed; see the gateway log", 1)]
    lat = s["all"]["latency_ms"]
    assert 0 < lat["p50"] <= lat["p90"] <= lat["p99"] <= lat["max"]
    assert s["prompt_ids"] == ["p1"] and not s["stopped_early"]
    assert out.getvalue().splitlines()[-1].startswith("4/4 sent: 2 student, 1 System One, 1 errors")

    text = traffic.format_summary(s, "toy", gw["url"])
    assert "1 x HTTP 502: System One model 'fake' failed" in text
    assert [line.split()[:3] for line in text.splitlines() if line.strip().startswith(("student", "errors"))] == [
        ["student", "2", "50.0%"],
        ["errors", "1", "25.0%"],
    ]


def test_cache_answers_get_their_own_row(gw):
    ds = Dataset("toy", "What does the customer want?", ["refund", "other"], ["refund again", "refund now"])
    out = io.StringIO()
    s = traffic.run(ds, gw["url"], n=2, concurrency=1, out=out)
    assert (s["student"]["answers"], s["cache"]["answers"], s["system_one"]["answers"]) == (1, 1, 0)
    assert out.getvalue().splitlines()[-1].startswith("2/2 sent: 1 student, 1 cache, 0 System One, 0 errors")
    rows = [line.split()[:2] for line in traffic.format_summary(s, "toy", gw["url"]).splitlines()[2:6]]
    assert rows == [["student", "1"], ["cache", "1"], ["System", "One"], ["all", "2"]]
    # Without cache answers, there is no cache row.
    s["cache"]["answers"] = 0
    assert "  cache " not in traffic.format_summary(s, "toy", gw["url"])


def test_run_honours_concurrency_and_rate(gw):
    gw["state"]["delay"] = 0.05
    s = traffic.run(TOY, gw["url"], n=9, concurrency=3, out=io.StringIO())
    assert s["requests"] == 9 and 2 <= gw["state"]["max_inflight"] <= 3
    gw["state"]["delay"] = 0.0
    s = traffic.run(TOY, gw["url"], n=5, concurrency=5, rate=20, out=io.StringIO())
    assert s["requests"] == 5 and s["seconds"] >= 0.19  # 4 gaps of 50 ms


def test_menu_picks_dataset_question_and_settings(gw, monkeypatch):
    monkeypatch.setattr("system_none.fastdecisions.load_heads", fake_heads)
    names = [n for n, _ in traffic.catalog()]
    shown = []
    ask = Scripted(str(names.index("fast-decisions/email_triage") + 1), "is_ph", "6", "2", "")
    s = traffic.main(args(gateway=gw["url"]), ask=ask, out=shown.append, err=io.StringIO())
    assert ask.asked[-3:] == [
        "How many requests? [100]: ",
        "How many at a time? [4]: ",
        "At most how many per second (0: no limit)? [0]: ",
    ]
    assert s is not None and s["requests"] == 6
    assert any("has 4 texts, so they repeat" in line for line in shown)
    prompts = {b["prompt"]["instructions"] for b in gw["state"]["bodies"]}
    assert len(prompts) == 1 and "is phishing" in prompts.pop()
    assert "fast-decisions/email_triage/is_phishing: 6 requests" in shown[-1]


def test_cli_runs_without_questions_when_flags_say_what_to_send(gw, capsys):
    cli(["traffic", "--dataset", "ai4privacy", "--n", "8", "--concurrency", "3", "--gateway", gw["url"] + "/"])
    out = capsys.readouterr().out
    assert "ai4privacy: 8 requests" in out and "agree: how often" not in out  # its labels stay with the tests
    assert len(gw["state"]["bodies"]) == 8 and all(check_body(b) is None for b in gw["state"]["bodies"])
    with pytest.raises(SystemExit, match="--concurrency must be at least 1"):
        cli(["traffic", "--dataset", "ai4privacy", "--concurrency", "0", "--gateway", gw["url"]])
    with pytest.raises(SystemExit, match="unknown dataset"):
        cli(["traffic", "--dataset", "nope", "--gateway", gw["url"]])


def test_cli_lists_datasets(capsys):
    cli(["traffic", "--list"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("banking77") and any(line.startswith("ai4privacy") for line in lines)
    assert sum(line.startswith("fast-decisions/") for line in lines) == 17


def test_no_gateway_fails_before_the_menu():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ask = Scripted()
    with pytest.raises(TrafficError, match="no gateway answers"):
        traffic.main(args(gateway=f"http://127.0.0.1:{port}"), ask=ask, out=lambda *_: None)
    assert ask.asked == []


def test_reference_labels_follow_the_official_split_or_a_stable_share(monkeypatch):
    rows = [(f"message {i}", f"intent_{i % 3}") for i in range(30)]
    monkeypatch.setattr("system_none.banking77.load", lambda split: rows[:20] if split == "train" else rows[20:])
    b77 = traffic.load("banking77")
    labels = traffic.reference_labels(b77)
    assert [r["split"] for r in labels] == ["train"] * 20 + ["audit"] * 10
    assert labels[0] == {"text": "message 0", "label": "intent_0", "split": "train"}
    many = Dataset("toy", "q", ["a", "b"], [f"text {i}" for i in range(1000)], ["a", "b"] * 500)
    shares = [r["split"] for r in traffic.reference_labels(many, audit_share=0.2)]
    assert 150 < shares.count("audit") < 250 and shares == [r["split"] for r in traffic.reference_labels(many, 0.2)]
    with pytest.raises(TrafficError, match="no public labels"):
        traffic.reference_labels(Dataset("x", "q", ["a"], ["t"]))


def test_labels_command_uploads_then_queues_training(monkeypatch, capsys):
    seen = {}

    class Client:
        def post(self, url, json=None, headers=None):
            seen.setdefault("calls", []).append((url, json, headers))

            class R:
                status_code = 200

                @staticmethod
                def json():
                    if url.endswith("/admin/api/labels"):
                        return {"prompt_id": "p1", "train": 3, "audit": 1}
                    return {"id": "job1"}

            return R()

    monkeypatch.setattr(traffic, "load", lambda name: TOY)
    monkeypatch.setenv("SN_ADMIN_TOKEN", "secret")
    args = argparse.Namespace(
        dataset="toy", gateway="http://gw", token_env="SN_ADMIN_TOKEN", audit_share=0.25, train=True
    )
    traffic.labels_main(args, client=Client())
    (labels_url, body, headers), (train_url, _, _) = seen["calls"]
    assert labels_url == "http://gw/admin/api/labels" and train_url == "http://gw/admin/api/prompts/p1/train"
    assert headers == {"Authorization": "Bearer secret"} and body["name"] == "toy"
    assert body["prompt"] == TOY.prompt() and len(body["labels"]) == 4
    out = capsys.readouterr().out
    assert "3 training and 1 audit answers" in out and "queued training job job1" in out
    monkeypatch.delenv("SN_ADMIN_TOKEN")
    with pytest.raises(TrafficError, match="SN_ADMIN_TOKEN"):
        traffic.labels_main(args, client=Client())
