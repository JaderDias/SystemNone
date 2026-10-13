import json
import types
from itertools import pairwise

import numpy as np
import pytest

from system_none.calibrate import expected_calibration_error, fit_temperature, softmax
from system_none.config import JudgeSpec, LicenseAck, TaskSpec
from system_none.judge import Judge, LicenseError, label_log, letter_from_text, probs_from_top_logprobs
from system_none.runtime import choose_threshold
from system_none.shadow import ShadowLogger, read_log, shadow

from conftest import LABELS, keyword_probs, synthetic_texts

TASK = TaskSpec(name="intent", question="What does the customer want?", labels=dict.fromkeys(LABELS, ""))


def judge_spec(**kw):
    return JudgeSpec(
        base_url="http://fake/v1",
        model=kw.pop("model", "fake-judge"),
        license=LicenseAck(allows_training_on_outputs=kw.pop("ack", True)),
        **kw,
    )


# --- shadow -------------------------------------------------------------------------------
def test_shadow_logs_inputs_only_and_dedupes(tmp_path):
    log = ShadowLogger(tmp_path / "s.jsonl")

    @shadow(log)
    def classify(text):
        return {"label": "refund", "secret_output": True}

    assert classify("I want my money back") == {"label": "refund", "secret_output": True}
    classify("I want my money back")
    classify("where is my parcel")
    recs = read_log(tmp_path / "s.jsonl")
    assert [r["text"] for r in recs] == ["I want my money back", "where is my parcel"]
    assert all(set(r) <= {"id", "text", "ts", "meta"} for r in recs)  # outputs never stored
    assert len(ShadowLogger(tmp_path / "s.jsonl")) == 2  # de-dup survives restarts


def test_shadow_sampling(tmp_path):
    log = ShadowLogger(tmp_path / "s.jsonl", sample_rate=0.2, seed=1)
    n = sum(log.log(f"message {i}") for i in range(2000))
    assert 300 < n < 500


# --- judge ----------------------------------------------------------------------------------
def test_license_gate():
    with pytest.raises(LicenseError):
        Judge(judge_spec(ack=False), TASK)
    with pytest.raises(LicenseError, match="distillation"):
        Judge(judge_spec(model="typesafe/jev-1.13"), TASK)
    Judge(judge_spec(), TASK)  # acknowledged, not denylisted


def test_own_credentials_are_never_api_keys():
    for name in ("SN_WORKER_TOKEN", "aws_secret_access_key", "sn_worker_token"):
        with pytest.raises(ValueError, match="own credentials"):
            judge_spec(api_key_env=name)
    assert judge_spec(api_key_env="JUDGE_API_KEY").api_key_env == "JUDGE_API_KEY"


def test_logprob_parsing_merges_spaced_tokens():
    top = [
        {"token": "A", "logprob": np.log(0.5)},
        {"token": " A", "logprob": np.log(0.1)},
        {"token": "B", "logprob": np.log(0.3)},
        {"token": "Z", "logprob": np.log(0.1)},
    ]
    probs, mass = probs_from_top_logprobs(top, 3)
    assert probs is not None and np.allclose(probs, [0.6 / 0.9, 0.3 / 0.9, 0.0]) and abs(mass - 0.9) < 1e-9


def test_letter_from_text():
    assert letter_from_text("B", 5) == 1
    assert letter_from_text("The answer is C.", 5) == 2
    assert letter_from_text("A) refund", 5) == 0
    assert letter_from_text("Z", 5) is None


@pytest.mark.parametrize("mode", ["logprobs", "sample"])
def test_judge_soft_labels_and_resume(tmp_path, mock_transport, mode):
    texts = synthetic_texts(30, seed=3)
    log = ShadowLogger(tmp_path / "s.jsonl")
    for t in texts:
        log.log(t)
    judge = Judge(judge_spec(mode=mode, samples=9), TASK, transport=mock_transport)
    r1 = label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", limit=10, progress=False)
    r2 = label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", progress=False)
    assert (r1["labeled"], r2["labeled"], r2["total"]) == (10, 20, 30)
    for rec in read_log(tmp_path / "l.jsonl"):
        p = np.array(rec["probs"])
        assert abs(p.sum() - 1) < 1e-4
        if mode == "logprobs":
            assert np.allclose(p, keyword_probs(rec["text"]), atol=1e-4)
        assert p.argmax() == int(np.argmax(keyword_probs(rec["text"]))) or mode == "sample"


def test_judge_counts_its_progress(tmp_path, mock_transport, capsys):
    log = ShadowLogger(tmp_path / "s.jsonl")
    for t in synthetic_texts(60, seed=4):
        log.log(t)
    judge = Judge(judge_spec(), TASK, transport=mock_transport)
    counts = []

    def count(done, total):
        counts.append((done, total))

    label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", limit=20, progress=False, on_progress=count)
    assert counts == [(i, 20) for i in range(1, 21)] and capsys.readouterr().out == ""
    # A resumed run counts what is left. `progress` still prints, every 50 texts.
    counts.clear()
    stats = label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", on_progress=count)
    assert stats["total"] == 60 and counts == [(i, 40) for i in range(1, 41)]
    assert capsys.readouterr().out == ""  # 40 texts: no line yet
    assert len(judge.label_many(synthetic_texts(50, seed=5), progress=True)) == 50
    assert capsys.readouterr().out.strip() == "judged 50/50"


def test_job_progress_reports_counts_every_couple_of_seconds():
    from system_none import worker

    class Gateway:
        def progress(self, job, message, done=None, total=None):
            reports.append((now, message, done, total))

    reports, now = [], 100.0
    report = worker.JobProgress(Gateway(), "j1", interval=2, clock=lambda: now)  # ty: ignore[invalid-argument-type]
    judging = "judging 3928 inputs with Mapika/decider-2b"
    report.stage(judging)
    for done in range(1, 3929):  # an input every 10 ms: 39 seconds
        now += 0.01
        report.count(done, 3928)
    report.stage("training a student")
    assert reports[0] == (100.0, judging, None, None)
    counted = [r for r in reports if r[2] is not None]
    assert 18 <= len(counted) <= 20, counted  # one every 2 seconds, not one per input
    assert all(b[0] - a[0] >= 2 for a, b in pairwise(reports[:-2])), "the stage, then counts 2 seconds apart"
    assert counted[-1][1:] == (judging, 3928, 3928), "the last count always goes"
    assert reports[-1][1:] == ("training a student", None, None)


def test_progress_reports_are_best_effort():
    import httpx

    from system_none import worker

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 2:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(409 if len(bodies) == 3 else 204)  # 409: an admin cancelled the job

    gw = worker.GatewayClient("http://gateway", "token")
    gw.http = httpx.Client(base_url="http://gateway", transport=httpx.MockTransport(handler))
    assert gw.progress("j1", "judging 20 inputs", 5, 20) is None
    # The gateway is down: judging goes on, and the worker is told why the report didn't go.
    assert gw.progress("j1", "judging 20 inputs", 10, 20) == "ConnectError: connection refused"
    assert gw.progress("j1", "training a student") == "409"
    assert bodies == [
        {"message": "judging 20 inputs", "done": 5, "total": 20},
        {"message": "judging 20 inputs", "done": 10, "total": 20},
        {"message": "training a student"},  # no counts: the gateway clears the judging ones
    ]


def test_job_progress_goes_to_the_log_too():
    from system_none import worker

    class Gateway:
        def progress(self, job, message, done=None, total=None):
            return refusals.pop(0) if refusals else None

    refusals, lines, now = [], [], 0.0
    report = worker.JobProgress(
        Gateway(),  # ty: ignore[invalid-argument-type]
        "j1",
        interval=2,
        clock=lambda: now,
        log=lines.append,
        log_interval=60,
    )
    report.stage("judging 3000 inputs with q")
    for done in range(1, 3001):  # an input every 250 ms: 12.5 minutes
        now += 0.25
        report.count(done, 3000)
    # The stage, a count a minute (every 240 inputs), and the last count.
    judging = "job j1: judging 3000 inputs with q"
    assert lines == [judging] + [f"{judging}: {d}/3000" for d in [*range(240, 3000, 240), 3000]]
    # A refused report is logged when the refusals start, and again when they stop, not at each one.
    lines.clear()
    refusals += ["503 restarting"] * 2
    report.stage("training a student")
    now += 2
    report.count(5, 10)
    now += 2
    report.count(10, 10)
    assert lines == [
        "job j1: training a student",
        "job j1: the gateway didn't take a progress report (503 restarting)",
        "job j1: training a student: 10/10",
        "job j1: progress reaches the gateway again",
    ]
    # A job an admin cancelled is no longer running: the worker stops working on it.
    refusals.append("409 job is not running")
    with pytest.raises(worker.JobCancelled):
        report.stage("exporting to ONNX")


def test_judge_failures_are_counted_with_their_reasons(tmp_path):
    import httpx

    log = ShadowLogger(tmp_path / "s.jsonl")
    for t in synthetic_texts(5, seed=6):
        log.log(t)
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "no such model"}))
    judge = Judge(judge_spec(), TASK, transport=transport)
    stats = label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", progress=False)
    assert (stats["labeled"], stats["failed"], stats["total"]) == (0, 5, 0)
    [(reason, times)] = stats["errors"].items()  # one reason, on one line, however many inputs failed
    assert times == 5 and reason.startswith("HTTPStatusError: Client error '400 Bad Request'") and "\n" not in reason


# --- calibration and cascade -----------------------------------------------------------------
def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 3, (20000, 5))
    p = softmax(true_logits)
    y = np.array([rng.choice(5, p=row) for row in p])
    student = 2.0 * true_logits  # overconfident by a factor of 2
    T = fit_temperature(student, np.eye(5)[y])
    assert 1.8 < T < 2.2
    assert expected_calibration_error(softmax(student, T), y) < expected_calibration_error(softmax(student), y) / 3


def test_choose_threshold_meets_target():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, (5000, 5))
    probs = softmax(logits)
    y = np.array([rng.choice(5, p=row) for row in probs])
    out = choose_threshold(probs, y, 0.95)
    served = probs.max(-1) >= out["threshold"]
    assert (probs[served].argmax(-1) == y[served]).mean() >= 0.95
    assert 0 < out["coverage"] < 1


def test_worker_waits_for_an_unreachable_gateway(monkeypatch):
    import httpx

    from system_none import worker

    class Slept(Exception):
        pass

    def sleep(seconds):
        raise Slept(seconds)

    msgs = []
    monkeypatch.setattr(worker.time, "sleep", sleep)
    with pytest.raises(Slept):  # a long-running worker waits for the gateway to come back
        worker.run_worker("http://127.0.0.1:9", "token", poll=15, log=msgs.append)
    assert msgs[0] == "worker: waiting for jobs from http://127.0.0.1:9, working in runs/worker"
    assert msgs[1].startswith("gateway unreachable") and msgs[1].endswith("retrying in 15s")
    with pytest.raises(httpx.TransportError):  # a one-off run reports it
        worker.run_worker("http://127.0.0.1:9", "token", once=True)


def test_worker_keeps_its_result_until_the_gateway_takes_it(monkeypatch, tmp_path):
    # verification/JobLifecycle.tla, ClaimedJobsFinish: when /complete can't reach the gateway
    # (it is restarting), the worker tries again instead of exiting with the job left running.
    import httpx

    from system_none import worker

    class Stop(Exception):
        pass

    class Gateway:
        def __init__(self, url, token):
            self.jobs = [{"job_id": "j1", "prompt_id": "p", "base_model": "m", "judge": {"model": "q"}}]
            self.calls = []

        def claim(self):
            if not self.jobs:
                raise Stop
            return self.jobs.pop()

        def complete(self, job_id, result):
            self.calls.append((job_id, result))
            if len(self.calls) == 1:
                raise httpx.ConnectError("connection refused")
            return {"deployed": True}

    gateways = []
    monkeypatch.setattr(
        worker, "GatewayClient", lambda url, token: gateways.append(Gateway(url, token)) or gateways[-1]
    )
    monkeypatch.setattr(worker, "process", lambda spec, root, gw, log=print: {"ok": True, "message": "trained"})
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    with pytest.raises(Stop):
        worker.run_worker("http://gateway", "token", workdir=tmp_path, log=lambda msg: None)
    assert gateways[0].calls == [("j1", {"ok": True, "message": "trained"})] * 2


def test_the_log_holds_a_failed_jobs_traceback(monkeypatch, tmp_path):
    from system_none import worker

    class Stop(Exception):
        pass

    class Gateway:
        def __init__(self, url, token):
            self.jobs = [{"job_id": "j1", "prompt_id": "p", "base_model": "m", "judge": {"model": "q"}}]
            self.results = []

        def claim(self):
            if not self.jobs:
                raise Stop
            return self.jobs.pop()

        def complete(self, job_id, result):
            self.results.append(result)
            return {"deployed": False, "retry": {"job_id": "j2", "attempt": 2, "not_before": "2026-09-30T12:01:00Z"}}

    def process(spec, root, gw, log=print):
        log("job j1: training a student on m with 300 judged inputs")
        raise RuntimeError("ONNX export failed verification")

    gateways, lines = [], []
    monkeypatch.setattr(
        worker, "GatewayClient", lambda url, token: gateways.append(Gateway(url, token)) or gateways[-1]
    )
    monkeypatch.setattr(worker, "process", process)
    with pytest.raises(Stop):
        worker.run_worker("http://gateway", "token", workdir=tmp_path, log=lines.append)
    assert gateways[0].results == [{"ok": False, "message": "RuntimeError: ONNX export failed verification"}]
    # Where it failed, then the traceback, then what the gateway made of it.
    failed = next(i for i, line in enumerate(lines) if line.startswith("job j1: failed after "))
    assert lines[failed - 1] == "job j1: training a student on m with 300 judged inputs"
    assert "\nTraceback (most recent call last):\n" in lines[failed]
    assert lines[failed].endswith("RuntimeError: ONNX export failed verification")
    assert lines[-1] == "job j1: reported to the gateway: deployed=False, retry 2 not before 2026-09-30T12:01:00Z"


def test_too_few_labels_fail_the_job_not_the_worker(monkeypatch, tmp_path):
    # train_all ends the CLI with SystemExit when too few inputs were labeled; in the worker, that
    # must fail the job (which the gateway then retries), not stop the worker with the job running.
    from system_none import judge, pipeline, worker
    from system_none.shadow import text_id
    from system_none.stores import LocalStore

    store = LocalStore(tmp_path / "store")
    records = [json.dumps({"id": text_id(t), "text": t}) for t in synthetic_texts(30, seed=7)]
    store.put("inputs/p/2026-09-30/1-a.jsonl", "\n".join(records).encode())
    stats = {"labeled": 12, "failed": 18, "total": 12, "errors": {"ReadTimeout: timed out": 18}}

    def train_all(cfg, **kw):
        raise SystemExit(f"Only 12 labeled examples in {cfg.labels_path}; run `system-none judge` first.")

    monkeypatch.setattr(worker, "open_store", lambda url: store)
    monkeypatch.setattr(judge, "Judge", lambda spec, task: types.SimpleNamespace(answered_as=lambda text: None))
    monkeypatch.setattr(judge, "label_log", lambda *a, **kw: stats)
    monkeypatch.setattr(pipeline, "train_all", train_all)

    class Gateway:
        def progress(self, job, message, done=None, total=None):
            return None

    spec = {
        "job_id": "j",
        "prompt_id": "p",
        "base_model": "some/model",
        "prompt": {"instructions": "What does the customer want?", "labels": [{"name": label} for label in LABELS]},
        "judge": {"base_url": "http://judge/v1", "model": "q", "mode": "logprobs", "license_ack": True},
        "store": "file:///unused",
        "inputs_prefix": "inputs/p/",
        "models_prefix": "models/p/j",
    }
    lines = []
    result = worker.process(spec, tmp_path / "work", Gateway(), log=lines.append)  # ty: ignore[invalid-argument-type]
    assert not result["ok"] and result["message"].startswith("not enough labeled data to train: Only 12 labeled")
    assert lines[:2] == ["job j: 30 distinct inputs logged under inputs/p/", "job j: judging 30 inputs with q"]
    assert lines[2].startswith("job j: judging done in ") and lines[2].endswith(
        "s: 12 labeled, 18 failed, 12 labeled in all; most common failures: ReadTimeout: timed out (18)"
    )
    assert lines[3:] == ["job j: training a student on some/model with 12 judged inputs"]


def test_worker_reads_its_token_variable(monkeypatch):
    from system_none import cli, worker

    tokens = []
    monkeypatch.setattr(worker, "run_worker", lambda gateway, token, *a, **k: tokens.append(token) or 0)
    monkeypatch.delenv("SN_WORKER_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="set SN_WORKER_TOKEN"):
        cli.main(["worker", "--once"])
    monkeypatch.setenv("SN_WORKER_TOKEN", "token")
    cli.main(["worker", "--once"])
    assert tokens == ["token"]


def test_threshold_does_not_split_equal_confidence_predictions():
    probs = np.tile([0.9, 0.1], (40, 1))
    labels = np.array([0] * 20 + [1] * 20)
    out = choose_threshold(probs, labels, 0.95)
    assert out["coverage"] == 0
    assert out["threshold"] > 1


def test_exact_calibration_accounts_for_sample_size_and_confidence():
    from scipy.stats import beta

    probs = np.tile([0.9, 0.1], (1000, 1))
    labels = np.zeros(1000, dtype=int)
    out = choose_threshold(probs, labels, 0.98, method="clopper_pearson", confidence_level=0.95)
    assert out["coverage"] == 1
    assert out["correction"] == "bonferroni"
    expected = beta.ppf(0.05 / 1000, 1000, 1)
    assert abs(out["agreement_lower_bound"] - expected) < 1e-12
    assert out["agreement_lower_bound"] >= 0.98
    stricter = choose_threshold(probs, labels, 0.98, method="clopper_pearson", confidence_level=0.99)
    assert stricter["agreement_lower_bound"] < out["agreement_lower_bound"]
    small = choose_threshold(probs[:20], labels[:20], 0.95, method="clopper_pearson")
    assert small["coverage"] == 0
    assert choose_threshold(probs[:20], labels[:20], 0.95)["coverage"] == 1
    assert choose_threshold(probs, np.ones(1000, dtype=int), 0.95, method="clopper_pearson")["coverage"] == 0
    assert choose_threshold(probs[:0], labels[:0], 0.95, method="clopper_pearson")["coverage"] == 0


def test_worker_receives_calibration_settings(tmp_path):
    from system_none.worker import build_config

    spec = {
        "prompt_id": "p",
        "prompt": {"instructions": "Choose", "labels": [{"name": "yes"}, {"name": "no"}]},
        "judge": {"base_url": "http://judge/v1", "model": "judge", "license_ack": True},
        "calibration": {"method": "clopper_pearson", "confidence_level": 0.99},
    }
    cfg = build_config(spec, tmp_path, "jhu-clsp/ettin-encoder-17m")
    assert cfg.cascade.method == "clopper_pearson"
    assert cfg.cascade.confidence_level == 0.99
    del spec["calibration"]
    assert build_config(spec, tmp_path, "jhu-clsp/ettin-encoder-17m").cascade.method == "empirical"
