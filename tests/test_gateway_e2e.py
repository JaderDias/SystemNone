"""Tiny end-to-end run across both languages: Rust gateway + Python worker.

System One answers at first; every input is logged; at the request threshold the gateway
queues a job; the worker judges the logged inputs, fine-tunes a tiny BERT and publishes it;
the gateway then answers confident requests itself, without calling System One.

Needs the gateway binary: set SN_GATEWAY_BIN, or have `cargo` on PATH (it is built here).
"""

import glob
import json
import os
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from system_none.stores import open_store
from system_none.worker import run_worker

from conftest import LABELS, keyword_probs, synthetic_texts

pytestmark = pytest.mark.gateway

ROOT = Path(__file__).resolve().parents[1]
ADMIN, WORKER = "admin-token-for-tests", "worker-token-for-tests"
PROMPT = {"instructions": "What does the customer want?", "labels": LABELS}
N_REQUESTS = 400


def _gateway_binary() -> str:
    if os.environ.get("SN_GATEWAY_BIN"):
        return os.environ["SN_GATEWAY_BIN"]
    if not shutil.which("cargo"):
        pytest.skip("no gateway binary: set SN_GATEWAY_BIN or install Rust")
    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--manifest-path", str(ROOT / "gateway" / "Cargo.toml")], check=True
    )
    return str(ROOT / "gateway" / "target" / "release" / "system-none-gateway")


def _onnxruntime_library() -> str:
    import onnxruntime

    capi = Path(onnxruntime.__file__).parent / "capi"
    for pattern in ("libonnxruntime.so*", "libonnxruntime*.dylib", "onnxruntime.dll"):
        hits = sorted(glob.glob(str(capi / pattern)))
        if hits:
            return hits[0]
    pytest.skip("ONNX Runtime shared library not found in the onnxruntime package")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    d = tmp_path_factory.mktemp("gateway")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "SN_ADMIN_TOKEN": ADMIN,
        "SN_WORKER_TOKEN": WORKER,
        "ORT_DYLIB_PATH": _onnxruntime_library(),
        "RUST_LOG": "warn",
    }
    store = f"file://{d / 'store'}"
    proc = subprocess.Popen(
        [
            _gateway_binary(),
            "--listen",
            f"127.0.0.1:{port}",
            "--db",
            str(d / "gateway.db"),
            "--store",
            store,
            "--cache",
            str(d / "cache"),
            "--flush-secs",
            "1",
        ],
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("gateway did not start")
    yield {"url": url, "store": store, "dir": d}
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def admin():
    def call(method, url, **kw):
        return httpx.request(method, url, headers={"Authorization": f"Bearer {ADMIN}"}, timeout=30, **kw)

    return call


@pytest.fixture(scope="module")
def served(gateway, admin, judge_server, tiny_model):
    """Configure the gateway and send the first wave of traffic, answered by System One."""
    url = gateway["url"]
    assert (
        admin(
            "PUT",
            f"{url}/admin/api/system-one-models/fake-s1",
            json={"kind": "openai_chat", "base_url": judge_server["url"], "model": "fake-system-one"},
        ).status_code
        == 200
    )
    assert (
        admin(
            "PUT",
            f"{url}/admin/api/judges/fake-judge",
            json={"base_url": judge_server["url"], "model": "fake-judge", "license_ack": True},
        ).status_code
        == 200
    )
    r = admin(
        "PUT",
        f"{url}/admin/api/settings",
        json={
            "default_system_one": "fake-s1",
            "default_judge": "fake-judge",
            "default_threshold": N_REQUESTS,
            "default_base_model": tiny_model,
            "default_target_accuracy": 0.9,
            "monitor": {"enabled": False},  # this fixture isolates training and serving; monitoring has its own test
            "train": {
                "epochs": 12,
                "min_steps": 0,
                "lr": 2e-3,
                "batch_size": 16,
                "max_length": 64,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "device": "cpu",
            },
        },
    )
    assert r.status_code == 200, r.text
    texts = synthetic_texts(N_REQUESTS, seed=7)
    with httpx.Client(timeout=30) as http, ThreadPoolExecutor(8) as pool:
        answers = list(
            pool.map(lambda t: http.post(f"{url}/v1/classify", json={"prompt": PROMPT, "input": t}).json(), texts)
        )
    return texts, answers


def test_system_one_answers_first(served, judge_server):
    texts, answers = served
    assert {a["source"] for a in answers} == {"system_one"}
    agree = sum(
        a["label"] == LABELS[max(range(5), key=keyword_probs(t).__getitem__)]
        for t, a in zip(texts, answers, strict=True)
    )
    assert agree == len(texts)
    assert judge_server["state"]["calls"] >= N_REQUESTS


def test_usage_and_queue(served, gateway):
    time.sleep(1.5)  # counters flush every second
    prompts = httpx.get(f"{gateway['url']}/api/prompts").json()
    assert len(prompts) == 1
    p = prompts[0]
    assert p["total"] == N_REQUESTS and p["today"] == N_REQUESTS
    assert p["status"] == "queued" and p["threshold"] == N_REQUESTS
    assert p["daily"][-1]["system_one"] == N_REQUESTS


def test_admin_rules(gateway, admin):
    url = gateway["url"]
    assert httpx.get(f"{url}/admin/api/state").status_code == 401
    r = admin(
        "PUT",
        f"{url}/admin/api/judges/jev",
        json={"base_url": "https://api.typesafe.example/v1", "model": "typesafe/jev-2", "license_ack": True},
    )
    assert r.status_code == 400 and "2.3(b)" in r.json()["error"]
    r = admin("PUT", f"{url}/admin/api/judges/unchecked", json={"base_url": "http://x/v1", "model": "some-llm"})
    assert r.status_code == 400  # the license must be acknowledged


def test_worker_trains_and_gateway_serves_the_student(served, gateway, admin, judge_server, tmp_path):
    url = gateway["url"]
    assert run_worker(url, WORKER, tmp_path / "worker", once=True, log=lambda *a: None) == 1

    p = httpx.get(f"{url}/api/prompts").json()[0]
    assert p["status"] == "deployed", p
    metrics = p["deployment"]["metrics"]
    assert metrics["test"]["accuracy"] >= 0.8 and metrics["inputs"] == N_REQUESTS
    assert p["deployment"]["threshold"] <= 1.0, "the cascade found a threshold that meets the target"

    # Only inputs were logged, with the language router's view of them, and the published student
    # sits next to them.
    store = open_store(gateway["store"])
    logged = [json.loads(line) for k in store.list(f"inputs/{p['id']}/") for line in store.get(k).decode().splitlines()]
    assert len(logged) == N_REQUESTS
    assert all({"id", "text", "ts"} <= set(r) <= {"id", "text", "ts", "lang", "lang_confidence"} for r in logged)
    assert any(k.endswith("model.onnx") for k in store.list(f"models/{p['id']}/"))

    before = judge_server["state"]["calls"]
    with httpx.Client(timeout=30) as http:
        answers = [
            http.post(f"{url}/v1/classify", json={"prompt_id": p["id"], "input": t}).json()
            for t in synthetic_texts(200, seed=11)
        ]
    by_student = [a for a in answers if a["source"] == "student"]
    assert len(by_student) >= 60, "confident requests are answered in-process"
    assert judge_server["state"]["calls"] - before == len(answers) - len(by_student), "System One only sees the rest"
    right = sum(
        a["label"] == LABELS[max(range(5), key=keyword_probs(t).__getitem__)]
        for t, a in zip(synthetic_texts(200, seed=11), answers, strict=True)
        if a["source"] == "student"
    )
    assert right / len(by_student) >= 0.85
    latency = sorted(a["latency_ms"] for a in by_student)
    print(f"student answers: {len(by_student)}/200, median gateway latency {latency[len(latency) // 2]:.2f} ms")
    assert latency[len(latency) // 2] < 50, "in-process inference, no network hop"

    # Stopping the student sends everything back to System One.
    assert admin("POST", f"{url}/admin/api/prompts/{p['id']}/undeploy").status_code == 200
    a = httpx.post(f"{url}/v1/classify", json={"prompt_id": p["id"], "input": "please refund refund money back"}).json()
    assert a["source"] == "system_one"


def test_web_pages(gateway):
    for page, needle in (("/", "/api/prompts"), ("/admin", "Admin token"), ("/jobs", "Training jobs")):
        r = httpx.get(gateway["url"] + page)
        assert r.status_code == 200 and needle in r.text
