import httpx
import pytest
from fastapi.testclient import TestClient

from system_none.config import JudgeSpec, LicenseAck, TaskSpec
from system_none.decision_server import DEFAULT_MODEL, create_app, load_model, resolve_device
from system_none.judge import Judge


class Model:
    def __init__(self):
        self.seen = []

    def system_one(self, *, state, questions):
        self.seen.append((state, questions))
        if state == "invalid":
            raise ValueError("Input too long")
        return {
            "answers": {
                key: {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "other": 0.2}}
                for key in questions
            }
        }


def test_default_model_lifecycle_and_training_probabilities(monkeypatch):
    monkeypatch.delenv("DECIDER_DEVICE", raising=False)
    monkeypatch.setattr("system_none.decision_server.resolve_device", lambda device: "cpu")
    model, loaded = Model(), []

    def loader(name, device):
        loaded.append((name, device))
        return model

    app = create_app(loader=loader)
    assert not loaded  # Importing the server does not download model weights.
    with TestClient(app) as client:
        assert loaded == [(DEFAULT_MODEL, "cpu")]
        assert client.get("/health").json()["model"] == DEFAULT_MODEL
        assert client.get("/health").json()["ok"] is True
        assert client.get("/health").json()["device"] == "cpu"
        task = TaskSpec("intent", "Which intent?", {"refund": "Money back", "other": "Other requests"})
        spec = JudgeSpec(
            base_url="http://testserver",
            model=DEFAULT_MODEL,
            mode="systemone",
            license=LicenseAck(allows_training_on_outputs=True),
        )

        def forward(request):
            response = client.post(
                request.url.path, content=request.content, headers={"Content-Type": "application/json"}
            )
            return httpx.Response(response.status_code, json=response.json())

        judged = Judge(spec, task, transport=httpx.MockTransport(forward)).label_many(["Refund please"])[0]
        assert judged["probs"] == pytest.approx([0.8, 0.2])
        assert model.seen == [
            (
                "Refund please",
                {
                    "label": {"type": "choice", "instructions": task.question, "criteria": task.labels},
                },
            )
        ]
    assert not hasattr(app.state, "model")


def test_request_validation_and_model_override():
    app = create_app("local/model", "auto", loader=lambda name, device: Model())
    with TestClient(app) as client:
        body = {"state": "hello", "questions": {"label": {"type": "choice", "criteria": ["refund", "other"]}}}
        assert client.post("/v1/systemone", json=body).status_code == 200
        assert client.post("/v1/systemone", json={**body, "model": DEFAULT_MODEL}).status_code == 400
        assert client.post("/v1/systemone", json={**body, "questions": {}}).status_code == 422
        assert client.post("/v1/systemone", json={**body, "state": "invalid"}).status_code == 422


def test_loader_passes_device_to_the_official_model_api(monkeypatch):
    import torch
    from transformers import AutoModel

    seen = []
    model = Model()

    def load(name, **kwargs):
        seen.append((name, kwargs))
        return model

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.delenv("DECIDER_THREADS", raising=False)
    monkeypatch.setenv("DECIDER_CPU_CONCURRENCY", "2")
    assert load_model(DEFAULT_MODEL, "cpu") is model
    assert seen == [(DEFAULT_MODEL, {"trust_remote_code": True, "device_map": "cpu", "threads": 4})]


@pytest.mark.parametrize("available,bf16", [(False, False), (True, False)])
def test_auto_device_uses_cpu_without_required_cuda_support(monkeypatch, available, bf16):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    assert resolve_device("auto") == "cpu"


@pytest.mark.parametrize("failure", [None, "kernel", "initialization", "synchronize"])
def test_auto_device_checks_kernels_before_selecting_cuda(monkeypatch, failure):
    import torch

    probe = torch.ones((2, 2))
    seen = []

    def available():
        if failure == "initialization":
            raise RuntimeError("driver unavailable")
        return True

    def allocate(shape, **kwargs):
        seen.append((shape, kwargs))
        if failure == "kernel":
            raise RuntimeError("no kernel image available")
        return probe

    def synchronize():
        if failure == "synchronize":
            raise RuntimeError("asynchronous CUDA failure")

    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch, "ones", allocate)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert resolve_device("auto") == ("cpu" if failure else "cuda")
    if failure != "initialization":
        assert seen == [((2, 2), {"device": "cuda", "dtype": torch.bfloat16})]


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps"])
def test_explicit_device_bypasses_automatic_selection(monkeypatch, device):
    import torch

    def unexpected_probe():
        pytest.fail("explicit device must not probe or fall back")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    assert resolve_device(device) == device


def test_auto_selection_is_reported_by_health_and_passed_to_loader(monkeypatch):
    monkeypatch.delenv("DECIDER_DEVICE", raising=False)
    seen = []

    def select(device):
        assert device == "auto"
        return "cuda"

    def load(name, device):
        seen.append((name, device))
        return Model()

    monkeypatch.setattr("system_none.decision_server.resolve_device", select)
    with TestClient(create_app(loader=load)) as client:
        assert client.get("/health").json() == {
            "ok": True,
            "status": "ok",
            "model": DEFAULT_MODEL,
            "device": "cuda",
            "concurrency": 1,
            "model_workers": 1,
        }
    assert seen == [(DEFAULT_MODEL, "cuda")]


@pytest.mark.parametrize("device,expected", [("cpu", 2), ("cuda", 1)])
def test_parallel_cpu_requests_are_bounded_and_gpu_is_serialized(monkeypatch, device, expected):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    monkeypatch.setenv("DECIDER_CPU_CONCURRENCY", "2")
    guard, overlap = Lock(), Event()

    class ParallelModel(Model):
        active = 0
        maximum = 0

        def system_one(self, **kwargs):
            with guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                if self.active == expected:
                    overlap.set()
            assert overlap.wait(5)
            time.sleep(0.02)
            result = super().system_one(**kwargs)
            with guard:
                self.active -= 1
            return result

    model = ParallelModel()
    with TestClient(create_app(device=device, loader=lambda name, device: model)) as client:
        body = {"state": "hello", "questions": {"label": {"type": "choice", "criteria": ["refund", "other"]}}}
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: client.post("/v1/systemone", json=body), range(8)))
        assert all(response.status_code == 200 for response in responses)
        assert model.maximum == expected
        assert client.get("/health").json()["concurrency"] == expected


@pytest.mark.parametrize(
    "setting,value", [("DECIDER_CPU_CONCURRENCY", "0"), ("DECIDER_CPU_CONCURRENCY", "33"), ("DECIDER_THREADS", "-1")]
)
def test_invalid_cpu_policy_is_rejected(monkeypatch, setting, value):
    from system_none.decision_server import cpu_concurrency, cpu_threads

    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError):
        cpu_concurrency() if setting == "DECIDER_CPU_CONCURRENCY" else cpu_threads()


def test_explicit_thread_budget_is_honored(monkeypatch):
    from system_none.decision_server import cpu_threads

    monkeypatch.setenv("DECIDER_THREADS", "6")
    assert cpu_threads() == 6


class PoolModel(Model):
    def system_one(self, **kwargs):
        import os
        import time

        time.sleep(0.1)
        return {**super().system_one(**kwargs), "worker_pid": os.getpid()}


def pool_loader(name, device):
    import os
    import time
    from pathlib import Path

    root = Path(name)
    (root / str(os.getpid())).touch()
    deadline = time.monotonic() + 10
    while len(list(root.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError("second process did not start")
        time.sleep(0.01)
    return PoolModel()


def test_process_queue_distributes_requests_and_propagates_validation_errors(monkeypatch, tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("DECIDER_WORKERS", "2")
    monkeypatch.setenv("DECIDER_CPU_CONCURRENCY", "1")
    with TestClient(create_app(str(tmp_path), device="cpu", loader=pool_loader)) as client:
        assert client.get("/health").json()["model_workers"] == 2
        body = {"state": "hello", "questions": {"label": {}}}
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(lambda _: client.post("/v1/systemone", json=body), range(8)))
        assert all(response.status_code == 200 for response in responses)
        pids = {response.json()["worker_pid"] for response in responses}
        assert len(pids) == 2 and os.getpid() not in pids
        assert client.post("/v1/systemone", json={**body, "state": "invalid"}).status_code == 422
