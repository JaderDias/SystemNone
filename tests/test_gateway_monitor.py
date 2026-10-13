"""Independent production gateway monitoring with real ONNX inference."""

import json
import time

import httpx
import pytest
from test_gateway_e2e import WORKER, admin as admin, gateway as gateway

from system_none.stores import open_store

from conftest import LABELS

pytestmark = pytest.mark.gateway


def test_gateway_shadow_promotion_audits_drift_and_retraining(gateway, admin, tmp_path, judge_server):
    """Real ONNX inference with an independent judge: shadow -> live -> drift -> fallback."""
    import asyncio

    import onnx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    from conftest import serve

    url = gateway["url"]
    teacher = {"label": "A", "fail": False}
    delay = [0.0]
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def judge_endpoint():
        await asyncio.sleep(delay[0])
        if teacher["fail"]:
            return JSONResponse(status_code=503, content={"error": "temporarily unavailable"})
        return JSONResponse(content={"choices": [{"message": {"content": teacher["label"]}}]})

    # A constant, confident student makes the routing and audit transitions deterministic.
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["logits"],
                value=helper.make_tensor("value", TensorProto.FLOAT, [1, 5], [9, 0, 0, 0, 0]),
            )
        ],
        "constant-student",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "n"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model_dir = tmp_path / "constant"
    model_dir.mkdir()
    onnx.save(model, model_dir / "model.onnx")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "refund": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "inputs": ["input_ids"],
                "max_length": 32,
                "source_model": "constant-student",
            }
        )
    )
    store = open_store(gateway["store"])

    def prompt_state(pid):
        return next(p for p in httpx.get(f"{url}/api/prompts").json() if p["id"] == pid)

    def wait_for(pid, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = prompt_state(pid)
            if predicate(state):
                return state
            time.sleep(0.02)
        pytest.fail(f"monitor did not reach expected state: {state}")

    with serve(app) as teacher_url:
        assert (
            admin(
                "PUT",
                f"{url}/admin/api/system-one-models/fake-s1",
                json={
                    "kind": "openai_chat",
                    "base_url": judge_server["url"],
                    "model": "fake-system-one",
                },
            ).status_code
            == 200
        )
        assert (
            admin(
                "PUT",
                f"{url}/admin/api/judges/monitor-judge",
                json={
                    "base_url": teacher_url,
                    "model": "independent-teacher",
                    "license_ack": True,
                },
            ).status_code
            == 200
        )
        assert (
            admin(
                "PUT",
                f"{url}/admin/api/settings",
                json={
                    "default_system_one": "fake-s1",
                    "monitor": {
                        "enabled": True,
                        "sample_rate": 0.02,
                        "uncertain_sample_rate": 1,
                        "min_samples": 3,
                        "window": 3,
                        "tolerance": 0,
                    },
                },
            ).status_code
            == 200
        )
        req = {"prompt": {"instructions": "Monitoring test: classify the request", "labels": LABELS}, "input": "refund"}
        pid = httpx.post(f"{url}/v1/classify", json=req).json()["prompt_id"]
        assert admin("PUT", f"{url}/admin/api/prompts/{pid}", json={"judge": "monitor-judge"}).status_code == 200
        assert admin("POST", f"{url}/admin/api/prompts/{pid}/train").status_code == 200
        spec = httpx.post(f"{url}/internal/jobs/claim", headers={"Authorization": f"Bearer {WORKER}"}).json()
        assert spec["prompt_id"] == pid
        for name in ("model.onnx", "tokenizer.json", "manifest.json"):
            store.put(f"{spec['models_prefix']}/{name}", (model_dir / name).read_bytes())
        completion = httpx.post(
            f"{url}/internal/jobs/{spec['job_id']}/complete",
            headers={"Authorization": f"Bearer {WORKER}"},
            json={"ok": True, "threshold": 0.8, "metrics": {"cascade": {"accuracy": 1.0}}},
        ).json()
        assert completion["shadow"] and not completion["deployed"]

        def classify():
            return httpx.post(f"{url}/v1/classify", json={"prompt_id": pid, "input": "refund"}).json()

        # The judge outage doesn't count as agreement or disagreement, or allow promotion.
        teacher["fail"] = True
        assert classify()["source"] == "system_one"
        state = wait_for(pid, lambda p: p["monitor"]["errors"] == 1)
        assert state["status"] == "shadow" and state["monitor"]["samples"] == 0
        assert state["monitor"]["traffic_samples"] == 0
        assert state["monitor"]["effective_sample_rate"] == 1
        # Even if serving fallback is unavailable, a shadow candidate must not answer. A request that
        # forces another System One model asks for that model: the student isn't consulted at all.
        r = httpx.post(f"{url}/v1/classify", json={"prompt_id": pid, "input": "refund", "model": "missing"})
        assert r.status_code == 503
        assert prompt_state(pid)["monitor"]["errors"] == 1
        teacher["fail"] = False
        for n in range(1, 4):
            assert classify()["source"] == "system_one"
            wait_for(pid, lambda p, n=n: p["monitor"]["audits"] == n)
        state = wait_for(pid, lambda p: p["monitor"]["phase"] == "live")
        assert state["status"] == "deployed"
        assert state["monitor"]["samples"] == 0
        assert state["monitor"]["traffic_samples"] == 3
        assert state["monitor"]["traffic_agreement"] == 1
        assert state["monitor"]["evidence"] == "insufficient"
        assert classify()["source"] == "student"
        wait_for(pid, lambda p: p["monitor"]["audits"] == 4)

        # Serving fallback still says refund. Only the separately configured judge changes.
        teacher["label"] = "B"
        for n in range(5, 7):
            classify()
            wait_for(pid, lambda p, n=n: p["monitor"]["audits"] == n)
        state = wait_for(pid, lambda p: p["monitor"]["phase"] == "demoted")
        assert not state["deployment"]["active"] and state["status"] == "queued"
        assert state["monitor"]["last_student_label"] == 0 and state["monitor"]["last_reference_label"] == 1
        assert state["monitor"]["failure"]["phase"] == "live"
        assert classify()["source"] == "system_one"
        retry = httpx.post(f"{url}/internal/jobs/claim", headers={"Authorization": f"Bearer {WORKER}"}).json()
        assert retry["prompt_id"] == pid and retry["job_id"] != spec["job_id"]
        assert retry["judge"]["name"] == "monitor-judge"
        # A replacement gets fresh shadow evidence. Slow results can't promote it after undeploy.
        for name in ("model.onnx", "tokenizer.json", "manifest.json"):
            store.put(f"{retry['models_prefix']}/{name}", (model_dir / name).read_bytes())
        assert httpx.post(
            f"{url}/internal/jobs/{retry['job_id']}/complete",
            headers={"Authorization": f"Bearer {WORKER}"},
            json={"ok": True, "threshold": 0.99999, "metrics": {"cascade": {"accuracy": 1.0}}},
        ).json()["shadow"]
        teacher["label"] = "A"
        assert classify()["source"] == "system_one"
        state = wait_for(pid, lambda p: p["monitor"]["audits"] == 1)
        assert state["monitor"]["samples"] == 0
        assert state["monitor"]["traffic_samples"] == 1
        assert state["monitor"]["traffic_agreement"] == 1
        delay[0] = 0.5
        assert classify()["source"] == "system_one"
        assert admin("POST", f"{url}/admin/api/prompts/{pid}/undeploy").status_code == 200
        time.sleep(0.6)
        state = prompt_state(pid)
        assert not state["deployment"]["active"] and state["monitor"]["audits"] == 1
        assert state["status"] == "collecting"


def test_known_answers_audit_the_student_instead_of_the_judge(gateway, admin, tmp_path, judge_server):
    """A student published with reference labels is audited against the held-out answers, not the
    judge (which disagrees with everything here), and not at all on texts it trained on."""
    import onnx
    from fastapi import FastAPI
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    from conftest import serve

    url = gateway["url"]
    judge_calls = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def judge_endpoint():
        judge_calls.append(1)
        return {"choices": [{"message": {"content": "B"}}]}

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["logits"],
                value=helper.make_tensor("value", TensorProto.FLOAT, [1, 5], [9, 0, 0, 0, 0]),
            )
        ],
        "constant-student",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "n"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 5])],
    )
    model_dir = tmp_path / "constant"
    model_dir.mkdir()
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model_dir / "model.onnx"
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "refund": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "manifest.json").write_text(
        json.dumps({"labels": LABELS, "inputs": ["input_ids"], "max_length": 32, "source_model": "constant-student"})
    )
    (model_dir / "reference_labels.jsonl").write_text(
        json.dumps({"text": "refund", "label": 0, "use": "audit"})
        + "\n"
        + json.dumps({"text": "cancel my order", "label": 0, "use": "train"})
        + "\n"
    )
    store = open_store(gateway["store"])

    def prompt_state(pid):
        return next(p for p in httpx.get(f"{url}/api/prompts").json() if p["id"] == pid)

    def wait_for(pid, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = prompt_state(pid)
            if predicate(state):
                return state
            time.sleep(0.02)
        pytest.fail(f"monitor did not reach expected state: {state}")

    with serve(app) as teacher_url:
        admin(
            "PUT",
            f"{url}/admin/api/system-one-models/fake-s1",
            json={"kind": "openai_chat", "base_url": judge_server["url"], "model": "fake-system-one"},
        ).raise_for_status()
        admin(
            "PUT",
            f"{url}/admin/api/judges/disagreeing-judge",
            json={"base_url": teacher_url, "model": "disagreeing-teacher", "license_ack": True},
        ).raise_for_status()
        admin(
            "PUT",
            f"{url}/admin/api/settings",
            json={
                "default_system_one": "fake-s1",
                "monitor": {
                    "enabled": True,
                    "sample_rate": 0.02,
                    "uncertain_sample_rate": 1,
                    "min_samples": 3,
                    "window": 10,
                    "tolerance": 0,
                },
            },
        ).raise_for_status()
        req = {"prompt": {"instructions": "Known answers test: classify the request", "labels": LABELS}, "input": "x"}
        pid = httpx.post(f"{url}/v1/classify", json=req).json()["prompt_id"]
        admin("PUT", f"{url}/admin/api/prompts/{pid}", json={"judge": "disagreeing-judge"}).raise_for_status()
        admin("POST", f"{url}/admin/api/prompts/{pid}/train").raise_for_status()
        spec = httpx.post(f"{url}/internal/jobs/claim", headers={"Authorization": f"Bearer {WORKER}"}).json()
        for name in ("model.onnx", "tokenizer.json", "manifest.json", "reference_labels.jsonl"):
            store.put(f"{spec['models_prefix']}/{name}", (model_dir / name).read_bytes())
        assert httpx.post(
            f"{url}/internal/jobs/{spec['job_id']}/complete",
            headers={"Authorization": f"Bearer {WORKER}"},
            json={"ok": True, "threshold": 0.8, "metrics": {"cascade": {"target_accuracy": 0.9}}},
        ).json()["shadow"]

        def classify(text):
            return httpx.post(f"{url}/v1/classify", json={"prompt_id": pid, "input": text}).json()

        # The student trained on this text's answer: agreeing with it proves nothing.
        assert classify("cancel my order")["source"] == "system_one"
        assert prompt_state(pid)["monitor"]["audits"] == 0
        # Known answers are checked at once, and the judge isn't asked.
        for n in range(1, 4):
            assert classify("refund")["source"] == "system_one"
            monitor = prompt_state(pid)["monitor"]
            assert (monitor["audits"], monitor["known_answer_audits"]) == (n, n)
        state = prompt_state(pid)
        assert state["monitor"]["phase"] == "live" and state["status"] == "deployed"
        assert state["monitor"]["agreement_boundary"] == pytest.approx(0.9)
        assert judge_calls == []
        assert classify("refund")["source"] == "student"
        # A text without a known answer is checked against the judge.
        assert classify("where is my parcel")["source"] == "student"
        state = wait_for(pid, lambda p: p["monitor"]["judge_audits"] == 1)
        assert len(judge_calls) == 1 and state["monitor"]["last_reference_label"] == 1
