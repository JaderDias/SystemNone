"""Serve Decision 2.0 through the System One interface used by the gateway and training judge."""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from threading import BoundedSemaphore
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL = "vllm-sr/Decision-2.0-Kai-0.6B"
logger = logging.getLogger(__name__)
process_model: Any = None


def load_process_model(name, device, loader):
    global process_model
    process_model = loader(name, device)


def process_ready():
    return os.getpid()


def process_decide(state, questions):
    return process_model.system_one(state=state, questions=questions)


def model_workers() -> int:
    value = int(os.environ.get("DECIDER_WORKERS", "1"))
    if not 1 <= value <= 8:
        raise ValueError("DECIDER_WORKERS must be between 1 and 8")
    return value


def resolve_device(device: str) -> str:
    """Kai's CUDA runtime needs working BF16 kernels, not just a visible GPU."""
    if device != "auto":
        return device
    import torch

    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            # A CUDA build can see a GPU yet lack kernels for its architecture.
            probe = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            _ = probe @ probe
            torch.cuda.synchronize()
            return "cuda"
    except (RuntimeError, AssertionError) as error:
        logger.warning("Kai CUDA readiness check failed: %s; using CPU", error)
        return "cpu"
    logger.warning("Kai needs usable CUDA BF16 support; using CPU")
    return "cpu"


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=64)
    model: str | None = None


def load_model(name: str, device: str):
    from transformers import AutoModel

    # Decision 2.0 ships its loader and inference runtime with the model package.
    device = resolve_device(device)
    options = {"threads": cpu_threads()} if device == "cpu" else {}
    return AutoModel.from_pretrained(name, trust_remote_code=True, device_map=device, **options)


def cpu_concurrency() -> int:
    value = int(os.environ.get("DECIDER_CPU_CONCURRENCY", "2"))
    if not 1 <= value <= 32:
        raise ValueError("DECIDER_CPU_CONCURRENCY must be between 1 and 32")
    return value


def cpu_threads() -> int:
    import torch

    value = int(os.environ.get("DECIDER_THREADS", "0"))
    if value < 0:
        raise ValueError("DECIDER_THREADS must be nonnegative (0 selects automatically)")
    return value or max(1, torch.get_num_threads() // (cpu_concurrency() * model_workers()))


def create_app(model_name=None, device=None, loader=load_model):
    name = model_name or os.environ.get("DECIDER_MODEL", DEFAULT_MODEL)
    device = device or os.environ.get("DECIDER_DEVICE", "auto")

    @asynccontextmanager
    async def lifespan(app):
        app.state.device = resolve_device(device)
        # CPU inference has no shared CUDA graphs/buffers. Keep the GPU runtime serialized.
        app.state.concurrency = cpu_concurrency() if app.state.device == "cpu" else 1
        app.state.workers = model_workers() if app.state.device == "cpu" else 1
        app.state.slots = BoundedSemaphore(app.state.concurrency * app.state.workers)
        logger.info("Loading %s on %s, concurrent requests: %s", name, app.state.device, app.state.concurrency)
        if app.state.workers > 1:
            # A shared queue sends each request to an available model process; HTTP
            # keep-alive connections cannot pin the client's workload to one model.
            with ProcessPoolExecutor(
                max_workers=app.state.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=load_process_model,
                initargs=(name, app.state.device, loader),
            ) as pool:
                app.state.pool = pool
                startup = [pool.submit(process_ready) for _ in range(app.state.workers)]
                for ready in startup:
                    ready.result()
                yield
        else:
            app.state.pool = None
            app.state.model = loader(name, app.state.device)
            try:
                yield
            finally:
                del app.state.model

    app = FastAPI(title="SystemNone Decision server", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "status": "ok",
            "model": name,
            "device": app.state.device,
            "concurrency": app.state.concurrency,
            "model_workers": app.state.workers,
        }

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest):
        if request.model is not None and request.model != name:
            raise HTTPException(status_code=400, detail=f"This server serves {name}")
        try:
            with app.state.slots:
                if app.state.pool is not None:
                    result = app.state.pool.submit(process_decide, request.state, request.questions).result()
                else:
                    result = app.state.model.system_one(state=request.state, questions=request.questions)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {**result, "model": name}

    return app


app = create_app()
