"""Minimal HTTP server: POST /classify -> cascade decision. Logs inputs (never outputs) in shadow mode."""

from .config import Config
from .judge import Judge
from .runtime import ReviewSampler, load_deployment
from .shadow import ShadowLogger


def create_app(cfg: Config, threads=None):
    # (no `from __future__ import annotations` here: FastAPI must see the real request model type)
    from fastapi import FastAPI
    from pydantic import BaseModel

    fallback = None
    if cfg.incumbent is not None:
        incumbent = Judge(cfg.incumbent, cfg.task, enforce_license=False)  # serve-only: its answers are never stored
        fallback = lambda text: incumbent.classify(text)[0]  # noqa: E731
    shadow = ShadowLogger(cfg.shadow_log, sample_rate=cfg.shadow.sample_rate, max_chars=cfg.shadow.max_chars)
    review = ReviewSampler(cfg.review_queue, sample_rate=cfg.monitor.sample_rate)
    pipeline = load_deployment(cfg.workdir, fallback=fallback, review=review, threads=threads)

    app = FastAPI(title=f"system-none: {cfg.task.name}")

    class Req(BaseModel):
        text: str

    @app.get("/health")
    def health():
        return {"ok": True, "task": cfg.task.name, "labels": cfg.task.label_names}

    @app.post("/classify")
    def classify(req: Req):
        shadow.log(req.text)
        return pipeline(req.text).to_dict()

    return app
