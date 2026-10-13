"""Drift monitoring: the judge scores a sample of live student decisions; retrain when agreement drops."""

from __future__ import annotations

import json

from .config import Config
from .judge import Judge
from .shadow import read_log


def run_monitor(cfg: Config, judge: Judge, log=print) -> dict:
    scored_path = cfg.workdir / "review_scored.jsonl"
    done = {r["id"] for r in read_log(scored_path)}
    queue = [r for r in read_log(cfg.review_queue) if r["id"] not in done]
    labels = cfg.task.label_names
    if queue:
        results = judge.label_many([r["text"] for r in queue], progress=False)
        known = {r["id"] for r in read_log(cfg.labels_path)}
        with open(scored_path, "a") as fs, open(cfg.labels_path, "a") as fl:
            for rec, res in zip(queue, results, strict=True):
                probs = res.get("probs")
                if not probs:
                    continue
                judge_label = labels[max(range(len(probs)), key=probs.__getitem__)]
                fs.write(json.dumps({**rec, "judge_label": judge_label, "agree": judge_label == rec["label"]}) + "\n")
                if rec["id"] not in known:  # judged live traffic also becomes training data for the next retrain
                    fl.write(
                        json.dumps(
                            {
                                "id": rec["id"],
                                "text": rec["text"],
                                "probs": probs,
                                "judge": judge.spec.model,
                                "mode": res.get("mode"),
                                "source": "monitor",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    scored = read_log(scored_path)[-cfg.monitor.window :]
    deploy_path = cfg.workdir / "deploy.json"
    baseline = None
    if deploy_path.exists():
        # Served answers are the confident ones, so compare with the cascade's validation accuracy.
        accs = [g["cascade"].get("accuracy") for g in json.loads(deploy_path.read_text())["groups"].values()]
        accs = [a for a in accs if a is not None]
        baseline = min(accs) if accs else None
    n = len(scored)
    agreement = sum(r["agree"] for r in scored) / n if n else None
    drift = bool(n >= cfg.monitor.min_samples and baseline is not None and agreement < baseline - cfg.monitor.tolerance)
    status = {
        "scored_new": len(queue),
        "window": n,
        "agreement": None if agreement is None else round(agreement, 4),
        "baseline": baseline,
        "tolerance": cfg.monitor.tolerance,
        "drift": drift,
        "action": "retrain: run `system-none train`" if drift else "ok",
    }
    log(json.dumps(status, indent=1))
    return status
