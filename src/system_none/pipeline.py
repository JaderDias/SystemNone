"""`system-none train`: split, fine-tune candidates, calibrate, export, benchmark, pick, write deploy.json."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .bench import evaluate, score_against_humans
from .calibrate import expected_calibration_error, fit_temperature, softmax
from .config import Config, TrainSpec
from .data import Example, label_distribution, load_labeled, split
from .human import apply_human_labels, read_human_labels
from .runtime import Classifier, choose_threshold

MAX_EPOCHS = 100  # the step budget never asks for more passes than this


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip("/")).strip("_")


def plan_epochs(spec: TrainSpec, n_train: int) -> int:
    """`spec.epochs`, or more over a small training set, so training takes at least `spec.min_steps` steps.

    A fixed number of epochs gives a few hundred examples only a few dozen optimizer steps. A step
    budget keeps the amount of training about the same whatever the data size; the best epoch is
    still picked on validation data, so the extra passes can't make the student worse on it.
    """
    per_epoch = max(1, math.ceil(n_train / spec.batch_size))
    return max(spec.epochs, min(MAX_EPOCHS, math.ceil(spec.min_steps / per_epoch)))


ROUTES_FILE = "routes.json"


def load_examples(cfg: Config, log=print) -> tuple[list[Example], list[Example]]:
    """Judged examples with human labels applied, and the human-labeled ones held out of training.

    With `routes.json` in the workdir ({id: language}, the languages the gateway's router is sure
    of), each example gets its `route`, for students per language."""
    k = cfg.task.num_labels
    human = read_human_labels(cfg.human_labels_path, cfg.task.label_names)
    examples, held_out = apply_human_labels(
        load_labeled(cfg.labels_path, k), human, k, cfg.train.human_holdout, cfg.train.seed
    )
    routes_path = cfg.workdir / ROUTES_FILE
    if routes_path.exists():
        routes = json.loads(routes_path.read_text(encoding="utf-8"))
        for e in (*examples, *held_out):
            e.route = routes.get(e.id)
    if human:
        log(
            f"{len(human)} human labels: {len(held_out)} held out to score against people, "
            f"{len(human) - len(held_out)} train instead of the judge's"
        )
    return examples, held_out


def groups_for(cfg: Config, examples: list[Example]) -> dict[str, tuple[list[str], list[Example]]]:
    """English candidates train on English data; multilingual candidates on everything. With
    `students.per_language`, each language routed in enough examples also gets a group of its own,
    `lang-{code}`, trained on those examples alone."""
    eng, multi = cfg.students.english, cfg.students.multilingual
    if eng and multi:
        groups = {"english": (eng, [e for e in examples if e.lang == "en"]), "multilingual": (multi, examples)}
    elif multi:
        groups = {"multilingual": (multi, examples)}
    else:
        groups = {"english": (eng, examples)}
    if cfg.students.per_language:
        counts = Counter(e.route for e in examples if e.route)
        for code, n in sorted(counts.items()):
            if n >= cfg.train.per_language_min_examples:
                groups[f"lang-{code}"] = (cfg.students.per_language, [e for e in examples if e.route == code])
    return groups


def train_all(
    cfg: Config,
    candidates: list[str] | None = None,
    keep_hf: bool = True,
    log=print,
    on_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict:
    """Report per-candidate training counts and stage changes through ONNX validation."""
    from .export import export_onnx
    from .train import predict_logits, save_hf, train_student

    k = cfg.task.num_labels
    examples, held_out = load_examples(cfg, log=log)
    if len(examples) < 20:
        raise SystemExit(f"Only {len(examples)} labeled examples in {cfg.labels_path}; run `system-none judge` first.")
    log(f"{len(examples)} labeled examples; label mix {np.round(label_distribution(examples, k), 3).tolist()}")

    deploy: dict[str, Any] = {"task": cfg.task.name, "labels": cfg.task.label_names, "groups": {}}
    board: dict[str, Any] = {}
    people = {g: exs for g, (_, exs) in groups_for(cfg, held_out).items()} if held_out else {}
    for group, (models, exs) in groups_for(cfg, examples).items():
        if candidates:
            models = [m for m in models if m in candidates] or candidates
        tr, va, te = split(exs, cfg.train.val_fraction, cfg.train.test_fraction, cfg.train.seed)
        log(f"\n[{group}] train {len(tr)} / val {len(va)} / test {len(te)}")
        if len(tr) < 10 or len(va) < 5 or len(te) < 5:
            log(f"[{group}] not enough data, skipping")
            continue
        val_p = np.stack([e.probs for e in va])
        test_texts, test_p = [e.text for e in te], np.stack([e.probs for e in te])
        spec = dataclasses.replace(cfg.train, epochs=plan_epochs(cfg.train, len(tr)))
        if spec.epochs > cfg.train.epochs:
            log(f"[{group}] {spec.epochs} epochs instead of {cfg.train.epochs} (min_steps {spec.min_steps})")
        rows: list[dict[str, Any]] = []
        for name in models:
            if on_stage is not None:
                on_stage(f"training a student on {name}")
            log(f"[{group}] fine-tuning {name}")
            res = train_student(name, cfg.task, tr, va, spec, log=log, on_progress=on_progress)
            log(f"[{group}] {name}: trained in {res.seconds:.0f}s, best epoch {res.best_epoch}; calibrating, exporting")
            if on_stage is not None:
                on_stage(f"calibrating probabilities for {name}")
            vlog = predict_logits(res.model, res.tokenizer, [e.text for e in va], cfg.train.max_length)
            targets = np.eye(k)[val_p.argmax(-1)] if cfg.train.calibrate_on == "hard" else val_p
            T = fit_temperature(vlog, targets)
            ece_before = expected_calibration_error(softmax(vlog), val_p.argmax(-1))
            ece_after = expected_calibration_error(softmax(vlog, T), val_p.argmax(-1))
            out = cfg.models_dir / group / _slug(name)
            if out.exists():
                shutil.rmtree(out)
            if on_stage is not None:
                on_stage(f"exporting {name} to ONNX")
            export_onnx(
                res.model,
                res.tokenizer,
                cfg.task.label_names,
                out / "onnx",
                cfg.train.max_length,
                T,
                sample_texts=[e.text for e in va[:8]],
                meta={"source_model": name, "task": cfg.task.name, "question": cfg.task.question, "group": group},
                log=log,
            )
            if keep_hf:
                save_hf(res, out / "hf")
            log(f"[{group}] {name}: scoring on {len(te)} test inputs, choosing the threshold")
            if on_stage is not None:
                on_stage(f"validating ONNX inference and confidence threshold for {name}")
            clf = Classifier(out / "onnx")
            metrics = evaluate(clf, test_texts, test_p)
            thr = choose_threshold(
                clf.predict_proba([e.text for e in va]),
                val_p.argmax(-1),
                cfg.cascade.target_accuracy,
                method=cfg.cascade.method,
                confidence_level=cfg.cascade.confidence_level,
                candidate_models=len(models),
            )
            row = {
                "model": name,
                "dir": str((out / "onnx").relative_to(cfg.workdir)),
                "temperature": round(T, 4),
                "val_ece_before": round(ece_before, 4),
                "val_ece_after": round(ece_after, 4),
                "epochs": spec.epochs,
                "train_seconds": round(res.seconds, 1),
                "cascade": thr,
                "test": metrics,
            }
            if people.get(group):
                row["human"] = score_against_humans(clf, people[group], thr["threshold"])
            rows.append(row)
            log(
                f"[{group}] {name}: test {metrics} | T={T:.3f} | cascade {thr}"
                + (f" | against people {row['human']}" if "human" in row else "")
            )
        if not rows:
            continue
        best_acc = max(r["test"]["accuracy"] for r in rows)
        pool = [r for r in rows if r["test"]["accuracy"] >= best_acc - cfg.train.accuracy_tolerance]
        best = min(pool, key=lambda r: r["test"].get("p50_ms", 0))
        board[group] = {"candidates": rows, "chosen": best["model"]}
        deploy["groups"][group] = {
            "model": best["model"],
            "model_dir": best["dir"],
            "threshold": best["cascade"]["threshold"],
            "cascade": best["cascade"],
            "test": best["test"],
            **({"human": best["human"]} if "human" in best else {}),
        }
        log(f"[{group}] chosen: {best['model']}")

    if {"english", "multilingual"} <= set(deploy["groups"]):
        deploy["routes"], deploy["routing"] = _route_english(cfg, examples, deploy, log)

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / "leaderboard.json").write_text(json.dumps(board, indent=1))
    (cfg.workdir / "deploy.json").write_text(json.dumps(deploy, indent=1))
    return deploy


def _route_english(cfg: Config, examples: list[Example], deploy: dict, log=print) -> tuple[dict, dict]:
    """Send English to whichever student is better on held-out English text.

    The multilingual student sees every language, so with little English data it can beat the
    English specialist. Splits hash the example id, so the English test rows were held out of both
    groups' training. The English student keeps the route unless it trails by more than
    `accuracy_tolerance`.
    """
    english = [e for e in examples if e.lang == "en"]
    test = split(english, cfg.train.val_fraction, cfg.train.test_fraction, cfg.train.seed)[2]
    if not test:
        return {"en": "english"}, {}
    targets = np.stack([e.probs for e in test])
    scores = {}
    for group in ("english", "multilingual"):
        clf = Classifier(cfg.workdir / deploy["groups"][group]["model_dir"])
        scores[group] = evaluate(clf, [e.text for e in test], targets, measure_latency=False)["accuracy"]
    pick = "multilingual" if scores["multilingual"] > scores["english"] + cfg.train.accuracy_tolerance else "english"
    log(
        f"English test accuracy: english {scores['english']:.3f}, multilingual {scores['multilingual']:.3f} "
        f"-> route English to {pick}"
    )
    return {"en": pick}, {"en": {**scores, "n": len(test), "chosen": pick}}
