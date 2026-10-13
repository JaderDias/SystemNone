"""Human labels: score the judge, the students and the cascade against people, or train on them."""

import json

import numpy as np
import pytest

from system_none import worker
from system_none.bench import score_against_humans
from system_none.cli import main
from system_none.config import load_config
from system_none.data import Example, load_labeled, split
from system_none.human import (
    add_human_labels,
    add_store_labels,
    apply_human_labels,
    describe,
    is_held_out,
    judge_agreement,
    read_human_labels,
)
from system_none.pipeline import plan_epochs, train_all
from system_none.shadow import read_log, text_id
from system_none.stores import LocalStore

from conftest import LABELS, keyword_probs, synthetic_texts, write_config


def top(text: str) -> str:
    return LABELS[int(np.argmax(keyword_probs(text)))]


def wrong(text: str) -> str:
    return LABELS[(LABELS.index(top(text)) + 1) % len(LABELS)]


def ex(i, label, lang="en"):
    return Example(id=f"id{i}", text=f"text {i}", probs=np.eye(5, dtype=np.float32)[label] * 0.8 + 0.04, lang=lang)


# --- recording ------------------------------------------------------------------------------------
def test_add_and_read_human_labels(tmp_path):
    rows = [
        ("I want my money back", "refund"),
        ("where is my parcel", "Shipping"),
        ("hello", "complaint"),
        ("  ", "refund"),
        ("I want my money back", "refund"),
        ("where is my parcel", "other"),
    ]
    stats = add_human_labels(tmp_path / "h.jsonl", rows, LABELS, shadow_log=tmp_path / "s.jsonl")
    assert stats == {"added": 3, "unchanged": 1, "total": 2, "unknown_labels": {"complaint": 1}}
    human = read_human_labels(tmp_path / "h.jsonl", LABELS)
    assert human == {
        text_id("I want my money back"): {"text": "I want my money back", "label": 0, "split": None},
        text_id("where is my parcel"): {"text": "where is my parcel", "label": 4, "split": None},
    }  # the later answer wins
    assert [r["text"] for r in read_log(tmp_path / "s.jsonl")] == ["I want my money back", "where is my parcel"]
    assert read_human_labels(tmp_path / "h.jsonl", ["refund", "cancel"]) == {  # a label the task dropped
        text_id("I want my money back"): {"text": "I want my money back", "label": 0, "split": None}
    }


def test_long_texts_get_the_shadow_logs_id(tmp_path):
    text = "refund " * 20
    add_human_labels(tmp_path / "h.jsonl", [(text, "refund")], LABELS, shadow_log=tmp_path / "s.jsonl", max_chars=30)
    assert set(read_human_labels(tmp_path / "h.jsonl", LABELS)) == {r["id"] for r in read_log(tmp_path / "s.jsonl")}


# --- hold-out and replacement -------------------------------------------------------------------------
def test_held_out_human_labels_leave_the_training_pool():
    judged = [ex(0, 0), ex(1, 1), ex(2, 2)]
    human = {"id1": {"text": "text 1", "label": 3}, "new": {"text": "text new", "label": 2}}
    train, held = apply_human_labels(judged, human, 5, holdout=1.0, seed=13, detect_lang=False)
    assert [e.id for e in train] == ["id0", "id2"] and {e.id for e in held} == {"id1", "new"}
    by_id = {e.id: e for e in held}
    assert by_id["id1"].label == 3 and np.allclose(by_id["id1"].meta["judge_probs"], judged[1].probs)
    assert by_id["id1"].lang == "en"  # the judged example's language
    assert by_id["new"].meta["judge_probs"] is None and by_id["new"].label == 2


def test_trained_human_labels_replace_the_judges():
    judged = [ex(0, 0), ex(1, 1), ex(2, 2)]
    human = {"id1": {"text": "text 1", "label": 3}, "new": {"text": "text new", "label": 2}}
    train, held = apply_human_labels(judged, human, 5, holdout=0.0, seed=13, detect_lang=False)
    assert held == [] and [e.id for e in train] == ["id0", "id1", "id2", "new"]  # order kept, new ones last
    assert train[1].probs.tolist() == [0, 0, 0, 1, 0] and train[1].meta["human"]
    assert train[0] is judged[0]
    same, none = apply_human_labels(judged, {}, 5, holdout=0.5, seed=13)
    assert same is judged and none == []


def test_partial_holdout_is_deterministic():
    ids = [f"x{i}" for i in range(1000)]
    held = [i for i in ids if is_held_out(i, 0.3, seed=13)]
    assert 240 < len(held) < 360 and held == [i for i in ids if is_held_out(i, 0.3, seed=13)]
    assert all(is_held_out(i, 1.0, 13) for i in ids) and not any(is_held_out(i, 0.0, 13) for i in ids)


# --- scoring ----------------------------------------------------------------------------------------
class FixedClassifier:
    def __init__(self, probs):
        self.probs = np.array(probs)

    def predict_proba(self, texts):
        return self.probs[: len(texts)]


def test_score_against_humans():
    people = [ex(i, lab) for i, lab in enumerate([0, 1, 2, 3])]
    for e, judge_label in zip(people, [0, 1, 2, 0], strict=True):  # the judge is wrong on the last one
        e.meta["judge_probs"] = np.eye(5)[judge_label]
    people.append(Example(id="unjudged", text="t", probs=np.eye(5)[4], meta={"judge_probs": None}))
    clf = FixedClassifier(
        [[0.9, 0.1, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0.6, 0, 0.4, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.3, 0.7]]
    )
    res = score_against_humans(clf, people, threshold=0.7)
    assert res["n"] == 5 and res["accuracy"] == 0.8  # wrong only on the third
    assert res["student_answers"] == 0.6 and res["accuracy_when_student_answers"] == 1.0
    assert res["judge_accuracy"] == 0.75 and res["judge_n"] == 4
    assert "student_answers" not in score_against_humans(clf, people)
    assert describe(res) == "; right 0.800 against 5 human labels (the judge: 0.750)" and describe(None) == ""


def test_judge_agreement(tmp_path):
    texts = synthetic_texts(10, seed=4)
    with open(tmp_path / "l.jsonl", "w") as f:
        for t in texts[:8]:
            f.write(json.dumps({"id": text_id(t), "text": t, "probs": keyword_probs(t)}) + "\n")
    add_human_labels(
        tmp_path / "h.jsonl",
        [(texts[0], wrong(texts[0]))] + [(t, top(t)) for t in texts[1:4]] + [(texts[9], top(texts[9]))],
        LABELS,
    )
    assert judge_agreement(tmp_path / "l.jsonl", tmp_path / "h.jsonl", LABELS) == {
        "n": 4,
        "accuracy": 0.75,
        "human_labels": 5,
    }
    assert judge_agreement(tmp_path / "l.jsonl", tmp_path / "none.jsonl", LABELS) is None


# --- CLI ----------------------------------------------------------------------------------------------
def test_log_records_human_labels_and_judge_reports_agreement(tmp_path, tiny_model, judge_server, capsys):
    cfg_path = write_config(tmp_path / "c.yaml", tmp_path / "run", tiny_model, judge_url=judge_server["url"])
    texts = synthetic_texts(40, seed=21)
    rows = [{"text": t, "label": top(t)} for t in texts[:20]] + [{"text": t} for t in texts[20:]]
    rows[0]["label"] = wrong(texts[0])  # a person disagrees with the judge
    rows[1]["label"] = "complaint"  # not a label of this task
    (tmp_path / "in.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    main(["log", "-c", str(cfg_path), str(tmp_path / "in.jsonl"), "--label-field", "label"])
    out = capsys.readouterr()
    assert "logged 40 new inputs" in out.out and "recorded 19 human labels" in out.out and "complaint" in out.err
    cfg = load_config(cfg_path)
    assert len(read_human_labels(cfg.human_labels_path, LABELS)) == 19 and len(read_log(cfg.shadow_log)) == 40

    main(["judge", "-c", str(cfg_path)])
    res = json.loads(capsys.readouterr().out)
    assert res["labeled"] == 40 and res["judge_vs_humans"] == {
        "n": 19,
        "accuracy": round(18 / 19, 4),
        "human_labels": 19,
    }

    (tmp_path / "in.txt").write_text("hello")
    with pytest.raises(SystemExit, match=r"needs a \.jsonl"):
        main(["log", "-c", str(cfg_path), str(tmp_path / "in.txt"), "--label-field", "label"])


# --- training ----------------------------------------------------------------------------------------
def test_training_scores_students_against_people(tmp_path, tiny_model):
    cfg = load_config(write_config(tmp_path / "c.yaml", tmp_path / "run", tiny_model))
    cfg.train.human_holdout, cfg.train.epochs, cfg.train.min_steps = 0.5, 3, 150
    texts = synthetic_texts(400, seed=31)
    judge = {t: keyword_probs(t) for t in texts}
    for t in texts[::10]:  # the judge's answers are shifted on every tenth text
        judge[t] = judge[t][1:] + judge[t][:1]
    cfg.workdir.mkdir(parents=True)
    with open(cfg.labels_path, "w") as f:
        for t in texts:
            f.write(json.dumps({"id": text_id(t), "text": t, "probs": judge[t]}) + "\n")
    add_human_labels(cfg.human_labels_path, [(t, top(t)) for t in texts[:120]], LABELS)

    stages = []
    deploy = train_all(cfg, keep_hf=True, log=lambda *a: None, on_stage=stages.append)
    assert [stage.split()[0] for stage in stages] == ["training", "calibrating", "exporting", "validating"]

    held = [t for t in texts[:120] if is_held_out(text_id(t), 0.5, cfg.train.seed)]
    judge_right = [t for t in held if LABELS[int(np.argmax(judge[t]))] == top(t)]
    human = deploy["groups"]["english"]["human"]
    assert human["n"] == len(held) > 30 and human["judge_n"] == len(held)
    assert human["judge_accuracy"] == round(len(judge_right) / len(held), 4) < 1
    assert human["accuracy"] >= 0.75 and 0 < human["student_answers"] <= 1, human
    # Held-out texts never reach training, validation or test; the other human labels do.
    pool, _ = apply_human_labels(
        load_labeled(cfg.labels_path, 5), read_human_labels(cfg.human_labels_path, LABELS), 5, 0.5, cfg.train.seed
    )
    tr, _, te = split(pool, cfg.train.val_fraction, cfg.train.test_fraction, cfg.train.seed)
    assert deploy["groups"]["english"]["test"]["n"] == len(te) and len(pool) == 400 - len(held)
    # Few examples: epochs were raised to reach the step budget.
    row = json.loads((cfg.workdir / "leaderboard.json").read_text())["english"]["candidates"][0]
    assert row["epochs"] == plan_epochs(cfg.train, len(tr)) > 3 and row["human"] == human
    history = json.loads((cfg.workdir / row["dir"]).parent.joinpath("hf", "history.json").read_text())["history"]
    assert len(history) == row["epochs"]


def test_worker_reads_human_labels_from_the_store(tmp_path, tiny_model, judge_server, monkeypatch):
    store = LocalStore(tmp_path / "store")
    monkeypatch.setattr(worker, "open_store", lambda url: store)
    texts = synthetic_texts(300, seed=41)
    store.put(
        "inputs/p1/2026-09-27/1-a.jsonl",
        "\n".join(json.dumps({"id": text_id(t), "text": t, "ts": 0}) for t in texts).encode(),
    )
    extra = synthetic_texts(10, seed=42)  # checked by a person, never sent through the gateway
    people = [{"text": t, "label": wrong(t) if i < 5 else top(t)} for i, t in enumerate(texts[:40])]
    people += [{"input": t, "label": top(t)} for t in extra]
    store.put("human_labels/p1/batch-1.jsonl", "\n".join(json.dumps(r) for r in people).encode())

    class Gateway:
        def progress(self, job, message, done=None, total=None):
            reports.append((message, done, total))

    reports = []

    spec = {
        "job_id": "j1",
        "prompt_id": "p1",
        "base_model": tiny_model,
        "target_accuracy": 0.9,
        "prompt": {"instructions": "What does the customer want?", "labels": [{"name": label} for label in LABELS]},
        "judge": {"base_url": judge_server["url"], "model": "fake-judge", "mode": "logprobs", "license_ack": True},
        "train": {
            "epochs": 12,
            "min_steps": 0,
            "lr": 2e-3,
            "batch_size": 16,
            "max_length": 64,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "device": "cpu",
            "not_a_setting": 1,
        },
        "store": "file:///unused",
        "inputs_prefix": "inputs/p1/",
        "models_prefix": "models/p1/j1",
    }
    # Gateway stands in for GatewayClient, with the one method process() calls.
    result = worker.process(spec, tmp_path / "work", Gateway(), log=lambda *a: None)  # ty: ignore[invalid-argument-type]

    assert result["ok"], result
    human = result["metrics"]["human"]
    assert human["n"] == 50 and human["judge_n"] == 50 and human["judge_accuracy"] == 0.9  # all held out by default
    assert human["accuracy"] >= 0.75
    assert "against 50 human labels (the judge: 0.900)" in result["message"]
    assert result["metrics"]["inputs"] == 300 and any(k.endswith("model.onnx") for k in store.list("models/p1/"))
    # The texts of the human labels the gateway never saw are judged too, and counted.
    judging = "judging 310 inputs with fake-judge"
    assert reports[0] == (judging, None, None) and (judging, 310, 310) in reports


def test_add_store_labels(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put(
        "human_labels/p/a.jsonl", b'{"text": "refund please", "label": "refund"}\n\n{"input": "hi", "label": "x"}\n'
    )
    store.put("human_labels/p/readme.txt", b"ignored")
    stats = add_store_labels(store, "human_labels/p/", tmp_path / "h.jsonl", LABELS, shadow_log=tmp_path / "s.jsonl")
    assert stats["added"] == 1 and stats["unknown_labels"] == {"x": 1}
    assert [r["text"] for r in read_log(tmp_path / "s.jsonl")] == ["refund please"]


# --- reference labels with a split ------------------------------------------------------------------
def test_a_labels_split_overrides_the_holdout_share(tmp_path):
    rows = [("I want my money back", "refund", "train"), ("where is my parcel", "shipping", "audit"), ("hi", "refund")]
    add_human_labels(tmp_path / "h.jsonl", rows, LABELS)
    human = read_human_labels(tmp_path / "h.jsonl", LABELS)
    assert [h["split"] for h in human.values()] == ["train", "audit", None]
    judged = [
        Example(id=i, text=h["text"], probs=np.full(5, 0.2, dtype=np.float32), lang="en") for i, h in human.items()
    ]
    for holdout in (0.0, 1.0):
        train, held = apply_human_labels(judged, human, 5, holdout=holdout, seed=13, detect_lang=False)
        assert text_id("I want my money back") in {e.id for e in train}
        assert text_id("where is my parcel") in {e.id for e in held}
        assert (text_id("hi") in {e.id for e in held}) == (holdout == 1.0)
    # Changing only the split records the label again.
    stats = add_human_labels(tmp_path / "h.jsonl", [("hi", "refund", "train")], LABELS)
    assert stats["added"] == 1 and read_human_labels(tmp_path / "h.jsonl", LABELS)[text_id("hi")]["split"] == "train"


def test_store_labels_keep_their_split(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put(
        "human_labels/p/b.jsonl",
        b'{"text": "refund please", "label": "refund", "split": "audit"}\n{"text": "late", "label": "shipping", "split": "x"}\n',
    )
    add_store_labels(store, "human_labels/p/", tmp_path / "h.jsonl", LABELS)
    human = read_human_labels(tmp_path / "h.jsonl", LABELS)
    assert human[text_id("refund please")]["split"] == "audit" and human[text_id("late")]["split"] is None


def test_worker_publishes_known_answers_and_skips_judging_them(tmp_path):
    from system_none.config import config_from_dict

    cfg = config_from_dict({"workdir": str(tmp_path / "w"), "task": {"name": "t", "question": "q", "labels": LABELS}})
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    rows = [("refund please", "refund", "train"), ("where is it", "shipping", "audit"), ("hello", "refund")]
    add_human_labels(cfg.human_labels_path, rows, LABELS, shadow_log=cfg.shadow_log)
    # "where is it" was judged by an earlier job: it stays, to score the judge.
    skipped = worker.skip_known_answers(cfg, saved={text_id("where is it")})
    assert skipped == 1
    assert [r["text"] for r in read_log(cfg.shadow_log)] == ["where is it", "hello"]
    counts = worker.write_reference_labels(cfg, tmp_path / "reference_labels.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "reference_labels.jsonl").read_text().splitlines()]
    # Unsplit labels follow human_holdout, 1.0 by default: held out, so audited.
    assert counts == {"train": 1, "audit": 2}
    assert {(r["text"], r["label"], r["use"]) for r in lines} == {
        ("refund please", 0, "train"),
        ("where is it", LABELS.index("shipping"), "audit"),
        ("hello", 0, "audit"),
    }
