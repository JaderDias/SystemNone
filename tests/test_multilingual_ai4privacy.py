"""Multilingual end-to-end run on a real slice of ai4privacy/open-pii-masking-500k-ai4privacy.

3,928 short texts in English, Hindi and Telugu (see data/ai4privacy/README.md). The task: which
kind of personal data is the most sensitive item in the text (id_number, contact, address,
identity_only).

The students only ever see `inputs.jsonl` (text, no labels), exactly what shadow mode would log.
Labels live in `judge_key.jsonl`, which only the hardcoded judge reads; they reach training as
soft labels through the ordinary `system-none judge` step.

Sizes (SN_TEST_SIZE):
  medium (default)  2-layer BERTs trained from scratch here; about 2 minutes on CPU.
  big               pretrained microsoft/deberta-v3-small (English) and microsoft/mdeberta-v3-base
                    (multilingual) from the Hugging Face Hub; tens of minutes on CPU, minutes on a GPU.
                    SN_BIG_ENGLISH / SN_BIG_MULTILINGUAL override the checkpoints.
"""

import collections
import json
import os
from typing import Any

import numpy as np
import pytest
import yaml

from system_none.cli import main
from system_none.config import load_config
from system_none.data import load_labeled, split
from system_none.runtime import Classifier, LanguageRouter, load_deployment
from system_none.shadow import read_log

from conftest import build_tiny_bert, serve
from hardcoded_judge import INPUTS, KEY, PII_LABELS, PII_QUESTION, HardcodedJudge, load_inputs, text_hash

pytestmark = pytest.mark.slow

SIZE = os.environ.get("SN_TEST_SIZE", "medium")
if SIZE not in ("medium", "big"):
    raise ValueError(f"SN_TEST_SIZE must be medium or big, not {SIZE!r}")
# Held-out bars. Medium: tiny models from scratch reach 0.73-0.79 overall (always-majority: 0.40).
LIMITS = {
    "medium": {"overall": 0.65, "per_language": 0.55, "english": 0.65, "coverage": 0.2},
    "big": {"overall": 0.85, "per_language": 0.75, "english": 0.80, "coverage": 0.4},
}[SIZE]


@pytest.fixture(scope="module")
def pii_judge():
    judge = HardcodedJudge()
    with serve(judge.app()) as url:
        yield judge, url


@pytest.fixture(scope="module")
def pii_run(tmp_path_factory, pii_judge):
    _, url = pii_judge
    d = tmp_path_factory.mktemp("pii")
    rows = load_inputs()
    if SIZE == "big":
        english = os.environ.get("SN_BIG_ENGLISH", "microsoft/deberta-v3-small")
        multi = os.environ.get("SN_BIG_MULTILINGUAL", "microsoft/mdeberta-v3-base")
        # Usual fine-tuning settings for pretrained encoders; device "auto" picks a GPU when there is one.
        train = {
            "epochs": 3,
            "min_steps": 0,
            "lr": 3e-5,
            "batch_size": 16,
            "max_length": 96,
            "val_fraction": 0.1,
            "test_fraction": 0.2,
        }
    else:
        # Tokenizers are fitted on unlabeled text only, the way a pretrained vocabulary would be.
        english = build_tiny_bert(d / "tiny-en", [r["text"] for r in rows if r["lang"] == "en"], vocab_size=2000)
        multi = build_tiny_bert(d / "tiny-multilingual", [r["text"] for r in rows], vocab_size=2000)
        # Settings from a sweep on this slice: the best validation KL comes at epoch 3-6.
        train = {
            "epochs": 8,
            "min_steps": 0,
            "lr": 2e-3,
            "batch_size": 32,
            "max_length": 96,
            "val_fraction": 0.1,
            "test_fraction": 0.2,
            "device": "cpu",
        }
    cfg = {
        "project": "pii-kind",
        "workdir": str(d / "run"),
        "task": {"name": "pii_kind", "question": PII_QUESTION, "labels": PII_LABELS},
        "judge": {
            "base_url": url,
            "model": "hardcoded-judge",
            "mode": "logprobs",
            "concurrency": 32,
            "license": {"allows_training_on_outputs": True},
        },
        "students": {"english": [english], "multilingual": [multi]},
        "train": train,
        "cascade": {"target_accuracy": 0.9},
    }
    cfg_path = d / "system-none.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    main(["log", "-c", str(cfg_path), str(INPUTS), "--field", "text"])
    main(["judge", "-c", str(cfg_path)])
    main(["train", "-c", str(cfg_path)])
    return load_config(cfg_path), rows


def _truth():
    """Test-side reference: the key, used only to score results after the fact."""
    return {r["sha256"]: r["label"] for r in map(json.loads, KEY.read_text().splitlines())}


def _held_out(cfg):
    """The test split. Splits hash the example id, so English rows land in the same split in both groups."""
    examples = load_labeled(cfg.labels_path, len(PII_LABELS))
    return split(examples, cfg.train.val_fraction, cfg.train.test_fraction, cfg.train.seed)[2]


def test_fixture_keeps_labels_away_from_inputs():
    rows = load_inputs()
    assert len(rows) == 3928 and {r["lang"] for r in rows} == {"en", "hi", "te"}
    assert all(set(r) == {"uid", "lang", "text"} for r in rows)  # no label anywhere on the student side
    key = [json.loads(line) for line in KEY.read_text().splitlines()]
    assert all(set(r) == {"sha256", "label"} for r in key)  # and no text in the key
    assert {text_hash(r["text"]) for r in rows} == {r["sha256"] for r in key}
    assert {r["label"] for r in key} == set(PII_LABELS)


def test_labels_arrive_only_through_the_judge(pii_run, pii_judge):
    cfg, rows = pii_run
    judge, _ = pii_judge
    assert all(set(r) <= {"id", "text", "ts", "meta"} for r in read_log(cfg.shadow_log))  # inputs only
    labeled = read_log(cfg.labels_path)
    assert len(labeled) == len(rows) and judge.calls == len(rows) and judge.misses == 0
    truth, names = _truth(), list(PII_LABELS)
    agree = [names[int(np.argmax(r["probs"]))] == truth[text_hash(r["text"])] for r in labeled]
    assert all(agree)
    assert all(abs(max(r["probs"]) - 0.9) < 1e-3 for r in labeled)  # the soft label survived the logprobs path


def test_groups_split_by_language(pii_run):
    cfg, rows = pii_run
    deploy = json.loads((cfg.workdir / "deploy.json").read_text())
    assert set(deploy["groups"]) == {"english", "multilingual"}
    examples = load_labeled(cfg.labels_path, len(PII_LABELS))
    detected = {e.text: e.lang for e in examples}
    for lang in ("hi", "te"):  # langid sees the script, not the Latin-script names inside the text
        texts = [r["text"] for r in rows if r["lang"] == lang]
        non_english = sum(detected[t] != "en" for t in texts) / len(texts)
        assert non_english >= 0.95, (lang, non_english)  # some Hindi reads as Marathi: same script, same route
    english_group = sum(e.lang == "en" for e in examples)
    assert 0.9 * sum(r["lang"] == "en" for r in rows) <= english_group <= len(rows)


def test_router_sends_hindi_and_telugu_to_the_multilingual_student(pii_run):
    cfg, rows = pii_run
    deploy = json.loads((cfg.workdir / "deploy.json").read_text())
    english_route = deploy["routes"]["en"]  # whichever student scored better on held-out English
    pipe = load_deployment(cfg.workdir)
    assert isinstance(pipe, LanguageRouter)
    seen = []
    english, multilingual = pipe.routes["en"], pipe.default
    pipe.routes["en"] = lambda t: seen.append(english_route) or english(t)
    pipe.default = lambda t: seen.append("multilingual") or multilingual(t)
    by_lang = collections.defaultdict(collections.Counter)
    for r in rows:
        pipe(r["text"])
        by_lang[r["lang"]][seen[-1]] += 1
    for lang in ("hi", "te"):
        assert by_lang[lang]["multilingual"] / sum(by_lang[lang].values()) >= 0.95, by_lang
    assert by_lang["en"][english_route] / sum(by_lang["en"].values()) >= 0.95, by_lang


def test_multilingual_vocabulary_covers_the_scripts(pii_run):
    """Why the multilingual group exists: an English vocabulary turns Devanagari and Telugu into [UNK]."""
    from tokenizers import Tokenizer

    if SIZE == "big":
        pytest.skip("pretrained vocabularies cover these scripts; this checks the tiny tokenizers")
    cfg, rows = pii_run

    def unk_share(model_dir, lang):
        tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        ids = [i for r in rows if r["lang"] == lang for i in tok.encode(r["text"]).ids]
        return np.mean([i == tok.token_to_id("[UNK]") for i in ids])

    for lang in ("hi", "te"):
        assert unk_share(cfg.students.english[0], lang) > 0.2  # the Latin-script names inside still tokenize
        assert unk_share(cfg.students.multilingual[0], lang) == 0


def test_students_learn_every_language(pii_run):
    """Held-out accuracy against the hidden key, per language, against the bars in LIMITS."""
    cfg, _ = pii_run
    deploy = json.loads((cfg.workdir / "deploy.json").read_text())
    truth, names = _truth(), list(PII_LABELS)
    test = _held_out(cfg)
    gold = np.array([names.index(truth[text_hash(e.text)]) for e in test])
    majority = np.bincount(gold, minlength=len(names)).max() / len(gold)

    multi = Classifier(cfg.workdir / deploy["groups"]["multilingual"]["model_dir"])
    pred = multi.predict_proba([e.text for e in test]).argmax(-1)
    overall = float((pred == gold).mean())
    report: dict[str, Any] = {
        "size": SIZE,
        "overall": round(overall, 3),
        "majority": round(float(majority), 3),
        "n": len(test),
    }
    for lang in ("en", "hi", "te"):
        m = np.array([e.lang == lang for e in test])
        report[lang] = round(float((pred[m] == gold[m]).mean()), 3)
        assert m.sum() >= 80, report
    print(report)
    assert overall >= LIMITS["overall"] and overall >= majority + 0.2, report
    assert min(report["en"], report["hi"], report["te"]) >= LIMITS["per_language"], report
    # judge agreement is accuracy against the key, because this judge reads the key
    assert deploy["groups"]["multilingual"]["test"]["accuracy"] == pytest.approx(overall, abs=1e-3)

    # English goes to whichever student did better on held-out English, and that choice holds up
    m = np.array([e.lang == "en" for e in test])
    en_texts = [e.text for e, keep in zip(test, m, strict=True) if keep]
    en_acc = {}
    for group in ("english", "multilingual"):
        clf = Classifier(cfg.workdir / deploy["groups"][group]["model_dir"])
        en_acc[group] = float((clf.predict_proba(en_texts).argmax(-1) == gold[m]).mean())
    routing = deploy["routing"]["en"]
    print({"english_route": routing, **{k: round(v, 3) for k, v in en_acc.items()}})
    assert en_acc[routing["chosen"]] >= max(en_acc.values()) - cfg.train.accuracy_tolerance - 1e-9
    assert en_acc[routing["chosen"]] >= LIMITS["english"]


def test_cascade_serves_confident_answers_and_holds_quality(pii_run, pii_judge):
    """Confident answers come from the students; the rest go to today's model (here: the judge)."""
    cfg, _ = pii_run
    judge, _ = pii_judge
    names = list(PII_LABELS)
    incumbent = lambda text: names[int(np.argmax(judge.probs(text)))]  # noqa: E731  (today's model)
    pipe = load_deployment(cfg.workdir, fallback=incumbent)
    truth = _truth()
    decisions = [(pipe(e.text), truth[text_hash(e.text)]) for e in _held_out(cfg)]
    served = [d.label == gold for d, gold in decisions if d.source == "student"]
    overall = float(np.mean([d.label == gold for d, gold in decisions]))
    coverage = len(served) / len(decisions)
    report = {
        "size": SIZE,
        "overall": round(overall, 3),
        "coverage": round(coverage, 3),
        "served_accuracy": round(float(np.mean(served)), 3) if served else None,
    }
    print(report)
    assert overall >= cfg.cascade.target_accuracy - 0.05, report
    assert coverage >= LIMITS["coverage"], report  # the student takes a real share of traffic
    # The threshold is chosen on the validation split, which early stopping also used, so it runs
    # optimistic: 0.9 on validation came out as 0.83-0.86 on held-out text in our runs.
    assert np.mean(served) >= cfg.cascade.target_accuracy - 0.1, report
    deploy = json.loads((cfg.workdir / "deploy.json").read_text())
    for g in deploy["groups"].values():
        c = g["cascade"]
        assert c["coverage"] > 0 and c["accuracy"] >= c["target_accuracy"], c


def test_router_decisions_on_held_out_text(pii_run):
    cfg, _ = pii_run
    pipe = load_deployment(cfg.workdir)
    d = pipe("మా ప్రయాణంలో ఫోన్ +91 98480 22338 ద్వారా సంప్రదించండి, లేదా XY@gmail.com కు ఇమెయిల్ పంపండి.")
    assert d.label in PII_LABELS and 0 < d.confidence <= 1 and len(d.probs) == len(PII_LABELS)
