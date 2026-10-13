"""`system-none banking77` offline: synthetic rows with human labels stand in for the download."""

import json

import numpy as np
import pytest

from system_none import banking77
from system_none.config import JudgeSpec, LicenseAck, TaskSpec
from system_none.judge import Judge

from conftest import LABELS, keyword_probs, synthetic_texts


def fake_load(split):
    texts = synthetic_texts(500 if split == "train" else 200, seed=1 if split == "train" else 2)
    return [(t, LABELS[int(np.argmax(keyword_probs(t)))]) for t in texts]


def test_trains_on_human_labels_and_scores_the_test_split(monkeypatch, tmp_path, tiny_model):
    monkeypatch.setattr(banking77, "load", fake_load)
    res = banking77.run(
        tiny_model,
        tmp_path / "run",
        {
            "epochs": 12,
            "min_steps": 0,
            "lr": 2e-3,
            "batch_size": 16,
            "max_length": 64,
            "device": "cpu",
            "val_fraction": 0.15,
            "test_fraction": 0.1,
        },
        log=lambda *a: None,
    )
    assert res["train_rows"] == 500 and res["test"]["n"] == 200
    assert res["test"]["accuracy"] >= 0.8, res
    assert 0 < res["cascade"]["threshold"] <= 1 and 0 < res["cascade"]["student_answers"] <= 1
    assert json.loads((tmp_path / "run" / "banking77.json").read_text()) == res


def test_more_than_26_labels_are_judged_by_name():
    task = TaskSpec("intents", "Which intent?", {f"intent_{i}": "" for i in range(77)})
    task.validate()
    ack = LicenseAck(allows_training_on_outputs=True)
    assert Judge(JudgeSpec(model="open-model", license=ack), task).answer == "name"  # letters stop at Z
    with pytest.raises(ValueError, match="up to 26 labels"):
        Judge(JudgeSpec(model="open-model", answer="letter", license=ack), task)
