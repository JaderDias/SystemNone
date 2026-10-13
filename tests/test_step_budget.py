"""The step budget: small training sets get more epochs, so training takes at least `min_steps` steps."""

import pytest

from system_none.config import TrainSpec, config_from_dict
from system_none.pipeline import MAX_EPOCHS, plan_epochs


@pytest.mark.parametrize(
    "n_train, batch, epochs, min_steps, expected",
    [
        (8000, 16, 3, 1000, 3),  # 500 steps an epoch: 3 epochs already make 1,500 steps
        (800, 16, 3, 1000, 20),  # 50 steps an epoch
        (801, 16, 3, 1000, 20),  # 51 steps an epoch: 20 epochs make 1,020
        (1760, 32, 10, 1780, 33),  # a step budget at batch 32
        (800, 16, 3, 0, 3),  # off
        (800, 16, 30, 1000, 30),  # never fewer epochs than asked for
        (10, 16, 3, 1000, MAX_EPOCHS),  # a handful of examples: capped
    ],
)
def test_plan_epochs(n_train, batch, epochs, min_steps, expected):
    spec = TrainSpec(epochs=epochs, batch_size=batch, min_steps=min_steps)
    assert plan_epochs(spec, n_train) == expected


def test_defaults_and_validation():
    spec = TrainSpec()
    assert spec.min_steps == 1000 and spec.human_holdout == 1.0
    with pytest.raises(ValueError, match="min_steps"):
        TrainSpec(min_steps=-1)
    with pytest.raises(ValueError, match="human_holdout"):
        TrainSpec(human_holdout=1.5)
    cfg = config_from_dict(
        {"task": {"name": "t", "question": "q", "labels": ["a", "b"]}, "train": {"min_steps": 0, "human_holdout": 0.25}}
    )
    assert (cfg.train.min_steps, cfg.train.human_holdout) == (0, 0.25)
    assert cfg.human_labels_path == cfg.workdir / "human_labels.jsonl"
