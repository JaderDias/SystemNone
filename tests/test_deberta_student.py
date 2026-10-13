"""DeBERTa-v3's architecture (DeBERTa-v2 in transformers) through fine-tuning, calibration and ONNX export.

The big run (make test-big) fine-tunes the real microsoft/mdeberta-v3-base; this tiny random-weight
version of the same architecture catches export problems on every push. DeBERTa-v3 ignores token-type
ids, so the exported graph (and the manifest the Rust gateway reads) must list only the inputs it keeps.
"""

import json

import numpy as np
import pytest

from system_none.config import TaskSpec, TrainSpec
from system_none.data import Example
from system_none.export import export_onnx
from system_none.runtime import Classifier
from system_none.train import train_student

from conftest import LABELS, build_tiny_bert, keyword_probs, synthetic_texts


@pytest.mark.parametrize("bad_loss", [False, True])
def test_half_precision_checkpoint_trains_in_float32(tmp_path, monkeypatch, bad_loss):
    import torch
    from transformers import AutoModel

    texts = synthetic_texts(16)
    base = build_tiny_bert(tmp_path / "half-deberta", texts, arch="deberta-v2")
    AutoModel.from_pretrained(base).half().save_pretrained(base)
    examples = [
        Example(id=str(i), text=t, probs=np.array(keyword_probs(t), dtype=np.float32)) for i, t in enumerate(texts)
    ]
    task = TaskSpec(name="intent", question="Classify", labels=dict.fromkeys(LABELS, ""))
    spec = TrainSpec(epochs=2, batch_size=4, max_length=32, device="cpu")
    if bad_loss:
        monkeypatch.setattr(torch.nn.functional, "kl_div", lambda *a, **kw: torch.tensor(float("nan")))
        with pytest.raises(RuntimeError, match=r"Nonfinite training loss.*epoch 1, step 1"):
            train_student(base, task, examples[:12], examples[12:], spec, log=lambda s: None)
        return
    result = train_student(base, task, examples[:12], examples[12:], spec, log=lambda s: None)
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in result.model.parameters())
    assert result.best_epoch > 0
    assert all(np.isfinite(row["train_kl"]) and np.isfinite(row["val_kl"]) for row in result.history)


def test_tiny_deberta_trains_and_exports(tmp_path):
    base = build_tiny_bert(tmp_path / "tiny-deberta", synthetic_texts(2000, seed=99), arch="deberta-v2")
    texts = synthetic_texts(480, seed=3)
    ex = [Example(id=str(i), text=t, probs=np.array(keyword_probs(t), dtype=np.float32)) for i, t in enumerate(texts)]
    task = TaskSpec(name="intent", question="What does the customer want?", labels=dict.fromkeys(LABELS, ""))
    steps = []
    res = train_student(
        base,
        task,
        ex[:360],
        ex[360:420],
        TrainSpec(epochs=8, lr=2e-3, batch_size=16, max_length=64, device="cpu"),
        log=lambda *a: None,
        on_progress=lambda done, total: steps.append((done, total)),
    )
    assert steps == [(i, 8 * 23) for i in range(1, 8 * 23 + 1)]  # every step of 8 epochs of 360 / 16 batches
    out = export_onnx(res.model, res.tokenizer, LABELS, tmp_path / "onnx", 64, 1.0, sample_texts=texts[:8])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"] == ["input_ids", "attention_mask"]
    assert manifest["export"]["max_abs_diff"] < 1e-3
    clf = Classifier(out)
    pred = clf.predict_proba([e.text for e in ex[420:]]).argmax(-1)
    assert (pred == np.array([e.label for e in ex[420:]])).mean() >= 0.8
