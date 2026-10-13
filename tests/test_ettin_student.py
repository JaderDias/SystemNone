"""Ettin's ModernBERT architecture must train and export with dynamic batch/sequence lengths."""

import numpy as np
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from system_none.config import TaskSpec, TrainSpec
from system_none.data import Example
from system_none.export import export_onnx
from system_none.runtime import Classifier
from system_none.train import train_student

from conftest import LABELS, build_tiny_bert, keyword_probs, synthetic_texts


def test_ettin_architecture_trains_and_exports(tmp_path):
    texts = synthetic_texts(60, seed=3)
    base = build_tiny_bert(tmp_path / "ettin", texts)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(base)
    cfg = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=16,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
    )
    ModernBertModel(cfg).save_pretrained(base)
    examples = [
        Example(id=str(i), text=t, probs=np.array(keyword_probs(t), dtype=np.float32)) for i, t in enumerate(texts)
    ]
    result = train_student(
        base,
        TaskSpec(name="intent", question="Choose", labels=dict.fromkeys(LABELS, "")),
        examples[:40],
        examples[40:50],
        TrainSpec(epochs=1, min_steps=0, batch_size=8, max_length=64, device="cpu"),
    )
    out = export_onnx(result.model, result.tokenizer, LABELS, tmp_path / "onnx", 64, 1.0, texts[:8])
    clf = Classifier(out)
    for batch in (["refund"], ["refund please", "cancel", "where is my delivery today please"]):
        actual = clf.predict_proba(batch)
        assert actual.shape == (len(batch), len(LABELS))
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-5)
