"""Students on a frozen sentence encoder (EmbeddingGemma's recipe): only a linear head trains.

Tiny random-weight encoders stand in for the real checkpoints: a BERT, and a Gemma 3 with
bidirectional attention and sliding windows, the architecture of EmbeddingGemma, with its mean
pooling, two dense layers, normalization and classification prompt.
"""

import json
import shutil

import numpy as np
import pytest

from system_none import worker
from system_none.calibrate import softmax
from system_none.config import load_config
from system_none.embed import EmbeddingStudent, PrefixTokenizer, is_embedding_model
from system_none.pipeline import _slug, train_all
from system_none.runtime import Classifier
from system_none.shadow import text_id

from conftest import LABELS, keyword_probs, synthetic_texts, write_config

PROMPT = "task: classification | query: "


def build_tiny_embedder(d, bert_dir, arch):
    import torch

    # sentence-transformers 6 moved `models` and keeps the old name as an alias only at run time.
    from sentence_transformers import SentenceTransformer, models  # ty: ignore[unresolved-import]

    enc = d / "encoder"
    shutil.copytree(bert_dir, enc)
    if arch == "gemma3":
        from transformers import Gemma3TextConfig, Gemma3TextModel

        vocab = json.loads((enc / "config.json").read_text())["vocab_size"]
        for f in [*enc.glob("model*.safetensors"), enc / "config.json"]:
            f.unlink()
        torch.manual_seed(0)
        cfg = Gemma3TextConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            query_pre_attn_scalar=16,
            max_position_embeddings=128,
            sliding_window=8,
            use_bidirectional_attention=True,
            layer_types=["sliding_attention", "full_attention"],
            pad_token_id=0,
        )
        gemma = Gemma3TextModel(cfg)
        # Gemma normalizes each sublayer's output before adding it to the residual stream, so random
        # sublayers would drown the tokens. Scale them to 0.01, as quiet as a trained model's.
        # (The ty ignores: torch types a submodule's attributes as Tensor | Module.)
        with torch.no_grad():
            for layer in gemma.layers:
                layer.post_attention_layernorm.weight.fill_(-0.99)  # ty: ignore[call-non-callable, unresolved-attribute]
                layer.post_feedforward_layernorm.weight.fill_(-0.99)  # ty: ignore[call-non-callable, unresolved-attribute]
        gemma.save_pretrained(enc)
    word = models.Transformer(str(enc), max_seq_length=64)
    dim = word.get_word_embedding_dimension()
    modules = [word, models.Pooling(dim, "mean")]
    if arch == "gemma3":  # EmbeddingGemma: 768 -> 3072 -> 768, no bias, no activation
        modules += [
            models.Dense(dim, 4 * dim, bias=False, activation_function=torch.nn.Identity()),
            models.Dense(4 * dim, dim, bias=False, activation_function=torch.nn.Identity()),
        ]
    SentenceTransformer(modules=[*modules, models.Normalize()], prompts={"Classification": PROMPT}).save(str(d / "st"))
    return str(d / "st")


@pytest.fixture(scope="module", params=["bert", "gemma3"])
def trained_head(request, tmp_path_factory, tiny_model):
    d = tmp_path_factory.mktemp(f"embed-{request.param}")
    model = build_tiny_embedder(d, tiny_model, request.param)
    cfg = load_config(write_config(d / "system-none.yaml", d / "run", model))
    cfg.labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.labels_path, "w") as f:
        for t in synthetic_texts(600, seed=7):
            f.write(json.dumps({"id": text_id(t), "text": t, "probs": keyword_probs(t), "lang": "en"}) + "\n")
    embedded = []
    deploy = train_all(cfg, log=lambda *a: None, on_progress=lambda done, total: embedded.append((done, total)))
    return model, cfg, deploy, embedded


def test_only_a_head_is_trained_and_it_serves(trained_head, tiny_model):
    import torch
    from sentence_transformers import SentenceTransformer

    model, cfg, deploy, embedded = trained_head
    assert is_embedding_model(model) and not is_embedding_model(tiny_model)
    g = deploy["groups"]["english"]
    assert g["test"]["accuracy"] >= 0.7, g  # random-weight encoders, 5 labels: chance is 0.2
    assert 0 < g["threshold"] <= 1 and g["cascade"]["accuracy"] >= 0.9
    # Progress counts the training and validation texts embedded, batch by batch.
    done, totals = [d for d, _ in embedded], {t for _, t in embedded}
    assert len(totals) == 1 and done == sorted(set(done)) and done[-1] == totals.pop() == 600 - g["test"]["n"]

    onnx_dir = cfg.workdir / g["model_dir"]
    manifest = json.loads((onnx_dir / "manifest.json").read_text())
    assert manifest["prefix"] == PROMPT and manifest["inputs"] == ["input_ids", "attention_mask"]

    # The ONNX runtime, which puts the prompt in front, matches the PyTorch encoder + head.
    hf = cfg.models_dir / "english" / _slug(model) / "hf"
    student = EmbeddingStudent(SentenceTransformer(str(hf), device="cpu"), len(LABELS)).eval()
    student.head.load_state_dict(torch.load(hf / "head.pt", weights_only=True))
    texts = synthetic_texts(20, seed=3)
    enc = PrefixTokenizer(student.encoder.tokenizer, PROMPT)(
        texts, truncation=True, max_length=64, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        ref = softmax(student(**enc).logits.numpy(), manifest["temperature"])
    clf = Classifier(onnx_dir)
    assert np.allclose(clf.predict_proba(texts), ref, atol=1e-4)
    clf.prefix = ""
    assert not np.allclose(clf.predict_proba(texts), ref, atol=1e-4), "the prompt changes the embedding"


def test_auto_base_model():
    assert worker.choose_base_model("my/model", 3) == "my/model"
    for labels in (3, 30):
        assert worker.choose_base_model("auto", labels) == "jhu-clsp/ettin-encoder-17m"
