"""Offline fixtures: a tiny BERT built locally, synthetic support traffic, and a fake judge endpoint."""

from __future__ import annotations

import collections
import contextlib
import json
import math
import random
import re
import socket
import threading
import time
from typing import Any

import httpx
import pytest
from fastapi import Request  # module level: annotations are resolved from module globals

LABELS = ["refund", "cancel", "shipping", "login", "other"]
KEYWORDS = {
    "refund": ["refund", "money back", "charged twice", "reimburse me"],
    "cancel": ["cancel", "stop the order", "unsubscribe", "terminate my plan"],
    "shipping": ["where is my package", "tracking number", "delivery date", "not shipped"],
    "login": ["password", "sign in", "log in", "locked out"],
    "other": ["opening hours", "gift card", "store location", "feedback"],
}
FILLER = "hi hello please thanks today again quick question about my account order item app yesterday urgently".split()


def make_text(rng: random.Random, label: str, ambiguous_with: str | None = None) -> str:
    words = [*rng.sample(FILLER, 3), rng.choice(KEYWORDS[label])]
    if ambiguous_with:
        words.append(rng.choice(KEYWORDS[ambiguous_with]))
    words += rng.sample(FILLER, 2)
    rng.shuffle(words)
    return " ".join(words) + f" #{rng.randint(0, 10**6)}"


def keyword_probs(text: str) -> list[float]:
    """The fake judge: soft label from keyword hits, with a little uncertainty."""
    hits = [sum(kw in text for kw in KEYWORDS[label]) for label in LABELS]
    if sum(hits) == 0:
        hits[LABELS.index("other")] = 1
    raw = [0.03 + 0.9 * h / sum(hits) for h in hits]
    s = sum(raw)
    return [r / s for r in raw]


def synthetic_texts(n: int, seed: int = 0, ambiguous_share: float = 0.15) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lab = rng.choice(LABELS)
        amb = rng.choice([other for other in LABELS if other != lab]) if rng.random() < ambiguous_share else None
        out.append(make_text(rng, lab, amb))
    return out


def fake_completion(body: dict, flip: bool = False) -> dict:
    content = body["messages"][-1]["content"]
    match = re.search(r'"""\n(.*)\n"""', content, re.S)
    assert match, content
    text = match.group(1)
    probs = keyword_probs(text)
    if flip:  # simulate drift: the world changed and the judge now disagrees
        probs = probs[1:] + probs[:1]
    if body.get("logprobs"):
        top = []
        for i, p in enumerate(probs):
            top.append({"token": "ABCDE"[i], "logprob": math.log(p * 0.9)})
            top.append({"token": " " + "ABCDE"[i], "logprob": math.log(p * 0.1)})
        top.append({"token": "The", "logprob": math.log(1e-4)})
        return {
            "choices": [
                {
                    "message": {"content": "ABCDE"[max(range(5), key=probs.__getitem__)]},
                    "logprobs": {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": top}]},
                }
            ]
        }
    n = body.get("n", 1)
    rng = random.Random(text)
    letters = rng.choices("ABCDE", weights=probs, k=n)
    return {"choices": [{"message": {"content": letter}} for letter in letters]}


@pytest.fixture
def mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fake_completion(json.loads(request.content)))

    return httpx.MockTransport(handler)


@contextlib.contextmanager
def serve(app):
    """Run a FastAPI app on a free local port in a background thread; yields its base URL."""
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True


@pytest.fixture(scope="session")
def judge_server():
    """A real HTTP server speaking the OpenAI chat-completions shape (for CLI tests)."""
    from fastapi import FastAPI

    app = FastAPI()
    state: dict[str, Any] = {"flip": False, "calls": 0}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        state["calls"] += 1
        return fake_completion(await request.json(), flip=state["flip"])

    with serve(app) as url:
        yield {"url": url, "state": state}


def wordpiece_vocab(corpus, vocab_size, normalizer, pre_tokenizer, specials):
    """A deterministic WordPiece vocabulary: every character (word-initial and ##continuation), then the
    most frequent words and ##suffixes. (The `tokenizers` trainer breaks ties differently on each run.)"""
    words = collections.Counter()
    for text in corpus:
        words.update(w for w, _ in pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text)))
    chars, pieces = collections.Counter(), collections.Counter()
    for w, n in words.items():
        chars[w[0]] += n
        for i in range(1, len(w)):
            chars["##" + w[i]] += n
            if 2 <= len(w) - i <= 4:
                pieces["##" + w[i:]] += n
    vocab = list(specials) + sorted(chars, key=lambda t: (-chars[t], t))
    extra = [(n, t) for t, n in words.items() if len(t) > 1] + [(n, t) for t, n in pieces.items()]
    for _, t in sorted(extra, key=lambda x: (-x[0], x[1])):
        if len(vocab) >= vocab_size:
            break
        vocab.append(t)
    return {t: i for i, t in enumerate(dict.fromkeys(vocab))}


def build_tiny_bert(d, corpus, vocab_size=600, max_positions=128, arch="bert"):
    """A 2-layer BERT (or DeBERTa-v2, the architecture of DeBERTa-v3 and mDeBERTa-v3) with a WordPiece
    tokenizer built from `corpus`, saved like a Hub checkpoint.

    Accents are kept (strip_accents=False): stripping them would delete Devanagari and Telugu vowel signs.
    Tokenizer and weights are deterministic, so test scores repeat from run to run.
    """
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=False)
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    vocab = wordpiece_vocab(corpus, vocab_size, normalizer, pre_tokenizer, specials)
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer, tok.pre_tokenizer = normalizer, pre_tokenizer
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    )
    fast.save_pretrained(d)
    torch.manual_seed(0)  # the same "pretrained" weights on every run
    size: dict[str, Any] = dict(
        vocab_size=tok.get_vocab_size(),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=max_positions,
    )
    if arch == "deberta-v2":  # DeBERTa-v3 settings: relative attention, no token types, no absolute positions
        cfg = DebertaV2Config(
            **size,
            type_vocab_size=0,
            relative_attention=True,
            # Not a field of transformers 5's config class, which keeps extra keywords as attributes;
            # the model reads it from there.
            position_buckets=32,  # ty: ignore[unknown-argument]
            pos_att_type=["p2c", "c2p"],
            max_relative_positions=-1,
            position_biased_input=False,
        )
        DebertaV2Model(cfg).save_pretrained(d)
    else:
        BertModel(BertConfig(**size)).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return build_tiny_bert(tmp_path_factory.mktemp("tiny-bert"), synthetic_texts(2000, seed=99))


def write_config(
    path, workdir, model_path, judge_url="http://unused/v1", mode="logprobs", ack=True, multilingual=None, target=0.9
):
    import yaml

    cfg = {
        "project": "test",
        "workdir": str(workdir),
        "task": {"name": "intent", "question": "What does the customer want?", "labels": dict.fromkeys(LABELS, "")},
        "judge": {
            "base_url": judge_url,
            "model": "fake-judge",
            "mode": mode,
            "concurrency": 4,
            "samples": 7,
            "license": {"allows_training_on_outputs": ack},
        },
        "students": {"english": [model_path], "multilingual": multilingual or []},
        "train": {
            "epochs": 12,
            "min_steps": 0,
            "lr": 2e-3,
            "batch_size": 16,
            "max_length": 64,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "device": "cpu",
        },
        "cascade": {"target_accuracy": target},
        "monitor": {"sample_rate": 1.0, "window": 200, "min_samples": 20, "tolerance": 0.05},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
