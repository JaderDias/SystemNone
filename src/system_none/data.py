"""Turn judged logs into train / validation / test splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .shadow import read_log


@dataclass
class Example:
    id: str
    text: str
    probs: np.ndarray  # judge's soft label, sums to 1
    lang: str = "und"
    meta: dict = field(default_factory=dict)
    # The language the gateway's router is sure the text is in (an ISO 639-3 code), for training a
    # student per language; None when it isn't sure, or the prompt trains one student.
    route: str | None = None

    @property
    def label(self) -> int:
        return int(np.argmax(self.probs))


@lru_cache(maxsize=1)
def _langid():
    import langid

    return langid


def detect_language(text: str) -> str:
    try:
        return _langid().classify(text)[0]
    except Exception:
        return "und"


def load_labeled(path: str | Path, num_labels: int, detect_lang: bool = True) -> list[Example]:
    out = []
    for r in read_log(path):
        probs = np.asarray(r["probs"], dtype=np.float32)
        if probs.shape != (num_labels,) or not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        probs = probs / probs.sum()
        lang = r.get("lang") or (detect_language(r["text"]) if detect_lang else "und")
        out.append(Example(id=r["id"], text=r["text"], probs=probs, lang=lang))
    return out


def _bucket(example_id: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{example_id}".encode(), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split(examples: list[Example], val_fraction: float, test_fraction: float, seed: int = 13):
    """Deterministic split by id hash: re-running after more logs keeps old examples where they were."""
    train, val, test = [], [], []
    for ex in examples:
        b = _bucket(ex.id, seed)
        if b < test_fraction:
            test.append(ex)
        elif b < test_fraction + val_fraction:
            val.append(ex)
        else:
            train.append(ex)
    return train, val, test


def label_distribution(examples: list[Example], num_labels: int) -> np.ndarray:
    counts = np.bincount([e.label for e in examples], minlength=num_labels)
    return counts / max(1, counts.sum())
