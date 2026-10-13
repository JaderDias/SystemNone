"""Shadow mode: copy the *inputs* of live traffic into a log.

The model you run today keeps answering users. Its outputs are never written here,
so they can never become training data (some licenses forbid exactly that).
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path


def text_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class ShadowLogger:
    """Append-only JSONL log of request inputs, with sampling and de-duplication."""

    def __init__(
        self,
        path: str | Path,
        sample_rate: float = 1.0,
        max_chars: int = 8000,
        redact: Callable[[str], str] | None = None,
        seed: int | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.max_chars = max_chars
        self.redact = redact
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self._seen.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def log(self, text: str, **meta) -> bool:
        """Record one input. Returns True if it was written."""
        if not isinstance(text, str) or not text.strip():
            return False
        if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
            return False
        if self.redact is not None:
            text = self.redact(text)
        text = text[: self.max_chars]
        tid = text_id(text)
        with self._lock:
            if tid in self._seen:
                return False
            self._seen.add(tid)
            rec = {"id": tid, "text": text, "ts": time.time()}
            if meta:
                rec["meta"] = meta
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True

    def __len__(self) -> int:
        return len(self._seen)


def shadow(logger: ShadowLogger, text_arg: int | str = 0):
    """Decorator: log the text input of an existing classify function, then call it unchanged.

    Only the input is logged; the wrapped function's return value is passed through untouched.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            text = (
                kwargs.get(text_arg)
                if isinstance(text_arg, str)
                else (args[text_arg] if len(args) > text_arg else None)
            )
            try:
                if text is not None:
                    logger.log(text)
            except Exception:  # logging must never break serving
                pass
            return fn(*args, **kwargs)

        return wrapper

    return deco


def read_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
