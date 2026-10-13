"""A hardcoded judge for the ai4privacy fixture.

The fixture is split in two so the labels can only reach training through the judge, as in production:

  data/ai4privacy/inputs.jsonl     uid, lang, text        what shadow mode would log (no labels)
  data/ai4privacy/judge_key.jsonl  sha256(text), label    read only by judges (no text)

Only judges open the key: this module, and the mock judge (src/system_none/mock_judge.py). This one
serves an OpenAI-compatible /v1/chat/completions endpoint, finds the text inside the prompt, looks
up its label by hash and answers with option-letter logprobs, like a real judge would.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from fastapi import Request  # module level: annotations are resolved from module globals

DATA = Path(__file__).parent / "data" / "ai4privacy"
INPUTS = DATA / "inputs.jsonl"
KEY = DATA / "judge_key.jsonl"

PII_LABELS = {
    "id_number": "the most sensitive item is an identifier: passport, licence, tax, account, card or other ID number",
    "contact": "the most sensitive item is contact data: an email address or phone number",
    "address": "the most sensitive item is a location: street address, postcode, city or area",
    "identity_only": "only names or demographic details (age, gender, title), no number, contact or place",
}
PII_QUESTION = "Which kind of personal data is the most sensitive item in this text?"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_inputs() -> list[dict]:
    """The student-side view of the data: text and language, never a label."""
    return [json.loads(line) for line in INPUTS.read_text(encoding="utf-8").splitlines() if line.strip()]


class HardcodedJudge:
    """Answers from the key with fixed confidence; unknown texts get a uniform (maximally unsure) answer."""

    def __init__(self, labels=tuple(PII_LABELS), confidence: float = 0.9):
        self.labels = list(labels)
        self.confidence = confidence
        self.key = {r["sha256"]: r["label"] for r in map(json.loads, KEY.read_text().splitlines()) if r}
        self.calls = 0
        self.misses = 0

    def probs(self, text: str) -> list[float]:
        k = len(self.labels)
        label = self.key.get(text_hash(text))
        if label is None:
            self.misses += 1
            return [1 / k] * k
        rest = (1 - self.confidence) / (k - 1)
        return [self.confidence if name == label else rest for name in self.labels]

    def completion(self, body: dict) -> dict:
        self.calls += 1
        content = body["messages"][-1]["content"]
        match = re.search(r'"""\n(.*)\n"""', content, re.S)
        assert match, content
        text = match.group(1)
        probs = self.probs(text)
        best = LETTERS[max(range(len(probs)), key=probs.__getitem__)]
        if body.get("logprobs"):
            top = [{"token": LETTERS[i], "logprob": math.log(p)} for i, p in enumerate(probs)]
            return {
                "choices": [
                    {
                        "message": {"content": best},
                        "logprobs": {"content": [{"token": best, "logprob": 0.0, "top_logprobs": top}]},
                    }
                ]
            }
        return {"choices": [{"message": {"content": best}} for _ in range(body.get("n", 1))]}

    def app(self):
        from fastapi import FastAPI

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            return self.completion(await request.json())

        return app
