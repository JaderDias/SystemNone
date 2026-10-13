"""Judging tasks with more than 26 labels: the judge answers with the label's name."""

import json
import math
import re

import httpx
import numpy as np
import pytest

from system_none.config import JudgeSpec, LicenseAck, TaskSpec, config_from_dict
from system_none.judge import Judge, answer_probability, build_messages, label_from_text, label_log, probs_from_answer
from system_none.shadow import ShadowLogger, read_log

NAMES = [f"intent_{i:02d}" for i in range(30)] + ["card_arrival", "card", "Sci/Tech"]
TASK = TaskSpec("intents", "Which intent?", dict.fromkeys(NAMES, ""))


def spec(**kw):
    return JudgeSpec(
        base_url="http://fake/v1", model="fake-judge", license=LicenseAck(allows_training_on_outputs=True), **kw
    )


def gold(text: str) -> str:
    """The fake judge's answer: the text names its intent."""
    match = re.search(r"intent_\d\d", text)
    assert match, text
    return match.group(0)


def name_transport(p: float = 0.8, answers=None):
    """An OpenAI-compatible judge that answers with a label name, and its logprobs when asked."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        match = re.search(r'"""\n(.*)\n"""', body["messages"][-1]["content"], re.S)
        assert match, body
        name = gold(match.group(1))
        if body.get("logprobs"):  # "intent", "_", "07", then a full stop that must not count
            tokens = [
                {"token": "intent", "logprob": math.log(p)},
                {"token": "_", "logprob": 0.0},
                {"token": name[-2:], "logprob": 0.0},
                {"token": ".", "logprob": math.log(0.5)},
            ]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": name + "."}, "logprobs": {"content": tokens}}]}
            )
        outs = answers or [name, f"**{name}**", "Answer: " + name, "not sure", "intent_00"]
        return httpx.Response(200, json={"choices": [{"message": {"content": c}} for c in outs[: body.get("n", 1)]]})

    return httpx.MockTransport(handler), seen


def test_answer_style_follows_the_number_of_labels():
    few = TaskSpec("few", "Which?", {"a": "", "b": ""})
    assert Judge(spec(), few).answer == "letter"
    assert Judge(spec(answer="name"), few).answer == "name"
    assert Judge(spec(), TASK).answer == "name"
    with pytest.raises(ValueError, match="up to 26 labels"):
        Judge(spec(answer="letter"), TASK)
    with pytest.raises(ValueError, match="answer must be"):
        spec(answer="number")


def test_name_prompt_lists_the_labels_and_keeps_the_text_block():
    msgs = build_messages(TaskSpec("t", "Which?", {"refund": "wants money back", "other": ""}), "hello", "name")
    user = msgs[-1]["content"]
    assert "- refund: wants money back\n- other" in user and '"""\nhello\n"""' in user
    assert "name of the best option" in user and "letter" not in user
    assert "A) refund" in build_messages(TaskSpec("t", "Which?", {"refund": "", "other": ""}), "hello")[-1]["content"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("intent_07", "intent_07"),
        ("INTENT_07", "intent_07"),
        ("intent 07", "intent_07"),  # spaces for underscores
        ("`intent_07`.", "intent_07"),
        ("**intent_07**", "intent_07"),
        ("Answer: intent_07", "intent_07"),
        ("The answer is intent_07", "intent_07"),
        ("intent_07\nbecause the customer asks about it", "intent_07"),
        ("card_arrival", "card_arrival"),  # not also "card"
        ("It is about card arrival.", "card_arrival"),
        ("card", "card"),
        ("Sci/Tech", "Sci/Tech"),
        ("either intent_07 or intent_08", None),  # two labels: no guess
        ("no idea", None),
        ("", None),
    ],
)
def test_label_from_text(answer, expected):
    idx = label_from_text(answer, NAMES)
    assert (NAMES[idx] if idx is not None else None) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("refund", "refund"),
        ("**Refund**", "Refund"),
        ("REFUND", None),  # "Refund" or "refund"?
        ("card arrival", "card arrival"),
        ("It is a card-arrival question", None),
        ("_", "_"),
        ("", None),
    ],
)
def test_labels_that_differ_only_in_case_or_separators_need_their_exact_name(answer, expected):
    """The same cases as `parses_names_like_the_python_judge` in gateway/src/prompt.rs."""
    names = ["Refund", "refund", "card_arrival", "card arrival", "_"]
    idx = label_from_text(answer, names)
    assert (names[idx] if idx is not None else None) == expected


def test_answer_probability_stops_at_the_end_of_the_name():
    tokens = [
        {"token": "card", "logprob": math.log(0.5)},
        {"token": "_arrival", "logprob": math.log(0.8)},
        {"token": ".", "logprob": math.log(0.1)},
    ]
    assert answer_probability(tokens, "card_arrival") == pytest.approx(0.4)
    assert answer_probability(tokens, "card") == pytest.approx(0.5)


def test_probs_from_answer():
    p = probs_from_answer(4, 2, 0.7)
    assert p == pytest.approx([0.1, 0.1, 0.7, 0.1]) and sum(p) == pytest.approx(1)
    low = probs_from_answer(4, 0, 0.01)  # a very unsure answer still keeps the most mass
    assert np.argmax(low) == 0 and low[0] > low[1] and sum(low) == pytest.approx(1)


@pytest.mark.parametrize("mode", ["logprobs", "sample"])
def test_judge_labels_more_than_26_labels(tmp_path, mode):
    transport, seen = name_transport(p=0.8)
    judge = Judge(spec(mode=mode, samples=5), TASK, transport=transport)
    log = ShadowLogger(tmp_path / "s.jsonl")
    for i in range(1, 13):
        log.log(f"message {i} about intent_{i:02d}")
    stats = label_log(judge, tmp_path / "s.jsonl", tmp_path / "l.jsonl", progress=False)
    assert stats == {"labeled": 12, "failed": 0, "total": 12}
    for rec in read_log(tmp_path / "l.jsonl"):
        p = np.array(rec["probs"])
        assert len(p) == len(NAMES) and abs(p.sum() - 1) < 1e-4
        assert NAMES[int(p.argmax())] == gold(rec["text"])
        if mode == "logprobs":  # the answer's own probability; the full stop after it doesn't count
            assert p.max() == pytest.approx(0.8, abs=1e-4) and rec["letter_mass"] == pytest.approx(0.8)
        else:  # 3 of the 5 samples name the right label, 1 names another, 1 names none
            assert p.max() == pytest.approx(0.75) and rec["letter_mass"] == pytest.approx(0.8)
    assert all(b["max_tokens"] >= len("card_arrival") for b in seen)


def test_a_name_judge_without_logprobs_says_what_to_do():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "card"}}]}))
    res = Judge(spec(), TASK, transport=transport).label_many(["about a card"])[0]
    assert res["probs"] is None and "judge.mode: sample" in res["error"]


def test_answer_is_read_from_the_config():
    cfg = config_from_dict(
        {
            "task": {"name": "t", "question": "q", "labels": NAMES},
            "judge": {"model": "m", "answer": "name", "license": {"allows_training_on_outputs": True}},
        }
    )
    assert cfg.judge.answer == "name"
