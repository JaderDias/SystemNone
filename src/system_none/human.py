"""Human labels: answers a person checked, to measure the judge and the students against people.

The judge sets the students' ceiling, and agreement with the judge can't show how good the judge
is. A few hundred inputs with checked answers can: they score the judge, every candidate student
and the cascade against people. By default they are all held out of training for that
(`train.human_holdout: 1.0`). With more of them, a share can train instead, with the human label
replacing the judge's, which lifts the ceiling on those inputs.

`human_labels.jsonl` holds one {"id", "text", "label"} per input, the label by name; a later line
for the same input replaces an earlier one. A line may also say how to use it, overriding
`human_holdout`: `"split": "train"` trains on it, `"split": "audit"` holds it out (a labeled
dataset's test split, say). The gateway audits the student against the held-out answers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .data import Example, _bucket, detect_language
from .shadow import ShadowLogger, read_log, text_id


def match_label(label, names: list[str]) -> str | None:
    """The task label a person wrote: exact, else ignoring case."""
    if not isinstance(label, str):
        return None
    if label in names:
        return label
    folded = {n.casefold(): n for n in names}
    return folded.get(label.strip().casefold())


SPLITS = ("train", "audit")


def read_human_labels(path: str | Path, names: list[str]) -> dict[str, dict]:
    """id -> {"text", "label": index, "split"}. Lines whose label the task no longer has are skipped."""
    out = {}
    for r in read_log(path):
        label = match_label(r.get("label"), names)
        if label is not None and r.get("text", "").strip():
            split = r.get("split") if r.get("split") in SPLITS else None
            out[r["id"]] = {"text": r["text"], "label": names.index(label), "split": split}
    return out


def add_human_labels(
    path: str | Path,
    rows: Iterable[tuple],
    names: list[str],
    shadow_log: str | Path | None = None,
    max_chars: int = 8000,
) -> dict:
    """Record (text, label) or (text, label, split) rows. Their texts also go to the shadow log, so
    the judge labels them too.

    Returns counts, and the labels the task doesn't have (those rows are not recorded).
    """
    path = Path(path)
    known = {i: (names[h["label"]], h["split"]) for i, h in read_human_labels(path, names).items()}
    logger = ShadowLogger(shadow_log, max_chars=max_chars) if shadow_log else None
    added = unchanged = 0
    unknown: dict[str, int] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            text, raw = row[0], row[1]
            split = row[2] if len(row) > 2 and row[2] in SPLITS else None
            if not isinstance(text, str) or not text.strip():
                continue
            text = text[:max_chars]  # as the shadow log stores it, so the ids match
            label = match_label(raw, names)
            if label is None:
                unknown[str(raw)] = unknown.get(str(raw), 0) + 1
                continue
            if logger is not None:
                logger.log(text)
            tid = text_id(text)
            if known.get(tid) == (label, split):
                unchanged += 1
                continue
            known[tid] = (label, split)
            rec = {"id": tid, "text": text, "label": label, **({"split": split} if split else {})}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1
    return {"added": added, "unchanged": unchanged, "total": len(known), "unknown_labels": unknown}


def is_held_out(example_id: str, holdout: float, seed: int, split: str | None = None) -> bool:
    """Deterministic, and independent of the train / validation / test split. A label's own
    `split` decides first."""
    if split in SPLITS:
        return split == "audit"
    return holdout >= 1 or (holdout > 0 and _bucket(f"human:{example_id}", seed) < holdout)


def apply_human_labels(
    examples: list[Example],
    human: dict[str, dict],
    num_labels: int,
    holdout: float,
    seed: int,
    detect_lang: bool = True,
) -> tuple[list[Example], list[Example]]:
    """Returns (examples to split and train on, examples held out to score against people).

    Human labels replace the judge's. Held-out examples keep the judge's soft label, if it gave
    one, in `meta["judge_probs"]`; their `probs` is the human label.
    """
    if not human:
        return examples, []
    eye = np.eye(num_labels, dtype=np.float32)
    train, held_out, seen = [], [], set()

    def place(ex: Example) -> None:
        (held_out if is_held_out(ex.id, holdout, seed, human[ex.id].get("split")) else train).append(ex)

    for e in examples:
        h = human.get(e.id)
        if h is None:
            train.append(e)
            continue
        seen.add(e.id)
        place(
            Example(
                id=e.id,
                text=e.text,
                probs=eye[h["label"]],
                lang=e.lang,
                meta={**e.meta, "human": True, "judge_probs": e.probs},
                route=e.route,
            )
        )
    for hid, h in human.items():  # labeled by people, not (yet) by the judge
        if hid not in seen:
            lang = detect_language(h["text"]) if detect_lang else "und"
            place(
                Example(
                    id=hid, text=h["text"], probs=eye[h["label"]], lang=lang, meta={"human": True, "judge_probs": None}
                )
            )
    return train, held_out


def judge_agreement(labels_path: str | Path, human_path: str | Path, names: list[str]) -> dict | None:
    """How often the judge's top answer matches the human label, on the inputs both labeled."""
    human = read_human_labels(human_path, names)
    if not human:
        return None
    pairs = []
    for r in read_log(labels_path):
        h = human.get(r["id"])
        probs = r.get("probs")
        if h is not None and probs and len(probs) == len(names):
            pairs.append((int(np.argmax(probs)), h["label"]))
    if not pairs:
        return {"n": 0, "accuracy": None, "human_labels": len(human)}
    return {
        "n": len(pairs),
        "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 4),
        "human_labels": len(human),
    }


def describe(scores: dict | None) -> str:
    """ "; right 0.912 against 300 human labels (the judge: 0.930)", or "" without human labels."""
    if not scores:
        return ""
    judge = f" (the judge: {scores['judge_accuracy']:.3f})" if scores.get("judge_accuracy") is not None else ""
    return f"; right {scores['accuracy']:.3f} against {scores['n']} human labels{judge}"


def add_store_labels(
    store,
    prefix: str,
    human_path: str | Path,
    names: list[str],
    shadow_log: str | Path | None = None,
    max_chars: int = 8000,
) -> dict:
    """The gateway worker's way in: `{"text", "label"}` lines under `prefix` in the shared store."""
    rows = []
    for key in store.list(prefix):
        if key.endswith(".jsonl"):
            for line in store.get(key).decode("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append((r.get("text", r.get("input")), r.get("label"), r.get("split")))
    return add_human_labels(human_path, rows, names, shadow_log=shadow_log, max_chars=max_chars)
