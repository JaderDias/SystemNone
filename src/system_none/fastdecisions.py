"""Benchmark data: Fastino's Fast Decisions suite (Apache-2.0, 17 domains, public dev split of 100 rows each).

The published split is the *development* split; Fastino's reported numbers use a held-out
test split. Report scores on these files as dev-split results, not as the benchmark.
"""

from __future__ import annotations

DOMAINS = [
    "support_intent",
    "support_topic",
    "document_type",
    "review_sentiment",
    "agent_handoff",
    "email_triage",
    "ticket_route",
    "product_feedback",
    "banking_intent",
    "clinic_request",
    "travel_request",
    "news_topic",
    "paper_field",
    "sports_recap",
    "restaurant_review",
    "benefits_request",
    "screen_tags",
]


def load_heads(domains: list[str] | None = None) -> dict[str, dict]:
    """Return {"<domain>/<head>": {"labels": [...], "texts": [...], "gold": [...]}} for single-label heads."""
    from datasets import load_dataset

    out: dict[str, dict] = {}
    for domain in domains or DOMAINS:
        ds = load_dataset("fastino/fast-decisions", domain, split="train")
        for row in ds:
            for head in row["output"]["classifications"]:
                if head["multi_label"]:
                    continue  # this package handles single-label tasks
                key = f"{domain}/{head['task']}"
                h = out.setdefault(key, {"labels": list(head["labels"]), "texts": [], "gold": []})
                h["texts"].append(row["input"])
                h["gold"].append(h["labels"].index(head["true_label"][0]))
    return out


def task_yaml(key: str, head: dict) -> str:
    """A starter config for one head (edit the question and descriptions before judging)."""
    domain, name = key.split("/")
    labels = "\n".join(f'    {label}: ""' for label in head["labels"])
    return (
        f"project: {domain}-{name}\nworkdir: runs/{domain}-{name}\ntask:\n  name: {name}\n"
        f'  question: "Which {name.replace("_", " ")} best fits this text?"\n  labels:\n{labels}\n'
    )
