"""Project configuration (one YAML file per classification task)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Environment variables that may hold SystemNone's own tokens (SN_*) and store credentials (AWS_*),
# so never an endpoint's API key. QS_* is on the list too: nothing reads it, but a deployment from
# before the project was renamed SystemNone may still hold the same tokens under that prefix.
OWN_CREDENTIAL_PREFIXES = ("SN_", "QS_", "AWS_")


@dataclass
class TaskSpec:
    """A single-label classification task: one question, a fixed set of labels."""

    name: str
    question: str
    labels: dict[str, str]  # label -> short description ("" allowed)

    @property
    def label_names(self) -> list[str]:
        return list(self.labels)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def validate(self) -> None:
        if len(self.labels) < 2:
            raise ValueError("A task needs at least two labels.")


@dataclass
class ModelEndpoint:
    """An OpenAI-compatible chat-completions endpoint (vLLM, Ollama, TGI, OpenRouter, ...), or with
    mode "systemone" a server speaking TypeSafe's POST /v1/systemone (Decider); base_url is then its root.
    Mode "openai_decisions" uses {base_url}/decisions (base_url includes /v1).
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key_env: str | None = None
    # "logprobs" (one call, soft labels), "sample" (n calls, vote shares) or "systemone" (one call to
    # {base_url}/v1/systemone, soft labels from the answer's probabilities)
    mode: str = "logprobs"  # also "openai_decisions": typed questions and complete soft labels
    # How the model names its answer: "letter" (A-Z, so up to 26 labels), "name" (the label itself,
    # any number of labels) or "auto" (letters up to 26 labels, names above).
    answer: str = "auto"
    samples: int = 5
    temperature: float = 1.0  # used in "sample" mode
    concurrency: int = 8
    timeout: float = 60.0
    max_retries: int = 3
    extra_body: dict[str, Any] = field(
        default_factory=dict
    )  # e.g. {"chat_template_kwargs": {"enable_thinking": false}}

    def __post_init__(self):
        if self.answer not in ("auto", "letter", "name"):
            raise ValueError(f"answer must be 'auto', 'letter' or 'name', not {self.answer!r}")
        # The key is sent to base_url.
        if self.api_key_env and self.api_key_env.upper().startswith(OWN_CREDENTIAL_PREFIXES):
            raise ValueError(
                f"api_key_env {self.api_key_env!r} names one of SystemNone's own credentials "
                "(SN_*, QS_*, AWS_*); use a variable that holds only this endpoint's key."
            )

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class LicenseAck:
    allows_training_on_outputs: bool = False
    note: str = ""


@dataclass
class JudgeSpec(ModelEndpoint):
    license: LicenseAck = field(default_factory=LicenseAck)


@dataclass
class ShadowSpec:
    sample_rate: float = 1.0
    max_chars: int = 8000


@dataclass
class StudentSpec:
    english: list[str] = field(default_factory=lambda: ["microsoft/deberta-v3-base", "answerdotai/ModernBERT-base"])
    multilingual: list[str] = field(default_factory=list)
    # Candidates for a student per language: each language the router is sure of in at least
    # `train.per_language_min_examples` examples gets one, trained on those examples alone.
    per_language: list[str] = field(default_factory=list)


@dataclass
class TrainSpec:
    epochs: int = 3
    # More passes over small datasets: epochs are raised until training takes at least this many
    # optimizer steps (at most MAX_EPOCHS passes). 0 trains for exactly `epochs`.
    min_steps: int = 1000
    lr: float = 3e-5
    batch_size: int = 16
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 13
    device: str | None = None  # "cuda", "mps", "cpu"; auto when unset
    accuracy_tolerance: float = 0.005  # candidates within this of the best compete on latency
    # Fit the temperature so confidence = P(student agrees with the judge's top answer) ("hard"),
    # or to reproduce the judge's own probabilities ("soft").
    calibrate_on: str = "hard"
    # Share of the human-labeled inputs (human_labels.jsonl) held out to score the judge, the students
    # and the cascade against people. The rest train on their human label instead of the judge's.
    human_holdout: float = 1.0
    # Examples a language needs for a student of its own (`students.per_language`).
    per_language_min_examples: int = 300

    def __post_init__(self):
        if self.min_steps < 0:
            raise ValueError("min_steps must be >= 0")
        if not 0.0 <= self.human_holdout <= 1.0:
            raise ValueError("human_holdout must be between 0 and 1")


@dataclass
class CascadeSpec:
    target_accuracy: float = 0.95
    method: str = "empirical"
    confidence_level: float = 0.95

    def __post_init__(self):
        if not 0.5 <= self.target_accuracy <= 1.0:
            raise ValueError("target_accuracy must be between 0.5 and 1")
        if self.method not in {"empirical", "clopper_pearson"}:
            raise ValueError("calibration method must be empirical or clopper_pearson")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive)")


@dataclass
class MonitorSpec:
    sample_rate: float = 0.01
    window: int = 500
    min_samples: int = 100
    tolerance: float = 0.03


@dataclass
class Config:
    project: str
    workdir: Path
    task: TaskSpec
    judge: JudgeSpec
    incumbent: ModelEndpoint | None = None  # today's model; serve-only, never a source of training data
    shadow: ShadowSpec = field(default_factory=ShadowSpec)
    students: StudentSpec = field(default_factory=StudentSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    cascade: CascadeSpec = field(default_factory=CascadeSpec)
    monitor: MonitorSpec = field(default_factory=MonitorSpec)

    # --- file layout -------------------------------------------------------
    @property
    def shadow_log(self) -> Path:
        return self.workdir / "shadow.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.workdir / "labels.jsonl"

    @property
    def human_labels_path(self) -> Path:
        return self.workdir / "human_labels.jsonl"

    @property
    def models_dir(self) -> Path:
        return self.workdir / "models"

    @property
    def review_queue(self) -> Path:
        return self.workdir / "review_queue.jsonl"


def _build(cls, data: dict | None):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(d: dict, base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    task_d = dict(d["task"])
    labels = task_d.get("labels")
    if isinstance(labels, list):
        task_d["labels"] = {str(label): "" for label in labels}
    task = _build(TaskSpec, task_d)
    task.validate()

    judge_d = dict(d.get("judge") or {})
    judge_d["license"] = _build(LicenseAck, judge_d.get("license"))
    judge = _build(JudgeSpec, judge_d)

    workdir = Path(d.get("workdir", f"runs/{d.get('project', task.name)}"))
    if not workdir.is_absolute():
        workdir = base_dir / workdir

    return Config(
        project=d.get("project", task.name),
        workdir=workdir,
        task=task,
        judge=judge,
        incumbent=_build(ModelEndpoint, d["incumbent"]) if d.get("incumbent") else None,
        shadow=_build(ShadowSpec, d.get("shadow")),
        students=_build(StudentSpec, d.get("students")),
        train=_build(TrainSpec, d.get("train")),
        cascade=_build(CascadeSpec, d.get("cascade")),
        monitor=_build(MonitorSpec, d.get("monitor")),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f)
    return config_from_dict(d, base_dir=path.resolve().parent)
