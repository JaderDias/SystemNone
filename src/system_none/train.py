"""Fine-tune an encoder to match the judge's soft labels (minimize KL(judge || student))."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import fit_temperature, soft_nll
from .config import TaskSpec, TrainSpec
from .data import Example


def pick_device(pref: str | None = None) -> str:
    import torch

    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    model: Any  # a transformers model, or an EmbeddingStudent
    tokenizer: Any  # its tokenizer, or a PrefixTokenizer
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def predict_logits(
    model, tokenizer, texts: list[str], max_length: int, batch_size: int = 64, device: str = "cpu"
) -> np.ndarray:
    import torch

    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for chunk in _batches(order, batch_size):
            enc = tokenizer(
                [texts[i] for i in chunk], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu().numpy()
            out[chunk] = logits
    return out


def train_student(
    model_name: str,
    task: TaskSpec,
    train: list[Example],
    val: list[Example],
    spec: TrainSpec,
    log=print,
    on_progress: Callable[[int, int], None] | None = None,
) -> TrainResult:
    """Fine-tune `model_name`. `on_progress` is told (done, total) optimizer steps after each one;
    a frozen embedding model's head tells it the texts embedded instead, which take most of its time."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .embed import is_embedding_model, train_head

    if is_embedding_model(model_name):  # EmbeddingGemma and other sentence encoders stay frozen
        return train_head(model_name, task, train, val, spec, log=log, on_progress=on_progress)
    set_seed(spec.seed)
    device = pick_device(spec.device)
    labels = task.label_names
    # Downloading the checkpoint, or building a fast tokenizer for it, can take minutes.
    log(f"  loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
        # Transformers 5 defaults to the checkpoint's dtype. AdamW's moments must stay
        # float32: mDeBERTa's fp16 weights otherwise become nonfinite on the first update.
        torch_dtype=torch.float32,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": spec.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=spec.lr)
    steps_per_epoch = math.ceil(len(train) / spec.batch_size)
    total = max(1, steps_per_epoch * spec.epochs)
    warm = int(spec.warmup_ratio * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm))
    )
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    log(f"  {total} steps: {spec.epochs} epochs of {steps_per_epoch} batches of {spec.batch_size}")

    val = val or train[: max(1, len(train) // 10)]
    val_texts = [e.text for e in val]
    val_p = np.stack([e.probs for e in val])
    val_entropy = float(-(val_p * np.log(np.clip(val_p, 1e-12, 1))).sum(-1).mean())
    best, best_state, history, best_epoch = float("inf"), None, [], 0
    t0 = time.time()
    rng = random.Random(spec.seed)
    step = 0
    for epoch in range(1, spec.epochs + 1):
        model.train()
        idx = list(range(len(train)))
        rng.shuffle(idx)
        running = 0.0
        for chunk in _batches(idx, spec.batch_size):
            # AutoTokenizer.from_pretrained is typed as possibly None (transformers' stand-in for its
            # sentencepiece tokenizers when sentencepiece is missing, which the train extra installs).
            enc = tokenizer(  # ty: ignore[call-non-callable]
                [train[i].text for i in chunk],
                truncation=True,
                max_length=spec.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            target = torch.tensor(np.stack([train[i].probs for i in chunk]), dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits
            # KL(judge || student): the student matches the judge's whole distribution
            loss = F.kl_div(F.log_softmax(logits.float(), dim=-1), target, reduction="batchmean")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss for {model_name} at epoch {epoch}, step {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            if on_progress is not None:
                on_progress(step, total)
        logits = predict_logits(model, tokenizer, val_texts, spec.max_length, device=device)
        if not np.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite validation logits for {model_name} at epoch {epoch}")
        q = torch.log_softmax(torch.tensor(logits), -1).numpy()
        val_kl = float((val_p * (np.log(np.clip(val_p, 1e-12, 1)) - q)).sum(-1).mean())
        # The student is temperature-scaled after training, so epochs compete after scaling too.
        # Raw KL would favour early, less confident epochs over later, more accurate ones.
        scaled_kl = soft_nll(logits, val_p, fit_temperature(logits, val_p)) - val_entropy
        val_acc = float((logits.argmax(-1) == val_p.argmax(-1)).mean())
        row = {
            "epoch": epoch,
            "train_kl": round(running / max(1, steps_per_epoch), 4),
            "val_kl": round(val_kl, 4),
            "val_kl_scaled": round(scaled_kl, 4),
            "val_agreement": round(val_acc, 4),
        }
        history.append(row)
        log(f"  epoch {epoch}: {row}")
        if scaled_kl < best:
            best = scaled_kl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()
    return TrainResult(
        model=model, tokenizer=tokenizer, history=history, best_epoch=best_epoch, seconds=time.time() - t0
    )


def save_hf(result: TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_pretrained(out_dir)
    result.tokenizer.save_pretrained(out_dir)
    (out_dir / "history.json").write_text(
        json.dumps(
            {"history": result.history, "best_epoch": result.best_epoch, "seconds": round(result.seconds, 1)}, indent=1
        )
    )
