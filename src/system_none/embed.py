"""Students on a frozen sentence-embedding model (EmbeddingGemma, ...): only a linear head is trained.

With few labels and no GPU this is the cheap option. Each input is embedded once, and a softmax
head is fitted to the judge's soft labels in seconds, instead of fine-tuning a whole encoder.
The encoder and the head export together to one ONNX graph, so the gateway serves the result
like any other student. Models that ship a Sentence Transformers `modules.json` are trained this
way; their "Classification" prompt, if any, is put in front of every input.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import TaskSpec, TrainSpec
from .data import Example
from .train import TrainResult, pick_device, set_seed

HEAD_EPOCHS = 200
HEAD_LR = 1e-2
HEAD_BATCH = 1024


def is_embedding_model(name: str) -> bool:
    """Sentence Transformers checkpoints ship modules.json. Gated repos raise GatedRepoError."""
    if Path(name).is_dir():
        return (Path(name) / "modules.json").is_file()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        hf_hub_download(name, "modules.json")
        return True
    except EntryNotFoundError:
        return False


class PrefixTokenizer:
    """The encoder's tokenizer, with the model's classification prompt in front of every text."""

    def __init__(self, tokenizer, prefix: str):
        self.tokenizer, self.prefix = tokenizer, prefix

    def __call__(self, texts, **kw):
        texts = [texts] if isinstance(texts, str) else texts
        return self.tokenizer([self.prefix + t for t in texts], **kw)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class EmbeddingStudent(torch.nn.Module):
    """A frozen Sentence Transformers encoder and a linear head, called like a Hugging Face classifier."""

    def __init__(self, encoder, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.get_sentence_embedding_dimension(), num_labels)
        self.config = SimpleNamespace(num_labels=num_labels, type_vocab_size=0)

    def embed(self, input_ids, attention_mask):
        return self.encoder({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]

    def forward(self, input_ids, attention_mask, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids, attention_mask)))

    def set_attn_implementation(self, implementation: str) -> None:
        self.encoder[0].auto_model.set_attn_implementation(implementation)

    def save_pretrained(self, out_dir) -> None:
        self.encoder.save(str(out_dir))
        torch.save(self.head.state_dict(), Path(out_dir) / "head.pt")


def embed_texts(
    student: EmbeddingStudent,
    tokenizer,
    texts: list[str],
    max_length: int,
    device: str,
    batch_size: int = 32,
    on_batch: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """The texts' embeddings, in their order. `on_batch` is told how many texts each batch embedded."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.zeros(len(texts), student.head.in_features)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            chunk = order[i : i + batch_size]
            enc = tokenizer(
                [texts[j] for j in chunk], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            )
            out[chunk] = student.embed(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
            if on_batch is not None:
                on_batch(len(chunk))
    return out


def train_head(
    model_name: str,
    task: TaskSpec,
    train: list[Example],
    val: list[Example],
    spec: TrainSpec,
    log=print,
    on_progress: Callable[[int, int], None] | None = None,
) -> TrainResult:
    """Fit a head on the frozen encoder's embeddings. `on_progress` is told (done, total) texts
    embedded, training and validation together: embedding takes most of the time, the head seconds."""
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    set_seed(spec.seed)
    device = pick_device(spec.device)
    log(f"  loading {model_name} on {device}, frozen: only a head is trained")
    encoder = SentenceTransformer(model_name, device=device)
    tokenizer = PrefixTokenizer(encoder.tokenizer, (encoder.prompts or {}).get("Classification") or "")
    student = EmbeddingStudent(encoder, task.num_labels).eval()
    val = val or train[: max(1, len(train) // 10)]
    embedded, total = 0, len(train) + len(val)
    log(f"  embedding {total} texts")

    def on_batch(n: int) -> None:
        nonlocal embedded
        embedded += n
        if on_progress is not None:
            on_progress(embedded, total)

    t0 = time.time()
    x_tr = embed_texts(student, tokenizer, [e.text for e in train], spec.max_length, device, on_batch=on_batch)
    x_va = embed_texts(student, tokenizer, [e.text for e in val], spec.max_length, device, on_batch=on_batch)
    log(f"  embedded {len(train) + len(val)} texts in {time.time() - t0:.1f}s")
    p_tr = torch.tensor(np.stack([e.probs for e in train]), dtype=torch.float32)
    p_va = torch.tensor(np.stack([e.probs for e in val]), dtype=torch.float32)

    # Fit on standardized embeddings, then fold the scaling into the head's weights.
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    z_tr, z_va = (x_tr - mu) / sd, (x_va - mu) / sd
    head = torch.nn.Linear(x_tr.shape[1], task.num_labels)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=spec.weight_decay)
    rng = random.Random(spec.seed)
    best, best_state, history, best_epoch = float("inf"), None, [], 0
    for epoch in range(1, HEAD_EPOCHS + 1):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        for i in range(0, len(idx), HEAD_BATCH):
            chunk = idx[i : i + HEAD_BATCH]
            # KL(judge || student), as for fine-tuned students
            loss = F.kl_div(F.log_softmax(head(z_tr[chunk]), -1), p_tr[chunk], reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            q = F.log_softmax(head(z_va), -1)
            val_kl = float(F.kl_div(q, p_va, reduction="batchmean"))
        if val_kl < best:
            best, best_epoch = val_kl, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if epoch % 20 == 0:
            row = {
                "epoch": epoch,
                "train_kl": round(float(loss), 4),
                "val_kl": round(val_kl, 4),
                "val_agreement": round(float((q.argmax(-1) == p_va.argmax(-1)).float().mean()), 4),
            }
            history.append(row)
            log(f"  head {row}")
    if best_state is None:  # no epoch had a finite validation loss
        raise RuntimeError(f"training the head on {model_name} gave no finite validation loss")
    head.load_state_dict(best_state)
    with torch.no_grad():
        w = head.weight / sd
        student.head.weight.copy_(w)
        student.head.bias.copy_(head.bias - w @ mu)
    student.to("cpu").eval()
    return TrainResult(
        model=student, tokenizer=tokenizer, history=history, best_epoch=best_epoch, seconds=time.time() - t0
    )
