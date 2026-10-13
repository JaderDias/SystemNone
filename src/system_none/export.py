"""Export a trained student to ONNX + tokenizer.json + manifest.json (runs without PyTorch)."""

from __future__ import annotations

import contextlib
import json
import time
import warnings
from pathlib import Path
from typing import cast

import numpy as np


def _input_names(tokenizer, model=None) -> list[str]:
    names = ["input_ids", "attention_mask"]
    uses_types = getattr(getattr(model, "config", None), "type_vocab_size", 2) or 0  # DeBERTa-v3: 0 (ignored)
    if "token_type_ids" in getattr(tokenizer, "model_input_names", []) and uses_types > 0:
        names.append("token_type_ids")
    return names


def _graph_inputs(onnx_path: Path) -> list[str]:
    """The inputs the exported graph actually kept (tracing drops unused ones)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return [i.name for i in sess.get_inputs()]


def export_onnx(
    model,
    tokenizer,
    labels: list[str],
    out_dir: Path,
    max_length: int,
    temperature: float,
    sample_texts: list[str],
    meta: dict | None = None,
    opset: int = 18,
    log=print,
) -> Path:
    """`log` is told each attempt: which exporter, and how far its logits are from PyTorch's, or its error."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    with contextlib.suppress(Exception):  # traced attention is more portable than fused kernels
        model.set_attn_implementation("eager")
    names = _input_names(tokenizer, model)

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            # One arg per name, always; not strict=True, which the exporters would have to trace too.
            return self.m(**dict(zip(names, args, strict=False))).logits

    wrapped = Wrapped(model).eval()
    texts = (sample_texts or ["hello world"])[:8]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    args = tuple(enc[n] for n in names)
    onnx_path = out_dir / "model.onnx"

    attempts = []
    for dynamo in (True, False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if dynamo:
                    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=max(max_length, 16))
                    # forward(*args) is one input to torch.export: a tuple holding every tensor
                    torch.onnx.export(
                        wrapped,
                        args,
                        str(onnx_path),
                        input_names=names,
                        output_names=["logits"],
                        dynamic_shapes=(tuple({0: batch, 1: seq} for _ in names),),
                        opset_version=opset,
                        dynamo=True,
                        external_data=False,
                        verbose=False,
                    )  # one file: MODEL_FILES
                else:
                    axes = {n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}}
                    torch.onnx.export(
                        wrapped,
                        args,
                        str(onnx_path),
                        input_names=names,
                        output_names=["logits"],
                        dynamic_axes=axes,
                        opset_version=17,
                        dynamo=False,
                    )
            diff = verify(onnx_path, model, tokenizer, _graph_inputs(onnx_path), texts, max_length)
            attempts.append({"exporter": "dynamo" if dynamo else "torchscript", "max_abs_diff": diff})
            log(f"  ONNX export: {attempts[-1]}")
            if diff < 1e-2:
                break
        except Exception as e:  # try the other exporter
            msg = " ".join(f"{type(e).__name__}: {e}".split())
            attempts.append({"exporter": "dynamo" if dynamo else "torchscript", "error": msg[:600]})
            log(f"  ONNX export: {attempts[-1]}")
    else:
        raise RuntimeError(f"ONNX export failed verification: {attempts}")

    tok_path = out_dir / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tok_path))
    manifest = {
        "labels": labels,
        "temperature": temperature,
        "max_length": max_length,
        "inputs": _graph_inputs(onnx_path),
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "pad_token": tokenizer.pad_token or "[PAD]",
        **({"prefix": tokenizer.prefix} if getattr(tokenizer, "prefix", "") else {}),  # prepended to every input
        "export": attempts[-1],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(meta or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir


def verify(onnx_path: Path, model, tokenizer, names: list[str], texts: list[str], max_length: int) -> float:
    """Max |logit difference| between ONNX and PyTorch on a padded batch of different lengths."""
    import onnxruntime as ort
    import torch

    probe = [*texts, texts[0] + " " + texts[-1], (texts[0] + " ") * 3]
    enc = tokenizer(probe, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**{n: enc[n] for n in names}).logits.numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # run() is typed for any ONNX output (sparse tensors, sequences, maps); a classifier's logits are dense.
    out = cast(np.ndarray, sess.run(None, {n: enc[n].numpy().astype(np.int64) for n in names})[0])
    if (out.argmax(-1) != ref.argmax(-1)).any():
        return float("inf")
    return float(np.abs(out - ref).max())
