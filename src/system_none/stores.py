"""The storage shared with the Rust gateway: a local folder (file:///...) or S3 (s3://bucket/prefix).

Layout:
  prompts/{prompt_id}.json                          prompt definition
  inputs/{prompt_id}/{YYYY-MM-DD}/{ms}-{rand}.jsonl  logged inputs {"id", "text", "ts"} (never answers)
  judgments/{task-and-judge-hash}/{input-hash}.jsonl reusable soft labels, checkpointed per input
  models/{prompt_id}/{job_id}/                      model.onnx, tokenizer.json, manifest.json
"""

from __future__ import annotations

import builtins  # the stores' list() method hides the type in their class bodies
from pathlib import Path
from urllib.parse import unquote, urlparse

MODEL_FILES = ("model.onnx", "tokenizer.json", "manifest.json")


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self, prefix: str) -> builtins.list[str]:
        base = self.root / prefix
        start = base if base.is_dir() else base.parent
        if not start.exists():
            return []
        keys = [p.relative_to(self.root).as_posix() for p in start.rglob("*") if p.is_file()]
        return sorted(k for k in keys if k.startswith(prefix))

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class S3Store:
    """Needs boto3 (`pip install system-none[s3]`) and the usual AWS_* credentials."""

    def __init__(self, bucket: str, prefix: str = ""):
        import boto3

        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def list(self, prefix: str) -> builtins.list[str]:
        keys, strip = [], (self.prefix + "/") if self.prefix else ""
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys += [o["Key"][len(strip) :] for o in page.get("Contents", [])]
        return sorted(keys)

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)


def open_store(url: str):
    u = urlparse(url)
    if u.scheme == "file":
        return LocalStore(unquote(u.path))
    if u.scheme == "s3":
        return S3Store(u.netloc, u.path)
    raise ValueError(f"unsupported store URL {url!r} (use file:///abs/path or s3://bucket/prefix)")
