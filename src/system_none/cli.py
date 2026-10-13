"""Command line: system-none {init,log,judge,train,predict,bench,monitor,serve,worker,mock-judge,banking77,
fast-decisions,traffic,labels}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLE = """\
project: support-intent
workdir: runs/support-intent

task:
  name: intent
  question: "What does the customer want?"
  labels:
    refund: "Wants money back for a charge or an order"
    cancel: "Wants to cancel an order or a subscription"
    shipping: "Asks where an order is, or about delivery"
    login: "Cannot sign in or access the account"
    other: "Anything else"

# The judge labels logged inputs offline. Pick the strongest model whose license lets you
# train on its outputs (e.g. open weights served with vLLM), and confirm it below.
judge:
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-32B
  api_key_env: JUDGE_API_KEY
  mode: logprobs          # or "sample" if the server has no logprobs, or "systemone" for a
                          # POST /v1/systemone server such as Decider (base_url: its root URL)
  answer: auto            # a letter up to 26 labels, the label's name above
  concurrency: 8
  extra_body: {chat_template_kwargs: {enable_thinking: false}}
  license:
    allows_training_on_outputs: false   # set to true once you've checked the license
    note: ""

# Today's model (an LLM or a System One model). Serve-only: used as the cascade fallback,
# never as a source of training data.
incumbent:
  base_url: https://api.example.com/v1
  model: your-current-model
  api_key_env: INCUMBENT_API_KEY
  mode: logprobs

shadow: {sample_rate: 1.0}

students:
  english: [microsoft/deberta-v3-base, answerdotai/ModernBERT-base]
  multilingual: []        # e.g. [microsoft/mdeberta-v3-base, jhu-clsp/mmBERT-base, FacebookAI/xlm-roberta-base]

# min_steps: more epochs over small training sets, so training takes at least this many steps.
# human_holdout: share of the human labels (`system-none log --label-field`) kept out of training to
# score the judge, the students and the cascade against people; the rest train.
train: {epochs: 3, min_steps: 1000, lr: 3.0e-5, batch_size: 16, max_length: 256, human_holdout: 1.0}
cascade: {target_accuracy: 0.95}
monitor: {sample_rate: 0.01, window: 500, min_samples: 100, tolerance: 0.03}
"""


def _cfg(args):
    from .config import load_config

    return load_config(args.config)


def cmd_init(args):
    p = Path(args.path)
    if p.exists() and not args.force:
        sys.exit(f"{p} exists (use --force to overwrite)")
    p.write_text(EXAMPLE)
    print(f"wrote {p}")


def cmd_log(args):
    """Import inputs from a text/JSONL file into the shadow log (for bootstrapping from existing traffic).

    With --label-field, the JSONL lines that carry a label also become human labels.
    """
    from .human import add_human_labels
    from .shadow import ShadowLogger

    cfg = _cfg(args)
    if args.label_field and not args.file.endswith(".jsonl"):
        sys.exit("--label-field needs a .jsonl file")
    logger = ShadowLogger(cfg.shadow_log, max_chars=cfg.shadow.max_chars)
    n, labeled = 0, []
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rec = json.loads(line) if args.file.endswith(".jsonl") else None
            text = rec[args.field] if rec is not None else line
            n += logger.log(text)
            if args.label_field and rec is not None and rec.get(args.label_field) not in (None, ""):
                labeled.append((text, rec[args.label_field]))
    print(f"logged {n} new inputs -> {cfg.shadow_log} ({len(logger)} total)")
    if args.label_field:
        stats = add_human_labels(cfg.human_labels_path, labeled, cfg.task.label_names, max_chars=cfg.shadow.max_chars)
        print(f"recorded {stats['added']} human labels -> {cfg.human_labels_path} ({stats['total']} total)")
        if stats["unknown_labels"]:
            print(f"skipped labels the task doesn't have: {stats['unknown_labels']}", file=sys.stderr)


def cmd_judge(args):
    from .human import judge_agreement
    from .judge import Judge, label_log

    cfg = _cfg(args)
    judge = Judge(cfg.judge, cfg.task)
    stats = label_log(judge, cfg.shadow_log, cfg.labels_path, limit=args.limit)
    agreement = judge_agreement(cfg.labels_path, cfg.human_labels_path, cfg.task.label_names)
    print(json.dumps({**stats, **({"judge_vs_humans": agreement} if agreement else {})}))


def cmd_train(args):
    from .pipeline import train_all

    cfg = _cfg(args)
    deploy = train_all(cfg, candidates=args.candidates, keep_hf=not args.no_hf)
    print(json.dumps(deploy, indent=1))


def cmd_predict(args):
    from .runtime import load_deployment

    cfg = _cfg(args)
    pipe = load_deployment(cfg.workdir)
    for text in args.text:
        print(json.dumps(pipe(text).to_dict(), ensure_ascii=False))


def cmd_bench(args):
    import numpy as np

    from .bench import evaluate, score_against_humans
    from .data import split
    from .pipeline import groups_for, load_examples
    from .runtime import Classifier

    cfg = _cfg(args)
    if args.model:
        group, model_dir = None, Path(args.model)
    else:
        group = json.loads((cfg.workdir / "deploy.json").read_text())["groups"][args.group]
        model_dir = cfg.workdir / group["model_dir"]
    clf = Classifier(model_dir, threads=args.threads)
    examples, held_out = load_examples(cfg, log=lambda *a: None)
    _, _, test = split(examples, cfg.train.val_fraction, cfg.train.test_fraction, cfg.train.seed)
    res = evaluate(clf, [e.text for e in test], np.stack([e.probs for e in test]))
    people = groups_for(cfg, held_out).get(args.group, (None, held_out))[1] if held_out else []
    if people:
        res["human"] = score_against_humans(clf, people, group["threshold"] if group else None)
    print(json.dumps(res, indent=1))


def cmd_monitor(args):
    from .judge import Judge
    from .monitor import run_monitor

    cfg = _cfg(args)
    status = run_monitor(cfg, Judge(cfg.judge, cfg.task))
    sys.exit(2 if status["drift"] else 0)


def cmd_serve(args):
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(_cfg(args), threads=args.threads), host=args.host, port=args.port)


def cmd_fast_decisions(args):
    import numpy as np

    from .bench import evaluate
    from .fastdecisions import load_heads, task_yaml
    from .runtime import Classifier

    heads = load_heads(args.domains)
    if args.action == "configs":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for key, head in heads.items():
            (out / f"{key.replace('/', '__')}.yaml").write_text(task_yaml(key, head))
        print(f"wrote {len(heads)} starter configs to {out}")
        return
    head = heads[args.head]
    clf = Classifier(args.model)
    if clf.labels != head["labels"]:
        sys.exit(f"label mismatch: model {clf.labels} vs head {head['labels']}")
    res = evaluate(clf, head["texts"], np.array(head["gold"]))
    print(json.dumps({"head": args.head, "split": "dev (published)", **res}, indent=1))


def cmd_mock_judge(args):
    import uvicorn

    from .mock_judge import create_app, load_labelled_data

    data = load_labelled_data(args.datasets, args.labels)
    uvicorn.run(create_app(data, confidence=args.confidence), host=args.host, port=args.port)


def cmd_banking77(args):
    from .banking77 import run

    train = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "device": args.device}
    print(json.dumps(run(args.model, args.workdir, {k: v for k, v in train.items() if v is not None}), indent=1))


def cmd_worker(args):
    import faulthandler
    import os

    from .worker import run_worker

    token = os.environ.get(args.token_env, "")
    if not token:
        sys.exit(f"set {args.token_env} to the gateway's worker token (its SN_WORKER_TOKEN)")
    # A crash in native code (PyTorch, ONNX Runtime) then leaves Python's tracebacks in the log,
    # instead of a log that just stops.
    if not faulthandler.is_enabled():
        faulthandler.enable()
    n = run_worker(args.gateway, token, args.workdir, once=args.once, poll=args.poll)
    print(f"processed {n} job(s)")


def cmd_traffic(args):
    from .traffic import TrafficError, main as traffic

    try:
        traffic(args)
    except TrafficError as e:
        sys.exit(str(e))
    except (EOFError, KeyboardInterrupt):  # the menu, closed or interrupted
        sys.exit("\ncancelled")


def cmd_labels(args):
    from .traffic import TrafficError, labels_main

    try:
        labels_main(args)
    except TrafficError as e:
        sys.exit(str(e))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="system-none", description="Distill LLM / System One classification into a fast encoder you own."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write an example config")
    p.add_argument("path", nargs="?", default="system-none.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    def with_cfg(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("-c", "--config", default="system-none.yaml")
        p.set_defaults(fn=fn)
        return p

    p = with_cfg("log", cmd_log, "import existing inputs (.txt one per line, or .jsonl) into the shadow log")
    p.add_argument("file")
    p.add_argument("--field", default="text")
    p.add_argument("--label-field", help="JSONL field with a label a person checked: recorded as a human label")
    p = with_cfg("judge", cmd_judge, "soft-label logged inputs with the judge (resumable)")
    p.add_argument("--limit", type=int)
    p = with_cfg("train", cmd_train, "fine-tune, calibrate, export and pick students")
    p.add_argument("--candidates", nargs="*")
    p.add_argument("--no-hf", action="store_true", help="don't keep the PyTorch checkpoints")
    p = with_cfg("predict", cmd_predict, "classify text with the deployed cascade")
    p.add_argument("text", nargs="+")
    p = with_cfg("bench", cmd_bench, "accuracy, ECE and latency on the test split")
    p.add_argument("--model")
    p.add_argument("--group", default="english")
    p.add_argument("--threads", type=int)
    with_cfg("monitor", cmd_monitor, "judge the review queue; exit code 2 on drift")
    p = with_cfg("serve", cmd_serve, "HTTP server (POST /classify)")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines (no authentication)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--threads", type=int)

    p = sub.add_parser("worker", help="train students for the Rust gateway's queued jobs")
    p.add_argument("--gateway", default="http://127.0.0.1:8080")
    p.add_argument("--token-env", default="SN_WORKER_TOKEN", help="environment variable holding the worker token")
    p.add_argument("--workdir", default="runs/worker")
    p.add_argument("--once", action="store_true", help="exit when the queue is empty")
    p.add_argument("--poll", type=float, default=15, help="seconds between polls when idle")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser(
        "mock-judge",
        help="serve POST /v1/systemone from labelled data instead of a model: a judge (and System One model) "
        "that answers each text with its label",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["banking77", "ai4privacy"],
        help="datasets of `traffic --list` whose labels it answers with (default: banking77 ai4privacy; "
        "the flag alone: none, only --labels)",
    )
    p.add_argument(
        "--labels", action="append", default=[], help='a JSONL file of {"text": ..., "label": ...} rows (repeatable)'
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="the probability it gives the label; the rest is spread over the other options (default 0.9)",
    )
    p.set_defaults(fn=cmd_mock_judge)

    p = sub.add_parser("banking77", help="banking77: train a student on its human labels, score it on its test split")
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--workdir", default="runs/banking77")
    p.add_argument("--epochs", type=int, default=10)  # these defaults reach 91.95% with Ettin 17M
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    p.set_defaults(fn=cmd_banking77)

    p = sub.add_parser("fast-decisions", help="Fastino Fast Decisions dev split: starter configs or evaluation")
    p.add_argument("action", choices=["configs", "eval"])
    p.add_argument("--domains", nargs="*")
    p.add_argument("--out", default="fast_decisions_configs")
    p.add_argument("--head", help="e.g. support_intent/intent")
    p.add_argument("--model", help="exported ONNX dir")
    p.set_defaults(fn=cmd_fast_decisions)

    p = sub.add_parser(
        "traffic",
        help="send texts from a public dataset to the gateway's POST /v1/classify "
        "(without --dataset, a menu asks what to send)",
    )
    p.add_argument(
        "--dataset",
        help="e.g. banking77, ai4privacy, fast-decisions/email_triage/category (see --list); "
        "without it, a menu asks for the dataset and the settings not given as flags",
    )
    p.add_argument("--n", type=int, help="number of requests (default 100)")
    p.add_argument("--concurrency", type=int, help="requests in flight at once (default 4)")
    p.add_argument("--rate", type=float, help="at most this many requests per second; 0 for no limit (default)")
    p.add_argument("--gateway", default="http://127.0.0.1:8080")
    p.add_argument("--model", help='force this System One model (the request\'s "model")')
    p.add_argument("--seed", type=int, default=0, help="shuffles the texts")
    p.add_argument("--timeout", type=float, default=60, help="seconds to wait for each answer")
    p.add_argument("--list", action="store_true", help="list the datasets and exit")
    p.set_defaults(fn=cmd_traffic)

    p = sub.add_parser(
        "labels",
        help="upload a dataset's own answers to the gateway: its training split trains students, "
        "its test split is what they are audited against",
    )
    p.add_argument("--dataset", required=True, help="e.g. banking77 (any dataset of `traffic --list` with labels)")
    p.add_argument("--gateway", default="http://127.0.0.1:8080")
    p.add_argument("--token-env", default="SN_ADMIN_TOKEN", help="environment variable holding the admin token")
    p.add_argument(
        "--audit-share",
        type=float,
        default=0.2,
        help="share held out for audits when the dataset has no official test split (default 0.2)",
    )
    p.add_argument("--train", action="store_true", help="queue a training job for the prompt afterwards")
    p.set_defaults(fn=cmd_labels)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
