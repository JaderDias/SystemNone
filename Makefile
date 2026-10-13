# The whole system on your own Linux, macOS or WSL machine, tests, and test traffic for a running
# gateway. `make` alone starts the gateway, the training worker and the mock judge
# (src/system_none/mock_judge.py), which answers from labelled data instead of running a model, and
# makes it the judge the worker trains students from and the gateway's System One model. Given a
# TypeSafe API key (it asks for one once, in a terminal), it makes TypeSafe's API the System One
# model instead. Local models (Decision-2.0-Kai-0.6B, Decider-2B) are set up, downloaded and started
# only with JUDGE=decider. `make help` lists the targets. CI runs what lint (ruff, ty and bandit on
# the Python code, cargo fmt and pedantic clippy on the gateway), test, test-tiny and verify run on
# every push; the medium and big end-to-end runs are only here. `make fmt` formats the Python code
# and sorts its imports, as ruff wants them.
#
# The Python targets first set up $(VENV): system-none[dev] in a virtualenv made with uv when it is
# installed (the easy way: https://docs.astral.sh/uv/, which also gets the right Python), otherwise
# with $(PYTHON) -m venv and pip (on Debian and Ubuntu, that needs the python3-venv package); a
# virtualenv that $(PYTHON) fails to make is removed. Delete $(VENV) to rebuild it. The Rust
# targets, and make itself, need cargo (https://rustup.rs). verify needs tlc on PATH, or Java 11+
# with TLA2TOOLS=/path/to/tla2tools.jar, or Nix (see verification/README.md).

.DEFAULT_GOAL := stack

VENV ?= .venv
# With uv: the Python in $(VENV) and $(DECIDER_VENV), as in CI (uv downloads it when needed).
PYTHON_VERSION ?= 3.11
# Without uv: the interpreter that makes them (3.10 or later; 3.11 or 3.12 for $(DECIDER_VENV),
# as decider-ai needs numpy<2, which has no wheels for later versions).
PYTHON ?= python3
UV ?= $(shell command -v uv 2>/dev/null)
# Where a new $(VENV) gets PyTorch: the default index (PyPI's CUDA build on Linux) when nvidia-smi
# sees a GPU, otherwise the much smaller CPU-only build. A GPU too old for the default build needs
# another, e.g. https://download.pytorch.org/whl/cu126 for a GTX 10xx (local.mk can set it with =).
TORCH_INDEX ?= $(if $(shell nvidia-smi -L 2>/dev/null),,https://download.pytorch.org/whl/cpu)
# Extra pytest arguments, e.g. PYTEST_ARGS="-k router".
PYTEST_ARGS ?=
# Where make (make stack) runs the gateway: an IP address and a port.
GATEWAY_HOST ?= 127.0.0.1
GATEWAY_PORT ?= 8080
# Where make configures the gateway and make traffic sends requests.
GATEWAY_URL ?= http://$(GATEWAY_HOST):$(GATEWAY_PORT)
# The judge make starts and configures: mock, the mock judge, which answers each text with its
# label in labelled data and runs no model; or decider, the local model server below, which make
# only sets up, downloads and starts when asked for it here (make JUDGE=decider).
JUDGE ?= mock
# The mock judge: where it listens, the datasets of `system-none traffic --list` whose labels it
# answers with, JSONL files of {"text": ..., "label": ...} rows it answers with too, and the
# probability it gives the label (the rest is spread over the other options). A text it has no
# label for gets an error, never a guess. Its answers are the datasets' own labels, so training on
# them is as their licenses allow: banking77 is CC BY 4.0, and ai4privacy CC BY 4.0 and the Llama
# Community License (see tests/data/ai4privacy/README.md); for your own files, that is yours to check.
MOCK_JUDGE_HOST ?= 127.0.0.1
MOCK_JUDGE_PORT ?= 8001
MOCK_JUDGE_URL ?= http://$(MOCK_JUDGE_HOST):$(MOCK_JUDGE_PORT)
MOCK_JUDGE_DATASETS ?= banking77 ai4privacy
MOCK_JUDGE_LABELS ?=
MOCK_JUDGE_CONFIDENCE ?= 0.9
MOCK_JUDGE_MODEL := mock-judge
# The Decider server make starts with JUDGE=decider: the Hub model (or a local folder), where it
# listens, and its device (cpu, cuda, mps or auto). Its own virtualenv gets the CPU-only PyTorch
# with DECIDER_DEVICE=cpu and the default build otherwise; delete $(DECIDER_VENV) after changing it.
DECIDER_MODEL ?= vllm-sr/Decision-2.0-Kai-0.6B
# Decision 2.0 needs its own loader; retain Decider compatibility for explicit overrides.
DECIDER_APP ?= $(if $(filter vllm-sr/Decision-2.0-%,$(DECIDER_MODEL)),system_none.decision_server:app,decider.serve:app)
DECISION_RUNTIME = $(filter system_none.decision_server:app,$(DECIDER_APP))
DECIDER_HOST ?= 127.0.0.1
DECIDER_PORT ?= 8000
DECIDER_DEVICE ?= $(if $(DECISION_RUNTIME),auto,cpu)
DECIDER_CPU_CONCURRENCY ?= 2
DECIDER_THREADS ?= 0
DECIDER_WORKERS ?= 1
DECIDER_URL ?= http://$(DECIDER_HOST):$(DECIDER_PORT)
# Decider runs in its own virtualenv, so its Transformers version and the legacy
# Decider dependencies never change the worker's.
DECIDER_VENV ?= .venv-decider
DECIDER_PACKAGE ?= decider-ai[serve]==1.6.0
DECIDER_TORCH_INDEX ?= $(if $(filter cpu,$(DECIDER_DEVICE)),https://download.pytorch.org/whl/cpu,$(if $(filter auto,$(DECIDER_DEVICE)),$(TORCH_INDEX)))
# Judges need someone to have checked that the model's license allows training students on its
# outputs. Decision-2.0-Kai-0.6B and its Qwen3-0.6B-Base backbone are Apache-2.0,
# with no restriction on training from outputs. Also checked for Mapika/decider-2b: its model card, the decider-ai package and
# github.com/Mapika/decider are Apache-2.0, as is its base model Qwen/Qwen3.5-2B-Base, and none
# restricts using the model's outputs. Another DECIDER_MODEL needs its own check before
# DECIDER_LICENSE_ACK=yes; with no, Decider only answers and the gateway gets no judge.
DECIDER_LICENSE_ACK ?= $(if $(filter vllm-sr/Decision-2.0-Kai-0.6B Mapika/decider-2b,$(DECIDER_MODEL)),yes,no)
# Seconds make waits for Decider to load its weights (after downloading them).
DECIDER_TIMEOUT ?= 900
# The judge JUDGE picks: where it answers, its model, and what make calls it.
STACK_JUDGE_URL = $(if $(filter decider,$(JUDGE)),$(DECIDER_URL),$(MOCK_JUDGE_URL))
STACK_JUDGE_MODEL = $(if $(filter decider,$(JUDGE)),$(DECIDER_MODEL),$(MOCK_JUDGE_MODEL))
STACK_JUDGE_ABOUT = $(if $(filter decider,$(JUDGE)),$(DECIDER_MODEL),the mock judge)
# TypeSafe's API, the gateway's System One model instead of the judge when .env or the environment
# has a TYPESAFE_API_KEY (make asks for one once, in a terminal), which the gateway sends as a
# bearer token: its root URL, under which it serves POST /v1/systemone, and the model to ask for,
# which the API requires (GET /v1/models lists them).
TYPESAFE_URL ?= https://api.typesafe.ai
TYPESAFE_MODEL ?= jev-latest
# The pids and logs of what make starts.
RUN_DIR ?= runs/stack
# The CLI make traffic runs: the one in $(VENV), which it sets up first; SYSTEM_NONE=system-none
# runs the one on PATH instead, without setting up $(VENV).
SYSTEM_NONE ?= $(VENV)/bin/system-none
# Flags skip the menu, e.g. make traffic ARGS="--dataset banking77 --n 500"
ARGS ?=

PY := $(VENV)/bin/python
PYTEST := $(PY) -m pytest
INSTALLED := $(VENV)/.installed
DECIDER_INSTALLED := $(DECIDER_VENV)/.installed-$(if $(DECISION_RUNTIME),decision2,decider)
GATEWAY_BIN := gateway/target/release/system-none-gateway
# The ONNX Runtime library from the Python package, which the gateway loads at run time.
ORT_LIBRARY = $(PY) -c "import glob, onnxruntime, os; capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'); print(sorted(glob.glob(os.path.join(capi, 'libonnxruntime.so*')) + glob.glob(os.path.join(capi, 'libonnxruntime*.dylib')))[0])"
# Detaches what make starts from the terminal, so Ctrl-C in a later make doesn't stop it (Linux).
SETSID := $(shell command -v setsid 2>/dev/null)

# $(call new_venv,dir) makes a virtualenv; $(call pip_install,dir,args) installs args into it.
# $(export_lock) writes the versions uv.lock pins to $(LOCK_CONSTRAINTS), which $(constrain)
# passes to pip install. --locked: after a change to pyproject.toml's dependencies, uv lock first.
LOCK_CONSTRAINTS := $(VENV)/uv-lock-constraints.txt
ifneq ($(UV),)
new_venv = $(UV) venv --python $(PYTHON_VERSION) $(1)
pip_install = $(UV) pip install --python $(1)/bin/python $(2)
export_lock = $(UV) export --locked --all-extras --no-emit-project --no-hashes --quiet -o $(LOCK_CONSTRAINTS)
constrain = -c $(LOCK_CONSTRAINTS)
else
# Nothing reads uv.lock without uv: pip installs what pyproject.toml's ranges allow today.
export_lock = :
constrain =
# A failed $(PYTHON) -m venv (Debian and Ubuntu ship its ensurepip in python3-venv) leaves a
# virtualenv without pip behind, which the next make would take as made: remove it.
new_venv = $(PYTHON) -m venv $(1) || { rm -rf $(1); $(call no_uv,$(1)); }
pip_install = $(1)/bin/python -m pip install $(2) || $(call no_uv,$(1))
endif
# $(call no_uv,dir): says what to install when setting up dir without uv failed, and fails.
no_uv = { echo "$(1): $$no_uv_hint"; exit 1; } >&2
define no_uv_hint
setting it up without uv failed. The easy way is to install uv (https://docs.astral.sh/uv/),
which also gets the right Python version: $(DECIDER_VENV) needs 3.11 or 3.12, which a newer $(PYTHON)
can't make. Or install the venv package for $(PYTHON) (python3-venv, or python3.X-venv for Python 3.X,
on Debian and Ubuntu), and delete the virtualenv if it has no pip. Then run make again.
endef
export no_uv_hint
# Without uv, $(DECIDER_VENV) needs $(PYTHON) to be 3.11 or 3.12: decider-ai needs numpy<2.
decider_python_check = $(PYTHON) -c 'import sys; v = sys.version_info[:2]; \
	sys.exit(v not in [(3, 11), (3, 12)] and "$(PYTHON) is Python %d.%d." % v)' || $(call no_uv,$(DECIDER_VENV))

.PHONY: help venv lint fmt test test-tiny test-medium test-big verify traffic labels \
	stack stack-down stack-status stack-logs stack-configure stack-tokens stack-gateway stack-worker stack-model \
	stack-mock-judge stack-decider

help: ## List the targets
	@echo "Usage: make [target] [NAME=value ...]"
	@echo
	@awk -F ':.*## ' '/^[a-zA-Z0-9_.-]+:.*## / {sub(/\r$$/, "", $$2); printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables, with their current values:"
	@echo "  VENV=$(VENV)"
	@echo "      the virtualenv the Python targets use; delete it to rebuild it"
	@echo "  TORCH_INDEX=$(TORCH_INDEX)"
	@echo "      where a new VENV gets PyTorch (empty: the default index, CUDA on Linux)"
	@echo "  PYTEST_ARGS=$(PYTEST_ARGS)"
	@echo "      extra pytest arguments, e.g. PYTEST_ARGS=\"-k router\""
	@echo "  GATEWAY_HOST=$(GATEWAY_HOST) GATEWAY_PORT=$(GATEWAY_PORT)"
	@echo "      where make runs the gateway"
	@echo "  GATEWAY_URL=$(GATEWAY_URL)"
	@echo "      the gateway make configures and make traffic sends to; ARGS=\"--dataset banking77 --n 500\" skips its menu"
	@echo "  JUDGE=$(JUDGE)"
	@echo "      mock: the mock judge, which answers from labelled data and runs no model;"
	@echo "      decider: the local DECIDER_MODEL, which make only sets up and starts then"
	@echo "  MOCK_JUDGE_HOST=$(MOCK_JUDGE_HOST) MOCK_JUDGE_PORT=$(MOCK_JUDGE_PORT)"
	@echo "      where make runs the mock judge"
	@echo "  MOCK_JUDGE_DATASETS=$(MOCK_JUDGE_DATASETS) MOCK_JUDGE_LABELS=$(MOCK_JUDGE_LABELS)"
	@echo "      the datasets whose labels it answers with, and JSONL files of {\"text\", \"label\"} rows"
	@echo "  MOCK_JUDGE_CONFIDENCE=$(MOCK_JUDGE_CONFIDENCE)"
	@echo "      the probability it gives the label"
	@echo "  DECIDER_MODEL=$(DECIDER_MODEL) DECIDER_DEVICE=$(DECIDER_DEVICE)"
	@echo "      with JUDGE=decider: the model make serves (a Hub id or a folder), and its device: cpu, cuda, mps or auto"
	@echo "  DECIDER_HOST=$(DECIDER_HOST) DECIDER_PORT=$(DECIDER_PORT)"
	@echo "      where make runs Decider's server"
	@echo "  DECIDER_LICENSE_ACK=$(DECIDER_LICENSE_ACK)"
	@echo "      yes: the local model is also the judge (Kai-0.6B and Decider-2B: Apache-2.0)"
	@echo "  TYPESAFE_URL=$(TYPESAFE_URL) TYPESAFE_MODEL=$(TYPESAFE_MODEL)"
	@echo "      TypeSafe's API (its root URL) and model: the System One model instead"
	@echo "      of the judge, when .env has a TYPESAFE_API_KEY (make asks, in a terminal)"
	@echo "  RUN_DIR=$(RUN_DIR)"
	@echo "      the pids and logs of what make starts"

venv: $(INSTALLED) ## Set up the virtualenv with system-none[dev] (the Python targets do it first)

# ruff, ty and bandit are the versions the lint extra pins, which CI runs too. clippy runs its
# pedantic lints as well, which gateway/Cargo.toml's [lints] turns on.
lint: $(INSTALLED) ## ruff, ty and bandit on the Python code, cargo fmt and pedantic clippy on the gateway, as in CI
	$(VENV)/bin/ruff check
	$(VENV)/bin/ruff format --check
	$(VENV)/bin/ty check --python $(VENV)
	$(VENV)/bin/bandit -c pyproject.toml -r src tests -ll
	cargo fmt --manifest-path gateway/Cargo.toml --check
	cargo clippy --manifest-path gateway/Cargo.toml --all-targets --locked -- -D warnings

fmt: $(INSTALLED) ## Format the Python code and sort its imports with ruff
	$(VENV)/bin/ruff check --select I --fix
	$(VENV)/bin/ruff format

test: $(INSTALLED) ## Python unit tests and the Rust tests, as in CI
	$(PYTEST) -m "not slow and not gateway" $(PYTEST_ARGS)
	ORT_DYLIB_PATH="$$($(ORT_LIBRARY))" cargo test --manifest-path gateway/Cargo.toml --locked

test-tiny: $(INSTALLED) ## Tiny end to end: Rust gateway + Python worker on synthetic traffic, as in CI
	cargo build --manifest-path gateway/Cargo.toml --release --locked
	SN_GATEWAY_BIN=gateway/target/release/system-none-gateway $(PYTEST) -m gateway -v $(PYTEST_ARGS)

test-medium: $(INSTALLED) ## Medium end to end: 3,928 real texts, tiny models from scratch (about 5 minutes)
	SN_TEST_SIZE=medium $(PYTEST) -m slow -v -s $(PYTEST_ARGS)

test-big: $(INSTALLED) ## Big end to end: pretrained DeBERTa-v3, on the GPU if any (about an hour on CPU)
	@$(PY) -c "$$check_gpu"
	SN_TEST_SIZE=big $(PYTEST) -m slow -v -s tests/test_multilingual_ai4privacy.py $(PYTEST_ARGS)

verify: ## TLA+ models of the gateway and worker, checked with TLC (see verification/), as in CI
	bash verification/check.sh
	bash verification/check.sh --original

# Needs $(VENV) only when it runs the CLI from there.
traffic: $(if $(filter $(VENV)/bin/system-none,$(SYSTEM_NONE)),$(INSTALLED)) ## Send texts from a public dataset, picked in a menu, to the gateway at GATEWAY_URL
	$(SYSTEM_NONE) traffic --gateway $(GATEWAY_URL) $(ARGS)

# The admin token comes from .env, which make writes on the first run.
labels: $(if $(filter $(VENV)/bin/system-none,$(SYSTEM_NONE)),$(INSTALLED)) ## Upload a dataset's answers (DATASET=banking77): its train split trains students, its test split audits them
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	  $(SYSTEM_NONE) labels --gateway $(GATEWAY_URL) --dataset $(or $(DATASET),banking77) $(ARGS)

# ---- make: the whole system ------------------------------------------------------------------
# Each part starts in the background unless it already answers (or is still starting), with its
# pid and log in $(RUN_DIR), and make waits until it is healthy. The admin and worker tokens are
# written to .env once, and so is the answer to make's question for a TypeSafe API key; docker
# compose reads the same file.

stack: stack-gateway stack-worker stack-model stack-configure ## Start the gateway, the worker and the mock judge: the judge, and the System One model without a TYPESAFE_API_KEY (the default)
	@echo "ready: usage $(GATEWAY_URL)/ and admin $(GATEWAY_URL)/admin (SN_ADMIN_TOKEN in .env);"
	@echo "make traffic sends test requests; make stack-status, stack-logs and stack-down check, follow and stop"

# Accept both the gateway's readiness flag and older Kai servers' status string.
stack_healthy = curl -sf -m 3 $(1)/health 2>/dev/null | grep -Eq '"(ok|status)": *(true|"ok")'
# $(call stack_alive,name): the process make started for name is running.
stack_alive = { [ -f $(RUN_DIR)/$(1).pid ] && kill -0 "$$(cat $(RUN_DIR)/$(1).pid)" 2>/dev/null; }
# $(call stack_start,name,command): runs command in the background, its output appended to
# $(RUN_DIR)/name.log, so starting it again keeps the log of how the last run ended.
stack_start = nohup $(SETSID) $(2) >> $(RUN_DIR)/$(1).log 2>&1 < /dev/null & echo $$! > $(RUN_DIR)/$(1).pid
# $(call stack_wait,name,url,seconds): waits until url is healthy; fails if name's process stops.
stack_wait = $(call stack_healthy,$(2)) && exit 0; echo "$(1): waiting for $(2)/health (at most $(3)s)"; \
	i=0; until $(call stack_healthy,$(2)); do \
	  if [ -f $(RUN_DIR)/$(1).pid ] && ! $(call stack_alive,$(1)); then \
	    echo "$(1) stopped. The end of $(RUN_DIR)/$(1).log:" >&2; tail -n 20 $(RUN_DIR)/$(1).log >&2; exit 1; fi; \
	  if [ $$i -ge $(3) ]; then echo "$(1) isn't healthy after $(3)s; see $(RUN_DIR)/$(1).log" >&2; exit 1; fi; \
	  i=$$((i + 1)); sleep 1; done; echo "$(1): up at $(2)"
# The variables in .env, for what make starts and configures (a dry run, make -n, may have none).
stack_env = set -a; [ ! -e .env ] || . ./.env; set +a

# A TypeSafe API key in the environment (or on the command line) goes to .env too. Without one,
# make asks for it once, in a terminal: .env keeps the answer, empty for none (delete that line to
# be asked again). Without a terminal (CI, scripts), it never asks. The key only goes through
# shell builtins, never a command's arguments, which any process list shows.
stack-tokens:
	@[ -e .env ] || (umask 077; : > .env)
	@for var in SN_ADMIN_TOKEN SN_WORKER_TOKEN; do grep -q "^$$var=." .env && continue; \
	  if [ -s .env ] && [ -n "$$(tail -c 1 .env)" ]; then echo >> .env; fi; \
	  echo "$$var=$$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')" >> .env; echo "wrote a new $$var to .env"; done
	@save() { case $$1 in *[!A-Za-z0-9._+/=-]*) echo "A TypeSafe API key has only letters, digits and ._+/=-" >&2; exit 1;; esac; \
	  rest=$$(grep -v '^TYPESAFE_API_KEY=' .env); { [ -z "$$rest" ] || printf '%s\n' "$$rest"; printf 'TYPESAFE_API_KEY=%s\n' "$$1"; } > .env; \
	  if [ -z "$$1" ]; then echo "no TypeSafe API key: $(STACK_JUDGE_ABOUT) answers (delete TYPESAFE_API_KEY from .env to be asked again)"; return; fi; \
	  echo "wrote TYPESAFE_API_KEY to .env"; if $(call stack_healthy,$(GATEWAY_URL)); then \
	    echo "The gateway already running at $(GATEWAY_URL) doesn't read it until it restarts (make stack-down, then make, if make started it)." >&2; fi; }; \
	key=$$(sed -n 's/^TYPESAFE_API_KEY=//p' .env | tail -n 1); \
	if [ -n "$$TYPESAFE_API_KEY" ]; then [ "$$TYPESAFE_API_KEY" = "$$key" ] || { key=$$TYPESAFE_API_KEY; save "$$key"; }; \
	elif ! grep -q '^TYPESAFE_API_KEY=' .env && [ -t 0 ]; then \
	  echo "With a TypeSafe API key, TypeSafe's API answers as the gateway's System One model ($(STACK_JUDGE_ABOUT) stays the judge)."; \
	  printf 'TypeSafe API key (Enter for none: $(STACK_JUDGE_ABOUT) answers): '; trap 'stty echo 2>/dev/null; exit 130' INT TERM HUP; \
	  stty -echo 2>/dev/null; read -r key; stty echo 2>/dev/null; echo; save "$$key"; fi

# Its database, inputs and students go to gateway-data/, as when you run it by hand.
stack-gateway: stack-tokens $(INSTALLED)
	@mkdir -p $(RUN_DIR)
	@if $(call stack_healthy,$(GATEWAY_URL)); then echo "gateway: already running at $(GATEWAY_URL)"; \
	elif ! $(call stack_alive,gateway); then \
	  cargo build --manifest-path gateway/Cargo.toml --release --locked || exit 1; \
	  echo "gateway: starting at $(GATEWAY_URL), log in $(RUN_DIR)/gateway.log"; \
	  $(stack_env); ORT_DYLIB_PATH="$$($(ORT_LIBRARY))" \
	    $(call stack_start,gateway,$(GATEWAY_BIN) --listen $(GATEWAY_HOST):$(GATEWAY_PORT)); fi
	@$(call stack_wait,gateway,$(GATEWAY_URL),60)

stack-worker: stack-gateway
	@if $(call stack_alive,worker); then echo "worker: already running (pid $$(cat $(RUN_DIR)/worker.pid))"; else \
	  echo "worker: starting, log in $(RUN_DIR)/worker.log"; \
	  $(stack_env); PYTHONUNBUFFERED=1 $(call stack_start,worker,$(VENV)/bin/system-none worker --gateway $(GATEWAY_URL)); \
	  sleep 3; $(call stack_alive,worker) || { echo "worker stopped. The end of $(RUN_DIR)/worker.log:" >&2; \
	    tail -n 20 $(RUN_DIR)/worker.log >&2; exit 1; }; fi

# The selected model weights are downloaded to the Hugging Face cache first, so
# the download shows its progress here.
DECIDER_DOWNLOAD = $(DECIDER_VENV)/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$(DECIDER_MODEL)"

# The judge: the mock judge, or Decider with JUDGE=decider. Only then does make set up
# $(DECIDER_VENV), download DECIDER_MODEL and start it, with or without a TypeSafe API key.
stack-model: $(if $(filter decider,$(JUDGE)),stack-decider,stack-mock-judge)

MOCK_JUDGE_CMD = $(VENV)/bin/system-none mock-judge --host $(MOCK_JUDGE_HOST) --port $(MOCK_JUDGE_PORT) \
	--confidence $(MOCK_JUDGE_CONFIDENCE) --datasets $(MOCK_JUDGE_DATASETS) $(foreach f,$(MOCK_JUDGE_LABELS),--labels $(f))

stack-mock-judge: $(INSTALLED) ## Start the mock judge, which answers each text with its label in MOCK_JUDGE_DATASETS and MOCK_JUDGE_LABELS
	@mkdir -p $(RUN_DIR)
	@if $(call stack_healthy,$(MOCK_JUDGE_URL)); then echo "mock-judge: already running at $(MOCK_JUDGE_URL)"; \
	elif ! $(call stack_alive,mock-judge); then \
	  echo "mock-judge: starting at $(MOCK_JUDGE_URL), log in $(RUN_DIR)/mock-judge.log"; \
	  PYTHONUNBUFFERED=1 $(call stack_start,mock-judge,$(MOCK_JUDGE_CMD)); fi
	@$(call stack_wait,mock-judge,$(MOCK_JUDGE_URL),120)

stack-decider: $(DECIDER_INSTALLED) ## Start DECIDER_MODEL (Decision-2.0-Kai-0.6B) only; make JUDGE=decider also makes it the judge
	@mkdir -p $(RUN_DIR)
	@if $(call stack_healthy,$(DECIDER_URL)); then echo "decider: already running at $(DECIDER_URL)"; \
	elif ! $(call stack_alive,decider); then \
	  [ -d "$(DECIDER_MODEL)" ] || $(DECIDER_DOWNLOAD) || exit 1; \
	  echo "decider: starting $(DECIDER_MODEL) on $(DECIDER_DEVICE) at $(DECIDER_URL), log in $(RUN_DIR)/decider.log"; \
	  DECIDER_MODEL="$(DECIDER_MODEL)" DECIDER_DEVICE="$(DECIDER_DEVICE)" \
	    DECIDER_CPU_CONCURRENCY="$(DECIDER_CPU_CONCURRENCY)" DECIDER_THREADS="$(DECIDER_THREADS)" \
	    DECIDER_WORKERS="$(DECIDER_WORKERS)" \
	    $(call stack_start,decider,$(DECIDER_VENV)/bin/uvicorn $(DECIDER_APP) --host $(DECIDER_HOST) --port $(DECIDER_PORT)); fi
	@$(call stack_wait,decider,$(DECIDER_URL),$(DECIDER_TIMEOUT))

# Through the admin API (the token is piped to curl, so it never shows in a process list).
# Every run applies the same settings again: the judge (named mock, or decider with
# JUDGE=decider) is the gateway's default judge and, without a TypeSafe API key, its default
# System One model. With one, the System One model is TypeSafe's API, named typesafe, whose key
# the gateway reads from TYPESAFE_API_KEY; it can't be the judge (TypeSafe's terms forbid training
# on Jev's outputs).
stack-configure: stack-gateway ## Make the mock judge (Decider with JUDGE=decider) the gateway's judge and default System One model (TypeSafe's API, with a TYPESAFE_API_KEY), again
	@$(stack_env); \
	put() { code=$$(printf 'header = "authorization: Bearer %s"\n' "$$SN_ADMIN_TOKEN" | curl -s -K - -X PUT \
	    -H 'content-type: application/json' -d "$$2" -o $(RUN_DIR)/configure.out -w '%{http_code}' "$(GATEWAY_URL)/admin/api/$$1"); \
	  if [ "$$code" = 200 ]; then echo "configured $$1"; return; fi; \
	  echo "PUT $(GATEWAY_URL)/admin/api/$$1 -> $$code: $$(cat $(RUN_DIR)/configure.out)" >&2; \
	  [ "$$code" != 401 ] || echo "The gateway at $(GATEWAY_URL) doesn't take the SN_ADMIN_TOKEN in .env." >&2; exit 1; }; \
	if [ -n "$$TYPESAFE_API_KEY" ]; then one=typesafe; \
	  put system-one-models/typesafe '{"kind": "systemone", "base_url": "$(TYPESAFE_URL)", "model": "$(TYPESAFE_MODEL)", "api_key_env": "TYPESAFE_API_KEY"}'; \
	else one=$(JUDGE); \
	  put system-one-models/$(JUDGE) '{"kind": "systemone", "base_url": "$(STACK_JUDGE_URL)", "model": "$(STACK_JUDGE_MODEL)"}'; fi; \
	if [ $(JUDGE) = decider ] && [ "$(DECIDER_LICENSE_ACK)" != yes ]; then \
	  put settings "{\"default_system_one\": \"$$one\"}"; \
	  echo "judge: not configured. Once you've checked that the license of $(DECIDER_MODEL) allows training on its outputs: make JUDGE=decider DECIDER_LICENSE_ACK=yes"; \
	else \
	  put judges/$(JUDGE) '{"base_url": "$(STACK_JUDGE_URL)", "model": "$(STACK_JUDGE_MODEL)", "mode": "systemone", "license_ack": true}'; \
	  put settings "{\"default_system_one\": \"$$one\", \"default_judge\": \"$(JUDGE)\"}"; fi

# The judge JUDGE doesn't pick shows only when something answers for it, or make started it.
stack-status: ## Which parts of the system are up
	@for s in "gateway $(GATEWAY_URL) mock decider" "mock-judge $(MOCK_JUDGE_URL) mock" "decider $(DECIDER_URL) decider"; do \
	  set -- $$s; if $(call stack_healthy,$$2); then st="up at $$2"; elif $(call stack_alive,$$1); then st=starting; \
	  else case " $$* " in *" $(JUDGE) "*) st=down;; *) continue;; esac; fi; \
	  if $(call stack_alive,$$1); then st="$$st (pid $$(cat $(RUN_DIR)/$$1.pid))"; fi; echo "$$1: $$st"; done
	@if $(call stack_alive,worker); then echo "worker: up (pid $$(cat $(RUN_DIR)/worker.pid))"; else echo "worker: down"; fi

stack-logs: ## Follow the logs of what make started
	tail -n 20 -F $(RUN_DIR)/*.log

stack-down: ## Stop what make started (not what it found running)
	@for s in worker gateway mock-judge decider; do \
	  if $(call stack_alive,$$s); then pid=$$(cat $(RUN_DIR)/$$s.pid); kill $$pid; i=0; \
	    while kill -0 $$pid 2>/dev/null && [ $$i -lt 30 ]; do sleep 1; i=$$((i + 1)); done; \
	    if kill -0 $$pid 2>/dev/null; then echo "$$s: still stopping (pid $$pid)"; else echo "$$s: stopped"; fi; fi; \
	  rm -f $(RUN_DIR)/$$s.pid; done

# The big run trains with device "auto": on a GPU when PyTorch can use one, otherwise on the CPU.
# Stop before the run, not an hour into it, when a CUDA build of PyTorch can't use the GPU that
# nvidia-smi sees: its driver is too old for the build, or the build has no kernels for it.
define check_gpu
import os, subprocess, torch
fix = ("Train on the CPU with: CUDA_VISIBLE_DEVICES= make test-big\n"
       "or rebuild $(VENV) with a PyTorch build for this GPU and driver (see pytorch.org/get-started), e.g.:\n"
       "  rm -rf $(VENV) && make test-big TORCH_INDEX=https://download.pytorch.org/whl/cu126")
if not torch.cuda.is_available():
    try:
        gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except OSError:
        gpus = ""
    if torch.version.cuda and gpus and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit(f"nvidia-smi sees {gpus}\nbut PyTorch {torch.__version__} can't use it (see any warning above).\n{fix}")
    print(f"No GPU for PyTorch {torch.__version__}: training on the CPU")
    raise SystemExit
name = torch.cuda.get_device_name()
try:
    torch.ones(1, device="cuda").add_(1).item()
except Exception as e:
    raise SystemExit(f"PyTorch {torch.__version__} sees {name} but can't run on it: {e}\n{fix}")
print(f"Training on {name} with PyTorch {torch.__version__}")
endef
export check_gpu

$(PY):
	$(call new_venv,$(VENV))

# PyTorch first, from TORCH_INDEX (the build for this machine), at the version uv.lock pins; its
# own dependencies come with everything else, from PyPI, where every locked version is.
$(INSTALLED): pyproject.toml uv.lock | $(PY)
	$(export_lock)
	$(call pip_install,$(VENV),--no-deps torch $(constrain) $(if $(TORCH_INDEX),--index-url $(TORCH_INDEX)))
	$(call pip_install,$(VENV),-e ".[dev]" $(constrain))
	touch $@

$(DECIDER_VENV)/bin/python:
	$(if $(DECISION_RUNTIME),,$(if $(UV),,$(decider_python_check)))
	$(call new_venv,$(DECIDER_VENV))

$(DECIDER_INSTALLED): Makefile src/system_none/decision_server.py | $(DECIDER_VENV)/bin/python
	$(call pip_install,$(DECIDER_VENV),torch $(if $(DECIDER_TORCH_INDEX),--index-url $(DECIDER_TORCH_INDEX)))
	$(if $(DECISION_RUNTIME),$(call pip_install,$(DECIDER_VENV),-e ".[serve]" "transformers>=5.17" safetensors),$(call pip_install,$(DECIDER_VENV),"$(DECIDER_PACKAGE)"))
	touch $@

# Machine-specific targets (such as starting local model servers) can live in an untracked local.mk.
-include local.mk

ifeq ($(filter mock decider,$(JUDGE)),)
$(error JUDGE is mock (the default) or decider, not '$(JUDGE)')
endif
