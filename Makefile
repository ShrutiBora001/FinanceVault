.DEFAULT_GOAL := help
SHELL := /bin/bash

# Docker Desktop does not always land on the inherited PATH. Put its whole bin directory
# there rather than just the binary: `docker` shells out to `docker-credential-desktop`,
# which lives alongside it and fails obscurely when only the binary is reachable.
DOCKER_BIN := /Applications/Docker.app/Contents/Resources/bin
export PATH := $(PATH):$(DOCKER_BIN)

# Absolute path for the binary (make execs recipes directly, ignoring the PATH it just
# exported), plus the exported PATH above so the child process finds its credential helper.
DOCKER := $(shell command -v docker 2>/dev/null || echo $(DOCKER_BIN)/docker)
COMPOSE := $(DOCKER) compose -f infra/docker-compose.yml

VENV   := .venv
PY     := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff

.PHONY: help venv up down logs psql redis-cli migrate reset test lint fmt clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment

venv:  ## create .venv (python 3.12) and install the package with dev extras
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	@echo "ok — activate with: source $(VENV)/bin/activate"

venv-embed:  ## add the embedding extra (pulls torch, ~2.5GB); needed from `make ingest` on
	uv pip install --python $(PY) -e ".[dev,embed]"

up:  ## start postgres+pgvector and redis, wait for both to pass healthcheck
	$(COMPOSE) up -d --wait
	@echo "ok — postgres on :55432, redis on :55433"

down:  ## stop the stack, keep the volume
	$(COMPOSE) down

logs:  ## tail container logs
	$(COMPOSE) logs -f

psql:  ## open a psql shell in the postgres container
	$(COMPOSE) exec postgres psql -U financevault -d financevault

redis-cli:  ## open a redis shell
	$(COMPOSE) exec redis redis-cli

# ---------------------------------------------------------------- schema

migrate:  ## apply schema.sql and print row counts (idempotent)
	$(PY) scripts/migrate.py

ingest:  ## fetch filings, chunks, XBRL facts and prices (TICKERS=AAPL by default)
	$(PY) scripts/ingest.py --tickers $(or $(TICKERS),AAPL)

# ---------------------------------------------------------------- evaluation

split:  ## regenerate the frozen eval split (changes the benchmark; needs --force)
	$(PY) scripts/make_split.py

eval:  ## run every policy over the frozen split, live
	$(PY) -m eval.harness $(ARGS)

replay:  ## re-run the same sweep from the journal; must cost $0.00
	FV_REPLAY=true $(PY) -m eval.harness --out eval/results/mvp1_replay.json $(ARGS)

determinism:  ## F1/F2: compare the live and replayed metric tables
	$(PY) scripts/determinism.py

reset:  ## DESTRUCTIVE: drop the postgres volume and re-migrate from empty
	$(COMPOSE) down -v
	$(MAKE) up
	$(MAKE) migrate

# ---------------------------------------------------------------- quality

test:  ## run the test suite
	$(PYTEST)

lint:  ## check formatting and lint rules
	$(RUFF) check src tests scripts
	$(RUFF) format --check src tests scripts

fmt:  ## apply formatting and autofixable lint rules
	$(RUFF) check --fix src tests scripts
	$(RUFF) format src tests scripts

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
