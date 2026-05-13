.PHONY: install dev test lint type fmt clean data run paper all
.DEFAULT_GOAL := help

PYTHON := python3
VENV := .venv
ACT := source $(VENV)/bin/activate &&

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(ACT) pip install -U pip wheel setuptools

install: $(VENV)/bin/activate  ## Install runtime deps
	$(ACT) pip install -e ".[agents,viz]"

dev: $(VENV)/bin/activate  ## Install dev deps
	$(ACT) pip install -e ".[agents,viz,dev]"
	$(ACT) pre-commit install || true

test:  ## Run all tests
	$(ACT) pytest -v

test-unit:  ## Run unit tests only (fast)
	$(ACT) pytest -v -m unit

test-int:  ## Run integration tests (loads real models)
	$(ACT) pytest -v -m integration

test-e2e:  ## Run end-to-end tests (needs API key)
	$(ACT) pytest -v -m e2e

cov:  ## Test with coverage
	$(ACT) pytest --cov --cov-report=term-missing --cov-report=html

lint:  ## Lint with ruff
	$(ACT) ruff check src tests

fmt:  ## Format with ruff
	$(ACT) ruff format src tests
	$(ACT) ruff check --fix src tests

type:  ## Type check with mypy
	$(ACT) mypy src

check: lint type test-unit  ## Quick CI check

data:  ## Download and prepare datasets
	$(ACT) python scripts/download_jobsdf.py
	$(ACT) python scripts/build_synthetic.py

run-demo:  ## Run end-to-end demo on sample data
	$(ACT) python scripts/demo.py

eval:  ## Run full ablation experiments
	$(ACT) python scripts/run_experiments.py --config configs/experiments.yaml

eval-quick:  ## Quick smoke run (5 jobs, no Multi-Agent)
	$(ACT) python scripts/run_experiments.py --max-jobs 5 --filter "bm25,tfidf,semantic_only,semantic_plus_bidirectional,full_no_agent"

summary:  ## Print the latest ablation table
	$(ACT) python scripts/summarize_ablation.py

demo-async:  ## Live event-stream demo against real API
	$(ACT) python scripts/demo_async.py

paper:  ## Compile arXiv paper
	cd paper && latexmk -pdf main.tex

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
