.PHONY: help install dev lint format test ingest eval serve pipeline clean

PY_PATHS = backend graph ingest evaluation ui run_backend.py pipeline.py tests

help:
	@echo "Targets: install dev lint format test ingest eval serve pipeline clean"

install:  ## Install the package
	pip install -e .

dev:  ## Install with dev tools and git hooks
	pip install -e ".[dev]"
	pre-commit install

lint:  ## Run ruff, black --check, and mypy
	ruff check $(PY_PATHS)
	black --check $(PY_PATHS)
	mypy backend graph ingest evaluation ui run_backend.py pipeline.py

format:  ## Auto-fix lint issues and format
	ruff check --fix $(PY_PATHS)
	black $(PY_PATHS)

test:  ## Run the unit tests
	RETRIEVAL_EVAL_REPORT=evaluation/results/retrieval_eval.json python -m unittest discover -s tests -v

ingest:  ## Build the retrieval index
	graph-rag-ingest

eval:  ## Evaluate retrieval against the baseline
	graph-rag-eval --fail-under evaluation/baseline.json

serve:  ## Run the FastAPI backend
	graph-rag-serve

pipeline:  ## Build the index and evaluate in one step
	graph-rag-pipeline

clean:  ## Remove generated artifacts and caches
	rm -rf data/index evaluation/results .pytest_cache .ruff_cache .mypy_cache
