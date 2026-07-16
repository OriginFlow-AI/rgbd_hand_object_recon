PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: help setup demo verify test lint check clean

help:
	@echo "setup   Create a virtual environment and install development dependencies"
	@echo "demo    Run the mock RGB-D reconstruction"
	@echo "verify  Run the synthetic ICP verification"
	@echo "test    Run the pytest suite"
	@echo "lint    Run Ruff and byte-compile source files"
	@echo "check   Run lint, tests, and verification"
	@echo "clean   Remove only Python caches (keeps data and outputs)"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e '.[dev]'

demo:
	PYTHONPATH=src $(PYTHON) -m hand_recon demo --config configs/mock_rgbd.json

verify:
	PYTHONPATH=src $(PYTHON) -m hand_recon verify

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests demo examples scripts
	$(PYTHON) -m compileall -q src tests demo examples scripts

check: lint test verify

clean:
	find src tests demo examples scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
