.PHONY: help install install-dev test test-fast lint format typecheck clean ci

help:
	@echo "Layra development commands:"
	@echo "  make install       - Install project in editable mode"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-fast     - Run fast tests only (exclude slow/mlx/sam3/rife)"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Run ruff formatter"
	@echo "  make typecheck     - Run mypy"
	@echo "  make ci            - Run lint + typecheck + test-fast"
	@echo "  make clean         - Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -m "not slow and not mlx and not sam3 and not rife"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

ci: lint typecheck test-fast

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
