.PHONY: install lint typecheck test test-cov bench check

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/tsfeat

test:
	pytest

test-cov:
	pytest --cov=tsfeat --cov-report=term-missing

bench:
	python -m benchmarks.benchmark_rolling
	python -m benchmarks.benchmark_cross_sectional
	python -m benchmarks.benchmark_portfolio

check: lint typecheck test
