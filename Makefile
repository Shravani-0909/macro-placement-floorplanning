.PHONY: test lint

test:
	uv run --frozen pytest

lint:
	uv run --frozen ruff check .
	uv run --frozen mypy src tests
