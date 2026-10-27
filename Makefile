.PHONY: test lint

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy src tests
