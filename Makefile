.PHONY: all
all: build


.PHONY: build
build:
	uv sync --all-packages


.PHONY: test
test: build
	uv run pytest


.PHONY: check
check: build
	uv run ruff check
	uv run mypy .
	uv run bandit -c pyproject.toml -r .
	make test


.PHONY: check-fix
check-fix: build
	uv run ruff format
	uv run ruff check --fix
