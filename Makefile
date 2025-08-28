.PHONY: all
all: build


.PHONY: build
build:
	uv sync --all-packages


.PHONY: test
test: build
	uv run pytest


.PHONY: check
check: build test
	uv run mypy .
	uv run ruff check
	uv run bandit -c pyproject.toml -r .
	uv export --frozen --no-hashes | uv run pip-audit -r /dev/stdin


.PHONY: check-fix
check-fix: build
	uv run ruff format
	uv run ruff check --fix
