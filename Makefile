.PHONY: all
all: build


.PHONY: build
build:
	uv sync --all-packages
	cd msaf && uv sync


.PHONY: test
test: build
	uv run pytest
	cd msaf && uv run pytest


.PHONY: check
check: build
	uv run ruff check
	uv run mypy .
	uv run bandit -c pyproject.toml -r .
	cd msaf && uv run ruff check
	cd msaf && uv run mypy .
	cd msaf && uv run bandit -c pyproject.toml -r .
	make test


.PHONY: check-fix
check-fix: build
	uv run ruff format
	uv run ruff check --fix
	cd msaf && uv run ruff format
	cd msaf && uv run ruff check --fix
