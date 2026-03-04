.PHONY: all
all: build


.PHONY: build
build:
	cd adk && uv sync
	cd msaf && uv sync


.PHONY: test
test: build
	cd adk && uv run pytest
	cd msaf && uv run pytest


.PHONY: check
check: build
	cd adk && uv run ruff check
	cd adk && uv run mypy .
	cd adk && uv run bandit -c pyproject.toml -r .
	cd msaf && uv run ruff check
	cd msaf && uv run mypy .
	cd msaf && uv run bandit -c pyproject.toml -r .
	make test


.PHONY: check-fix
check-fix: build
	cd adk && uv run ruff format
	cd adk && uv run ruff check --fix
	cd msaf && uv run ruff format
	cd msaf && uv run ruff check --fix
