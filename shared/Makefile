.PHONY: all
all: build


.PHONY: build
build:
	uv sync


.PHONY: test
test: build


.PHONY: check
check: build
	uv run ruff check
	uv run mypy --no-incremental .
	make test


.PHONY: check-fix
check-fix: build
	uv run ruff format
	uv run ruff check --fix


.PHONY: update
update:
	uv sync -U
