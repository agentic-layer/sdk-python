# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python SDK monorepo for the Agentic Layer platform that helps convert Google ADK agents into instrumented web applications. The main component is the ADK Python SDK located in the `adk/` directory.

## Architecture

### Key Components
- **agent_to_a2a.py**: Core conversion logic that transforms Google ADK agents into Starlette web applications with A2A protocol support
- **otel.py**: OpenTelemetry instrumentation setup (tracing, metrics, logging)
- **callback_tracer_plugin.py**: Custom tracing plugin for agent callbacks

### Package Structure
- Root package: `agentic-layer-sdk` (meta-package)
- Main SDK: `agentic-layer-sdk-adk` in `adk/agenticlayer/`
- Uses uv workspace configuration with `adk/` as a workspace member

## Development Commands

### Build and Dependencies
```bash
make build              # Install dependencies with uv sync --all-packages
```

### Testing
```bash
make test              # Run pytest tests
uv run pytest         # Run tests directly
```

### Type Checking and Linting
```bash
make check             # Run full check suite (mypy, ruff, bandit)
uv run mypy .          # Type check with mypy
uv run ruff check      # Lint with ruff
uv run ruff format     # Format code with ruff
make check-fix         # Format and auto-fix linting issues
```

### Security
```bash
uv run bandit -c pyproject.toml -r .  # Security analysis
```

## Testing

- Tests are located in `adk/tests/`
- Uses pytest with configuration in `pyproject.toml`
- Test coverage reporting with pytest-cov
- Single test file currently: `test_a2a_starlette.py`

## Key Dependencies

- **google-adk[a2a]**: Google Agent Development Kit with A2A protocol support
- **starlette**: ASGI web framework for creating the web application
- **opentelemetry**: Full observability stack (tracing, metrics, logging)
- **openinference-instrumentation-google-adk**: Specialized instrumentation for Google ADK

## Configuration

- Python 3.12+ required
- Line length: 120 characters (ruff configuration)
- Strict mypy type checking enabled
- Pre-commit hooks configured for code quality

## Release Process

Create and push a git tag with semantic versioning (e.g., `v0.1.0`) to trigger automatic PyPI publication via GitHub Actions.
