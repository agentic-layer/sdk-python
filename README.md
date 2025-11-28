# Python SDKs for Agentic Layer

SDKs for Python that helps to get agents configured in the Agentic Layer quickly.

## SDKs

This repository contains the following SDKs:

- [ADK Python SDK](./adk/README.md)

## Development

To build and test locally:

```shell
# Install dependencies on MacOS with Homebrew
brew bundle
# Install pre-commit hooks (if you want to commit code)
pre-commit install
# Install Python dependencies
make build
# Run tests and other checks
make check
```

## Creating a release

Create and push a GIT tag like `v0.1.0` and GitHub workflows will build and publish the package to PyPI.
Follow [Semantic Versioning](https://semver.org/).
