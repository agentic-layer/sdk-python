"""Pytest configuration for tests."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to only use asyncio backend (trio is not installed)."""
    return "asyncio"
