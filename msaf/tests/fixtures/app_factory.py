"""App factory fixture for creating Starlette apps with MSAF agents."""

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from agent_framework import Agent
from asgi_lifespan import LifespanManager

from agenticlayer_msaf.agent_to_a2a import to_a2a


@pytest_asyncio.fixture
def msaf_app_factory() -> Any:
    """
    Factory fixture to create Starlette apps with MSAF agents.

    Usage:
        async with msaf_app_factory(agent, name="MyAgent") as app:
            client = TestClient(app)
    """

    @contextlib.asynccontextmanager
    async def _create_app(
        agent: Agent[Any],
        name: str = "test_agent",
        description: str | None = "Test agent",
    ) -> AsyncIterator[Any]:
        rpc_url = "http://localhost:80/"
        app = to_a2a(
            agent=agent,
            name=name,
            rpc_url=rpc_url,
            description=description,
        )
        async with LifespanManager(app) as manager:
            yield manager.app

    return _create_app
