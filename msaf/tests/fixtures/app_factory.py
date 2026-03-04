"""App factory fixture for creating Starlette apps with MSAF agents."""

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from agent_framework import Agent
from agenticlayer_shared.config import McpTool, SubAgent
from asgi_lifespan import LifespanManager
from httpx_retries import Retry

from agenticlayer_msaf.agent import MsafAgentFactory
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
        sub_agents: list[SubAgent] | None = None,
        tools: list[McpTool] | None = None,
    ) -> AsyncIterator[Any]:
        rpc_url = "http://localhost:80/"
        app = to_a2a(
            agent=agent,
            name=name,
            rpc_url=rpc_url,
            description=description,
            sub_agents=sub_agents,
            tools=tools,
            agent_factory=MsafAgentFactory(retry=Retry(total=2)),
        )
        async with LifespanManager(app) as manager:
            yield manager.app

    return _create_app
