"""App factory fixture for creating Starlette apps with agents."""

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from agenticlayer.agent import AgentFactory
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.config import McpTool, SubAgent
from asgi_lifespan import LifespanManager
from google.adk.agents import LlmAgent
from httpx_retries import Retry


@pytest_asyncio.fixture
def app_factory() -> Any:
    """
    Factory fixture to create Starlette apps with agents.

    This fixture provides a context manager that:
    - Creates a Starlette app from an LlmAgent
    - Manages the app lifecycle (startup/shutdown)
    - Supports optional sub-agents and MCP tools configuration

    Usage:
        async with app_factory(agent) as app:
            client = TestClient(app)
            # Use the client for testing

        async with app_factory(agent, sub_agents=[...], tools=[...]) as app:
            client = TestClient(app)
            # Use the client for testing with sub-agents and tools
    """

    @contextlib.asynccontextmanager
    async def _create_app(
        agent: LlmAgent,
        sub_agents: list[SubAgent] | None = None,
        tools: list[McpTool] | None = None,
    ) -> AsyncIterator[Any]:
        rpc_url = "http://localhost:80/"
        app = to_a2a(
            agent=agent,
            rpc_url=rpc_url,
            sub_agents=sub_agents,
            tools=tools,
            agent_factory=AgentFactory(retry=Retry(total=2)),
        )
        async with LifespanManager(app) as manager:
            yield manager.app

    return _create_app
