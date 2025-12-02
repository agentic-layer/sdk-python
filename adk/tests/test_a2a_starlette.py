import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from agenticlayer.agent import load_agent
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.config import InteractionType, McpTool, SubAgent
from asgi_lifespan import LifespanManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import AnyHttpUrl
from starlette.testclient import TestClient


def create_mock_agent_card(
    agent_name: str,
    base_url: str,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Helper function to create a valid agent card response."""
    return {
        "name": agent_name,
        "description": f"Mock agent {agent_name}",
        "url": base_url,
        "version": "1.0.0",
        "capabilities": {},
        "skills": skills or [],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "supports_authenticated_extended_card": False,
    }


def create_send_message_request(
    message_text: str = "Hello, agent!",
) -> dict[str, Any]:
    """Helper function to create a valid A2A send message request."""
    message_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message_text}],
                "messageId": message_id,
                "contextId": context_id,
            },
            "metadata": {},
        },
    }


def create_agent(
    name: str = "test_agent",
    sub_agents: list[SubAgent] | None = None,
    tools: list[McpTool] | None = None,
) -> LlmAgent:
    agent = LlmAgent(
        name=name,
        model=LiteLlm(model="gemini/gemini-2.5-flash"),
        description="Test agent",
        instruction="You are a test agent.",
    )
    return load_agent(
        agent=agent,
        sub_agents=sub_agents or [],
        tools=tools or [],
    )


rpc_url = "http://localhost:80/"


@pytest_asyncio.fixture
def app_factory() -> Any:
    @contextlib.asynccontextmanager
    async def _create_app(agent: LlmAgent) -> AsyncIterator[Any]:
        app = to_a2a(agent, rpc_url)
        async with LifespanManager(app) as manager:
            yield manager.app

    return _create_app


class TestA2AStarlette:
    @pytest.mark.asyncio
    async def test_agent_card(self, app_factory: Any) -> None:
        """Test that the agent card is available at /.well-known/agent-card.json"""

        # Given:
        agent = create_agent()
        async with app_factory(agent) as app:
            client = TestClient(app)

            # When: Requesting the agent card endpoint
            response = client.get("/.well-known/agent-card.json")

            # Then: Agent card is returned
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict), "Agent card should return a JSON object"
            assert data.get("name") == agent.name
            assert data.get("description") == agent.description

    @pytest.mark.asyncio
    async def test_agent_rpc_send_message(self, app_factory: Any) -> None:
        """Test that the RPC url is working for send message."""

        # Given:
        agent = create_agent()
        async with app_factory(agent) as app:
            client = TestClient(app)

            # When: Sending an A2A RPC request
            rpc_response = client.post("", json=create_send_message_request())

            # Then: RPC response is returned
            assert rpc_response.status_code == 200
            rpc_data = rpc_response.json()
            assert rpc_data.get("jsonrpc") == "2.0"
            assert rpc_data.get("id") == 1

    @pytest.mark.asyncio
    async def test_sub_agents(self, app_factory: Any) -> None:
        """Test that sub-agents are integrated correctly."""

        # When: Creating an agent with sub-agents
        sub_agents = [
            SubAgent(
                name="sub_agent_1",
                url=AnyHttpUrl("http://sub-agent-1.local/.well-known/agent-card.json"),
                interaction_type=InteractionType.TRANSFER,
            ),
            SubAgent(
                name="sub_agent_2",
                url=AnyHttpUrl("http://sub-agent-2.local/.well-known/agent-card.json"),
                interaction_type=InteractionType.TOOL_CALL,
            ),
        ]
        agent = create_agent(sub_agents=sub_agents)

        # Then: Verify sub-agents and tools are integrated correctly
        assert len(agent.sub_agents) == 1, "There should be 1 sub-agent for transfer interaction type"
        assert len(agent.tools) == 1, "There should be 1 agent tool for tool_call interaction type"

        # When: Requesting the agent card endpoint
        async with app_factory(agent) as app:
            client = TestClient(app)
            response = client.get("/.well-known/agent-card.json")

            # Then: Agent card is returned
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_tools(self) -> None:
        """Test that tools are integrated correctly."""

        # When: Creating an agent with tools
        tools = [
            McpTool(name="tool_1", url=AnyHttpUrl("http://tool-1.local/mcp")),
            McpTool(name="tool_2", url=AnyHttpUrl("http://tool-2.local/mcp")),
        ]
        agent = create_agent(tools=tools)

        # Then: Verify McpToolsets are created and integrated correctly
        assert len(agent.tools) == 2, "There should be 2 McpToolset tools"

        # Note: Further integration tests would require mocking MCP tool behavior
