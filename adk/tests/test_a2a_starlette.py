import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from a2a.client.errors import A2AClientHTTPError
from agenticlayer.agent import AgentFactory
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.config import InteractionType, McpTool, SubAgent
from asgi_lifespan import LifespanManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from httpx import Response
from httpx_retries import Retry
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
) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=LiteLlm(model="gemini/gemini-2.5-flash"),
        description="Test agent",
        instruction="You are a test agent.",
    )


def create_mcp_tool_config(
    name: str = "test_mcp_tool",
    url: str = "http://mcp-server.local/mcp",
    timeout: int = 30,
) -> McpTool:
    """Helper function to create McpTool configuration object."""
    return McpTool(
        name=name,
        url=AnyHttpUrl(url),
        timeout=timeout,
    )


@pytest_asyncio.fixture
def app_factory() -> Any:
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

    @respx.mock
    @pytest.mark.asyncio
    async def test_sub_agents(self, app_factory: Any) -> None:
        """Test that sub-agents are integrated correctly."""

        # Given: Mock sub-agent agent card response
        route_sub_agent_1 = respx.get("http://sub-agent-1.local/.well-known/agent-card.json").mock(
            return_value=Response(
                status_code=200,
                json=create_mock_agent_card(
                    agent_name="sub-agent-1",
                    base_url="http://sub-agent-1.local",
                ),
            )
        )
        route_sub_agent_2 = respx.get("http://sub-agent-2.local/.well-known/agent-card.json").mock(
            return_value=Response(
                status_code=200,
                json=create_mock_agent_card(
                    agent_name="sub-agent-2",
                    base_url="http://sub-agent-2.local",
                ),
            )
        )

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
        agent = create_agent()

        # When: Requesting the agent card endpoint
        async with app_factory(agent=agent, sub_agents=sub_agents) as app:
            client = TestClient(app)
            response = client.get("/.well-known/agent-card.json")

            # Then: Agent card is returned
            assert response.status_code == 200

        # And: Sub-agent agent card endpoints were called
        assert route_sub_agent_1.called, "Sub-agent 1 agent card endpoint was not called"
        assert route_sub_agent_2.called, "Sub-agent 2 agent card endpoint was not called"

    @respx.mock
    @pytest.mark.asyncio
    async def test_sub_agent_unavailable_fails_startup(self, app_factory: Any) -> None:
        """Test that unavailable sub-agents cause app startup to fail with retries."""

        # Given: Mock sub-agent that returns connection errors
        route_unavailable = respx.get("http://unavailable-agent.local/.well-known/agent-card.json").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        agent = create_agent()

        sub_agents = [
            SubAgent(
                name="unavailable_agent",
                url=AnyHttpUrl("http://unavailable-agent.local/.well-known/agent-card.json"),
                interaction_type=InteractionType.TRANSFER,
            ),
        ]

        # Expect: App creation should fail with A2AClientHTTPError
        with pytest.raises(A2AClientHTTPError, match="Network communication error"):
            async with app_factory(agent=agent, sub_agents=sub_agents):
                pass

        # And: The retry mechanism should have been used (total=2 means initial + 2 retries = 3 calls)
        assert route_unavailable.call_count == 3, (
            f"Expected 3 calls (1 initial + 2 retries), got {route_unavailable.call_count}"
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_multiple_from_single_server(self, app_factory: Any, monkeypatch: Any) -> None:
        """Test that multiple tools from a single MCP server are all added to agent instructions."""

        # Given: Mock tools with names and descriptions
        mock_tools = [
            type("MockTool", (), {"name": "get_customer", "description": "Retrieves customer information"})(),
            type("MockTool", (), {"name": "update_customer", "description": "Updates customer records"})(),
            type("MockTool", (), {"name": "delete_customer", "description": "Deletes customer from database"})(),
        ]

        # And: Mock McpToolset.get_tools to return our mock tools
        async def mock_get_tools(self, readonly_context=None):
            return mock_tools

        monkeypatch.setattr("google.adk.tools.mcp_tool.mcp_toolset.McpToolset.get_tools", mock_get_tools)

        # And: Agent and MCP tool configuration
        agent = create_agent()
        tools = [create_mcp_tool_config(name="customer_api", url="http://mcp-server.local/mcp")]

        # When: Creating app with MCP tool
        async with app_factory(agent=agent, tools=tools) as _:
            # Then: All three tools should appear in instructions
            assert "- 'get_customer': Retrieves customer information" in agent.instruction
            assert "- 'update_customer': Updates customer records" in agent.instruction
            assert "- 'delete_customer': Deletes customer from database" in agent.instruction

            # And: MCP tools section should be present
            assert "\n\nFollowing MCP tools are available:\n" in agent.instruction

    @pytest.mark.asyncio
    async def test_mcp_tool_server_unavailable(self, app_factory: Any, monkeypatch: Any) -> None:
        """Test that unavailable MCP server causes app startup to fail with ConnectionError."""

        # Given: Mock McpToolset.get_tools to raise an exception
        async def mock_get_tools_error(self, readonly_context=None):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr("google.adk.tools.mcp_tool.mcp_toolset.McpToolset.get_tools", mock_get_tools_error)

        # And: Agent and MCP tool configuration
        agent = create_agent()
        tools = [create_mcp_tool_config(name="unavailable_tool", url="http://unreachable-mcp.local/mcp")]

        # Expect: App creation should fail with ConnectionError containing tool name, URL, and helpful message
        with pytest.raises(ConnectionError) as exc_info:
            async with app_factory(agent=agent, tools=tools):
                pass

        # Then: Error message should contain expected details
        error_message = str(exc_info.value)
        assert "Could not connect to MCP server 'unavailable_tool'" in error_message
        assert "http://unreachable-mcp.local/mcp" in error_message
        assert "Ensure the server is running and accessible" in error_message
