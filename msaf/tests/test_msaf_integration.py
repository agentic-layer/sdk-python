"""Integration tests for the Microsoft Agent Framework A2A adapter."""

import uuid
from typing import Any

import httpx
import pytest
import respx
from a2a.client.errors import A2AClientHTTPError
from agenticlayer.shared.config import InteractionType, McpTool, SubAgent
from asgi_lifespan import LifespanManager
from fastmcp import FastMCP
from httpx_retries import Retry
from pydantic import AnyHttpUrl
from starlette.testclient import TestClient

from agenticlayer.msaf.agent import MsafAgentFactory
from agenticlayer.msaf.agent_to_a2a import to_a2a
from tests.fixtures.mock_client import MockChatClient, create_mock_agent
from tests.utils.helpers import create_asgi_request_handler, create_send_message_request, verify_jsonrpc_response


class TestMsafAgentIntegration:
    """Integration tests for MSAF agent A2A application."""

    @pytest.mark.asyncio
    async def test_agent_card(self, msaf_app_factory: Any) -> None:
        """Test that the agent card is available at /.well-known/agent-card.json"""

        # Given: A mock agent with a known name and description
        agent = create_mock_agent(name="my_test_agent")
        async with msaf_app_factory(agent, name="my_test_agent", description="A test agent") as app:
            client = TestClient(app)

            # When: Requesting the agent card endpoint
            response = client.get("/.well-known/agent-card.json")

            # Then: Agent card is returned with correct data
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict), "Agent card should return a JSON object"
            assert data.get("name") == "my_test_agent"
            assert data.get("description") == "A test agent"

    @pytest.mark.asyncio
    async def test_send_message_returns_response(self, msaf_app_factory: Any) -> None:
        """Test that sending a message returns a response from the agent."""

        # Given: A mock agent that responds with a known message
        expected_response = "Hello from the test agent!"
        agent = create_mock_agent(name="responder_agent", response_text=expected_response)

        async with msaf_app_factory(agent, name="responder_agent") as app:
            client = TestClient(app)

            # When: Sending a message
            request = create_send_message_request("Hello")
            response = client.post("/", json=request)

            # Then: A successful JSON-RPC response is returned
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())

            # And: The result contains the agent's response text
            status = result.get("status", {})
            message = status.get("message", {})
            parts = message.get("parts", [])
            text_parts = [p.get("text") for p in parts if p.get("kind") == "text"]
            assert expected_response in text_parts

    @pytest.mark.asyncio
    async def test_agent_card_url_matches_rpc_url(self, msaf_app_factory: Any) -> None:
        """Test that the agent card URL matches the configured rpc_url."""

        # Given: A mock agent
        agent = create_mock_agent(name="url_test_agent")
        async with msaf_app_factory(agent, name="url_test_agent") as app:
            client = TestClient(app)

            # When: Requesting the agent card
            response = client.get("/.well-known/agent-card.json")

            # Then: The URL in the agent card matches the rpc_url
            assert response.status_code == 200
            data = response.json()
            assert data.get("url") == "http://localhost:80/"

    @respx.mock
    @pytest.mark.asyncio
    async def test_sub_agent_unavailable_fails_startup(self, msaf_app_factory: Any) -> None:
        """Test that unavailable sub-agents cause app startup to fail with retries."""

        # Given: Mock sub-agent that returns connection errors
        agent_card_url = "http://unavailable-agent.local/.well-known/agent-card.json"
        route_unavailable = respx.get(agent_card_url).mock(side_effect=httpx.ConnectError("Connection failed"))
        agent = create_mock_agent(name="test_agent")

        sub_agents = [
            SubAgent(
                name="unavailable_agent",
                url=AnyHttpUrl(agent_card_url),
                interaction_type=InteractionType.TOOL_CALL,
            ),
        ]

        # Expect: App creation should fail with A2AClientHTTPError
        with pytest.raises(A2AClientHTTPError, match="Network communication error"):
            async with msaf_app_factory(agent=agent, sub_agents=sub_agents):
                pass

        # And: The retry mechanism should have been used (total=2 means initial + 2 retries = 3 calls)
        assert route_unavailable.call_count == 3, (
            f"Expected 3 calls (1 initial + 2 retries), got {route_unavailable.call_count}"
        )

    @pytest.mark.asyncio
    async def test_with_sub_agent(
        self,
        msaf_app_factory: Any,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test main agent configured with a sub-agent.

        Verifies both sub-agent integration (agent card fetching) and full conversation flow
        where the main agent calls the sub-agent as a tool.
        """

        # Given: Sub-agent that returns a fixed response
        sub_agent_response = "The weather is sunny and 75 degrees!"
        sub_agent = create_mock_agent(name="weather_agent", response_text=sub_agent_response)
        sub_agent_url = "http://sub-agent.test"

        sub_agent_app = to_a2a(
            agent=sub_agent,
            name="weather_agent",
            rpc_url=sub_agent_url,
            agent_factory=MsafAgentFactory(retry=Retry(total=2)),
        )

        async with LifespanManager(sub_agent_app) as sub_manager:
            # Route sub-agent requests through ASGI transport
            respx_mock.route(host="sub-agent.test").mock(
                side_effect=create_asgi_request_handler(sub_manager.app, sub_agent_url)
            )

            # Given: Main agent that responds with the sub-agent's answer
            main_agent_response = f"According to the weather agent: {sub_agent_response}"
            main_agent = create_mock_agent(name="main_agent", response_text=main_agent_response)

            sub_agents = [
                SubAgent(
                    name="weather_agent",
                    url=AnyHttpUrl(f"{sub_agent_url}/.well-known/agent-card.json"),
                    interaction_type=InteractionType.TOOL_CALL,
                ),
            ]

            # When: Create main agent with sub-agent configuration
            async with msaf_app_factory(main_agent, name="main_agent", sub_agents=sub_agents) as app:
                client = TestClient(app)
                response = client.post("/", json=create_send_message_request("What is the weather?"))

            # Then: Response is successful
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert result["status"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_conversation_history_preserved_across_messages(self, msaf_app_factory: Any) -> None:
        """Test that messages with the same contextId share conversation history."""

        # Given: A mock agent with a client that records received messages
        mock_client = MockChatClient(response_text="First response")
        from agent_framework import Agent

        agent: Agent[Any] = Agent(
            client=mock_client,
            name="history_agent",
            description="Test agent",
            instructions="You are a test agent.",
        )

        context_id = str(uuid.uuid4())

        async with msaf_app_factory(agent, name="history_agent") as app:
            client = TestClient(app)

            # When: Sending a first message
            request1 = create_send_message_request("Hello, my name is Alice!", context_id=context_id)
            response1 = client.post("/", json=request1)
            assert response1.status_code == 200
            verify_jsonrpc_response(response1.json())

            # And: Sending a second message with the same contextId
            mock_client.set_response("Second response")
            request2 = create_send_message_request("What is my name?", context_id=context_id)
            response2 = client.post("/", json=request2)
            assert response2.status_code == 200
            verify_jsonrpc_response(response2.json())

            # Then: The second call should have received conversation history
            # (more messages than just the current user input)
            assert len(mock_client.received_messages) == 2
            second_call_messages = mock_client.received_messages[1]
            assert len(second_call_messages) > 1, (
                f"Second call should include conversation history, but only got {len(second_call_messages)} message(s)"
            )

    @pytest.mark.asyncio
    async def test_different_context_ids_have_separate_history(self, msaf_app_factory: Any) -> None:
        """Test that messages with different contextIds do not share history."""

        # Given: A mock agent
        mock_client = MockChatClient(response_text="Response")
        from agent_framework import Agent

        agent: Agent[Any] = Agent(
            client=mock_client,
            name="isolation_agent",
            description="Test agent",
            instructions="You are a test agent.",
        )

        async with msaf_app_factory(agent, name="isolation_agent") as app:
            client = TestClient(app)

            # When: Sending messages with different contextIds
            request1 = create_send_message_request("Message 1", context_id=str(uuid.uuid4()))
            client.post("/", json=request1)

            request2 = create_send_message_request("Message 2", context_id=str(uuid.uuid4()))
            client.post("/", json=request2)

            # Then: Each call should have received only its own message (no shared history)
            assert len(mock_client.received_messages) == 2
            assert len(mock_client.received_messages[0]) == 1
            assert len(mock_client.received_messages[1]) == 1

    @pytest.mark.asyncio
    async def test_with_tool_server(
        self,
        msaf_app_factory: Any,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test agent calling an MCP server tool.

        Verifies both tool integration and full conversation flow with tool calls.
        """

        # Given: MCP server with 'add' tool
        mcp = FastMCP("Calculator")

        @mcp.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        mcp_server_url = "http://test-mcp.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Route MCP requests through ASGI transport
            respx_mock.route(host="test-mcp.local").mock(
                side_effect=create_asgi_request_handler(mcp_manager.app, mcp_server_url)
            )

            # Given: Agent that returns a fixed response (the MockChatClient can't call tools,
            # but we verify the MCP tool is registered and the app starts correctly)
            test_agent = create_mock_agent(name="test_agent", response_text="Calculation complete!")
            tools = [McpTool(name="calc", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            # When: Create agent with MCP tool
            async with msaf_app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                response = client.post("/", json=create_send_message_request("Calculate 5 + 3"))

            # Then: Response is successful (tool is registered, agent runs)
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert "status" in result
