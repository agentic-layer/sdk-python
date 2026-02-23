"""Comprehensive integration and end-to-end tests for agent applications."""

from typing import Any

import httpx
import pytest
import respx
from a2a.client.errors import A2AClientHTTPError
from agenticlayer.agent import AgentFactory
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.config import InteractionType, McpTool, SubAgent
from agenticlayer.loguru_config import setup_logging
from asgi_lifespan import LifespanManager
from fastmcp import Context, FastMCP
from httpx_retries import Retry
from pydantic import AnyHttpUrl
from starlette.testclient import TestClient

from tests.fixtures.mock_llm import LLMMockController
from tests.utils.helpers import (
    create_asgi_request_handler,
    create_send_message_request,
    verify_jsonrpc_response,
)

setup_logging()


class TestAgentIntegration:
    """Integration tests for agent configuration and behavior."""

    @pytest.mark.asyncio
    async def test_agent_card(self, app_factory: Any, agent_factory: Any) -> None:
        """Test that the agent card is available at /.well-known/agent-card.json"""

        # Given:
        agent = agent_factory("test_agent")
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

    @respx.mock
    @pytest.mark.asyncio
    async def test_sub_agent_unavailable_fails_startup(self, app_factory: Any, agent_factory: Any) -> None:
        """Test that unavailable sub-agents cause app startup to fail with retries."""

        # Given: Mock sub-agent that returns connection errors
        agent_card_url = "http://unavailable-agent.local/.well-known/agent-card.json"
        route_unavailable = respx.get(agent_card_url).mock(side_effect=httpx.ConnectError("Connection failed"))
        agent = agent_factory("test_agent")

        sub_agents = [
            SubAgent(
                name="unavailable_agent",
                url=AnyHttpUrl(agent_card_url),
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
    async def test_simple_conversation(
        self, app_factory: Any, agent_factory: Any, llm_controller: LLMMockController
    ) -> None:
        """Test basic agent conversation with mocked LLM.

        Also verifies RPC endpoint functionality and JSONRPC response format.
        """
        # Given: Mock LLM response
        user_message = "Test message for simple conversation"
        expected_response = "Hello! This is a mocked response from the test agent."
        llm_controller.respond_with_message(user_message, expected_response)

        # When: Send message to agent
        test_agent = agent_factory("test_agent")
        async with app_factory(test_agent) as app:
            client = TestClient(app)
            response = client.post("", json=create_send_message_request(user_message))

        # Then: Response contains the mocked LLM response
        assert response.status_code == 200
        result = verify_jsonrpc_response(response.json())
        assert result["status"]["state"] == "completed", "Task should complete successfully"

        # Then: Verify conversation history structure
        history = result["history"]
        assert len(history) == 3, "History should contain: user message (2x), agent message"

        # Verify each message's role and parts content
        assert history[0]["role"] == "user"
        assert history[0]["parts"] == [{"kind": "text", "text": user_message}]

        assert history[1]["role"] == "user"  # Duplicate user message in A2A protocol
        assert history[1]["parts"] == [{"kind": "text", "text": user_message}]

        assert history[2]["role"] == "agent"
        assert history[2]["parts"] == [{"kind": "text", "text": expected_response}]

    @pytest.mark.asyncio
    async def test_with_sub_agent(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test main agent configured with a sub-agent.

        Verifies both sub-agent integration (agent card fetching) and full conversation flow
        where the main agent calls the sub-agent as a tool.
        """

        # Given: Mock LLM responses for BOTH agents (they share the same controller)
        # Main agent receives this and calls sub-agent
        user_message = "Ask the helper about the weather"

        # Sub-agent receives this message when called and responds
        sub_agent_message = "What is the weather today?"
        sub_agent_response = "The weather is sunny and 75 degrees!"
        llm_controller.respond_with_message(sub_agent_message, sub_agent_response)

        # Main agent calls sub-agent tool, then responds with final message
        main_agent_final = "The helper says: The weather is sunny and 75 degrees!"
        llm_controller.respond_with_tool_call(
            pattern=user_message,
            tool_name="sub_agent",
            tool_args={"request": sub_agent_message},  # AgentTool expects 'request' key
            final_message=main_agent_final,
        )

        # Given: Sub-agent running as ASGI app
        sub_agent_url = "http://sub-agent.test"
        sub_agent = agent_factory("sub_agent")

        # Create sub-agent app with matching rpc_url so RemoteA2aAgent uses correct URL
        sub_agent_app = to_a2a(
            agent=sub_agent,
            rpc_url=sub_agent_url,
            agent_factory=AgentFactory(retry=Retry(total=2)),
        )

        async with LifespanManager(sub_agent_app) as sub_manager:
            # Route sub-agent requests through ASGI transport
            respx_mock.route(host="sub-agent.test").mock(
                side_effect=create_asgi_request_handler(sub_manager.app, sub_agent_url)
            )

            # When: Create main agent with sub-agent configuration
            main_agent = agent_factory("main_agent")
            sub_agents = [
                SubAgent(
                    name="sub_agent",
                    url=AnyHttpUrl(f"{sub_agent_url}/.well-known/agent-card.json"),
                    interaction_type=InteractionType.TOOL_CALL,
                )
            ]

            # Create httpx client for respx interception and main agent app
            async with httpx.AsyncClient() as test_client:
                main_app = to_a2a(
                    agent=main_agent,
                    rpc_url="http://localhost:80/",
                    sub_agents=sub_agents,
                    agent_factory=AgentFactory(retry=Retry(total=2), httpx_client=test_client),
                )

                async with LifespanManager(main_app) as main_manager:
                    client = TestClient(main_manager.app)
                    response = client.post("", json=create_send_message_request(user_message))

                    # Then: Verify sub-agent card was fetched
                    agent_card_calls = [
                        call
                        for call in respx_mock.calls
                        if call.request.url.path == "/.well-known/agent-card.json"
                        and "sub-agent.test" in str(call.request.url)
                    ]
                    assert len(agent_card_calls) == 1, "Sub-agent card should be fetched during startup"

                    # Then: Verify sub-agent was actually called (not just the card)
                    sub_agent_calls = [
                        call
                        for call in respx_mock.calls
                        if call.request.url.path == "/" and "sub-agent.test" in str(call.request.url)
                    ]
                    assert len(sub_agent_calls) > 0, "Sub-agent should be called during conversation"

                    # Then: Verify main agent response
                    assert response.status_code == 200
                    result = verify_jsonrpc_response(response.json())
                    assert result["status"]["state"] == "completed", "Task should complete successfully"

                    # Then: Verify conversation history structure includes sub-agent interaction
                    history = result["history"]
                    assert len(history) >= 5, (
                        "History should contain: user message (2x), function call, function response, agent message"
                    )

                    # Verify initial user messages
                    assert history[0]["role"] == "user"
                    assert history[0]["parts"] == [{"kind": "text", "text": user_message}]

                    assert history[1]["role"] == "user"  # Duplicate user message in A2A protocol
                    assert history[1]["parts"] == [{"kind": "text", "text": user_message}]

                    # Verify sub-agent tool call
                    assert history[2]["role"] == "agent"
                    assert history[2]["parts"] == [
                        {
                            "kind": "data",
                            "data": {"id": "call_test123", "name": "sub_agent", "args": {"request": sub_agent_message}},
                            "metadata": {"adk_type": "function_call"},
                        }
                    ]

                    # Verify sub-agent response
                    assert history[3]["role"] == "agent"
                    assert history[3]["parts"] == [
                        {
                            "kind": "data",
                            "data": {
                                "id": "call_test123",
                                "name": "sub_agent",
                                "response": {"result": sub_agent_response},
                            },
                            "metadata": {"adk_type": "function_response"},
                        }
                    ]

                    # Verify final main agent message
                    assert history[4]["role"] == "agent"
                    assert history[4]["parts"] == [{"kind": "text", "text": main_agent_final}]

    @pytest.mark.asyncio
    async def test_with_tool_server(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test agent calling an MCP server tool.

        Verifies both tool integration and full conversation flow with tool calls.
        """

        # Given: Mock LLM to call 'add' tool
        llm_controller.respond_with_tool_call(
            pattern="",  # Match any message
            tool_name="add",
            tool_args={"a": 5, "b": 3},
            final_message="The calculation result is correct!",
        )

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
            handler = create_asgi_request_handler(mcp_manager.app, mcp_server_url)
            respx_mock.route(host="test-mcp.local").mock(side_effect=handler)

            # When: Create agent with MCP tool
            test_agent = agent_factory("test_agent")
            tools = [McpTool(name="calc", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                user_message = "Calculate 5 + 3"
                response = client.post("", json=create_send_message_request(user_message))

            # Then: Verify response structure
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert "status" in result
            assert "history" in result

            # Then: Verify task completed successfully
            assert result["status"]["state"] == "completed", "Task should complete successfully"

            # Then: Verify conversation history structure
            history = result["history"]
            assert len(history) == 5, (
                "History should contain: user message (2x), function call, function response, agent message"
            )

            # Verify each message's role and parts content
            assert history[0]["role"] == "user"
            assert history[0]["parts"] == [{"kind": "text", "text": "Calculate 5 + 3"}]

            assert history[1]["role"] == "user"  # Duplicate user message in A2A protocol
            assert history[1]["parts"] == [{"kind": "text", "text": "Calculate 5 + 3"}]

            assert history[2]["role"] == "agent"
            assert history[2]["parts"] == [
                {
                    "kind": "data",
                    "data": {"id": "call_test123", "name": "add", "args": {"a": 5, "b": 3}},
                    "metadata": {"adk_type": "function_call"},
                }
            ]

            assert history[3]["role"] == "agent"
            assert history[3]["parts"] == [
                {
                    "kind": "data",
                    "data": {
                        "id": "call_test123",
                        "name": "add",
                        "response": {
                            "content": [{"type": "text", "text": "8"}],
                            "structuredContent": {"result": 8},
                            "isError": False,
                        },
                    },
                    "metadata": {"adk_type": "function_response"},
                }
            ]

            assert history[4]["role"] == "agent"
            assert history[4]["parts"] == [{"kind": "text", "text": "The calculation result is correct!"}]

    @pytest.mark.asyncio
    async def test_external_token_passed_to_mcp_tools(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that X-External-Token header is passed from A2A request to MCP tool calls.

        Verifies end-to-end token passing through the agent to MCP servers.
        """

        # Given: Mock LLM to call 'echo' tool
        llm_controller.respond_with_tool_call(
            pattern="",  # Match any message
            tool_name="echo",
            tool_args={"message": "test"},
            final_message="Echo completed!",
        )

        # Given: MCP server with 'echo' tool that can access headers via Context
        mcp = FastMCP("TokenVerifier")
        received_headers: list[dict[str, str]] = []
        received_tokens_in_tool: list[str | None] = []

        @mcp.tool()
        def echo(message: str, ctx: Context) -> str:
            """Echo a message and verify token is accessible in tool context."""
            # Access headers from the MCP request context
            # The Context object provides access to the request_context which includes HTTP headers
            if ctx.request_context and hasattr(ctx.request_context, "request"):
                # Try to get the token from request headers if available
                request = ctx.request_context.request
                if request and hasattr(request, "headers"):
                    token = request.headers.get("x-external-token") or request.headers.get("X-External-Token")
                    received_tokens_in_tool.append(token)
            return f"Echoed: {message}"

        mcp_server_url = "http://test-mcp-token.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Create a custom handler that captures headers
            async def handler_with_header_capture(request: httpx.Request) -> httpx.Response:
                # Capture the headers from the request
                received_headers.append(dict(request.headers))

                # Forward to the MCP app
                transport = httpx.ASGITransport(app=mcp_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp_server_url) as client:
                    return await client.request(
                        method=request.method,
                        url=str(request.url),
                        headers=request.headers,
                        content=request.content,
                    )

            # Route MCP requests through our custom handler
            respx_mock.route(host="test-mcp-token.local").mock(side_effect=handler_with_header_capture)

            # When: Create agent with MCP tool and send request with X-External-Token header
            test_agent = agent_factory("test_agent")
            tools = [McpTool(name="verifier", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]
            external_token = "secret-api-token-12345"  # nosec B105

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                user_message = "Echo test message"
                response = client.post(
                    "",
                    json=create_send_message_request(user_message),
                    headers={"X-External-Token": external_token},
                )

            # Then: Verify response is successful
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert result["status"]["state"] == "completed", "Task should complete successfully"

            # Then: Verify X-External-Token header was passed to MCP server
            assert len(received_headers) > 0, "MCP server should have received requests"

            # Find the tool call request (not the initialization requests)
            # Header keys might be lowercase
            tool_call_headers = [h for h in received_headers if "x-external-token" in h or "X-External-Token" in h]
            assert len(tool_call_headers) > 0, (
                f"At least one request should have X-External-Token header. "
                f"Received {len(received_headers)} requests total."
            )

            # Verify the token value
            for headers in tool_call_headers:
                # Header might be lowercase in the dict
                token_value = headers.get("X-External-Token") or headers.get("x-external-token")
                assert token_value == external_token, f"Expected token '{external_token}', got '{token_value}'"

    @pytest.mark.asyncio
    async def test_multiple_headers_propagated_to_mcp_tools(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that multiple configured headers are passed from A2A request to MCP tool calls."""

        # Given: Mock LLM to call 'echo' tool
        llm_controller.respond_with_tool_call(
            pattern="",  # Match any message
            tool_name="echo",
            tool_args={"message": "test"},
            final_message="Echo completed!",
        )

        # Given: MCP server with 'echo' tool
        mcp = FastMCP("MultiHeaderVerifier")
        received_headers: list[dict[str, str]] = []

        @mcp.tool()
        def echo(message: str) -> str:
            """Echo a message."""
            return f"Echoed: {message}"

        mcp_server_url = "http://test-multi-header.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Create a custom handler that captures headers
            async def handler_with_header_capture(request: httpx.Request) -> httpx.Response:
                # Capture the headers from the request
                received_headers.append(dict(request.headers))

                # Forward to the MCP app
                transport = httpx.ASGITransport(app=mcp_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp_server_url) as client:
                    return await client.request(
                        method=request.method,
                        url=str(request.url),
                        headers=request.headers,
                        content=request.content,
                    )

            # Route MCP requests through our custom handler
            respx_mock.route(host="test-multi-header.local").mock(side_effect=handler_with_header_capture)

            # When: Create agent with MCP tool configured to propagate multiple headers
            test_agent = agent_factory("test_agent")
            tools = [
                McpTool(
                    name="verifier",
                    url=AnyHttpUrl(f"{mcp_server_url}/mcp"),
                    timeout=30,
                    propagate_headers=["X-API-Key", "X-Custom-Header", "Authorization"],
                )
            ]

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                user_message = "Echo test message"
                response = client.post(
                    "",
                    json=create_send_message_request(user_message),
                    headers={
                        "X-API-Key": "api-key-12345",
                        "X-Custom-Header": "custom-value",
                        "Authorization": "Bearer token123",
                        "X-Ignored-Header": "should-not-be-sent",
                    },
                )

            # Then: Verify response is successful
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert result["status"]["state"] == "completed", "Task should complete successfully"

            # Then: Verify only configured headers were passed to MCP server
            assert len(received_headers) > 0, "MCP server should have received requests"

            # Find the tool call request (check all headers)
            tool_call_headers = [
                h
                for h in received_headers
                if any(key.lower() in ["x-api-key", "x-custom-header", "authorization"] for key in h.keys())
            ]
            assert len(tool_call_headers) > 0, "At least one request should have configured headers"

            # Verify all configured headers are present
            for headers in tool_call_headers:
                # Check for each configured header (case-insensitive)
                headers_lower = {k.lower(): v for k, v in headers.items()}

                if "x-api-key" in headers_lower:
                    assert headers_lower["x-api-key"] == "api-key-12345"
                if "x-custom-header" in headers_lower:
                    assert headers_lower["x-custom-header"] == "custom-value"
                if "authorization" in headers_lower:
                    assert headers_lower["authorization"] == "Bearer token123"

                # Verify ignored header is NOT present
                assert "X-Ignored-Header" not in headers
                assert "x-ignored-header" not in headers_lower

    @pytest.mark.asyncio
    async def test_different_headers_per_mcp_server(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that different MCP servers receive only their configured headers."""

        # Given: Mock LLM to call both tools
        llm_controller.respond_with_tool_call(
            pattern="",
            tool_name="server1_echo",
            tool_args={"message": "test1"},
            final_message="First call done",
        )
        llm_controller.respond_with_tool_call(
            pattern="",
            tool_name="server2_echo",
            tool_args={"message": "test2"},
            final_message="Second call done",
        )

        # Given: Two MCP servers
        mcp1 = FastMCP("Server1")
        mcp2 = FastMCP("Server2")
        server1_headers: list[dict[str, str]] = []
        server2_headers: list[dict[str, str]] = []

        @mcp1.tool()
        def server1_echo(message: str) -> str:
            """Echo from server 1."""
            return f"Server1: {message}"

        @mcp2.tool()
        def server2_echo(message: str) -> str:
            """Echo from server 2."""
            return f"Server2: {message}"

        mcp1_url = "http://test-mcp1.local"
        mcp2_url = "http://test-mcp2.local"
        mcp1_app = mcp1.http_app(path="/mcp")
        mcp2_app = mcp2.http_app(path="/mcp")

        async with LifespanManager(mcp1_app) as mcp1_manager, LifespanManager(mcp2_app) as mcp2_manager:
            # Handlers for each server
            async def handler1(request: httpx.Request) -> httpx.Response:
                server1_headers.append(dict(request.headers))
                transport = httpx.ASGITransport(app=mcp1_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp1_url) as client:
                    return await client.request(
                        method=request.method, url=str(request.url), headers=request.headers, content=request.content
                    )

            async def handler2(request: httpx.Request) -> httpx.Response:
                server2_headers.append(dict(request.headers))
                transport = httpx.ASGITransport(app=mcp2_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp2_url) as client:
                    return await client.request(
                        method=request.method, url=str(request.url), headers=request.headers, content=request.content
                    )

            respx_mock.route(host="test-mcp1.local").mock(side_effect=handler1)
            respx_mock.route(host="test-mcp2.local").mock(side_effect=handler2)

            # When: Create agent with two MCP tools with different header configs
            test_agent = agent_factory("test_agent")
            tools = [
                McpTool(
                    name="server1",
                    url=AnyHttpUrl(f"{mcp1_url}/mcp"),
                    timeout=30,
                    propagate_headers=["X-Server1-Token"],  # Only X-Server1-Token
                ),
                McpTool(
                    name="server2",
                    url=AnyHttpUrl(f"{mcp2_url}/mcp"),
                    timeout=30,
                    propagate_headers=["X-Server2-Token", "Authorization"],  # Different headers
                ),
            ]

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                response = client.post(
                    "",
                    json=create_send_message_request("Test message"),
                    headers={
                        "X-Server1-Token": "token-for-server1",
                        "X-Server2-Token": "token-for-server2",
                        "Authorization": "Bearer shared-token",
                        "X-Common-Header": "should-not-be-sent",
                    },
                )

            # Then: Verify response is successful
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert result["status"]["state"] == "completed", "Task should complete successfully"

            # Then: Verify server1 only received X-Server1-Token
            assert len(server1_headers) > 0, "Server1 should have received requests"
            for headers in server1_headers:
                headers_lower = {k.lower(): v for k, v in headers.items()}
                if "x-server1-token" in headers_lower:
                    assert headers_lower["x-server1-token"] == "token-for-server1"
                # Should NOT have server2's headers
                assert "x-server2-token" not in headers_lower
                assert "authorization" not in headers_lower
                assert "x-common-header" not in headers_lower

            # Then: Verify server2 only received X-Server2-Token and Authorization
            assert len(server2_headers) > 0, "Server2 should have received requests"
            for headers in server2_headers:
                headers_lower = {k.lower(): v for k, v in headers.items()}
                if "x-server2-token" in headers_lower:
                    assert headers_lower["x-server2-token"] == "token-for-server2"
                if "authorization" in headers_lower:
                    assert headers_lower["authorization"] == "Bearer shared-token"
                # Should NOT have server1's headers
                assert "x-server1-token" not in headers_lower
                assert "x-common-header" not in headers_lower

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_propagate_headers(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test backward compatibility: when propagate_headers is not set, X-External-Token is still passed."""

        # Given: Mock LLM to call 'echo' tool
        llm_controller.respond_with_tool_call(
            pattern="",
            tool_name="echo",
            tool_args={"message": "test"},
            final_message="Echo completed!",
        )

        # Given: MCP server
        mcp = FastMCP("BackwardCompatVerifier")
        received_headers: list[dict[str, str]] = []

        @mcp.tool()
        def echo(message: str) -> str:
            """Echo a message."""
            return f"Echoed: {message}"

        mcp_server_url = "http://test-backward-compat.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Create a custom handler that captures headers
            async def handler_with_header_capture(request: httpx.Request) -> httpx.Response:
                received_headers.append(dict(request.headers))
                transport = httpx.ASGITransport(app=mcp_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp_server_url) as client:
                    return await client.request(
                        method=request.method,
                        url=str(request.url),
                        headers=request.headers,
                        content=request.content,
                    )

            respx_mock.route(host="test-backward-compat.local").mock(side_effect=handler_with_header_capture)

            # When: Create agent with MCP tool WITHOUT propagate_headers config
            test_agent = agent_factory("test_agent")
            tools = [
                McpTool(
                    name="verifier",
                    url=AnyHttpUrl(f"{mcp_server_url}/mcp"),
                    timeout=30,
                    # No propagate_headers specified - should use legacy behavior
                )
            ]
            external_token = "legacy-token-12345"  # nosec B105

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                response = client.post(
                    "",
                    json=create_send_message_request("Test message"),
                    headers={
                        "X-External-Token": external_token,
                        "X-Other-Header": "should-not-be-sent",
                    },
                )

            # Then: Verify response is successful
            assert response.status_code == 200
            result = verify_jsonrpc_response(response.json())
            assert result["status"]["state"] == "completed", "Task should complete successfully"

            # Then: Verify X-External-Token was passed (legacy behavior)
            assert len(received_headers) > 0, "MCP server should have received requests"
            tool_call_headers = [h for h in received_headers if "x-external-token" in h or "X-External-Token" in h]
            assert len(tool_call_headers) > 0, "At least one request should have X-External-Token header"

            for headers in tool_call_headers:
                headers_lower = {k.lower(): v for k, v in headers.items()}
                assert headers_lower.get("x-external-token") == external_token
                # Other headers should NOT be passed
                assert "x-other-header" not in headers_lower
