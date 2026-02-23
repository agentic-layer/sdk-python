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
    async def test_mcp_server_restart_causes_communication_failure(
            self,
            app_factory: Any,
            agent_factory: Any,
            llm_controller: LLMMockController,
            respx_mock: respx.MockRouter,
    ) -> None:
        """Test that agent fails to communicate with MCP server after server restart.

        This test reproduces the issue where:
        1. Agent successfully calls an MCP server tool
        2. MCP server restarts (losing session state)
        3. Agent tries to call the tool again with the SAME agent instance
        4. Communication fails because the cached MCP session is no longer valid

        This reproduces the real-world scenario where a server restarts and loses
        all session state, but the client still has cached session objects.

        Expected behavior after this issue is fixed:
        - The agent should automatically detect the invalid session
        - Create a new session with the restarted server
        - Successfully complete the second tool call
        """

        # Given: Mock LLM to call 'add' tool twice
        llm_controller.respond_with_tool_call(
            pattern="first call",
            tool_name="add",
            tool_args={"a": 2, "b": 3},
            final_message="First calculation done: 5",
        )
        llm_controller.respond_with_tool_call(
            pattern="second call",
            tool_name="add",
            tool_args={"a": 10, "b": 20},
            final_message="Second calculation done: 30",
        )

        # Given: MCP server with 'add' tool
        mcp = FastMCP("Calculator")

        @mcp.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        mcp_server_url = "http://test-mcp-restart.local"
        mcp_app = mcp.http_app(path="/mcp")

        # Track server state to simulate restart
        server_state: dict[str, Any] = {
            "accept_old_sessions": True,
            "old_session_ids": set(),
        }

        async with LifespanManager(mcp_app) as mcp_manager:
            # Handler that can reject old session IDs after "restart"
            async def session_handler(request: httpx.Request) -> httpx.Response:
                url_str = str(request.url)

                # Extract session ID from URL path (e.g., /mcp/messages/SESSION_ID)
                session_id = None
                if "/messages/" in url_str:
                    parts = url_str.split("/messages/")
                    if len(parts) > 1:
                        session_id = parts[1].split("/")[0].split("?")[0]

                # Check if we should reject this session
                if session_id and session_id in server_state["old_session_ids"]:
                    if not server_state["accept_old_sessions"]:
                        # Server has "restarted" and doesn't recognize old sessions
                        print(f"  [Server] Rejecting old session: {session_id}")
                        return httpx.Response(
                            status_code=404,
                            json={"error": "Session not found"},
                            headers={"content-type": "application/json"},
                        )

                # Forward request to MCP server
                transport = httpx.ASGITransport(app=mcp_manager.app)
                async with httpx.AsyncClient(transport=transport, base_url=mcp_server_url) as client:
                    response = await client.request(
                        method=request.method,
                        url=str(request.url),
                        headers=request.headers,
                        content=request.content,
                    )

                    # Track successful session IDs
                    if session_id and response.status_code == 200:
                        server_state["old_session_ids"].add(session_id)

                    return response

            respx_mock.route(host="test-mcp-restart.local").mock(side_effect=session_handler)

            # When: Create agent with MCP tool
            test_agent = agent_factory("test_agent")
            tools = [McpTool(name="calc", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)

                # ===== FIRST CALL =====
                print("\n=== FIRST CALL: Should succeed ===")
                response1 = client.post("", json=create_send_message_request("first call: Calculate 2 + 3"))

                # Then: Verify first call succeeded
                assert response1.status_code == 200
                result1 = verify_jsonrpc_response(response1.json())
                assert result1["status"]["state"] == "completed", "First task should complete successfully"
                print("✓ First call completed successfully")

                # ===== SIMULATE SERVER RESTART =====
                print("\n=== SIMULATING SERVER RESTART ===")
                print("  Server will reject all previously established sessions")
                server_state["accept_old_sessions"] = False

                # ===== SECOND CALL =====
                print("\n=== SECOND CALL: Should fail with current implementation ===")
                response2 = client.post("", json=create_send_message_request("second call: Calculate 10 + 20"))

                # Then: Verify the issue is reproduced
                assert response2.status_code == 200, "A2A response should be 200 even if task failed"
                result2 = verify_jsonrpc_response(response2.json())

                task_state = result2["status"]["state"]
                print(f"✓ Task state: {task_state}")

                # Document the current behavior: task should fail
                if task_state == "failed":
                    print("\n=== ISSUE SUCCESSFULLY REPRODUCED ===")
                    print("The agent failed to communicate with the MCP server after restart.")
                    print("This is the bug we're documenting.")
                    error_msg = result2["status"].get("message", {}).get("parts", [{}])[0].get("text", "")
                    print(f"Error message: {error_msg}")

                    # Assert that we got the expected failure
                    assert "Failed to create MCP session" in error_msg or "MCP" in error_msg, (
                        f"Expected MCP session error, got: {error_msg}"
                    )

                elif task_state == "completed":
                    print("\n=== ISSUE NOT REPRODUCED (or already fixed) ===")
                    print("The agent successfully reconnected despite the server restart.")
                    print("This suggests the MCP client auto-recovery is working.")
                    # If this happens, the issue might already be fixed or the test isn't right
                    # For now, we'll fail the test to investigate
                    pytest.fail(
                        "Expected task to fail after server restart, but it completed successfully. "
                        "Either the bug is already fixed, or the test needs adjustment."
                    )
                else:
                    print(f"\n=== UNEXPECTED STATE: {task_state} ===")
                    pytest.fail(f"Unexpected task state: {task_state}")
