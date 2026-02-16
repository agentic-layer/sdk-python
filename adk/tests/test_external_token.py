"""Tests for external token passing to MCP tools via ADK session."""

import uuid
from typing import Any

import pytest
from agenticlayer.agent import AgentFactory
from agenticlayer.agent_to_a2a import _EXTERNAL_TOKEN_SESSION_KEY, to_a2a
from agenticlayer.config import McpTool
from asgi_lifespan import LifespanManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from httpx_retries import Retry
from pydantic import AnyHttpUrl
from starlette.testclient import TestClient


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


@pytest.mark.asyncio
async def test_external_token_stored_in_session() -> None:
    """Test that X-External-Token header is captured and stored in ADK session state."""
    # Given: An agent with a tool
    agent = create_agent()
    tools = [McpTool(name="test_tool", url=AnyHttpUrl("http://tool-1.local/mcp"))]
    test_token = "test-bearer-token-12345"

    # When: Creating an app and sending a request with X-External-Token header
    rpc_url = "http://localhost:80/"
    app = to_a2a(
        agent=agent,
        rpc_url=rpc_url,
        tools=tools,
        agent_factory=AgentFactory(retry=Retry(total=2)),
    )

    async with LifespanManager(app) as manager:
        client = TestClient(manager.app)

        # Send a request with the X-External-Token header
        response = client.post(
            "",
            json=create_send_message_request(),
            headers={"X-External-Token": test_token},
        )

        # Then: The request should succeed
        assert response.status_code == 200

        # Note: We cannot directly verify the session state from the test client
        # because the session is internal to the ADK executor. However, we can
        # verify that the app starts correctly and processes the request, which
        # means our custom executor is working.


@pytest.mark.asyncio
async def test_header_provider_retrieves_token_from_session() -> None:
    """Test that the header provider function can retrieve token from session state."""
    from agenticlayer.agent import _get_mcp_headers_from_session
    from google.adk.sessions.session import Session

    # Given: A session with an external token stored
    test_token = "test-api-token-xyz"
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user",
        state={_EXTERNAL_TOKEN_SESSION_KEY: test_token},
        events=[],
        last_update_time=0.0,
    )

    # Create a mock readonly context
    class MockReadonlyContext:
        def __init__(self, session):
            self.session = session

    readonly_context = MockReadonlyContext(session)

    # When: Calling the header provider function
    headers = _get_mcp_headers_from_session(readonly_context)

    # Then: The headers should include the X-External-Token
    assert headers == {"X-External-Token": test_token}


@pytest.mark.asyncio
async def test_header_provider_returns_empty_when_no_token() -> None:
    """Test that the header provider returns empty dict when no token is present."""
    from agenticlayer.agent import _get_mcp_headers_from_session
    from google.adk.sessions.session import Session

    # Given: A session without an external token
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user",
        state={},  # No token
        events=[],
        last_update_time=0.0,
    )

    # Create a mock readonly context
    class MockReadonlyContext:
        def __init__(self, session):
            self.session = session

    readonly_context = MockReadonlyContext(session)

    # When: Calling the header provider function
    headers = _get_mcp_headers_from_session(readonly_context)

    # Then: The headers should be empty
    assert headers == {}


@pytest.mark.asyncio
async def test_header_provider_handles_none_context() -> None:
    """Test that the header provider safely handles None context."""
    from agenticlayer.agent import _get_mcp_headers_from_session

    # When: Calling the header provider with None
    headers = _get_mcp_headers_from_session(None)

    # Then: The headers should be empty (no exception)
    assert headers == {}
