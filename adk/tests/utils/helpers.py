"""Shared test helper functions for all tests."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from httpx import ASGITransport


def create_send_message_request(message_text: str = "Hello, agent!") -> dict[str, Any]:
    """Create an A2A send message request."""
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


def create_asgi_request_handler(app: Any, base_url: str) -> Callable[[httpx.Request], Awaitable[httpx.Response]]:
    """
    Create a request handler that forwards requests to an ASGI app.

    Args:
        app: The ASGI application to forward requests to
        base_url: The base URL for the app

    Returns:
        An async function that can be used as a respx side_effect
    """
    transport = ASGITransport(app=app)

    async def handle_request(request: httpx.Request) -> httpx.Response:
        """Forward requests to the ASGI app."""
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            return await client.request(
                method=request.method,
                url=str(request.url),
                headers=request.headers,
                content=request.content,
            )

    return handle_request


def extract_agent_text_messages(history: list[dict[str, Any]]) -> list[str]:
    """
    Extract text messages from agent responses in conversation history.

    Args:
        history: The conversation history from an A2A response

    Returns:
        List of text messages from agent responses
    """
    messages = []
    for msg in history:
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    messages.append(part.get("text", ""))
    return messages


def verify_jsonrpc_response(data: dict[str, Any], expected_id: int = 1) -> Any:
    """
    Verify basic JSON-RPC response structure and return the result.

    Args:
        data: The JSON-RPC response
        expected_id: Expected request ID

    Returns:
        The result field from the response
    """
    assert data.get("jsonrpc") == "2.0"
    assert data.get("id") == expected_id
    assert "result" in data
    result: Any = data["result"]
    return result
