"""Shared test helper functions for MSAF tests."""

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
    """Create a request handler that forwards requests to an ASGI app."""
    transport = ASGITransport(app=app)

    async def handle_request(request: httpx.Request) -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            return await client.request(
                method=request.method,
                url=str(request.url),
                headers=request.headers,
                content=request.content,
            )

    return handle_request


def verify_jsonrpc_response(data: dict[str, Any], expected_id: int = 1) -> Any:
    """Verify basic JSON-RPC response structure and return the result."""
    assert data.get("jsonrpc") == "2.0"
    assert data.get("id") == expected_id
    assert "result" in data
    result: Any = data["result"]
    return result
