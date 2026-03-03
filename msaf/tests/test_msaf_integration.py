"""Integration tests for the Microsoft Agent Framework A2A adapter."""

import uuid
from typing import Any

import pytest
from starlette.testclient import TestClient

from tests.fixtures.mock_client import create_mock_agent


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
            data = response.json()
            assert data.get("jsonrpc") == "2.0"
            assert "result" in data

            # And: The result contains the agent's response text
            result = data["result"]
            assert result is not None

            # Extract text from the status message
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
