"""Common test fixtures and helper functions for ADK tests."""

import uuid
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm


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
    """Helper function to create a test agent."""
    return LlmAgent(
        name=name,
        model=LiteLlm(model="gemini/gemini-2.5-flash"),
        description="Test agent",
        instruction="You are a test agent.",
    )
