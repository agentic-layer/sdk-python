"""Mock chat client for MSAF tests."""

from collections.abc import Awaitable, Sequence
from typing import Any

from agent_framework import Agent, BaseChatClient, ChatResponse, Message


class MockChatClient(BaseChatClient[Any]):
    """A mock chat client for testing that returns predefined responses."""

    def __init__(self, response_text: str = "Hello from mock agent!") -> None:
        self._response_text = response_text
        self.received_messages: list[Sequence[Message]] = []

    def set_response(self, response_text: str) -> None:
        """Set the response text for the mock client."""
        self._response_text = response_text

    def _inner_get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Any = None,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse[Any]]:
        self.received_messages.append(messages)

        async def _respond() -> ChatResponse[Any]:
            return ChatResponse(
                messages=[Message("assistant", [self._response_text])],
                finish_reason="stop",
                model_id="mock-model",
            )

        return _respond()


def create_mock_agent(name: str = "test_agent", response_text: str = "Hello from mock agent!") -> Agent[Any]:
    """Create a test Agent with a mock chat client."""
    client = MockChatClient(response_text=response_text)
    return Agent(
        client=client,
        name=name,
        description="Test agent",
        instructions="You are a test agent.",
    )
