"""Flexible LLM mocking fixture for e2e tests"""

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import pytest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from litellm import ModelResponse


@dataclass
class MessageResponse:
    """Configuration for a text message response"""

    content: str


@dataclass
class ToolResponse:
    """Configuration for a tool call response"""

    tool_name: str
    tool_args: dict[str, Any]
    final_message: str


@dataclass
class ResponseConfig:
    """Configuration for a single mock response. Exactly one of message_response or tool_response must be set."""

    message_response: MessageResponse | None = None
    tool_response: ToolResponse | None = None


def _get_last_user_message(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message from the message list"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return str(content)
    return ""


def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    """Check if the messages contain tool results (indicating second call after tool execution)"""
    return any(msg.get("role") == "tool" for msg in messages)


class LLMMockController:
    """
    Controller for configuring LLM mock responses.

    Provides a clean API for setting up different response types based on message patterns.
    """

    def __init__(self) -> None:
        self._responses: dict[str, ResponseConfig] = {}

    def reset(self) -> None:
        """Reset the mock to its initial state"""
        self._responses = {}

    def respond_with_message(self, pattern: str, message: str) -> None:
        """
        Configure the mock to respond with a text message when the pattern matches.

        Args:
            pattern: A substring to match in the last user message (case-insensitive)
            message: The text message to return
        """
        self._responses[pattern.lower()] = ResponseConfig(message_response=MessageResponse(content=message))

    def respond_with_tool_call(
        self, pattern: str, tool_name: str, tool_args: dict[str, Any], final_message: str | None = None
    ) -> None:
        """
        Configure the mock to respond with a tool call when the pattern matches.

        Args:
            pattern: A substring to match in the last user message (case-insensitive)
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            final_message: Optional message to return after tool execution completes
        """
        self._responses[pattern.lower()] = ResponseConfig(
            tool_response=ToolResponse(
                tool_name=tool_name,
                tool_args=tool_args,
                final_message=final_message or "Tool execution completed successfully.",
            )
        )

    def _find_matching_config(self, user_message: str) -> ResponseConfig | None:
        """Find the first matching response config for the given user message"""
        user_message_lower = user_message.lower()
        for pattern, config in self._responses.items():
            if pattern in user_message_lower:
                return config
        return None

    def create_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        """
        Create a mock response based on the configured patterns and current messages.

        Args:
            messages: The conversation messages
            tools: Available tools

        Returns:
            A mock ModelResponse
        """

        if len(self._responses) == 0:
            return ModelResponse(
                id="mock-completion-default",
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "No mock responses configured. Use the mock_llm fixture to configure responses.",
                        },
                    }
                ],
            )

        has_tool_results = _has_tool_results(messages)
        last_user_msg = _get_last_user_message(messages)
        config = self._find_matching_config(last_user_msg)

        # If we have tool results, return the final message
        if has_tool_results and config and config.tool_response:
            return ModelResponse(
                id="mock-completion-final",
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": config.tool_response.final_message,
                        },
                        "finish_reason": "stop",
                    }
                ],
                model="mock-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

        # If configured for tool call and tools are available, return tool call
        if config and config.tool_response and len(tools) > 0:
            return ModelResponse(
                id="mock-completion-toolcall",
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_test123",
                                    "type": "function",
                                    "function": {
                                        "name": config.tool_response.tool_name,
                                        "arguments": json.dumps(config.tool_response.tool_args),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                model="mock-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

        # Return configured message or default
        response_content = (
            config.message_response.content
            if config and config.message_response
            else f"No matching mock response configured for last user message: {last_user_msg}"
        )

        return ModelResponse(
            id="mock-completion-text",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


class MockLiteLLMClient(LiteLLMClient):
    """Mock LiteLLMClient that uses a controller for responses."""

    def __init__(self, controller: LLMMockController) -> None:
        # Don't call super().__init__() to avoid litellm initialization
        self._controller = controller

    async def acompletion(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> ModelResponse:
        """Mock async completion that delegates to the controller."""
        return self._controller.create_response(messages, tools)

    def completion(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], stream: Any = None, **kwargs: Any
    ) -> ModelResponse:
        """Mock sync completion that delegates to the controller."""
        return self._controller.create_response(messages, tools)


@pytest.fixture
def llm_controller() -> Generator[LLMMockController, None, None]:
    """
    Fixture that provides a fresh LLMMockController for configuring mock LLM responses.

    The controller is local to each test and automatically reset between tests.

    Usage:
        def test_example(llm_controller):
            llm_controller.respond_with_message("hello", "Hi there!")
    """
    controller = LLMMockController()
    yield controller
    # Reset after test (defensive cleanup)
    controller.reset()


@pytest.fixture
def llm_client(llm_controller: LLMMockController) -> MockLiteLLMClient:
    """
    Fixture that provides a MockLiteLLMClient configured with the llm_controller.

    This client can be passed to LiteLlm via the llm_client parameter.
    Depends on llm_controller fixture for response configuration.

    Usage:
        def test_example(llm_client, llm_controller):
            llm_controller.respond_with_message("hello", "Hi there!")
            # Use llm_client with your agent...
    """
    return MockLiteLLMClient(llm_controller)


@pytest.fixture
def agent_factory(llm_client: MockLiteLLMClient) -> Callable[[str], LlmAgent]:
    """
    Fixture that provides a factory function for creating test agents with a preconfigured mock LLM client.

    This factory allows tests to create multiple agents as needed.
    Configure responses via the llm_controller fixture.

    Usage:
        def test_example(agent_factory, llm_controller):
            llm_controller.respond_with_message("hello", "Hi there!")
            test_agent = agent_factory("test_agent")
            another_agent = agent_factory("another_agent")
            # Use agents in your test...
    """

    def _create_agent(name: str = "test_agent") -> LlmAgent:
        """Create a test agent with LiteLLM model and mock client."""
        model_kwargs: dict[str, Any] = {"model": "gemini/gemini-2.5-flash", "llm_client": llm_client}

        return LlmAgent(
            name=name,
            model=LiteLlm(**model_kwargs),
            description="Test agent",
            instruction="You are a test agent.",
        )

    return _create_agent
