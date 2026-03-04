"""
A custom plugin that records agent metrics using OpenTelemetry.
Tracks agent invocations, LLM calls, token usage, tool calls, and errors.
"""

from typing import Any, Dict, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import metrics

_meter = metrics.get_meter("agenticlayer.agent")


class MetricsPlugin(BasePlugin):
    """A custom ADK plugin that records agent metrics using OpenTelemetry."""

    def __init__(self) -> None:
        super().__init__("MetricsPlugin")
        self._agent_invocations = _meter.create_counter(
            "agent.invocations",
            unit="{invocation}",
            description="Number of agent invocations",
        )
        self._llm_calls = _meter.create_counter(
            "agent.llm.calls",
            unit="{call}",
            description="Number of LLM calls",
        )
        self._llm_input_tokens = _meter.create_histogram(
            "agent.llm.tokens.input",
            unit="{token}",
            description="Number of input tokens per LLM call",
        )
        self._llm_output_tokens = _meter.create_histogram(
            "agent.llm.tokens.output",
            unit="{token}",
            description="Number of output tokens per LLM call",
        )
        self._tool_calls = _meter.create_counter(
            "agent.tool.calls",
            unit="{call}",
            description="Number of tool calls",
        )
        self._agent_errors = _meter.create_counter(
            "agent.errors",
            unit="{error}",
            description="Number of agent errors",
        )

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        self._agent_invocations.add(1, {"agent_name": callback_context.agent_name})
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        model = getattr(llm_response, "model", "unknown") or "unknown"
        attrs = {"agent_name": callback_context.agent_name, "model": model}
        self._llm_calls.add(1, attrs)
        usage = getattr(llm_response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
            if prompt_tokens:
                self._llm_input_tokens.record(prompt_tokens, attrs)
            if candidates_tokens:
                self._llm_output_tokens.record(candidates_tokens, attrs)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self._tool_calls.add(1, {"agent_name": tool_context.agent_name, "tool_name": tool.name})
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        self._agent_errors.add(1, {"agent_name": callback_context.agent_name, "error_source": "model"})
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[Dict[str, Any]]:
        self._agent_errors.add(1, {"agent_name": tool_context.agent_name, "error_source": "tool"})
        return None
