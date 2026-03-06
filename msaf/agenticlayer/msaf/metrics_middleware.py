"""
Middleware that records agent metrics using OpenTelemetry.
Tracks agent invocations, LLM calls, tool calls, and errors.

Token usage and operation duration are already provided by the built-in
``agent_framework.observability`` telemetry layers and do not need to be
duplicated here.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from agent_framework import (
    AgentContext,
    AgentMiddleware,
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTypes,
)
from opentelemetry import metrics

_meter = metrics.get_meter("agenticlayer.agent")

_agent_invocations = _meter.create_counter(
    "agent.invocations",
    unit="{invocation}",
    description="Number of agent invocations",
)
_llm_calls = _meter.create_counter(
    "agent.llm.calls",
    unit="{call}",
    description="Number of LLM calls",
)
_tool_calls = _meter.create_counter(
    "agent.tool.calls",
    unit="{call}",
    description="Number of tool calls",
)
_agent_errors = _meter.create_counter(
    "agent.errors",
    unit="{error}",
    description="Number of agent errors",
)


class AgentInvocationMetrics(AgentMiddleware):
    """Counts agent invocations and errors."""

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        agent_name = getattr(context.agent, "name", None) or "unknown"
        _agent_invocations.add(1, {"agent_name": agent_name})
        try:
            await call_next()
        except Exception:
            _agent_errors.add(1, {"agent_name": agent_name, "error_source": "agent"})
            raise


class LlmCallMetrics(ChatMiddleware):
    """Counts LLM / chat-client calls and records model-level errors."""

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        options = context.options or {}
        model: str = options.get("model_id") or getattr(context.client, "model_id", None) or "unknown"
        attrs: dict[str, Any] = {"model": model}
        _llm_calls.add(1, attrs)
        try:
            await call_next()
        except Exception:
            _agent_errors.add(1, {**attrs, "error_source": "model"})
            raise


class ToolCallMetrics(FunctionMiddleware):
    """Counts tool / function calls and records tool-level errors."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        tool_name = getattr(context.function, "name", None) or "unknown"
        _tool_calls.add(1, {"tool_name": tool_name})
        try:
            await call_next()
        except Exception:
            _agent_errors.add(1, {"tool_name": tool_name, "error_source": "tool"})
            raise


def create_metrics_middleware() -> list[MiddlewareTypes]:
    """Return the full set of metrics middleware ready to pass to an Agent.

    Example::

        from agent_framework import Agent
        from agenticlayer.msaf.metrics_middleware import create_metrics_middleware

        agent = Agent(
            client=client,
            middleware=create_metrics_middleware(),
        )
    """
    return [AgentInvocationMetrics(), LlmCallMetrics(), ToolCallMetrics()]
