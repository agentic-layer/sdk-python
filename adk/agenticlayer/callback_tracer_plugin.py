import re
from typing import Any, Dict, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import trace

# Pattern to match any key containing 'structuredcontent' or 'structured_content', case-insensitive
STRUCTURED_CONTENT_PATTERN = re.compile(r"\.structured_?content", re.IGNORECASE)


def _span_attribute_item(key: str, data: Any) -> tuple[str, Any]:
    """Convert data to a span attribute-compatible type."""
    if isinstance(data, (str, bool, int, float)):  # only these types are supported by span attributes
        return key, data
    else:
        return key, str(data)


def _flatten_dict(
    data: Any, parent_key: str = "", sep: str = ".", parent_key_lower: Optional[str] = None
) -> Dict[str, Any]:
    if parent_key_lower is None:
        parent_key_lower = parent_key.lower()

    if STRUCTURED_CONTENT_PATTERN.search(parent_key_lower):
        return {}  # skip structured content as it can add too many attributes

    items: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            new_key_lower = new_key.lower()
            items.extend(_flatten_dict(v, new_key, sep=sep, parent_key_lower=new_key_lower).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}"
            new_key_lower = new_key.lower()
            items.extend(_flatten_dict(v, new_key, sep=sep, parent_key_lower=new_key_lower).items())
    elif data is not None:
        items.append(_span_attribute_item(parent_key, data))
    return dict(items)


def _set_span_attributes_from_callback_context(span: Any, callback_context: CallbackContext) -> None:
    conversation_id = (
        callback_context.state.to_dict().get("conversation_id") or callback_context._invocation_context.session.id
    )
    span.set_attribute("agent_name", callback_context.agent_name)
    span.set_attribute("conversation_id", conversation_id)
    span.set_attribute("invocation_id", callback_context.invocation_id)
    span.set_attributes(callback_context.state.to_dict())

    if callback_context.user_content:
        span.set_attributes(_flatten_dict(callback_context.user_content.model_dump(), parent_key="user_content"))


def _set_span_attributes_for_tool(span: Any, tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext) -> None:
    _set_span_attributes_from_callback_context(span, tool_context)
    span.set_attributes(_flatten_dict(tool_context.actions.model_dump(), parent_key="tool_context.actions"))
    span.set_attribute("tool_name", tool.name)
    span.set_attributes(_flatten_dict(args, parent_key="args"))


class CallbackTracerPlugin(BasePlugin):
    """A custom plugin class for the Observability Dashboard."""

    def __init__(self) -> None:
        super().__init__("AdkCallbackTracerPlugin")

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        with trace.get_tracer(__name__).start_as_current_span("before_agent_callback") as span:
            _set_span_attributes_from_callback_context(span, callback_context)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        with trace.get_tracer(__name__).start_as_current_span("after_agent_callback") as span:
            _set_span_attributes_from_callback_context(span, callback_context)
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        with trace.get_tracer(__name__).start_as_current_span("before_model_callback") as span:
            _set_span_attributes_from_callback_context(span, callback_context)
        span.set_attribute("model", llm_request.model or "unknown")
        if llm_request.contents:
            span.set_attributes(
                _flatten_dict(llm_request.contents[-1].model_dump(), parent_key="llm_request.content")
            )  # only send the last content part (last user input)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        with trace.get_tracer(__name__).start_as_current_span("after_model_callback") as span:
            _set_span_attributes_from_callback_context(span, callback_context)
            span.set_attributes(_flatten_dict(llm_response.model_dump(), parent_key="llm_response"))
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[Dict[str, Any]]:
        with trace.get_tracer(__name__).start_as_current_span("before_tool_callback") as span:
            _set_span_attributes_for_tool(span, tool, tool_args, tool_context)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        with trace.get_tracer(__name__).start_as_current_span("after_tool_callback") as span:
            _set_span_attributes_for_tool(span, tool, tool_args, tool_context)
        if isinstance(result, (dict, list)):
            span.set_attributes(_flatten_dict(result, parent_key="tool_response"))
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        with trace.get_tracer(__name__).start_as_current_span("on_model_error_callback") as span:
            _set_span_attributes_from_callback_context(span, callback_context)
            span.set_attribute("model", llm_request.model or "unknown")
            if llm_request.contents:
                span.set_attributes(
                    _flatten_dict(llm_request.contents[-1].model_dump(), parent_key="llm_request.content")
                )  # only send the last content part (last user input)
            span.set_attribute("error", str(error))
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[Dict[str, Any]]:
        with trace.get_tracer(__name__).start_as_current_span("on_tool_error_callback") as span:
            _set_span_attributes_for_tool(span, tool, tool_args, tool_context)
            span.set_attribute("error", str(error))
        return None
