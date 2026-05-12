"""
Convert a Microsoft Agent Framework Agent to an A2A Starlette application.
"""

import contextlib
import json
import logging
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable

from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_0_3, TransportProtocol
from agent_framework import Agent, AgentSession
from agent_framework import Content as MsafContent
from agent_framework import Message as MsafMessage
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework._tools import FunctionTool
from agenticlayer.shared.config import McpTool, SubAgent
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value
from httpx_retries import Retry
from starlette.applications import Starlette

from agenticlayer.msaf.agent import MsafAgentFactory

logger = logging.getLogger(__name__)


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Check if the log message contains the well known path of the card, which is used for health checks
        return record.getMessage().find(AGENT_CARD_WELL_KNOWN_PATH) == -1


def _to_jsonable(value: Any) -> Any:
    """Coerce a tool result into a JSON-friendly value for ``data.response``.

    - ``None`` / ``dict`` / ``list`` / ``int`` / ``float`` / ``bool`` pass through.
    - A Pydantic model is ``model_dump``-ed.
    - A ``str`` is best-effort JSON-decoded (MSAF's ``from_function_result``
      stores results as ``json.dumps(result, default=str)``, so most structured
      results arrive here as JSON strings); if decoding fails, the raw string
      is returned.
    - Anything else is stringified.
    """
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError, TypeError:
            return value
    return str(value)


def _data_part(data: dict[str, Any], metadata: dict[str, Any]) -> Part:
    return Part(data=ParseDict(data, Value()), metadata=metadata)


def _function_call_data_part(
    *,
    call_id: str | None,
    name: str | None,
    arguments: Any,
    metadata: dict[str, Any],
) -> Part:
    """Build a tool-call ``Part`` shaped as ``{id, name, args}``."""
    if isinstance(arguments, dict):
        args: Any = arguments
    elif arguments is None:
        args = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except ValueError, TypeError:
            args = arguments
    else:
        args = str(arguments)

    data: dict[str, Any] = {"args": args}
    if call_id is not None:
        data["id"] = call_id
    if name is not None:
        data["name"] = name
    return _data_part(data, metadata)


def _function_response_data_part(
    *,
    call_id: str | None,
    name: str | None,
    response: Any,
    metadata: dict[str, Any],
) -> Part:
    """Build a tool-response ``Part`` shaped as ``{id, name, response}``."""
    data: dict[str, Any] = {"response": _to_jsonable(response)}
    if call_id is not None:
        data["id"] = call_id
    if name is not None:
        data["name"] = name
    return _data_part(data, metadata)


def _msaf_content_to_a2a_part(content: MsafContent) -> Part:
    """Convert an MSAF Content object to an A2A Part."""
    if content.type == "text":
        return Part(text=content.text or "")

    if content.type == "function_call":
        metadata: dict[str, Any] = {"msaf_type": "function_call"}
        if content.exception is not None:
            metadata["exception"] = content.exception
        return _function_call_data_part(
            call_id=content.call_id,
            name=content.name,
            arguments=content.arguments,
            metadata=metadata,
        )

    if content.type == "mcp_server_tool_call":
        metadata = {"msaf_type": "mcp_server_tool_call"}
        if content.server_name is not None:
            metadata["server_name"] = content.server_name
        return _function_call_data_part(
            call_id=content.call_id,
            name=content.tool_name,
            arguments=content.arguments,
            metadata=metadata,
        )

    if content.type == "function_result":
        metadata = {"msaf_type": "function_result"}
        if not content.result and content.exception is not None:
            response: Any = {"error": str(content.exception)}
        else:
            response = content.result
            if content.exception is not None:
                metadata["exception"] = content.exception
        return _function_response_data_part(
            call_id=content.call_id,
            name=content.name,
            response=response,
            metadata=metadata,
        )

    if content.type == "mcp_server_tool_result":
        return _function_response_data_part(
            call_id=content.call_id,
            name=content.tool_name,
            response=content.output,
            metadata={"msaf_type": "mcp_server_tool_result"},
        )

    # Fallback for non-tool content types (error, usage, text_reasoning,
    # code_interpreter_*, image_generation_*, hosted_*, function_approval_*,
    # oauth_consent_*). Preserves the previous flat-DataPart behaviour.
    data: dict[str, Any] = {"type": content.type}
    if content.name is not None:
        data["name"] = content.name
    if content.call_id is not None:
        data["call_id"] = content.call_id
    if content.arguments is not None:
        data["arguments"] = content.arguments if isinstance(content.arguments, dict) else str(content.arguments)
    if content.result is not None:
        data["result"] = str(content.result)
    if content.tool_name is not None:
        data["tool_name"] = content.tool_name
    if content.server_name is not None:
        data["server_name"] = content.server_name
    if content.exception is not None:
        data["exception"] = content.exception
    if content.output is not None:
        data["output"] = str(content.output)

    return Part(data=ParseDict(data, Value()))


def _msaf_messages_to_a2a(messages: list[MsafMessage]) -> list[Message]:
    """Convert a list of MSAF Messages to A2A Messages."""
    a2a_messages: list[Message] = []
    for msg in messages:
        if msg.role == "system":
            continue
        role = Role.ROLE_USER if msg.role == "user" else Role.ROLE_AGENT
        parts = [_msaf_content_to_a2a_part(c) for c in msg.contents]
        if not parts:
            continue
        a2a_messages.append(
            Message(
                message_id=msg.message_id or str(uuid.uuid4()),
                role=role,
                parts=parts,
            )
        )
    return a2a_messages


class MsafAgentExecutor(AgentExecutor):
    """A2A AgentExecutor that wraps a Microsoft Agent Framework agent.

    Converts incoming A2A messages to agent-framework inputs and publishes
    the agent's response back to the A2A event queue.
    """

    def __init__(
        self,
        agent: Agent,
        sub_agent_tools: list[FunctionTool] | None = None,
        mcp_tool_configs: list[McpTool] | None = None,
        agent_factory: MsafAgentFactory | None = None,
    ) -> None:
        self._agent = agent
        self._sub_agent_tools: list[FunctionTool] = sub_agent_tools or []
        self._mcp_tool_configs: list[McpTool] = mcp_tool_configs or []
        self._agent_factory = agent_factory
        self._sessions: dict[str, AgentSession] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent and publish results to the event queue.

        Follows the v1.0 task-lifecycle streaming pattern: a Task object is
        enqueued first, followed by ``TaskStatusUpdateEvent`` entries until a
        terminal state is reached.
        """
        user_input = context.get_user_input()
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or str(uuid.uuid4())

        task = context.current_task or Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED, message=context.message),
            history=[context.message] if context.message is not None else [],
        )
        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        try:
            async with contextlib.AsyncExitStack() as stack:
                mcp_tools: list[MCPStreamableHTTPTool] = []
                if self._mcp_tool_configs and self._agent_factory:
                    for mcp_tool in self._agent_factory.create_mcp_tools(self._mcp_tool_configs):
                        await stack.enter_async_context(mcp_tool)
                        mcp_tools.append(mcp_tool)

                # Flatten MCPStreamableHTTPTool into their underlying FunctionTool
                # instances so that the session's context-provider pipeline does not
                # encounter raw MCPStreamableHTTPTool objects (which are not JSON
                # serializable) during option merging.
                all_tools: list[FunctionTool] = list(self._sub_agent_tools)
                for mcp_tool in mcp_tools:
                    all_tools.extend(mcp_tool.functions)

                # Look up or create a session for this context to preserve conversation history
                session = self._sessions.get(context_id)
                if session is None:
                    session = self._agent.create_session(session_id=context_id)
                    self._sessions[context_id] = session

                response = await self._agent.run(user_input, session=session, tools=all_tools if all_tools else None)

            a2a_messages = _msaf_messages_to_a2a(response.messages) if response.messages else []

            # Emit intermediate messages (tool calls/results) as working events
            # so they end up in the task history.
            for intermediate_msg in a2a_messages[:-1]:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_WORKING, message=intermediate_msg),
                    )
                )

            # Use the last converted message if available, otherwise fall back to text
            if a2a_messages:
                final_message = a2a_messages[-1]
            else:
                response_text = response.text if hasattr(response, "text") else str(response)
                final_message = new_text_message(text=response_text, role=Role.ROLE_AGENT)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED, message=final_message),
                )
            )
        except Exception as e:
            logger.error("Error running agent: %s", e, exc_info=True)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=new_text_message(text=str(e), role=Role.ROLE_AGENT),
                    ),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the ongoing task."""
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or str(uuid.uuid4())

        if context.current_task is None:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                )
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            )
        )


def _build_agent_card(name: str, description: str | None, rpc_url: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=description or "",
        version="0.1.0",
        capabilities=AgentCapabilities(),
        skills=[],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supported_interfaces=[
            AgentInterface(protocol_binding=TransportProtocol.JSONRPC.value, url=rpc_url),
            AgentInterface(
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_0_3,
                url=rpc_url,
            ),
        ],
    )


async def create_a2a_routes(
    agent: Agent,
    name: str,
    description: str | None,
    rpc_url: str,
    sub_agent_tools: list[FunctionTool],
    mcp_tool_configs: list[McpTool] | None = None,
    agent_factory: MsafAgentFactory | None = None,
) -> list[Any]:
    """Build A2A Starlette routes (agent card + JSON-RPC) for a Microsoft Agent Framework agent.

    Args:
        agent: The Microsoft Agent Framework agent to convert
        name: The name of the agent
        description: Optional description of the agent
        rpc_url: The URL where the agent will be available for A2A communication
        sub_agent_tools: Pre-loaded FunctionTools wrapping remote A2A sub-agents
        mcp_tool_configs: MCP tool configurations; per-request connections are created at execution time
        agent_factory: Factory used to create MCP tools per request

    Returns:
        A list of Starlette ``Route`` objects to mount on a Starlette/FastAPI app.
    """
    task_store = InMemoryTaskStore()
    agent_executor = MsafAgentExecutor(
        agent=agent,
        sub_agent_tools=sub_agent_tools if sub_agent_tools else None,
        mcp_tool_configs=mcp_tool_configs,
        agent_factory=agent_factory,
    )
    agent_card = _build_agent_card(name=name, description=description, rpc_url=rpc_url)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        agent_card=agent_card,
    )
    logger.info("Built agent card: %s", agent_card)

    routes: list[Any] = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True))
    return routes


def to_a2a(
    agent: Agent,
    name: str,
    rpc_url: str,
    description: str | None = None,
    sub_agents: list[SubAgent] | None = None,
    tools: list[McpTool] | None = None,
    agent_factory: MsafAgentFactory | None = None,
) -> Starlette:
    """Convert a Microsoft Agent Framework agent to a Starlette A2A application.

    Args:
        agent: The Microsoft Agent Framework agent to convert
        name: The name of the agent (used in the agent card)
        rpc_url: The URL where the agent will be available for A2A communication
        description: Optional description of the agent
        sub_agents: Optional list of A2A sub-agents to expose as tools
        tools: Optional list of MCP tool servers to expose as tools
        agent_factory: Factory for loading sub-agents and MCP tools. Defaults to
            :class:`MsafAgentFactory` with 2 retries.

    Returns:
        A Starlette application that can be run with uvicorn

    Example:
        from agent_framework import Agent
        from agenticlayer.msaf.agent_to_a2a import to_a2a
        from agenticlayer.msaf.client import create_openai_client

        agent = Agent(client=create_openai_client(), instructions="You are a helpful assistant.")
        app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
        # Then run with: uvicorn module:app
    """
    factory = agent_factory or MsafAgentFactory(retry=Retry(total=2))

    return to_starlette(
        lambda: _build_routes(agent, name, description, rpc_url, sub_agents or [], tools or [], factory)
    )


async def _build_routes(
    agent: Agent,
    name: str,
    description: str | None,
    rpc_url: str,
    sub_agents: list[SubAgent],
    tools: list[McpTool],
    factory: MsafAgentFactory,
) -> list[Any]:
    """Load sub-agents and return the A2A routes.

    MCP tools are created per-request inside the executor; no connections are
    established here.
    """
    sub_agent_tools = await factory.load_sub_agents(sub_agents)

    return await create_a2a_routes(
        agent=agent,
        name=name,
        description=description,
        rpc_url=rpc_url,
        sub_agent_tools=sub_agent_tools,
        mcp_tool_configs=tools if tools else None,
        agent_factory=factory,
    )


def to_starlette(
    routes_builder: Callable[[], Awaitable[list[Any]]],
) -> Starlette:
    """Wrap an async A2A-routes builder in a Starlette application.

    The builder runs during lifespan startup; its returned routes are appended
    to the application's router so they take effect once startup completes.
    """
    # Filter out health check logs from uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        routes = await routes_builder()
        app.router.routes.extend(routes)
        yield

    starlette_app = Starlette(lifespan=lifespan)

    # Instrument the Starlette app with OpenTelemetry
    from agenticlayer.shared.otel_starlette import instrument_starlette_app

    instrument_starlette_app(starlette_app)

    return starlette_app
