"""
Convert a Microsoft Agent Framework Agent to an A2A Starlette application.
"""

import contextlib
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from agent_framework import SupportsAgentRun
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework._tools import FunctionTool
from agenticlayer_shared.config import McpTool, SubAgent
from httpx_retries import Retry
from starlette.applications import Starlette

from agenticlayer_msaf.agent import MsafAgentFactory

logger = logging.getLogger(__name__)


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Check if the log message contains the well known path of the card, which is used for health checks
        return record.getMessage().find(AGENT_CARD_WELL_KNOWN_PATH) == -1


class MsafAgentExecutor(AgentExecutor):
    """A2A AgentExecutor that wraps a Microsoft Agent Framework agent.

    Converts incoming A2A messages to agent-framework inputs and publishes
    the agent's response back to the A2A event queue.
    """

    def __init__(
        self,
        agent: SupportsAgentRun,
        extra_tools: list[FunctionTool | MCPStreamableHTTPTool] | None = None,
    ) -> None:
        self._agent = agent
        self._extra_tools: list[FunctionTool | MCPStreamableHTTPTool] = extra_tools or []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent and publish results to the event queue."""
        user_input = context.get_user_input()
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or str(uuid.uuid4())

        if not context.current_task:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.submitted,
                        message=context.message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    final=False,
                )
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=False,
            )
        )

        try:
            response = await self._agent.run(user_input, tools=self._extra_tools if self._extra_tools else None)
            response_text = response.text if hasattr(response, "text") else str(response)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=response_text))],
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    final=True,
                )
            )
        except Exception as e:
            logger.error("Error running agent: %s", e, exc_info=True)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=str(e)))],
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    final=True,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the ongoing task."""
        task_id = context.task_id or str(uuid.uuid4())
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context.context_id or str(uuid.uuid4()),
                status=TaskStatus(
                    state=TaskState.canceled,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )


async def create_a2a_app(
    agent: SupportsAgentRun,
    name: str,
    description: str | None,
    rpc_url: str,
    sub_agent_tools: list[FunctionTool],
    mcp_tools: list[MCPStreamableHTTPTool],
) -> A2AStarletteApplication:
    """Create an A2A Starlette application from a Microsoft Agent Framework agent.

    Args:
        agent: The Microsoft Agent Framework agent to convert
        name: The name of the agent
        description: Optional description of the agent
        rpc_url: The URL where the agent will be available for A2A communication
        sub_agent_tools: Pre-loaded FunctionTools wrapping remote A2A sub-agents
        mcp_tools: Connected MCPStreamableHTTPTool instances

    Returns:
        An A2AStarletteApplication instance
    """
    task_store = InMemoryTaskStore()
    extra_tools: list[FunctionTool | MCPStreamableHTTPTool] = [*sub_agent_tools, *mcp_tools]
    agent_executor = MsafAgentExecutor(agent=agent, extra_tools=extra_tools if extra_tools else None)
    request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)

    agent_card = AgentCard(
        name=name,
        description=description or "",
        url=rpc_url,
        version="0.1.0",
        capabilities=AgentCapabilities(),
        skills=[],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supports_authenticated_extended_card=False,
    )
    logger.info("Built agent card: %s", agent_card.model_dump_json())

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


def to_a2a(
    agent: SupportsAgentRun,
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
        from agenticlayer_msaf.agent_to_a2a import to_a2a
        from agenticlayer_msaf.client import create_openai_client

        agent = Agent(client=create_openai_client(), instructions="You are a helpful assistant.")
        app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
        # Then run with: uvicorn module:app
    """
    factory = agent_factory or MsafAgentFactory(retry=Retry(total=2))

    return to_starlette(lambda: _build_app(agent, name, description, rpc_url, sub_agents or [], tools or [], factory))


async def _build_app(
    agent: SupportsAgentRun,
    name: str,
    description: str | None,
    rpc_url: str,
    sub_agents: list[SubAgent],
    tools: list[McpTool],
    factory: MsafAgentFactory,
) -> tuple[A2AStarletteApplication, list[MCPStreamableHTTPTool]]:
    """Load sub-agents, connect MCP tools, and return the A2A app plus tools to manage.

    The returned MCP tools have already been entered as async context managers;
    the caller (lifespan) is responsible for exiting them on shutdown.
    """
    sub_agent_tools = await factory.load_sub_agents(sub_agents)
    mcp_tools = factory.create_mcp_tools(tools)

    connected_mcp: list[MCPStreamableHTTPTool] = []
    try:
        for mcp_tool in mcp_tools:
            await mcp_tool.__aenter__()
            connected_mcp.append(mcp_tool)
    except Exception:
        for already_connected in reversed(connected_mcp):
            await already_connected.__aexit__(None, None, None)
        raise

    app = await create_a2a_app(
        agent=agent,
        name=name,
        description=description,
        rpc_url=rpc_url,
        sub_agent_tools=sub_agent_tools,
        mcp_tools=connected_mcp,
    )
    return app, connected_mcp


def to_starlette(
    a2a_app_creator: Callable[[], Awaitable[tuple[A2AStarletteApplication, list[MCPStreamableHTTPTool]]]],
) -> Starlette:
    """Convert an A2A application creator to a Starlette application.

    Args:
        a2a_app_creator: A callable that creates an A2AStarletteApplication and
            connected MCP tools asynchronously during startup.

    Returns:
        A Starlette application that can be run with uvicorn
    """
    # Filter out health check logs from uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        a2a_app, connected_mcp = await a2a_app_creator()
        # Add A2A routes to the main app
        a2a_app.add_routes_to_app(app)
        try:
            yield
        finally:
            for mcp_tool in connected_mcp:
                await mcp_tool.__aexit__(None, None, None)

    # Create a Starlette app that will be configured during startup
    starlette_app = Starlette(lifespan=lifespan)

    # Instrument the Starlette app with OpenTelemetry
    from agenticlayer_shared.otel_starlette import instrument_starlette_app

    instrument_starlette_app(starlette_app)

    return starlette_app
