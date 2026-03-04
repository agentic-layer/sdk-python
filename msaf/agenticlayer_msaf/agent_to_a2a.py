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
from starlette.applications import Starlette

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

    def __init__(self, agent: SupportsAgentRun) -> None:
        self._agent = agent

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
            response = await self._agent.run(user_input)
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
) -> A2AStarletteApplication:
    """Create an A2A Starlette application from a Microsoft Agent Framework agent.

    Args:
        agent: The Microsoft Agent Framework agent to convert
        name: The name of the agent
        description: Optional description of the agent
        rpc_url: The URL where the agent will be available for A2A communication

    Returns:
        An A2AStarletteApplication instance
    """
    task_store = InMemoryTaskStore()
    agent_executor = MsafAgentExecutor(agent=agent)
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
) -> Starlette:
    """Convert a Microsoft Agent Framework agent to a Starlette A2A application.

    Args:
        agent: The Microsoft Agent Framework agent to convert
        name: The name of the agent (used in the agent card)
        rpc_url: The URL where the agent will be available for A2A communication
        description: Optional description of the agent

    Returns:
        A Starlette application that can be run with uvicorn

    Example:
        from agent_framework import Agent
        from agent_framework.openai import OpenAIChatClient
        from agenticlayer_msaf.agent_to_a2a import to_a2a

        agent = Agent(client=OpenAIChatClient(), instructions="You are a helpful assistant.")
        app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
        # Then run with: uvicorn module:app
    """

    async def a2a_app_creator() -> A2AStarletteApplication:
        return await create_a2a_app(agent, name, description, rpc_url)

    return to_starlette(a2a_app_creator)


def to_starlette(a2a_starlette: Callable[[], Awaitable[A2AStarletteApplication]]) -> Starlette:
    """Convert an A2A application creator to a Starlette application.

    Args:
        a2a_starlette: A callable that creates an A2AStarletteApplication asynchronously during startup.

    Returns:
        A Starlette application that can be run with uvicorn
    """
    # Filter out health check logs from uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        a2a_app = await a2a_starlette()
        # Add A2A routes to the main app
        a2a_app.add_routes_to_app(app)
        yield

    # Create a Starlette app that will be configured during startup
    starlette_app = Starlette(lifespan=lifespan)

    # Instrument the Starlette app with OpenTelemetry
    from agenticlayer_shared.otel_starlette import instrument_starlette_app

    instrument_starlette_app(starlette_app)

    return starlette_app
