"""
Convert an ADK agent to an A2A Starlette application.
This is an adaption of google.adk.a2a.utils.agent_to_a2a.
"""

import contextlib
import logging
from typing import AsyncIterator, Awaitable, Callable

from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from starlette.applications import Starlette

from .agent import AgentFactory
from .callback_tracer_plugin import CallbackTracerPlugin
from .config import McpTool, SubAgent
from .constants import EXTERNAL_TOKEN_SESSION_KEY

logger = logging.getLogger(__name__)


class TokenCapturingA2aAgentExecutor(A2aAgentExecutor):
    """Custom A2A agent executor that captures and stores the X-External-Token header.

    This executor extends the standard A2aAgentExecutor to intercept the request
    and store the X-External-Token header in the ADK session state. This allows
    MCP tools to access the token via the header_provider hook, using ADK's
    built-in session management rather than external context variables.
    """

    async def _prepare_session(
        self,
        context: RequestContext,
        run_request: AgentRunRequest,
        runner: Runner,
    ) -> Session:
        """Prepare the session and store the external token if present.

        This method extends the parent implementation to capture the X-External-Token
        header from the request context and store it in the session state using ADK's
        recommended approach: creating an Event with state_delta and appending it to
        the session.

        Args:
            context: The A2A request context containing the call context with headers
            run_request: The agent run request
            runner: The ADK runner instance

        Returns:
            The prepared session with the external token stored in its state
        """
        # Call parent to get or create the session
        session: Session = await super()._prepare_session(context, run_request, runner)

        # Extract the X-External-Token header from the request context
        # The call_context.state contains headers from the original HTTP request
        if context.call_context and "headers" in context.call_context.state:
            headers = context.call_context.state["headers"]
            # Headers might be in different cases, check all variations
            external_token = (
                headers.get("x-external-token") 
                or headers.get("X-External-Token")
                or headers.get("X-EXTERNAL-TOKEN")
            )
            
            if external_token:
                # Store the token in the session state using ADK's recommended method:
                # Create an Event with a state_delta and append it to the session.
                # This follows ADK's pattern for updating session state as documented at:
                # https://google.github.io/adk-docs/sessions/state/#how-state-is-updated-recommended-methods
                event = Event(
                    author="system",
                    actions=EventActions(
                        state_delta={EXTERNAL_TOKEN_SESSION_KEY: external_token}
                    )
                )
                await runner.session_service.append_event(session, event)
                logger.debug("Stored external token in session %s via state_delta", session.id)

        return session


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Check if the log message contains the well known path of the card, which is used for health checks
        return record.getMessage().find(AGENT_CARD_WELL_KNOWN_PATH) == -1


async def create_a2a_app(agent: BaseAgent, rpc_url: str) -> A2AStarletteApplication:
    """Create an A2A Starlette application from an ADK agent.

    Args:
        agent: The ADK agent to convert
        rpc_url: The URL where the agent will be available for A2A communication
    Returns:
        An A2AStarletteApplication instance
    """

    async def create_runner() -> Runner:
        """Create a runner for the agent."""
        return Runner(
            app=App(
                name=agent.name or "adk_agent",
                root_agent=agent,
                plugins=[CallbackTracerPlugin()],
            ),
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),  # type: ignore[no-untyped-call]
            memory_service=InMemoryMemoryService(),  # type: ignore[no-untyped-call]
            credential_service=InMemoryCredentialService(),  # type: ignore[no-untyped-call]
        )

    # Create A2A components
    task_store = InMemoryTaskStore()

    # Use custom executor that captures X-External-Token and stores in session
    agent_executor = TokenCapturingA2aAgentExecutor(
        runner=create_runner,
    )

    request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)

    # Build agent card
    agent_card = AgentCard(
        name=agent.name,
        description=agent.description,
        url=rpc_url,
        version="0.1.0",
        capabilities=AgentCapabilities(),
        skills=[],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supports_authenticated_extended_card=False,
    )
    logger.info("Built agent card: %s", agent_card.model_dump_json())

    # Create the A2A Starlette application
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


def to_a2a(
    agent: LlmAgent,
    rpc_url: str,
    sub_agents: list[SubAgent] | None = None,
    tools: list[McpTool] | None = None,
    agent_factory: AgentFactory | None = None,
) -> Starlette:
    """Convert an ADK agent to a Starlette application.
    Resolves sub-agents and tools while starting the application.

    Args:
        :param agent: The ADK agent to convert
        :param rpc_url: The URL where the agent will be available for A2A communication
        :param sub_agents: The sub agents to add to the agent
        :param tools: The tools to add to the agent
        :param agent_factory: Agent factory to use for loading sub-agents and tools

    Returns:
        A Starlette application that can be run with uvicorn

    Example:
        agent = MyAgent()
        rpc_url = "http://localhost:8000/"
        app = to_a2a(root_agent, rpc_url)
        # Then run with: uvicorn module:app
    """

    agent_factory = agent_factory or AgentFactory()

    async def a2a_app_creator() -> A2AStarletteApplication:
        configured_agent = await agent_factory.load_agent(
            agent=agent,
            sub_agents=sub_agents or [],
            tools=tools or [],
        )
        return await create_a2a_app(configured_agent, rpc_url)

    return to_starlette(a2a_app_creator)


def to_starlette(a2a_starlette: Callable[[], Awaitable[A2AStarletteApplication]]) -> Starlette:
    """Convert an ADK agent to a A2A Starlette application.
    This is inspired by google.adk.a2a.utils.agent_to_a2a.

    Args:
        :param a2a_starlette: A callable that creates an A2AStarletteApplication asynchronously during startup.

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
    from .otel_starlette import instrument_starlette_app

    instrument_starlette_app(starlette_app)

    return starlette_app
