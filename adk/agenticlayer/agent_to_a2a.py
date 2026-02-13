"""
Convert an ADK agent to an A2A Starlette application.
This is an adaption of google.adk.a2a.utils.agent_to_a2a.
"""

import contextlib
import logging
from typing import AsyncIterator, Awaitable, Callable

from a2a.server.apps import A2AStarletteApplication
from a2a.server.apps.jsonrpc import CallContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from starlette.applications import Starlette
from starlette.requests import Request

from .agent import AgentFactory
from .callback_tracer_plugin import CallbackTracerPlugin
from .config import McpTool, SubAgent
from .token_context import set_external_token

logger = logging.getLogger(__name__)


class TokenCapturingCallContextBuilder(CallContextBuilder):
    """Custom CallContextBuilder that captures X-External-Token header and stores it in context.

    This builder extracts the X-External-Token header from incoming requests and stores it
    in a context variable for later use by MCP tools. The token is kept separate from the
    session state to prevent agent access while still being available for tool authentication.
    """

    def build(self, request: Request) -> ServerCallContext:
        """Build ServerCallContext and capture the X-External-Token header.

        Args:
            request: The incoming Starlette Request object

        Returns:
            A ServerCallContext with the token stored in context variables
        """
        # Extract and store the external token from the request headers
        token = request.headers.get("X-External-Token")
        set_external_token(token)

        # Build the standard context with headers and auth information
        # (following the pattern from DefaultCallContextBuilder)
        from a2a.auth.user import UnauthenticatedUser
        from a2a.auth.user import User as A2AUser
        from a2a.extensions.common import HTTP_EXTENSION_HEADER, get_requested_extensions
        from a2a.server.apps.jsonrpc import StarletteUserProxy

        user: A2AUser = UnauthenticatedUser()
        state = {}
        try:
            user = StarletteUserProxy(request.user)
            state["auth"] = request.auth
        except Exception:
            pass

        state["headers"] = dict(request.headers)

        return ServerCallContext(
            user=user,
            state=state,
            requested_extensions=get_requested_extensions(request.headers.getlist(HTTP_EXTENSION_HEADER)),
        )


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
            session_service=InMemorySessionService(),  # type: ignore
            memory_service=InMemoryMemoryService(),  # type: ignore
            credential_service=InMemoryCredentialService(),  # type: ignore
        )

    # Create A2A components
    task_store = InMemoryTaskStore()

    agent_executor = A2aAgentExecutor(
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
        context_builder=TokenCapturingCallContextBuilder(),
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
