import contextlib
import logging
import os
from typing import AsyncIterator

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.cli.utils.logs import setup_adk_logger
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from opentelemetry.instrumentation.starlette import StarletteInstrumentor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .callback_tracer_plugin import CallbackTracerPlugin


def to_a2a(agent: BaseAgent) -> Starlette:
    """Convert an ADK agent to a A2A Starlette application.
    This is an adaption of google.adk.a2a.utils.agent_to_a2a.

    Args:
        agent: The ADK agent to convert

    Returns:
        A Starlette application that can be run with uvicorn

    Example:
        agent = MyAgent()
        app = to_a2a(agent)
        # Then run with: uvicorn module:app
    """
    # Set up ADK logging to ensure logs are visible when using uvicorn directly
    log_level = os.environ.get("LOGLEVEL", "INFO")
    setup_adk_logger(log_level)  # type: ignore
    logger = logging.getLogger(__name__)

    async def create_runner() -> Runner:
        """Create a runner for the agent."""
        return Runner(
            app_name=agent.name or "adk_agent",
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),  # type: ignore
            memory_service=InMemoryMemoryService(),  # type: ignore
            credential_service=InMemoryCredentialService(),  # type: ignore
            plugins=[CallbackTracerPlugin()],
        )

    # Create A2A components
    task_store = InMemoryTaskStore()

    agent_executor = A2aAgentExecutor(
        runner=create_runner,
    )

    request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)

    # Add a simple health check endpoint for readiness/liveness probes
    def health(_: Request) -> JSONResponse:
        return JSONResponse(content={"status": "healthy"})

    # Get the agent card URL from environment variable *only*
    # At this point, we don't know the applications port and the host is unknown when running in k8s or similar
    # A2A_AGENT_CARD_URL is deprecated but still supported for backwards compatibility
    agent_card_url = os.environ.get("AGENT_A2A_RPC_URL", os.environ.get("A2A_AGENT_CARD_URL", None))
    logger.debug(f"Using agent card url: {agent_card_url}")

    # Add startup handler to build the agent card and configure A2A routes
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.debug("Setting up A2A app")
        # Build agent card
        card_builder = AgentCardBuilder(
            agent=agent,
            rpc_url=agent_card_url,
        )
        # Build the agent card asynchronously
        agent_card = await card_builder.build()

        # Create the A2A Starlette application
        a2a_app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        # Add A2A routes to the main app
        a2a_app.add_routes_to_app(
            app,
        )
        yield

    # Create a Starlette app that will be configured during startup
    starlette_app = Starlette(lifespan=lifespan, routes=[Route("/health", health)])

    # Instrument the Starlette app with OpenTelemetry
    StarletteInstrumentor().instrument_app(starlette_app)

    return starlette_app
