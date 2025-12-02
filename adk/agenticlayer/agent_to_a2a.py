"""
Convert an ADK agent to an A2A Starlette application.
This is an adaption of google.adk.a2a.utils.agent_to_a2a.
"""

import logging
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from starlette.applications import Starlette

from .callback_tracer_plugin import CallbackTracerPlugin

logger = logging.getLogger("agenticlayer")


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Check if the log message contains the well known path of the card, which is used for health checks
        return record.getMessage().find(AGENT_CARD_WELL_KNOWN_PATH) == -1


def to_a2a(agent: BaseAgent, rpc_url: str) -> Starlette:
    """Convert an ADK agent to a A2A Starlette application.
    This is an adaption of google.adk.a2a.utils.agent_to_a2a.

    Args:
        agent: The ADK agent to convert
        rpc_url: The URL where the agent will be available for A2A communication

    Returns:
        A Starlette application that can be run with uvicorn

    Example:
        agent = MyAgent()
        rpc_url = "http://localhost:8000/"
        app = to_a2a(root_agent, rpc_url)
        # Then run with: uvicorn module:app
    """

    # Filter out health check logs from uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())

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
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Create a Starlette app that will be configured during startup
    starlette_app = Starlette()

    # Add A2A routes to the main app
    a2a_app.add_routes_to_app(
        starlette_app,
    )

    # Instrument the Starlette app with OpenTelemetry
    # env needs to be set here since _excluded_urls is initialized at module import time
    os.environ.setdefault("OTEL_PYTHON_STARLETTE_EXCLUDED_URLS", AGENT_CARD_WELL_KNOWN_PATH)
    from opentelemetry.instrumentation.starlette import StarletteInstrumentor

    StarletteInstrumentor().instrument_app(starlette_app)

    return starlette_app
