import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.base_agent import BaseAgent
from opentelemetry.instrumentation.starlette import StarletteInstrumentor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse


def agent_to_a2a_starlette(agent: BaseAgent) -> Starlette:
    app = to_a2a(
        agent,
        host=os.environ.get("A2A_HOST", "localhost"),
        port=int(os.environ.get("A2A_HTTP_PORT", "8000")),
    )

    StarletteInstrumentor().instrument_app(app)

    def health(_: Request):
        return JSONResponse(content={"status": "healthy"})

    app.add_route("/health", health)

    return app
