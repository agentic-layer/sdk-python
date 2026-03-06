"""OpenTelemetry setup for a Microsoft Agent Framework Agent App."""

import logging

from agent_framework.observability import configure_otel_providers
from agenticlayer.shared.otel import request_hook, response_hook
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

__all__ = ["setup_otel"]

_logger = logging.getLogger(__name__)


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics for a Microsoft Agent Framework agent.

    Uses the built-in ``agent_framework`` OTLP provider setup (reads standard
    ``OTEL_EXPORTER_OTLP_*`` environment variables) and enables the telemetry
    layers that emit ``gen_ai.client.token.usage`` and
    ``gen_ai.client.operation.duration`` metrics.

    Additionally instruments HTTPX clients so outgoing HTTP calls (to
    sub-agents, MCP servers, LLM gateways) are traced with debug-level
    request/response body logging.

    Starlette server instrumentation is handled separately by
    :func:`agenticlayer.shared.otel_starlette.instrument_starlette_app`.
    """
    # Set log level for urllib to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    configure_otel_providers()

    HTTPXClientInstrumentor().instrument(
        request_hook=request_hook,
        response_hook=response_hook,
    )
