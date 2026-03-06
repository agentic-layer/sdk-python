"""OpenTelemetry setup for a Microsoft Agent Framework Agent App."""

from agent_framework.observability import enable_instrumentation
from agenticlayer.shared.otel import setup_otel as _setup_otel_shared

__all__ = ["setup_otel"]


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics for a Microsoft Agent Framework agent.

    Calls the shared OTLP provider/exporter setup (traces, logs, metrics, HTTPX
    instrumentation) and then enables the built-in ``agent_framework`` telemetry
    layers that emit ``gen_ai.client.token.usage`` and
    ``gen_ai.client.operation.duration`` metrics.

    Starlette server instrumentation is handled separately by
    :func:`agenticlayer.shared.otel_starlette.instrument_starlette_app`.
    """
    _setup_otel_shared()
    enable_instrumentation()
