"""OpenTelemetry setup for a Google ADK Agent App."""

from agenticlayer._otel import (
    _decode_body,
    _is_text_content,
    request_hook,
    response_hook,
)
from agenticlayer._otel import (
    setup_otel as _setup_otel_base,
)

__all__ = ["_decode_body", "_is_text_content", "request_hook", "response_hook", "setup_otel"]


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics, including Google ADK instrumentation."""
    _setup_otel_base()

    # Instrument Google ADK using openinference instrumentation
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

    GoogleADKInstrumentor().instrument()
