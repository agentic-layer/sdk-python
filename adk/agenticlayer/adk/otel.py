"""OpenTelemetry setup for a Google ADK Agent App."""

from agenticlayer.shared.otel import (
    setup_otel as _setup_otel_base,
)


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics, including Google ADK instrumentation."""
    _setup_otel_base()

    # Instrument Google ADK using openinference instrumentation
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

    GoogleADKInstrumentor().instrument()
