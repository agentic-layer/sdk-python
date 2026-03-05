"""OpenTelemetry setup for a Microsoft Agent Framework Agent App."""

from agenticlayer.shared.otel import setup_otel as _setup_otel_shared

__all__ = ["setup_otel"]


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics for a Microsoft Agent Framework agent."""
    _setup_otel_shared()
