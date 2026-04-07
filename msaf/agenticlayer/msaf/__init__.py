"""
Microsoft Agent Framework adapter for the Agentic Layer SDK.
Provides utilities to convert a Microsoft agent-framework Agent into
an instrumented A2A Starlette web application.
"""

from agenticlayer.msaf.metrics_middleware import create_metrics_middleware

__all__ = ["create_metrics_middleware"]
