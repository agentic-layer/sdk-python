"""
Microsoft Agent Framework adapter for the Agentic Layer SDK.
Provides utilities to convert a Microsoft agent-framework Agent into
an instrumented A2A Starlette web application.
"""

from agenticlayer.msaf.client import create_openai_client

__all__ = ["create_openai_client"]
