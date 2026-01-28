"""OpenTelemetry instrumentation for Starlette applications."""

import logging
import os
from typing import Any, Dict

from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from opentelemetry import trace
from starlette.applications import Starlette

from .otel import _capture_http_bodies, _is_text_content, _truncate_body

_logger = logging.getLogger(__name__)


def _starlette_server_request_hook(span: trace.Span, scope: Dict[str, Any]) -> None:
    """Hook to capture Starlette request body in traces if enabled.

    Note: This captures the body from the ASGI scope's cached body if available.
    It does not consume the request stream to avoid breaking request handling.
    """

    if not _capture_http_bodies:
        return

    try:
        # Only process HTTP requests
        if scope.get("type") != "http":
            return

        # Check if body is cached in scope (some middleware/frameworks cache it)
        # Don't try to read the stream directly as it would consume it
        if "body" in scope:
            body = scope["body"]
            if body:
                # Get content type from headers
                headers = dict(scope.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("latin1")

                if _is_text_content(content_type):
                    span.set_attribute("http.request.body", _truncate_body(body))
    except Exception:
        _logger.exception("Failed to capture Starlette request body in trace")


def _starlette_client_request_hook(span: trace.Span, scope: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Hook to capture Starlette client request body in traces if enabled."""
    # Import here to avoid circular dependency
    from .otel import _capture_http_bodies, _is_text_content, _truncate_body

    if not _capture_http_bodies:
        return

    try:
        # Capture body from the message if available and it's the body message
        if message.get("type") == "http.request" and "body" in message:
            body = message["body"]
            if body:
                # Get content type from scope headers
                headers = dict(scope.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("latin1")

                if _is_text_content(content_type):
                    span.set_attribute("http.request.body", _truncate_body(body))
    except Exception:
        _logger.exception("Failed to capture Starlette client request body in trace")


def _starlette_client_response_hook(span: trace.Span, scope: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Hook to capture Starlette client response body in traces if enabled."""

    if not _capture_http_bodies:
        return

    try:
        # Capture body from response message
        if message.get("type") == "http.response.body" and "body" in message:
            body = message["body"]
            if body:
                # We don't have easy access to response headers here
                # Could try to get from span attributes if set earlier
                span.set_attribute("http.response.body", _truncate_body(body))
    except Exception:
        _logger.exception("Failed to capture Starlette client response body in trace")


def instrument_starlette_app(app: Starlette) -> None:
    """Instrument a Starlette application with OpenTelemetry.

    Args:
        app: The Starlette application to instrument

    Note:
        Body logging is controlled by the enable_body_logging parameter passed to setup_otel().
        This should be called after setup_otel() has been called to set up the tracer provider.
        Body logging for Starlette is limited compared to HTTPX as it must avoid consuming
        request/response streams. Bodies are only captured when already buffered in the ASGI scope.
    """

    # env needs to be set here since _excluded_urls is initialized at module import time
    os.environ.setdefault("OTEL_PYTHON_STARLETTE_EXCLUDED_URLS", AGENT_CARD_WELL_KNOWN_PATH)
    from opentelemetry.instrumentation.starlette import StarletteInstrumentor

    StarletteInstrumentor().instrument_app(
        app,
        server_request_hook=_starlette_server_request_hook,
        client_request_hook=_starlette_client_request_hook,
        client_response_hook=_starlette_client_response_hook,
    )
