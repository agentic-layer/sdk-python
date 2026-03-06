"""Framework-independent OpenTelemetry setup."""

import logging
import os
from typing import Any

import httpx
from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import inject
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_logger = logging.getLogger(__name__)


def _is_text_content(content_type: str) -> bool:
    """Check if content type is text-based and safe to log."""
    text_types = ("application/json", "application/xml", "text/", "application/x-www-form-urlencoded")
    return any(ct in content_type.lower() for ct in text_types)


def _decode_body(body: bytes) -> str:
    """Decode body bytes to string."""
    return body.decode("utf-8", errors="replace")


def request_hook(span: trace.Span, request: httpx.Request) -> None:
    """Hook to log request body at DEBUG level."""
    try:
        # Skip streaming requests to avoid consuming the stream
        if hasattr(request, "stream") and request.stream is not None:
            return

        content_type = request.headers.get("content-type", "")
        if _is_text_content(content_type) and hasattr(request, "content") and request.content:
            _logger.debug("HTTP request body: %s", _decode_body(request.content))
    except Exception:
        _logger.exception("Failed to log request body")


def response_hook(span: trace.Span, request: httpx.Request, response: httpx.Response) -> None:
    """Hook to log response body at DEBUG level."""
    try:
        # Skip streaming responses to avoid consuming the stream
        if hasattr(response, "is_stream_consumed") and not response.is_stream_consumed:
            return

        content_type = response.headers.get("content-type", "")
        if _is_text_content(content_type) and hasattr(response, "content") and response.content:
            _logger.debug("HTTP response body: %s", _decode_body(response.content))
    except Exception:
        _logger.exception("Failed to log response body")


class TraceContextHttpClient(httpx.AsyncClient):
    """httpx client that propagates stored trace context headers into every request.

    Used for MCP clients where HTTP requests happen in background tasks
    that don't inherit the request handler's OTel span context.

    Call :meth:`capture_trace_context` from the request handler context
    (before agent execution) to snapshot the current trace context for
    later injection by the background ``post_writer`` task.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._trace_headers: dict[str, str] = {}
        # Apply MCP-compatible defaults (follow redirects, no env proxy lookup)
        kwargs.setdefault("follow_redirects", True)
        super().__init__(**kwargs)
        # Prepend our hook so stored headers are set first; the monkey-patched
        # _async_inject_trace_context hook (from setup_otel) runs after and will
        # overwrite with live context when an active span exists.
        self.event_hooks.setdefault("request", []).insert(0, self._inject_trace_headers)

    async def _inject_trace_headers(self, request: httpx.Request) -> None:
        """Inject stored trace context headers into the request."""
        for k, v in self._trace_headers.items():
            request.headers[k] = v

    def capture_trace_context(self) -> None:
        """Capture current OTel trace context for injection into future requests."""
        carrier: dict[str, str] = {}
        inject(carrier)
        if carrier:
            self._trace_headers = carrier


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics (framework-independent)."""
    # Set log level for urllib to WARNING to reduce noise (like sending logs to OTLP)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Traces
    trace_provider = TracerProvider()
    if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf") == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGrpc

        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporterGrpc()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHttp

        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporterHttp()))
    trace.set_tracer_provider(trace_provider)

    # Instrument HTTPX clients for span creation
    HTTPXClientInstrumentor().instrument(
        request_hook=request_hook,
        response_hook=response_hook,
    )

    # Logs - inject trace context into log records and export logs via OTLP
    LoggingInstrumentor().instrument()

    logger_provider = LoggerProvider()
    if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf") == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as OTLPLogExporterGrpc

        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporterGrpc()))
    else:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as OTLPLogExporterHttp

        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporterHttp()))
    _logs.set_logger_provider(logger_provider)

    logging.getLogger().addHandler(LoggingHandler(logger_provider=logger_provider))

    # Sets the global default meter provider
    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
        )
    )
