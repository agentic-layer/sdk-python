"""OpenTelemetry setup for a Google ADK Agent App."""

import logging

import httpx
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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


def setup_otel() -> None:
    """Set up OpenTelemetry tracing, logging and metrics."""
    # Set log level for urllib to WARNING to reduce noise (like sending logs to OTLP)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Traces
    _tracer_provider = trace_sdk.TracerProvider()
    _tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    # Sets the global default tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Instrument Google ADK using openinference instrumentation
    GoogleADKInstrumentor().instrument()
    # Instrument HTTPX clients (this also transfers the trace context automatically)
    HTTPXClientInstrumentor().instrument(
        request_hook=request_hook,
        response_hook=response_hook,
    )

    # Logs
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    # Sets the global default logger provider
    set_logger_provider(logger_provider)

    # Attach OTLP handler to root logger
    logging.getLogger().addHandler(LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider))

    # Sets the global default meter provider
    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
        )
    )
