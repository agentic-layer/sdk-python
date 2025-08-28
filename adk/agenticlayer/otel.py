import logging
import os

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


def setup_otel():
    # Traces
    _tracer_provider = trace_sdk.TracerProvider()
    _tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    # Sets the global default tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Instrument Google ADK using openinference instrumentation
    GoogleADKInstrumentor().instrument()
    # Instrument HTTPX clients (this also transfers the trace context automatically)
    HTTPXClientInstrumentor().instrument()

    # Logs
    provider = LoggerProvider()
    provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    # Sets the global default logger provider
    set_logger_provider(provider)

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
    log_level = os.environ.get("LOGLEVEL", "INFO")
    logging.getLogger().setLevel(log_level)
    # Attach OTLP handler to root logger
    logging.getLogger().addHandler(handler)

    # Metrics
    provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    # Sets the global default meter provider
    metrics.set_meter_provider(provider)
