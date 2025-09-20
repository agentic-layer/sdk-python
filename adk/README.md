# Agentic Layer Python SDK for Google ADK

SDK for Google ADK that helps to get agents configured in the Agentic Layer quickly.

## Features

- Configures OTEL (Tracing, Metrics, Logging)
- Converts an ADK agent into an instrumented starlette app with health endpoint
- Set log level via env var `LOGLEVEL` (default: `INFO`)

## Usage

Dependencies can be installed via pip or the tool of your choice:

```shell
pip install agentic-layer-sdk-adk
```

Basic usage example:

```python
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.otel import setup_otel

root_agent = ...  # Your ADK agent here

# Set up OpenTelemetry instrumentation, logging and metrics
setup_otel()

# Create starlette app with A2A protocol
app = to_a2a(root_agent)
```
