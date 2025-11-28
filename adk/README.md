# Agentic Layer Python SDK for Google ADK

SDK for Google ADK that helps to get agents configured in the Agentic Layer quickly.

## Features

- Configures OTEL (Tracing, Metrics, Logging)
- Converts an ADK agent into an instrumented starlette app
- Configures A2A protocol for inter-agent communication
- Offers parsing methods for sub agents and tools
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

# Set up OpenTelemetry instrumentation, logging and metrics
setup_otel()

# Declare your ADK root agent
root_agent = ...

# Define the URL where the agent will be available from outside
# This can not be determined automatically,
# because the port is only known at runtime,
# when the starlette app is started with Uvicorn.
rpc_url = "http://localhost:8000/"

# Create starlette app with A2A protocol
app = to_a2a(root_agent, rpc_url)
```

## OpenTelemetry Configuration

The SDK automatically configures OpenTelemetry observability when running `setup_otel()`. You can customize the OTLP
exporters using standard OpenTelemetry environment variables:
https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
