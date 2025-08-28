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
from agenticlayer.a2a_starlette import agent_to_a2a_starlette
from agenticlayer.otel import setup_otel

root_agent = ...  # Your ADK agent here

setup_otel()
app = agent_to_a2a_starlette(root_agent)
```
