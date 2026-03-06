# agentic-layer-sdk-msaf

Microsoft Agent Framework adapter for the Agentic Layer SDK.

This package provides utilities to convert a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) agent into an instrumented A2A Starlette web application.

## Usage

```python
from agent_framework import Agent
from agenticlayer.msaf import create_metrics_middleware, create_openai_client
from agenticlayer.msaf.agent_to_a2a import to_a2a

agent = Agent(
    client=create_openai_client(),
    name="MyAgent",
    instructions="You are a helpful assistant.",
    middleware=create_metrics_middleware(),
)
app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
# Then run with: uvicorn module:app
```

## Configuration

### OpenAI-compatible gateway (LiteLLM proxy)

Set the following environment variables to point the agent at an OpenAI-compatible gateway
such as [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start):

| Variable | Description |
|---|---|
| `LITELLM_PROXY_API_BASE` | Base URL of the gateway, e.g. `http://litellm-proxy:4000` |
| `LITELLM_PROXY_API_KEY` | API key for the gateway |
| `OPENAI_CHAT_MODEL_ID` | Model name to use, e.g. `gpt-4o` |

`create_openai_client()` reads these variables automatically and passes them to
`OpenAIChatClient` as `base_url` and `api_key`.

## Observability

### OpenTelemetry setup

Call `setup_otel()` before creating agents to configure OTLP exporters and enable instrumentation:

```python
from agenticlayer.msaf.otel import setup_otel

setup_otel()
```

This reads standard `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_PROTOCOL` environment
variables, sets up trace/log/metric providers, and enables the built-in Agent Framework telemetry
layers.

### Metrics

The SDK emits the following OpenTelemetry metrics:

**Built-in** (provided by Agent Framework telemetry layers, enabled by `setup_otel()`):

| Metric | Type | Description |
|---|---|---|
| `gen_ai.client.token.usage` | Histogram | Input and output token counts per LLM call |
| `gen_ai.client.operation.duration` | Histogram | Duration of LLM / agent operations |

**Custom** (provided by `create_metrics_middleware()`, must be added to the agent):

| Metric | Type | Description |
|---|---|---|
| `agent.invocations` | Counter | Number of agent invocations |
| `agent.llm.calls` | Counter | Number of LLM calls |
| `agent.tool.calls` | Counter | Number of tool calls |
| `agent.errors` | Counter | Number of errors (with `error_source` attribute) |

Add the metrics middleware to your agent:

```python
from agent_framework import Agent
from agenticlayer.msaf import create_metrics_middleware, create_openai_client

agent = Agent(
    client=create_openai_client(),
    instructions="You are a helpful assistant.",
    middleware=create_metrics_middleware(),
)
```

If you already have other middleware, combine them:

```python
agent = Agent(
    client=create_openai_client(),
    instructions="You are a helpful assistant.",
    middleware=[MyCustomMiddleware(), *create_metrics_middleware()],
)
```
