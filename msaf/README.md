# agentic-layer-sdk-msaf

Microsoft Agent Framework adapter for the Agentic Layer SDK.

This package provides utilities to convert a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
agent into an instrumented A2A Starlette web application.

## Usage

```python
import os
from agent_framework import Agent
from agent_framework_openai import OpenAIChatCompletionClient
from agenticlayer.msaf import create_metrics_middleware
from agenticlayer.msaf.agent_to_a2a import to_a2a

agent = Agent(
    client=OpenAIChatCompletionClient(
        model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
        base_url=os.environ.get("LITELLM_PROXY_API_BASE"),
        api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
    ),
    name="MyAgent",
    instructions="You are a helpful assistant.",
    middleware=create_metrics_middleware(),
)
app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
# Then run with: uvicorn module:app
```

## Configuration

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

| Metric                             | Type      | Description                                |
|------------------------------------|-----------|--------------------------------------------|
| `gen_ai.client.token.usage`        | Histogram | Input and output token counts per LLM call |
| `gen_ai.client.operation.duration` | Histogram | Duration of LLM / agent operations         |

**Custom** (provided by `create_metrics_middleware()`, must be added to the agent):

| Metric              | Type    | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `agent.invocations` | Counter | Number of agent invocations                      |
| `agent.llm.calls`   | Counter | Number of LLM calls                              |
| `agent.tool.calls`  | Counter | Number of tool calls                             |
| `agent.errors`      | Counter | Number of errors (with `error_source` attribute) |

Add the metrics middleware to your agent:

```python
import os
from agent_framework import Agent
from agent_framework_openai import OpenAIChatCompletionClient
from agenticlayer.msaf import create_metrics_middleware

agent = Agent(
    client=OpenAIChatCompletionClient(
        model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
        base_url=os.environ.get("LITELLM_PROXY_API_BASE"),
        api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
    ),
    instructions="You are a helpful assistant.",
    middleware=create_metrics_middleware(),
)
```

If you already have other middleware, combine them:

```python
import os
from agent_framework import Agent
from agent_framework_openai import OpenAIChatCompletionClient

agent = Agent(
    client=OpenAIChatCompletionClient(
        model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
        base_url=os.environ.get("LITELLM_PROXY_API_BASE"),
        api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
    ),
    instructions="You are a helpful assistant.",
    middleware=[MyCustomMiddleware(), *create_metrics_middleware()],
)
```
