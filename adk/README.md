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
from agenticlayer.config import parse_sub_agents, parse_tools
from agenticlayer.otel import setup_otel
from google.adk.agents import LlmAgent

# Set up OpenTelemetry instrumentation, logging and metrics
setup_otel()

# Parse sub agents and tools from JSON configuration
sub_agent, agent_tools = parse_sub_agents("{}")
mcp_tools = parse_tools("{}")
tools = agent_tools + mcp_tools

# Declare your ADK root agent
root_agent = LlmAgent(
    name="root-agent",
    sub_agents=sub_agent,
    tools=tools,
    # [...]
)

# Define the URL where the agent will be available from outside
# This can not be determined automatically,
# because the port is only known at runtime,
# when the starlette app is started with Uvicorn.
rpc_url = "http://localhost:8000/"

# Create starlette app with A2A protocol
app = to_a2a(root_agent, rpc_url)
```

## Configuration

The JSON configuration for sub agents should follow this structure:
```json5
{
  "agent_name": {
    "url": "http://agent-url/.well-known/agent-card.json",
    // Optional: interaction type, defaults to "tool_call"
    // "transfer" for full delegation, "tool_call" for tool-like usage
    "interaction_type": "transfer|tool_call"
  }
}
```

The JSON configuration for `AGENT_TOOLS` should follow this structure:
```json5
{
  "tool_name": {
    "url": "https://mcp-tool-endpoint:8000/mcp",
    "timeout": 30  // Optional: connect timeout in seconds
  }
}
```

## OpenTelemetry Configuration

The SDK automatically configures OpenTelemetry observability when running `setup_otel()`. You can customize the OTLP
exporters using standard OpenTelemetry environment variables:
https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
