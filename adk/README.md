# Agentic Layer Python SDK for Google ADK

SDK for Google ADK that helps to get agents configured in the Agentic Layer quickly.

## Features

- Configures OTEL (Tracing, Metrics, Logging)
- Converts an ADK agent into an instrumented starlette app
- Configures A2A protocol for inter-agent communication
- Offers parsing methods for sub agents and tools
- Set log level via env var `LOGLEVEL` (default: `INFO`)
- Automatically passes external API tokens to MCP tools via the `X-External-Token` header

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
setup_otel(capture_http_bodies=True)

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
    "timeout": 30,  // Optional: connect timeout in seconds (default: 30)
    "propagate_headers": ["X-API-Key", "Authorization"]  // Optional: list of headers to propagate
  }
}
```

### Header Propagation

You can configure which HTTP headers are passed from the incoming A2A request to each MCP server using the `propagate_headers` field. This provides fine-grained control over which headers each MCP server receives.

**Key features:**
- **Per-server configuration**: Each MCP server can receive different headers
- **Security**: Headers are only sent to servers explicitly configured to receive them
- **Case-insensitive matching**: Header names are matched case-insensitively
- **Backward compatibility**: When `propagate_headers` is not specified, the legacy behavior is used (only `X-External-Token` is passed)

**Example configuration:**
```json5
{
  "github_api": {
    "url": "https://github-mcp.example.com/mcp",
    "propagate_headers": ["Authorization", "X-GitHub-Token"]
  },
  "stripe_api": {
    "url": "https://stripe-mcp.example.com/mcp",
    "propagate_headers": ["X-Stripe-Key"]
  },
  "public_tool": {
    "url": "https://public-mcp.example.com/mcp"
    // No propagate_headers - only X-External-Token will be passed (legacy behavior)
  }
}
```

## OpenTelemetry Configuration

The SDK automatically configures OpenTelemetry observability when running `setup_otel()`. You can customize the OTLP
exporters using standard OpenTelemetry environment variables:
https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/

### HTTP Body Logging

By default, HTTP request/response bodies are not captured in traces for security and privacy reasons. To enable body
logging for debugging purposes, pass `enable_body_logging=True` to `setup_otel()`.

When enabled, body logging applies to both:
- **HTTPX client requests/responses** (outgoing HTTP calls)
- **Starlette server requests/responses** (incoming HTTP requests to your app)

Body logging behavior:
- Only text-based content types are logged (JSON, XML, plain text, form data)
- Bodies are truncated to 100KB to prevent memory issues
- Binary content (images, PDFs, etc.) is never logged
- Streaming requests/responses are skipped to avoid consuming streams
- All exceptions during body capture are logged but won't break HTTP requests

**Note**: Starlette body logging is more limited than HTTPX because it must avoid consuming request/response streams.
Bodies are only captured when already buffered in the ASGI scope.

## HTTP Header Propagation to MCP Tools

The SDK supports passing HTTP headers from A2A requests to MCP tools. This enables MCP servers to authenticate with external APIs on behalf of users, and provides flexible header-based configuration.

### How It Works

1. **Header Capture**: When an A2A request is received, all HTTP headers are captured and stored in the ADK session state
2. **Secure Storage**: Headers are stored in ADK's session state (not in memory state accessible to the LLM), ensuring the agent cannot directly access or leak sensitive information
3. **Per-Server Filtering**: Each MCP server receives only the headers configured in its `propagate_headers` list
4. **Automatic Injection**: When MCP tools are invoked, the SDK uses ADK's `header_provider` hook to retrieve the configured headers from the session and inject them into tool requests

### Configuration

Configure which headers to propagate using the `propagate_headers` field in your MCP tool configuration:

```json5
{
  "weather_api": {
    "url": "https://weather-mcp.example.com/mcp",
    "propagate_headers": ["X-API-Key", "X-User-Location"]
  },
  "database_tool": {
    "url": "https://db-mcp.example.com/mcp",
    "propagate_headers": ["Authorization"]
  }
}
```

### Usage Example

Include the headers you want to propagate in your A2A requests:

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "Authorization: Bearer your-token" \
  -H "X-User-Location: US-West" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is the weather?"}],
        "messageId": "msg-123",
        "contextId": "ctx-123"
      }
    }
  }'
```

Based on the configuration above:
- `weather_api` MCP server will receive `X-API-Key` and `X-User-Location` headers
- `database_tool` MCP server will receive only the `Authorization` header

### Backward Compatibility

For backward compatibility, if `propagate_headers` is not specified in the configuration, the SDK will use legacy behavior: only the `X-External-Token` header is passed to the MCP server.

```json5
{
  "legacy_tool": {
    "url": "https://legacy-mcp.example.com/mcp"
    // No propagate_headers - only X-External-Token will be passed
  }
}
```

**Limitations**: Header propagation is only supported for MCP servers. Propagation to sub-agents is not currently supported due to ADK limitations in passing custom HTTP headers in A2A requests.

### Security Considerations

- Tokens are stored in ADK session state (separate from memory state that the LLM can access)
- Tokens are not directly accessible to agent code through normal session state queries
- Tokens persist for the session duration and are managed by ADK's session lifecycle
- This is a simple authentication mechanism; for production use, consider implementing more sophisticated authentication and authorization schemes
