# agentic-layer-sdk-msaf

Microsoft Agent Framework adapter for the Agentic Layer SDK.

This package provides utilities to convert a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) agent into an instrumented A2A Starlette web application.

## Usage

```python
from agent_framework import Agent
from agenticlayer_msaf.agent_to_a2a import to_a2a
from agenticlayer_msaf.client import create_openai_client

agent = Agent(
    client=create_openai_client(),
    name="MyAgent",
    instructions="You are a helpful assistant.",
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
