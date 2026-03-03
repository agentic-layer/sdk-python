# agentic-layer-sdk-msaf

Microsoft Agent Framework adapter for the Agentic Layer SDK.

This package provides utilities to convert a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) agent into an instrumented A2A Starlette web application.

## Usage

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agenticlayer_msaf.agent_to_a2a import to_a2a

agent = Agent(
    client=OpenAIChatClient(),
    name="MyAgent",
    instructions="You are a helpful assistant.",
)
app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
# Then run with: uvicorn module:app
```
