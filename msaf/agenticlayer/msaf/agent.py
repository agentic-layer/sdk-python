"""
Convert Sub Agents and Tools into agent-framework FunctionTools and MCPStreamableHTTPTools.
"""

import logging

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import AgentCapabilities, AgentInterface, Role, SendMessageRequest, TaskState
from a2a.types import AgentCard as A2AAgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, TransportProtocol
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework._tools import FunctionTool
from agenticlayer.shared.config import McpTool, SubAgent
from httpx_retries import Retry, RetryTransport

logger = logging.getLogger(__name__)


def _make_a2a_tool(
    name: str,
    description: str,
    url: str,
    timeout: httpx.Timeout,
) -> FunctionTool:
    """Create a FunctionTool that calls a remote A2A agent via the A2A protocol.

    Args:
        name: The name to expose to the LLM.
        description: Description of what the sub-agent does.
        url: The A2A RPC URL of the sub-agent.
        timeout: HTTP timeout to use when calling the sub-agent.

    Returns:
        A FunctionTool whose invocation sends an A2A message to the remote agent.
    """

    async def call_agent(request: str) -> str:
        """Call the remote A2A sub-agent with a text request."""
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            minimal_card = A2AAgentCard(
                name=name,
                description=description,
                version="0.1.0",
                capabilities=AgentCapabilities(),
                skills=[],
                default_input_modes=["text/plain"],
                default_output_modes=["text/plain"],
                supported_interfaces=[
                    AgentInterface(protocol_binding=TransportProtocol.JSONRPC.value, url=url),
                ],
            )
            client = await create_client(minimal_card, client_config=ClientConfig(httpx_client=http_client))

            message = new_text_message(text=request, role=Role.ROLE_USER)
            response_text = f"No response from agent {name}"
            send_request = SendMessageRequest(message=message)
            async for chunk in client.send_message(send_request):
                if chunk.HasField("status_update"):
                    status = chunk.status_update.status
                    if status.state == TaskState.TASK_STATE_COMPLETED and status.HasField("message"):
                        texts = [p.text for p in status.message.parts if p.HasField("text")]
                        if texts:
                            response_text = "\n".join(texts)
                            break
                elif chunk.HasField("task"):
                    status = chunk.task.status
                    if status.state == TaskState.TASK_STATE_COMPLETED and status.HasField("message"):
                        texts = [p.text for p in status.message.parts if p.HasField("text")]
                        if texts:
                            response_text = "\n".join(texts)
                            break
        return response_text

    return FunctionTool(
        name=name,
        description=description or f"Call the {name} agent",
        func=call_agent,
    )


class MsafAgentFactory:
    """Factory for loading sub-agents and MCP tools into a Microsoft Agent Framework agent.

    Fetches A2A agent cards at startup (failing fast if unreachable) and wraps each
    sub-agent as a :class:`~agent_framework._tools.FunctionTool`.  MCP servers are
    wrapped as :class:`~agent_framework._mcp.MCPStreamableHTTPTool` instances which
    must be entered as async context managers before use.
    """

    def __init__(
        self,
        timeout: httpx.Timeout = httpx.Timeout(timeout=300),  # High timeout for potentially long-running agents
        retry: Retry = Retry(total=10, backoff_factor=0.5, max_backoff_wait=15),
    ) -> None:
        self.timeout = timeout
        self.transport = RetryTransport(retry=retry)

    async def load_sub_agents(self, sub_agents: list[SubAgent]) -> list[FunctionTool]:
        """Fetch agent cards and create FunctionTools for sub-agents.

        Raises a client HTTP error if any agent card is unreachable (after
        retries), causing app startup to fail early.

        Args:
            sub_agents: List of sub-agent configurations.

        Returns:
            A list of FunctionTool instances, one per sub-agent.
        """
        tools: list[FunctionTool] = []
        for sub_agent in sub_agents:
            base_url = str(sub_agent.url).replace(AGENT_CARD_WELL_KNOWN_PATH, "")
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
                agent_card = await resolver.get_agent_card()
            sub_agent_url = next(
                (
                    i.url
                    for i in agent_card.supported_interfaces
                    if i.protocol_binding == TransportProtocol.JSONRPC.value
                ),
                base_url,
            )
            tool = _make_a2a_tool(
                name=sub_agent.name,
                description=agent_card.description,
                url=sub_agent_url,
                timeout=self.timeout,
            )
            logger.info("Loaded sub-agent %s from %s", sub_agent.name, sub_agent_url)
            tools.append(tool)
        return tools

    def create_mcp_tools(self, mcp_tools: list[McpTool]) -> list[MCPStreamableHTTPTool]:
        """Create MCPStreamableHTTPTool instances (not yet connected).

        The returned tools must be entered as async context managers (i.e. ``async with tool``)
        before they can be used with an agent.

        Args:
            mcp_tools: List of MCP tool configurations.

        Returns:
            A list of unconnected MCPStreamableHTTPTool instances.
        """
        tools: list[MCPStreamableHTTPTool] = []
        for mcp_tool in mcp_tools:
            logger.info("Creating MCP tool %s at %s", mcp_tool.name, mcp_tool.url)
            tools.append(
                MCPStreamableHTTPTool(
                    name=mcp_tool.name,
                    url=str(mcp_tool.url),
                    request_timeout=mcp_tool.timeout,
                )
            )
        return tools
