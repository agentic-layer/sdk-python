"""
Convert Sub Agents and Tools into RemoteA2aAgents, AgentTools and McpToolsets.
"""

import logging
from typing import Callable

import httpx
from a2a.client import A2ACardResolver
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from httpx_retries import Retry, RetryTransport

from agenticlayer.config import InteractionType, McpTool, SubAgent
from agenticlayer.constants import HTTP_HEADERS_SESSION_KEY

logger = logging.getLogger(__name__)


def _create_header_provider(propagate_headers: list[str]) -> Callable[[ReadonlyContext], dict[str, str]]:
    """Create a header provider function for a specific MCP server.

    This factory function creates a header provider that filters headers based on
    the MCP server's configuration. Only headers listed in propagate_headers will
    be included in requests to that server.

    Headers are stored in the session state as flat primitive string keys of the form
    ``f"{HTTP_HEADERS_SESSION_KEY}.{header_name_lower}"``. The provider looks up each
    configured header by its lower-cased key and returns it with the casing specified
    in the configuration.

    Example:
        >>> provider = _create_header_provider(['Authorization', 'X-API-Key'])
        >>> # Session state has: {'http_headers.authorization': 'Bearer token', '__http_headers__.x-api-key': 'key123'}
        >>> # Output will be: {'Authorization': 'Bearer token', 'X-API-Key': 'key123'}

    Args:
        propagate_headers: List of header names to propagate to this MCP server

    Returns:
        A header provider function that can be passed to McpToolset
    """

    def header_provider(readonly_context: ReadonlyContext) -> dict[str, str]:
        """Header provider that reads per-header flat primitive keys from session state."""
        if not readonly_context or not readonly_context.state:
            return {}

        result_headers = {}
        for header_name in propagate_headers:
            key = f"{HTTP_HEADERS_SESSION_KEY}.{header_name.lower()}"
            value = readonly_context.state.get(key)
            if value is not None:
                result_headers[header_name] = value

        return result_headers

    return header_provider


class AgentFactory:
    def __init__(
        self,
        timeout: httpx.Timeout = httpx.Timeout(timeout=10),
        retry: Retry = Retry(total=10, backoff_factor=0.5, max_backoff_wait=15),
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self.transport = RetryTransport(retry=retry)
        self.httpx_client = httpx_client

    async def load_agent(self, agent: LlmAgent, sub_agents: list[SubAgent], tools: list[McpTool]) -> LlmAgent:
        """
        Load Sub Agents and Tools into the given agent.

        :param agent: The root agent to load sub agents and tools into
        :param sub_agents: The sub agents to load
        :param tools: The tools to load
        :return: The agent with loaded sub agents and tools
        """

        agents, agent_tools = await self.load_sub_agents(sub_agents)
        mcp_tools = self.load_tools(tools)
        all_tools: list[ToolUnion] = agent_tools + mcp_tools

        # The ADK currently only adds the agent as a function with the agent name to the instructions.
        # The description is not included. So we manually add the descriptions here.
        if agent_tools:
            agent_tool_instructions = "\n\nFollowing agents are available as tools:\n"
            agent_tool_instructions += "\n".join(
                [f"- '{agent_tool.name}': {agent_tool.description}" for agent_tool in agent_tools]
            )
            agent_tool_instructions += "\nYou can use them by calling the tool with the agent name.\n"
            agent.instruction = f"{agent.instruction}{agent_tool_instructions}"

        agent.sub_agents += agents
        agent.tools += all_tools
        return agent

    async def load_sub_agents(self, sub_agents: list[SubAgent]) -> tuple[list[BaseAgent], list[AgentTool]]:
        """
        Convert Sub Agents into RemoteA2aAgents and AgentTools.

        :param sub_agents: The sub agents to load
        :return: A tuple of:
            - list of sub agents for transfer interaction type
            - list of agent tools for tool_call interaction type
        """

        agents: list[BaseAgent] = []
        tools: list[AgentTool] = []
        for sub_agent in sub_agents:
            base_url = str(sub_agent.url).replace(AGENT_CARD_WELL_KNOWN_PATH, "")
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
                resolver = A2ACardResolver(
                    httpx_client=client,
                    base_url=base_url,
                )
                agent_card = await resolver.get_agent_card()
            agent = RemoteA2aAgent(
                name=sub_agent.name,
                agent_card=agent_card,
                httpx_client=self.httpx_client,  # Pass through custom httpx client for testing
            )
            # Set description from agent card, as this is currently done lazy on first RPC call to agent by ADK
            agent.description = agent_card.description
            if sub_agent.interaction_type == InteractionType.TOOL_CALL:
                tools.append(AgentTool(agent=agent))
            else:
                agents.append(agent)

        return agents, tools

    def load_tools(self, mcp_tools: list[McpTool]) -> list[ToolUnion]:
        """
        Convert Tools into McpToolsets.

        :param mcp_tools: The tools to load
        :return: A list of McpToolset tools
        """

        tools: list[ToolUnion] = []
        for tool in mcp_tools:
            logger.info(f"Loading tool {tool.model_dump_json()}")

            # Create header provider with configured headers
            header_provider = _create_header_provider(tool.propagate_headers)

            tools.append(
                McpToolset(
                    connection_params=StreamableHTTPConnectionParams(
                        url=str(tool.url),
                        timeout=tool.timeout,
                    ),
                    # Provide header provider to inject session-stored headers into tool requests
                    header_provider=header_provider,
                )
            )

        return tools
