"""
Convert Sub Agents and Tools into RemoteA2aAgents, AgentTools and McpToolsets.
"""

import logging

import httpx
from a2a.client import A2ACardResolver
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from httpx_retries import Retry, RetryTransport

from agenticlayer.config import InteractionType, McpTool, SubAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    def __init__(
        self,
        timeout: httpx.Timeout = httpx.Timeout(timeout=10),
        retry: Retry = Retry(total=10, backoff_factor=0.5, max_backoff_wait=15),
    ) -> None:
        self.timeout = timeout
        self.transport = RetryTransport(retry=retry)

    async def load_agent(self, agent: LlmAgent, sub_agents: list[SubAgent], tools: list[McpTool]) -> LlmAgent:
        """
        Load Sub Agents and Tools into the given agent.

        :param agent: The root agent to load sub agents and tools into
        :param sub_agents: The sub agents to load
        :param tools: The tools to load
        :return: The agent with loaded sub agents and tools
        """

        agents, agent_tools = await self.load_sub_agents(sub_agents)
        mcp_tools, mcp_tool_descriptions = await self.load_tools(tools)
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

        # Add MCP tool descriptions to instructions
        if mcp_tool_descriptions:
            mcp_tool_instructions = "\n\nFollowing MCP tools are available:\n"
            mcp_tool_instructions += "\n".join(
                [f"- '{name}': {description}" for name, description in mcp_tool_descriptions]
            )
            mcp_tool_instructions += "\nYou can use them by calling the tool with the tool name.\n"
            agent.instruction = f"{agent.instruction}{mcp_tool_instructions}"

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
            agent = RemoteA2aAgent(name=sub_agent.name, agent_card=agent_card)
            # Set description from agent card, as this is currently done lazy on first RPC call to agent by ADK
            agent.description = agent_card.description
            if sub_agent.interaction_type == InteractionType.TOOL_CALL:
                tools.append(AgentTool(agent=agent))
            else:
                agents.append(agent)

        return agents, tools

    async def load_tools(self, mcp_tools: list[McpTool]) -> tuple[list[ToolUnion], list[tuple[str, str]]]:
        """
        Convert Tools into McpToolsets and extract their descriptions.

        This method creates McpToolset instances for runtime use and introspects
        MCP servers using temporary connections to extract tool descriptions.

        We use TWO separate McpToolset instances per MCP server:
        1. Temporary instance for introspection - created, used to get tool schemas, then closed
        2. Runtime instance - long-lived, kept in agent's tool list for execution

        Why separate instances are necessary:
        - McpToolset internally uses anyio.create_task_group() for async connection management
        - When get_tools() is called during Starlette's lifespan startup, these task groups
          conflict with Starlette's own async context management
        - Closing the introspection instance ensures all internal task groups are properly
          cleaned up before the runtime instance is created

        :param mcp_tools: The tools to load
        :return: Tuple of (toolsets for runtime, tool descriptions for instructions)
        :raises ConnectionError: If any MCP server is unavailable or unreachable
        """

        toolsets: list[ToolUnion] = []
        tool_descriptions: list[tuple[str, str]] = []

        for mcp_tool in mcp_tools:
            logger.info(f"Loading tool {mcp_tool.model_dump_json()}")

            # Create connection params (reused for both introspection and runtime)
            connection_params = StreamableHTTPConnectionParams(
                url=str(mcp_tool.url),
                timeout=mcp_tool.timeout,
            )

            # PHASE 1: Introspection using temporary toolset
            # Create a temporary toolset just for introspection to avoid async context conflicts
            introspection_toolset = McpToolset(connection_params=connection_params)

            try:
                # Introspect the MCP server to get tool schemas with descriptions
                # This queries the MCP server's tools/list endpoint
                logger.debug(f"Introspecting MCP server at {mcp_tool.url}")
                tools = await introspection_toolset.get_tools()

                # Extract (name, description) for each tool
                for tool in tools:
                    name = tool.name
                    description = tool.description
                    if description:  # Only include tools with descriptions
                        tool_descriptions.append((name, description))
                    else:
                        logger.warning(f"Tool '{name}' from {mcp_tool.name} has no description")

            except Exception as e:
                logger.error(f"Failed to load MCP tool from {mcp_tool.url}: {e}")
                # Attempt cleanup before raising
                try:
                    await introspection_toolset.close()
                except Exception as cleanup_error:
                    logger.warning(f"Error during introspection toolset cleanup for {mcp_tool.name}: {cleanup_error}")

                raise ConnectionError(
                    f"Could not connect to MCP server '{mcp_tool.name}' at {mcp_tool.url}. "
                    f"Ensure the server is running and accessible."
                ) from e

            finally:
                # ALWAYS close the introspection toolset to clean up async task groups
                try:
                    logger.debug(f"Closing introspection connection to {mcp_tool.url}")
                    await introspection_toolset.close()
                except Exception as cleanup_error:
                    # Log but don't raise - cleanup failures shouldn't block startup
                    logger.warning(f"Error closing introspection toolset for {mcp_tool.name}: {cleanup_error}")

            # PHASE 2: Create separate runtime toolset
            # This toolset will be kept alive for the lifetime of the agent
            logger.debug(f"Creating runtime toolset for {mcp_tool.url}")
            runtime_toolset = McpToolset(connection_params=connection_params)

            # Add runtime toolset to the list (keep it alive for agent execution)
            toolsets.append(runtime_toolset)

        return toolsets, tool_descriptions
