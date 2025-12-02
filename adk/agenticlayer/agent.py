"""
Convert Sub Agents and Tools into RemoteA2aAgents, AgentTools and McpToolsets.
"""

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from agenticlayer.config import InteractionType, McpTool, SubAgent


def load_agent(agent: LlmAgent, sub_agents: list[SubAgent], tools: list[McpTool]) -> LlmAgent:
    agents, agent_tools = load_sub_agents(sub_agents)
    mcp_tools = load_tools(tools)
    all_tools: list[ToolUnion] = agent_tools + mcp_tools

    agent.sub_agents += agents
    agent.tools += all_tools
    return agent


def load_sub_agents(sub_agents: list[SubAgent]) -> tuple[list[BaseAgent], list[ToolUnion]]:
    """
    Convert Sub Agents into RemoteA2aAgents and AgentTools.

    :return: A tuple of:
        - list of sub agents for transfer interaction type
        - list of agent tools for tool_call interaction type
    """

    agents: list[BaseAgent] = []
    tools: list[ToolUnion] = []
    for sub_agent in sub_agents:
        agent_card = str(sub_agent.url)
        agent = RemoteA2aAgent(name=sub_agent.name, agent_card=agent_card)
        if sub_agent.interaction_type == InteractionType.TOOL_CALL:
            tools.append(AgentTool(agent=agent))
        else:
            agents.append(agent)

    return agents, tools


def load_tools(mcp_tools: list[McpTool]) -> list[ToolUnion]:
    """
    Convert Tools into McpToolsets.

    :return: A list of McpToolset tools
    """

    tools: list[ToolUnion] = []
    for tool in mcp_tools:
        tools.append(
            McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=str(tool.url),
                    timeout=tool.timeout,
                ),
            )
        )

    return tools
