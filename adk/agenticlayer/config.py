"""
Configuration parsing for sub-agents and tools.
Parses JSON configurations to create RemoteA2aAgents, AgentTools and McpToolsets.
"""

import json
import logging

from google.adk.agents import BaseAgent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def parse_sub_agents(sub_agents_config: str) -> tuple[list[BaseAgent], list[ToolUnion]]:
    """
    Get sub agents from JSON string.
    Format: {"agent_name": {"url": "http://agent_url", "interaction_type", "transfer|tool_call"}, ...}

    :return: A tuple of:
        - list of sub agents for transfer interaction type
        - list of agent tools for tool_call interaction type
    """

    try:
        agents_map = json.loads(sub_agents_config)
    except json.JSONDecodeError as e:
        raise ValueError("Warning: Invalid JSON in SUB_AGENTS environment variable: " + sub_agents_config, e)

    sub_agents: list[BaseAgent] = []
    tools: list[ToolUnion] = []
    for agent_name, config in agents_map.items():
        if "url" not in config:
            raise ValueError(f"Missing 'url' for agent '{agent_name}': " + str(config))

        interaction_type = config.get("interaction_type", "tool_call")

        logging.info("Adding sub-agent: %s (%s) with URL: %s", agent_name, interaction_type, config["url"])
        agent = RemoteA2aAgent(name=agent_name, agent_card=config["url"])
        if interaction_type == "tool_call":
            tools.append(AgentTool(agent=agent))
        else:
            sub_agents.append(agent)

    return sub_agents, tools


def parse_tools(tools_config: str) -> list[ToolUnion]:
    """
    Get tools from JSON string.
    Format: {"tool_name": {"url": "http://tool_url"}, ...}

    :return: A list of McpToolset tools
    """

    try:
        tools_map = json.loads(tools_config)
    except json.JSONDecodeError as e:
        raise ValueError("Warning: Invalid JSON in AGENT_TOOLS environment variable: " + tools_config, e)

    tools: list[ToolUnion] = []
    for name, config in tools_map.items():
        if "url" not in config:
            raise ValueError(f"Missing 'url' for tool '{name}': " + str(config))

        logging.info("Adding tool: %s with URL: %s", name, config["url"])
        tools.append(
            McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=config["url"],
                ),
            )
        )

    return tools
