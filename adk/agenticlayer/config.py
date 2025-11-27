import json
import logging
import os

from google.adk.agents import BaseAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools import BaseTool, agent_tool
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def get_sub_agents() -> tuple[list[BaseAgent], list[BaseTool]]:
    """
    Get sub agents from environment variable configuration.
    Format: {"agent_name": {"url": "http://agent_url", "interaction_type", "transfer|tool_call"}, ...}

    :return: A tuple of:
        - list of sub agents for transfer interaction type
        - list of agent tools for tool_call interaction type
    """

    sub_agents_config = os.environ.get("SUB_AGENTS", "{}")
    try:
        agents_map = json.loads(sub_agents_config)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in SUB_AGENTS environment variable. Using empty configuration.")
        agents_map = {}

    sub_agents: list[BaseAgent] = []
    tools: list[BaseTool] = []
    for agent_name, config in agents_map.items():
        if "url" not in config:
            print(f"Warning: Missing 'url' for agent '{agent_name}'. Skipping.")
            continue

        interaction_type = config.get("interaction_type", "tool_call")

        logging.info("Adding sub-agent: %s (%s) with URL: %s", agent_name, interaction_type, config["url"])
        agent = RemoteA2aAgent(name=agent_name, agent_card=config["url"])
        if interaction_type == "tool_call":
            tools.append(agent_tool.AgentTool(agent=agent))
        else:
            sub_agents.append(agent)

    return sub_agents, tools


def get_tools() -> list[McpToolset]:
    """
    Get tools from environment variable configuration.
    Format: {"tool_name": {"url": "http://tool_url"}, ...}

    :return: A list of McpToolset tools
    """

    tools_config = os.environ.get("AGENT_TOOLS", "{}")
    try:
        tools_map = json.loads(tools_config)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in AGENT_TOOLS environment variable. Using empty configuration.")
        tools_map = {}

    tools = []
    for name, config in tools_map.items():
        if "url" not in config:
            print(f"Warning: Missing 'url' for tool '{name}'. Skipping.")
            continue

        logging.info("Adding tool: %s with URL: %s", name, config["url"])
        tools.append(
            McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=config["url"],
                ),
            )
        )

    return tools
