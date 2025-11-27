import json
import logging
import os

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def get_sub_agents():
    """Create sub agents from environment variable configuration."""
    sub_agents_config = os.environ.get("SUB_AGENTS", "{}")
    try:
        agents_map = json.loads(sub_agents_config)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in SUB_AGENTS environment variable. Using empty configuration.")
        agents_map = {}

    sub_agents = []
    for agent_name, config in agents_map.items():
        if "url" not in config:
            print(f"Warning: Missing 'url' for agent '{agent_name}'. Skipping.")
            continue

        logging.info("Adding sub-agent: %s with URL: %s", agent_name, config["url"])
        sub_agents.append(
            RemoteA2aAgent(
                name=agent_name,
                agent_card=config["url"],
            )
        )

    return sub_agents


def get_tools():
    """Get tools from environment variable configuration."""
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
