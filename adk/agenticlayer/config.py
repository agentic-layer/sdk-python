"""
Configuration parsing for sub-agents and tools.
Parses JSON configurations to create RemoteA2aAgents, AgentTools and McpToolsets.
"""

import json
from enum import Enum

from pydantic import AnyHttpUrl, BaseModel


class InteractionType(str, Enum):
    TOOL_CALL = "tool_call"
    TRANSFER = "transfer"


class SubAgent(BaseModel):
    name: str
    url: AnyHttpUrl
    interaction_type: InteractionType = InteractionType.TOOL_CALL


class McpTool(BaseModel):
    name: str
    url: AnyHttpUrl
    timeout: int = 30
    propagate_headers: list[str] | None = None


def parse_sub_agents(sub_agents_config: str) -> list[SubAgent]:
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

    return [
        SubAgent(
            name=agent_name,
            url=config["url"],
            interaction_type=InteractionType(config.get("interaction_type", "tool_call")),
        )
        for agent_name, config in agents_map.items()
    ]


def parse_tools(tools_config: str) -> list[McpTool]:
    """
    Get tools from JSON string.
    Format: {"tool_name": {"url": "http://tool_url", "timeout": 30}, ...}

    :return: A list of McpToolset tools
    """

    try:
        tools_map = json.loads(tools_config)
    except json.JSONDecodeError as e:
        raise ValueError("Warning: Invalid JSON in AGENT_TOOLS environment variable: " + tools_config, e)

    return [
        McpTool(
            name=name,
            url=config["url"],
            timeout=config.get("timeout", 30),
        )
        for name, config in tools_map.items()
    ]
