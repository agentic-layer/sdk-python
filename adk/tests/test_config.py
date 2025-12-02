"""Unit tests for config parsing functions."""

import pytest
from agenticlayer.config import InteractionType, parse_sub_agents, parse_tools


class TestParseSubAgents:
    def test_parse_sub_agents_with_different_interaction_types(self) -> None:
        """Test parsing multiple sub-agents with both interaction types."""
        # Given: JSON config with both transfer and tool_call types
        config = """{
            "sub_agent_1": {
                "url": "http://sub-agent-1.local/.well-known/agent-card.json",
                "interaction_type": "transfer"
            },
            "sub_agent_2": {
                "url": "http://sub-agent-2.local/.well-known/agent-card.json",
                "interaction_type": "tool_call"
            }
        }"""

        # When: Parsing the config
        sub_agents = parse_sub_agents(config)

        # Then: Both sub-agents are parsed correctly
        assert len(sub_agents) == 2

        transfer_agent = next(a for a in sub_agents if a.name == "sub_agent_1")
        assert str(transfer_agent.url) == "http://sub-agent-1.local/.well-known/agent-card.json"
        assert transfer_agent.interaction_type == InteractionType.TRANSFER

        tool_call_agent = next(a for a in sub_agents if a.name == "sub_agent_2")
        assert str(tool_call_agent.url) == "http://sub-agent-2.local/.well-known/agent-card.json"
        assert tool_call_agent.interaction_type == InteractionType.TOOL_CALL

    def test_parse_sub_agents_empty_config(self) -> None:
        """Test parsing empty sub-agents config."""
        assert parse_sub_agents("{}") == []

    def test_parse_sub_agents_invalid_json(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON in SUB_AGENTS"):
            parse_sub_agents("not valid json")


class TestParseTools:
    def test_parse_tools_with_default_and_custom_timeout(self) -> None:
        """Test parsing multiple tools with default and custom timeouts."""
        # Given: JSON config with multiple tools
        config = """{
            "tool_1": {
                "url": "http://tool-1.local/mcp"
            },
            "tool_2": {
                "url": "http://tool-2.local/mcp",
                "timeout": 60
            }
        }"""

        # When: Parsing the config
        tools = parse_tools(config)

        # Then: Both tools are parsed correctly
        assert len(tools) == 2

        tool_1 = next(t for t in tools if t.name == "tool_1")
        assert str(tool_1.url) == "http://tool-1.local/mcp"
        assert tool_1.timeout == 30  # Default timeout

        tool_2 = next(t for t in tools if t.name == "tool_2")
        assert str(tool_2.url) == "http://tool-2.local/mcp"
        assert tool_2.timeout == 60  # Custom timeout

    def test_parse_tools_empty_config(self) -> None:
        """Test parsing empty tools config."""
        assert parse_tools("{}") == []

    def test_parse_tools_invalid_json(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON in AGENT_TOOLS"):
            parse_tools("invalid json string")
