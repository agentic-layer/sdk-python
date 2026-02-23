"""Integration test to verify MCP tool descriptions are included in agent instructions.

This test reproduces the issue described in the GitHub issue about MCP tool descriptions
not being contained in agent instructions.

The problem: When MCP tools are added to an agent, their descriptions should be
included in the agent's instructions so the LLM knows what each tool does. However,
currently the tool descriptions are not being added to the instructions, unlike
sub-agents which do have their descriptions added.

Expected behavior:
- Agent instructions should contain tool names and descriptions
- Format should be similar to sub-agent descriptions

Actual behavior:
- Agent instructions do not contain MCP tool descriptions
- Tools are available but LLM doesn't know what they do without explicit prompts
"""

from typing import Any

import pytest
import respx
from agenticlayer.config import McpTool
from asgi_lifespan import LifespanManager
from fastmcp import FastMCP
from pydantic import AnyHttpUrl

from tests.utils.helpers import create_asgi_request_handler


class TestMcpToolDescriptions:
    """Tests for MCP tool descriptions in agent instructions."""

    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_missing_from_agent_instructions(
        self,
        app_factory: Any,
        agent_factory: Any,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that reproduces the issue: MCP tool descriptions are not in agent instructions.

        This test demonstrates that when MCP tools are added to an agent, their
        descriptions are NOT included in the agent's instructions, making it harder
        for the LLM to know what tools are available and how to use them.

        This is the ACTUAL behavior (the bug we're documenting).
        """

        # Given: MCP server with tools that have detailed descriptions
        mcp = FastMCP("CustomerService")

        @mcp.tool()
        def get_customer_crm_data(customer_id: str) -> dict[str, Any]:
            """Retrieve customer CRM data including contact info and purchase history."""
            return {"customer_id": customer_id, "name": "John Doe"}

        @mcp.tool()
        def get_all_customer_data() -> list[dict[str, str]]:
            """Get a list of all customers in the system."""
            return [{"id": "1", "name": "Customer 1"}]

        @mcp.tool()
        def send_message(customer_id: str, subject: str, body: str) -> str:
            """Send a personalized message to a customer via email."""
            return f"Message sent to {customer_id}"

        @mcp.tool()
        def get_insurance_products() -> list[dict[str, str]]:
            """Get all available insurance products."""
            return [{"id": "prod1", "name": "Health Insurance"}]

        @mcp.tool()
        def get_product_details(product_id: str) -> dict[str, str]:
            """Get detailed information about a specific insurance product."""
            return {"id": product_id, "name": "Product Details"}

        @mcp.tool()
        def get_products_by_segment(segment: str) -> list[dict[str, str]]:
            """Get insurance products filtered by customer segment."""
            return [{"id": "prod1", "segment": segment}]

        # Start MCP server
        mcp_server_url = "http://test-crm-mcp.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Route MCP requests through ASGI transport
            handler = create_asgi_request_handler(mcp_manager.app, mcp_server_url)
            respx_mock.route(host="test-crm-mcp.local").mock(side_effect=handler)

            # When: Create agent with MCP tools
            test_agent = agent_factory("customer_service_agent")
            # Set a simple instruction
            test_agent.instruction = "You are a customer service agent."

            tools = [McpTool(name="crm_tools", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            # Create the app which loads the MCP tools
            async with app_factory(test_agent, tools=tools):
                # Access the agent to check its instructions
                # The agent is modified during app startup in agent_factory.load_agent()
                # We need to get the configured agent instance

                # ACTUAL BEHAVIOR (the bug):
                # The agent instructions do NOT contain the MCP tool descriptions
                agent_instructions = test_agent.instruction

                print("\n" + "=" * 80)
                print("AGENT INSTRUCTIONS:")
                print("=" * 80)
                print(agent_instructions)
                print("=" * 80)

                # These assertions document the CURRENT (buggy) behavior:
                # The tool descriptions are NOT present in the instructions

                # Check that the basic instruction is still there
                assert "customer service agent" in agent_instructions.lower()

                # DOCUMENT THE BUG: Tool descriptions are missing
                # If this were working correctly, we would expect to see something like:
                # "Following tools are available:
                #  - 'get_customer_crm_data': Retrieve customer CRM data...
                #  - 'send_message': Send a personalized message..."
                # But these are NOT present currently.

                assert "get_customer_crm_data" not in agent_instructions, (
                    "Tool names should NOT be in instructions yet (this documents the bug). "
                    "If this assertion fails, the issue might be fixed!"
                )
                assert "Retrieve customer CRM data" not in agent_instructions, (
                    "Tool descriptions should NOT be in instructions yet (this documents the bug). "
                    "If this assertion fails, the issue might be fixed!"
                )
                assert "send_message" not in agent_instructions, (
                    "Tool names should NOT be in instructions yet (this documents the bug)"
                )

                # When the issue is fixed, the agent instructions should include
                # tool descriptions similar to how sub-agent descriptions are added
                # in agent.py lines 74-80

    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_expected_behavior(
        self,
        app_factory: Any,
        agent_factory: Any,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that documents the EXPECTED behavior: MCP tool descriptions should be in instructions.

        This test is currently SKIPPED. It documents what the correct
        behavior should be once the issue is fixed.

        When the bug is fixed, this test should pass, and the agent instructions
        should contain the MCP tool descriptions, similar to how sub-agent
        descriptions are currently handled.
        """

        # Mark this test as expected to fail - remove this when the issue is fixed
        pytest.skip(
            "This test documents expected behavior and will fail until the issue is fixed. "
            "When MCP tool descriptions are properly added to agent instructions, "
            "remove the pytest.skip() and this test should pass."
        )

        # Given: MCP server with tools
        mcp = FastMCP("Calculator")

        @mcp.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        @mcp.tool()
        def multiply(x: float, y: float) -> float:
            """Multiply two numbers."""
            return x * y

        # Start MCP server
        mcp_server_url = "http://test-calc-mcp.local"
        mcp_app = mcp.http_app(path="/mcp")

        async with LifespanManager(mcp_app) as mcp_manager:
            # Route MCP requests through ASGI transport
            handler = create_asgi_request_handler(mcp_manager.app, mcp_server_url)
            respx_mock.route(host="test-calc-mcp.local").mock(side_effect=handler)

            # When: Create agent with MCP tools
            test_agent = agent_factory("calculator_agent")
            test_agent.instruction = "You are a calculator assistant."

            tools = [McpTool(name="calculator", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            async with app_factory(test_agent, tools=tools):
                agent_instructions = test_agent.instruction

                print("\n" + "=" * 80)
                print("EXPECTED AGENT INSTRUCTIONS (when bug is fixed):")
                print("=" * 80)
                print(agent_instructions)
                print("=" * 80)

                # These assertions define the expected behavior:
                assert "add" in agent_instructions, "Tool name 'add' should be in instructions"
                assert "multiply" in agent_instructions, "Tool name 'multiply' should be in instructions"

                # The full descriptions should be present
                assert "Add two numbers together" in agent_instructions, (
                    "Full tool description should be in instructions"
                )
                assert "Multiply two numbers" in agent_instructions, "Full tool description should be in instructions"

                # The format should be similar to sub-agent descriptions
                # (see agent.py lines 74-80 for reference)
                assert (
                    "tools are available" in agent_instructions.lower() or "functions" in agent_instructions.lower()
                ), "Instructions should mention available tools/functions"
