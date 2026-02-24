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

import logging
from typing import Any

import pytest
import respx
from agenticlayer.config import McpTool
from asgi_lifespan import LifespanManager
from fastmcp import FastMCP
from pydantic import AnyHttpUrl
from starlette.testclient import TestClient

from tests.fixtures.mock_llm import LLMMockController
from tests.utils.helpers import create_asgi_request_handler, create_send_message_request


class TestMcpToolDescriptions:
    """Tests for MCP tool descriptions in agent instructions."""

    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_missing_from_agent_instructions(
        self,
        app_factory: Any,
        agent_factory: Any,
        llm_controller: LLMMockController,
        respx_mock: respx.MockRouter,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that reproduces the issue: MCP tool descriptions are not in the logged functions.

        This test demonstrates that when MCP tools are added to an agent, their
        descriptions are NOT included in the function logs that the ADK generates.

        The ADK logs functions in this format:
        Functions:
        get_customer_crm_data: {'customer_id': {'title': 'Customer Id', 'type': <Type.STRING: 'STRING'>}} -> None

        Notice that the description ("Retrieve customer CRM data...") is MISSING.
        This is the bug reproduced from the GitHub issue.
        """

        # Enable DEBUG logging to capture the function logs
        # Need to set it at the root level to capture all ADK logs
        caplog.set_level(logging.DEBUG)

        # Configure LLM to respond with a simple message
        llm_controller.respond_with_message("", "I can help you with customer service tasks.")

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
            test_agent.instruction = "You are a customer service agent."

            tools = [McpTool(name="crm_tools", url=AnyHttpUrl(f"{mcp_server_url}/mcp"), timeout=30)]

            # Create the app and send a message to trigger LLM call with logging
            async with app_factory(test_agent, tools=tools) as app:
                client = TestClient(app)
                user_message = "What can you help me with?"
                response = client.post("", json=create_send_message_request(user_message))

            # Then: Verify we got a successful response
            assert response.status_code == 200

            # Find the "Functions:" section in the logs
            functions_log = None
            for record in caplog.records:
                if "Functions:" in record.message:
                    functions_log = record.message
                    break

            assert functions_log is not None, "Should have logged functions"

            print("\n" + "=" * 80)
            print("LOGGED FUNCTIONS (from ADK debug logs):")
            print("=" * 80)
            print(functions_log)
            print("=" * 80)

            # DOCUMENT THE BUG: The logged functions contain the exact format from the issue
            # Functions should be logged like:
            # get_customer_crm_data: {'customer_id': {'title': 'Customer Id', 'type': <Type.STRING: 'STRING'>}} -> None

            # Verify the bug - function names are present
            assert "get_customer_crm_data" in functions_log, "Function name should be in log"
            assert "send_message" in functions_log, "Function name should be in log"
            assert "get_insurance_products" in functions_log, "Function name should be in log"

            # BUG: The descriptions are MISSING from the log
            # The log shows parameter types but not the human-readable descriptions
            assert "Retrieve customer CRM data including contact info and purchase history" not in functions_log, (
                "Tool description should NOT be in log (this documents the bug). "
                "If this assertion fails, the issue might be fixed!"
            )
            assert "Get a list of all customers in the system" not in functions_log, (
                "Tool description should NOT be in log (this documents the bug)"
            )
            assert "Send a personalized message to a customer via email" not in functions_log, (
                "Tool description should NOT be in log (this documents the bug)"
            )

            # The log shows parameter information (like 'customer_id') but not descriptions
            assert "customer_id" in functions_log, "Parameter names should be in log"

            print("\n" + "=" * 80)
            print("BUG REPRODUCED: Functions logged without their descriptions!")
            print("The log shows function signatures like:")
            print("  get_customer_crm_data: {'customer_id': ...} -> None")
            print("But missing the description:")
            print("  'Retrieve customer CRM data including contact info and purchase history.'")
            print("=" * 80)

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
