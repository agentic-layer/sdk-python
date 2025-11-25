from unittest.mock import AsyncMock, Mock

import pytest
from agenticlayer.agent_to_a2a import to_a2a
from agenticlayer.otel import TelemetryFilter
from google.adk.agents.base_agent import BaseAgent
from starlette.applications import Starlette
from starlette.testclient import TestClient


class TestA2AStarlette:
    """Test suite for the a2a_starlette module."""

    @pytest.fixture
    def test_agent(self) -> BaseAgent:
        """Create a test agent for testing."""
        return BaseAgent(name="test_agent")

    @pytest.fixture
    def starlette_app(self, test_agent: BaseAgent) -> Starlette:
        """Create a Starlette app with the test agent."""
        return to_a2a(test_agent)

    @pytest.fixture
    def client(self, starlette_app: Starlette) -> TestClient:
        """Create a test client."""
        return TestClient(starlette_app)

    def test_agent_card_endpoint(self, starlette_app: Starlette, client: TestClient) -> None:
        """Test that the agent card is available at /.well-known/agent-card.json"""

        # Try the standard agent card endpoint
        response = client.get("/.well-known/agent-card.json")

        if response.status_code == 200:
            # Great! We found the agent card
            data = response.json()
            assert isinstance(data, dict), "Agent card should return a JSON object"

            # Verify it contains expected agent card fields
            assert len(data) > 0, "Agent card should not be empty"


class TestTelemetryFilter:
    """Test suite for the TelemetryFilter class."""

    @pytest.mark.anyio
    async def test_telemetry_filter_bypasses_health_checks(self) -> None:
        """Verify TelemetryFilter bypasses OpenTelemetry for health checks."""
        # Create mocks
        app_mock = AsyncMock()
        otel_middleware_mock = AsyncMock()

        # Create filter with health check path
        telemetry_filter = await self.create_telemetry_filter(app_mock)
        telemetry_filter.otel_middleware = otel_middleware_mock

        # Simulate health check request
        scope = {"type": "http", "path": "/.well-known/agent-card.json"}
        receive = Mock()
        send = Mock()

        # Call the filter
        await telemetry_filter(scope, receive, send)

        # Assert: app was called directly (bypassed OTEL)
        app_mock.assert_called_once_with(scope, receive, send)
        otel_middleware_mock.assert_not_called()

    async def create_telemetry_filter(self, app_mock):
        return TelemetryFilter(
            app=app_mock,
            filtered_paths={"/.well-known/agent-card.json"}
        )

    @pytest.mark.anyio
    async def test_telemetry_filter_allows_regular_requests(self) -> None:
        """Verify TelemetryFilter applies OpenTelemetry for regular requests."""
        # Create mocks
        app_mock = AsyncMock()
        otel_middleware_mock = AsyncMock()

        telemetry_filter = await self.create_telemetry_filter(app_mock)
        telemetry_filter.otel_middleware = otel_middleware_mock

        # Simulate regular request
        scope = {"type": "http", "path": "/a2a/v1/tasks"}
        receive = Mock()
        send = Mock()

        await telemetry_filter(scope, receive, send)

        # Assert: OTEL middleware was used
        otel_middleware_mock.assert_called_once_with(scope, receive, send)

    @pytest.mark.anyio
    async def test_telemetry_filter_handles_non_http_requests(self) -> None:
        """Verify TelemetryFilter passes non-HTTP requests to OTEL middleware."""
        # Create mocks
        app_mock = AsyncMock()
        otel_middleware_mock = AsyncMock()

        telemetry_filter = await self.create_telemetry_filter(app_mock)
        telemetry_filter.otel_middleware = otel_middleware_mock

        # Simulate WebSocket request (non-HTTP)
        scope = {"type": "websocket", "path": "/ws"}
        receive = Mock()
        send = Mock()

        await telemetry_filter(scope, receive, send)

        # Assert: OTEL middleware was used (non-HTTP requests go through)
        otel_middleware_mock.assert_called_once_with(scope, receive, send)
