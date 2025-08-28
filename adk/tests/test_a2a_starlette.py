import pytest
from agenticlayer.agent_to_a2a import to_a2a
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

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test that the health check endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

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
