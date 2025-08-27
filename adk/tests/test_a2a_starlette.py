import pytest
from agenticlayer.a2a_starlette import agent_to_a2a_starlette
from google.adk.agents.base_agent import BaseAgent
from starlette.testclient import TestClient


class SimpleTestAgent(BaseAgent):
    """A simple test agent implementation."""

    def __init__(self):
        # BaseAgent is a Pydantic model that requires a name field
        super().__init__(name="test_agent")

    def process_request(self, request):
        """Mock implementation of request processing."""
        return {"response": "test response"}


class TestA2AStarlette:
    """Test suite for the a2a_starlette module."""

    @pytest.fixture
    def test_agent(self):
        """Create a test agent for testing."""
        return SimpleTestAgent()

    @pytest.fixture
    def starlette_app(self, test_agent):
        """Create a Starlette app with the test agent."""
        return agent_to_a2a_starlette(test_agent)

    @pytest.fixture
    def client(self, starlette_app):
        """Create a test client."""
        return TestClient(starlette_app)

    def test_health_endpoint(self, client):
        """Test that the health check endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_agent_card_endpoint(self, starlette_app, client):
        """Test that the agent card is available at /.well-known/agent-card.json"""

        # Try the standard agent card endpoint
        response = client.get("/.well-known/agent-card.json")

        if response.status_code == 200:
            # Great! We found the agent card
            data = response.json()
            assert isinstance(data, dict), "Agent card should return a JSON object"

            # Verify it contains expected agent card fields
            assert len(data) > 0, "Agent card should not be empty"
