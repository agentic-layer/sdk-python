"""Tests for external token passing to MCP tools via ADK session."""

from agenticlayer.agent import _get_mcp_headers_from_session
from agenticlayer.constants import EXTERNAL_TOKEN_SESSION_KEY
from google.adk.sessions.session import Session


def test_header_provider_retrieves_token_from_session() -> None:
    """Test that the header provider function can retrieve token from session state."""
    # Given: A session with an external token stored
    test_token = "test-api-token-xyz"  # nosec B105
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user",
        state={EXTERNAL_TOKEN_SESSION_KEY: test_token},
        events=[],
        last_update_time=0.0,
    )

    # Create a mock readonly context
    class MockReadonlyContext:
        def __init__(self, session: Session) -> None:
            self.session = session

    readonly_context = MockReadonlyContext(session)

    # When: Calling the header provider function
    headers = _get_mcp_headers_from_session(readonly_context)  # type: ignore[arg-type]

    # Then: The headers should include the X-External-Token
    assert headers == {"X-External-Token": test_token}


def test_header_provider_returns_empty_when_no_token() -> None:
    """Test that the header provider returns empty dict when no token is present."""
    # Given: A session without an external token
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user",
        state={},  # No token
        events=[],
        last_update_time=0.0,
    )

    # Create a mock readonly context
    class MockReadonlyContext:
        def __init__(self, session: Session) -> None:
            self.session = session

    readonly_context = MockReadonlyContext(session)

    # When: Calling the header provider function
    headers = _get_mcp_headers_from_session(readonly_context)  # type: ignore[arg-type]

    # Then: The headers should be empty
    assert headers == {}


def test_header_provider_handles_none_context() -> None:
    """Test that the header provider safely handles None context."""
    # When: Calling the header provider with None
    headers = _get_mcp_headers_from_session(None)  # type: ignore[arg-type]

    # Then: The headers should be empty (no exception)
    assert headers == {}

