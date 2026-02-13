"""
Token context management for passing API tokens to MCP tools.
Uses contextvars to store tokens per-request in a way that's secure and
inaccessible to the agent but available for MCP tool authentication.
"""

from contextvars import ContextVar

# Context variable to store the external API token for the current request
# This is stored separately from the session to ensure agents cannot access it
_external_token: ContextVar[str | None] = ContextVar("external_token", default=None)


def set_external_token(token: str | None) -> None:
    """Store the external API token for the current request context.

    Args:
        token: The X-External-Token header value, or None to clear it
    """
    _external_token.set(token)


def get_external_token() -> str | None:
    """Retrieve the external API token for the current request context.

    Returns:
        The token if set, otherwise None
    """
    return _external_token.get()


def get_mcp_headers() -> dict[str, str]:
    """Get headers to be passed to MCP tool calls.

    This function is intended to be used as a header_provider for McpToolset.
    It retrieves the external token from the context and returns it in a format
    suitable for HTTP headers.

    Returns:
        A dictionary of headers to include in MCP tool requests.
        If a token is set, includes the X-External-Token header.
    """
    token = get_external_token()
    if token:
        return {"X-External-Token": token}
    return {}
