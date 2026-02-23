"""
Monkey-patch for Google ADK's MCPSessionManager to fix session invalidation on server restart.

This module patches both the MCPSessionManager.create_session method and the retry_on_errors
decorator to properly handle the case where an MCP server restarts and loses session state.

Root Cause:
-----------
When an MCP server restarts:
1. The server loses all session state
2. Client-side session streams remain open (not disconnected)
3. Cached session appears valid because _is_session_disconnected() only checks local streams
4. Server returns 404 for requests with old session IDs
5. Tool calls time out waiting for responses
6. On retry, the same bad cached session is reused

The Fix:
--------
We patch the retry_on_errors decorator to:
1. Detect when an error occurs during MCP operations
2. Force-close the streams of the cached session
3. This makes _is_session_disconnected() return True
4. On retry, create_session() sees the session is disconnected and creates a fresh one

This is a temporary workaround until the fix is merged upstream in Google ADK.

Issue: https://github.com/agentic-layer/sdk-python/issues/XXX
"""

import functools
import logging
from typing import Any, Callable

from google.adk.tools.mcp_tool import mcp_toolset
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

logger = logging.getLogger(__name__)

# Store the original decorator
_original_retry_on_errors = None


def _patched_retry_on_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Patched version of retry_on_errors that invalidates sessions on error.

    This wraps the original decorator and adds logic to close cached session streams
    when an error occurs, ensuring the session is marked as disconnected for retry.
    """
    # First, apply the original decorator if it exists
    if _original_retry_on_errors:
        func = _original_retry_on_errors(func)

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # When an error occurs, try to invalidate any cached MCP sessions
            # by closing their streams, so retry gets a fresh session
            if hasattr(self, "_mcp_session_manager"):
                session_manager: MCPSessionManager = self._mcp_session_manager
                logger.info(
                    f"[PATCH] Error in MCP operation ({func.__name__}), invalidating cached sessions: {type(e).__name__}"
                )

                # Access the session cache and close all sessions' streams
                if hasattr(session_manager, "_sessions"):
                    try:
                        # Use the lock to safely access sessions
                        num_sessions = len(session_manager._sessions)  # type: ignore
                        logger.debug(f"[PATCH] Found {num_sessions} cached sessions to invalidate")

                        # We can't use the lock here because we're already in an async context
                        # and the lock might be held. Instead, just try to close streams.
                        for session_key, (session, _, _) in list(session_manager._sessions.items()):  # type: ignore
                            try:
                                logger.debug(f"[PATCH] Invalidating session: {session_key}")

                                # Force-close the read stream
                                if hasattr(session, "_read_stream"):
                                    stream = session._read_stream
                                    logger.debug(
                                        f"[PATCH] Read stream type: {type(stream).__name__}, has aclose: {hasattr(stream, 'aclose')}"
                                    )
                                    if hasattr(stream, "aclose"):
                                        await stream.aclose()
                                        logger.debug("[PATCH] Closed read stream via aclose()")
                                    elif hasattr(stream, "close"):
                                        stream.close()
                                        logger.debug("[PATCH] Closed read stream via close()")
                                else:
                                    logger.debug("[PATCH] Session has no _read_stream")

                                # Force-close the write stream
                                if hasattr(session, "_write_stream"):
                                    stream = session._write_stream
                                    logger.debug(
                                        f"[PATCH] Write stream type: {type(stream).__name__}, has aclose: {hasattr(stream, 'aclose')}"
                                    )
                                    if hasattr(stream, "aclose"):
                                        await stream._write_stream.aclose()
                                        logger.debug("[PATCH] Closed write stream via aclose()")
                                    elif hasattr(stream, "close"):
                                        stream.close()
                                        logger.debug("[PATCH] Closed write stream via close()")
                                else:
                                    logger.debug("[PATCH] Session has no _write_stream")

                                logger.info(f"[PATCH] Successfully invalidated session {session_key}")
                            except Exception as close_err:
                                logger.warning(f"[PATCH] Could not close streams for {session_key}: {close_err}")
                    except Exception as invalidate_err:
                        logger.error(f"[PATCH] Error invalidating sessions: {invalidate_err}", exc_info=True)
                else:
                    logger.debug("[PATCH] Session manager has no _sessions attribute")
            else:
                logger.debug(f"[PATCH] Object {type(self).__name__} has no _mcp_session_manager attribute")

            # Re-raise the exception so the original decorator can handle retry
            raise

    return wrapper


def apply_mcp_session_patch() -> None:
    """Apply the monkey-patch to the retry_on_errors decorator.

    This should be called once during application initialization before
    any MCP tools are created.
    """
    global _original_retry_on_errors

    if _original_retry_on_errors is None:
        logger.info("Applying MCP session manager patch for server restart handling")

        # Store the original decorator
        from google.adk.tools.mcp_tool import mcp_session_manager
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

        _original_retry_on_errors = mcp_session_manager.retry_on_errors

        # Replace the decorator in the module
        mcp_session_manager.retry_on_errors = _patched_retry_on_errors

        # Re-decorate the methods in McpToolset that use @retry_on_errors
        # Find all methods that were decorated and re-decorate them
        for attr_name in dir(McpToolset):
            if not attr_name.startswith("_"):
                attr = getattr(McpToolset, attr_name)
                if callable(attr) and hasattr(attr, "__wrapped__"):
                    # This is likely a decorated method
                    # Re-decorate it with our patched decorator
                    original_func = attr.__wrapped__
                    setattr(McpToolset, attr_name, _patched_retry_on_errors(original_func))
                    logger.debug(f"Re-decorated McpToolset.{attr_name}")

        logger.info("MCP session manager patch applied successfully")
    else:
        logger.warning("MCP session manager patch already applied")


def remove_mcp_session_patch() -> None:
    """Remove the monkey-patch and restore original behavior.

    This is primarily for testing purposes.
    """
    global _original_retry_on_errors

    if _original_retry_on_errors is not None:
        logger.info("Removing MCP session manager patch")

        from google.adk.tools.mcp_tool import mcp_session_manager

        mcp_session_manager.retry_on_errors = _original_retry_on_errors

        if hasattr(mcp_toolset, "retry_on_errors"):
            mcp_toolset.retry_on_errors = _original_retry_on_errors

        _original_retry_on_errors = None
        logger.info("MCP session manager patch removed")
