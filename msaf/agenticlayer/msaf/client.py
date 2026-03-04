"""Factory for creating an OpenAI-compatible chat client from environment variables."""

import os

from agent_framework.openai import OpenAIChatClient


def create_openai_client(
    model_id: str | None = None,
) -> OpenAIChatClient:
    """Create an OpenAIChatClient configured from environment variables.

    Reads the following environment variables to configure an OpenAI-compatible gateway:

    - ``LITELLM_PROXY_API_BASE``: Base URL of the gateway
      (e.g. ``http://litellm-proxy:4000``). Maps to the ``base_url`` parameter.
      Falls back to the ``OPENAI_BASE_URL`` env var or the default OpenAI endpoint
      when not set.
    - ``LITELLM_PROXY_API_KEY``: API key for the gateway. Maps to the ``api_key``
      parameter. Falls back to the ``OPENAI_API_KEY`` env var when not set.
    - ``OPENAI_CHAT_MODEL_ID``: Model name to use (e.g. ``gpt-4o``). Handled
      natively by :class:`~agent_framework.openai.OpenAIChatClient`; takes effect
      even when ``model_id`` is not passed to this function.

    Args:
        model_id: Optional model identifier override. When ``None``, the
            ``OPENAI_CHAT_MODEL_ID`` environment variable is used automatically
            by :class:`~agent_framework.openai.OpenAIChatClient`.

    Returns:
        A configured :class:`~agent_framework.openai.OpenAIChatClient` instance.

    Example:
        .. code-block:: python

            import os
            from agent_framework import Agent
            from agenticlayer.msaf.client import create_openai_client
            from agenticlayer.msaf.agent_to_a2a import to_a2a

            os.environ["LITELLM_PROXY_API_BASE"] = "http://litellm-proxy:4000"
            os.environ["LITELLM_PROXY_API_KEY"] = "sk-my-key"
            os.environ["OPENAI_CHAT_MODEL_ID"] = "gpt-4o"

            agent = Agent(
                client=create_openai_client(),
                instructions="You are a helpful assistant.",
            )
            app = to_a2a(agent, name="MyAgent", rpc_url="http://localhost:8000/")
    """
    base_url = os.environ.get("LITELLM_PROXY_API_BASE")
    api_key = os.environ.get("LITELLM_PROXY_API_KEY")

    return OpenAIChatClient(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
    )
