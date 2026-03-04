import os

os.environ.setdefault(
    "LITELLM_PROXY_API_KEY", "test-key"
)  # Required by OpenAIChatClient when used via create_openai_client()

pytest_plugins = [
    "tests.fixtures.app_factory",
]
