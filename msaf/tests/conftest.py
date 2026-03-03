import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")  # Required by agent-framework to avoid validation errors

pytest_plugins = [
    "tests.fixtures.app_factory",
]
