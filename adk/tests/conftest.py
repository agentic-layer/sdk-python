import os

os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
os.environ.setdefault("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "true")

pytest_plugins = [
    "tests.fixtures.mock_llm",
    "tests.fixtures.app_factory",
]
