"""Global fixtures.

This module provides shared pytest fixtures for testing:
- app_factory: Factory for creating Starlette apps with agents
- llm_controller: Controller for configuring LLM mock responses
- llm_client: Mock LLM client
- agent_factory: Factory for creating test agents with mock LLM

All fixtures are automatically available in tests via pytest plugin registration.
"""
