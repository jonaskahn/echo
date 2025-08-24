"""Pytest configuration and shared fixtures for Echo tests."""

import asyncio
import os
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from echo.config.settings import Settings
from echo.core.orchestrator.state import AgentState
from echo.infrastructure.llm.factory import LLMModelFactory
from echo.infrastructure.plugins.sdk_manager import SDKPluginManager


@pytest.fixture
def settings() -> Settings:
    """Provide test settings with safe defaults."""
    return Settings(
        debug=True,
        default_llm_provider="openai",
        openai_api_key="test-key",
        max_agent_hops=5,
        max_tool_hops=10,
        graph_recursion_limit=25,
        finalizer_llm_provider="openai",
        finalizer_temperature=0.5,
        finalizer_max_tokens=1000,
    )


@pytest.fixture
def mock_llm_factory() -> LLMModelFactory:
    """Provide a mocked LLM factory for testing."""
    factory = MagicMock(spec=LLMModelFactory)
    mock_model = MagicMock()
    mock_model.invoke = AsyncMock(return_value=AIMessage(content="Test response"))
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    factory.create_base_model = MagicMock(return_value=mock_model)
    return factory


@pytest.fixture
def mock_plugin_manager() -> SDKPluginManager:
    """Provide a mocked plugin manager for testing."""
    manager = MagicMock(spec=SDKPluginManager)
    manager.plugin_bundles = {
        "math_agent": MagicMock(),
        "search_agent": MagicMock(),
    }
    manager.get_coordinator_tools = MagicMock(return_value=[])
    manager.get_plugin_routing_info = MagicMock(
        return_value={
            "math_agent": "Handles mathematical calculations",
            "search_agent": "Performs web searches",
        }
    )
    manager.get_available_plugins = MagicMock(return_value=["math_agent", "search_agent"])
    return manager


@pytest.fixture
def sample_agent_state() -> AgentState:
    """Provide a sample agent state for testing."""
    return AgentState(
        messages=[
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content="I'll help you calculate that."),
        ],
        current_agent="math_agent",
        agent_hops=1,
        tool_hops=2,
        last_tool_call="calculator",
        session_id="test-session-123",
        metadata={"user_preferences": {"math_level": "basic"}},
        parallel_results=None,
        routing_decision="math_calculation",
        plugin_context={"math_operation": "addition"},
    )


@pytest.fixture
def empty_agent_state() -> AgentState:
    """Provide an empty agent state for testing."""
    return AgentState(
        messages=[],
        current_agent=None,
        agent_hops=0,
        tool_hops=0,
        last_tool_call=None,
        session_id=None,
        metadata=None,
        parallel_results=None,
        routing_decision=None,
        plugin_context=None,
    )


@pytest.fixture
def mock_checkpointer():
    """Provide a mocked checkpointer for testing."""
    return MagicMock()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        SystemMessage(content="You are a helpful assistant."),
    ]


@pytest.fixture
def mock_tool_calls():
    """Provide mock tool calls for testing."""
    return [
        {
            "id": "call_123",
            "name": "calculator",
            "args": {"expression": "2 + 2"},
        }
    ]


@pytest.fixture
def mock_tool_response():
    """Provide mock tool response for testing."""
    return AIMessage(
        content="4",
        tool_call_id="call_123",
    )


@pytest.fixture
def test_env_vars():
    """Provide test environment variables."""
    return {
        "ECHO_DEBUG": "true",
        "ECHO_DEFAULT_LLM_PROVIDER": "openai",
        "ECHO_OPENAI_API_KEY": "test-key-123",
        "ECHO_MAX_AGENT_HOPS": "10",
        "ECHO_MAX_TOOL_HOPS": "20",
    }


@pytest.fixture(autouse=True)
def setup_test_env(test_env_vars):
    """Automatically set up test environment variables."""
    for key, value in test_env_vars.items():
        os.environ[key] = value
    yield
    # Clean up
    for key in test_env_vars:
        os.environ.pop(key, None)
