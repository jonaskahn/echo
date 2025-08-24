"""Basic tests to verify the test setup is working correctly."""

from unittest.mock import MagicMock

import pytest

from echo.config.settings import Settings
from echo.core.orchestrator.state import AgentState, _inc_agent_hops, _inc_tool_hops


class TestBasicSetup:
    """Basic tests to verify test setup."""

    def test_settings_import(self):
        """Test that Settings can be imported and instantiated."""
        settings = Settings()
        assert settings is not None
        assert hasattr(settings, "app_name")
        assert settings.app_name == "Echo 🤖 Multi-agents AI Framework"

    def test_agent_state_import(self):
        """Test that AgentState can be imported and used."""
        state = AgentState(
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
        assert state is not None
        assert state["agent_hops"] == 0
        assert state["tool_hops"] == 0

    def test_state_utility_functions(self):
        """Test state utility functions."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=5,
            tool_hops=3,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        assert _inc_agent_hops(state) == 6
        assert _inc_tool_hops(state) == 4

    def test_mock_functionality(self):
        """Test that mocking works correctly."""
        mock_obj = MagicMock()
        mock_obj.some_method.return_value = "test_value"

        result = mock_obj.some_method()
        assert result == "test_value"
        mock_obj.some_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_test_support(self):
        """Test that async tests are supported."""

        async def async_function():
            return "async_result"

        result = await async_function()
        assert result == "async_result"

    def test_pytest_fixtures(self, settings, sample_agent_state):
        """Test that pytest fixtures work correctly."""
        assert settings is not None
        assert hasattr(settings, "debug")
        assert sample_agent_state is not None
        assert "messages" in sample_agent_state

    def test_environment_variables(self, test_env_vars):
        """Test that environment variable fixtures work."""
        assert test_env_vars is not None
        assert "ECHO_DEBUG" in test_env_vars
        assert test_env_vars["ECHO_DEBUG"] == "true"

    def test_coverage_works(self):
        """Test that coverage tracking works."""
        # This test should be covered by coverage reporting
        value = 42
        assert value == 42

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True

    @pytest.mark.asyncio
    async def test_async_marker(self):
        """Test that async marker works."""
        assert True

    def test_assertions_work(self):
        """Test that assertions work correctly."""
        assert 1 + 1 == 2
        assert "hello" in "hello world"
        assert len([1, 2, 3]) == 3

    def test_exception_handling(self):
        """Test that exception testing works."""
        with pytest.raises(ValueError):
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            raise ValueError("Test exception")

    def test_parametrized_test(self, request):
        """Test parametrized testing."""
        test_data = [
            (1, 2, 3),
            (5, 5, 10),
            (0, 0, 0),
        ]

        for a, b, expected in test_data:
            assert a + b == expected

    def test_fixture_scope(self, settings):
        """Test that fixtures have correct scope."""
        # This should get a fresh settings instance
        assert settings.debug is True  # From conftest.py fixture

    def test_import_paths(self):
        """Test that all necessary modules can be imported."""
        # Test core imports
        from echo.config.settings import Settings
        from echo.core.orchestrator.coordinator import MultiAgentOrchestrator
        from echo.core.orchestrator.state import AgentState

        # Test infrastructure imports
        from echo.infrastructure.llm.factory import LLMModelFactory

        # If we get here, imports work
        assert True
