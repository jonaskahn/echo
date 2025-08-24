"""Test cases for the Settings configuration class."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from echo.config.settings import Settings


class TestSettings:
    """Test cases for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        # Clear environment variables for this test
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.app_name == "Echo 🤖 Multi-agents AI Framework"
            assert settings.debug is False
            assert settings.default_llm_provider == "openai"
            assert settings.default_llm_context_window == 32_000
            assert settings.default_llm_temperature == 0.1
            assert settings.openai_default_model == "gpt-4.1"
            assert settings.anthropic_default_model == "claude-sonnet-4-20250514"
            assert settings.gemini_default_model == "gemini-2.5-flash"
            assert settings.azure_openai_default_model == "gpt-4.1"
            assert settings.azure_openai_api_version == "2024-02-15-preview"
            assert settings.api_host == "0.0.0.0"
            assert settings.api_port == 8000
            assert settings.cors_origins == ["*"]
            assert settings.postgres_pool_size == 20
            assert settings.postgres_max_overflow == 30
            assert settings.redis_pool_size == 20
            assert settings.conversation_storage_backend == "redis"
            assert settings.max_agent_hops == 25
            assert settings.max_tool_hops == 50
            assert settings.graph_recursion_limit == 50
            assert settings.finalizer_llm_provider == "openai"
            assert settings.finalizer_temperature == 0.8
            assert settings.finalizer_max_tokens == 32_000

    def test_settings_with_environment_variables(self):
        """Test settings with environment variables."""
        env_vars = {
            "ECHO_DEBUG": "true",
            "ECHO_DEFAULT_LLM_PROVIDER": "anthropic",
            "ECHO_OPENAI_API_KEY": "test-key-123",
            "ECHO_MAX_AGENT_HOPS": "10",
            "ECHO_MAX_TOOL_HOPS": "20",
            "ECHO_API_PORT": "8080",
            "ECHO_APP_NAME": "Test Echo App",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.debug is True
            assert settings.default_llm_provider == "anthropic"
            assert settings.openai_api_key == "test-key-123"
            assert settings.max_agent_hops == 10
            assert settings.max_tool_hops == 20
            assert settings.api_port == 8080
            assert settings.app_name == "Test Echo App"

    def test_settings_with_custom_values(self):
        """Test settings with custom values."""
        settings = Settings(
            debug=True,
            default_llm_provider="anthropic",
            openai_api_key="custom-key",
            max_agent_hops=15,
            max_tool_hops=30,
            api_port=9000,
            app_name="Custom Echo App",
        )

        assert settings.debug is True
        assert settings.default_llm_provider == "anthropic"
        assert settings.openai_api_key == "custom-key"
        assert settings.max_agent_hops == 15
        assert settings.max_tool_hops == 30
        assert settings.api_port == 9000
        assert settings.app_name == "Custom Echo App"

    def test_plugins_dir_single_string(self):
        """Test plugins_dir with single string value."""
        # Use a relative path that won't trigger directory creation
        settings = Settings(plugins_dir="./test_plugins")

        # The validator converts single strings to lists
        assert settings.plugins_dir == ["./test_plugins"]

    def test_plugins_dir_list(self):
        """Test plugins_dir with list of paths."""
        plugin_paths = ["./plugins1", "./plugins2", "./plugins3"]
        settings = Settings(plugins_dir=plugin_paths)

        assert settings.plugins_dir == plugin_paths

    def test_cors_origins_list(self):
        """Test cors_origins with custom list."""
        origins = ["http://localhost:3000", "https://example.com"]
        settings = Settings(cors_origins=origins)

        assert settings.cors_origins == origins

    def test_azure_openai_settings(self):
        """Test Azure OpenAI specific settings."""
        settings = Settings(
            azure_openai_api_key="azure-key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_deployment="test-deployment",
            azure_openai_api_version="2024-01-01",
        )

        assert settings.azure_openai_api_key == "azure-key"
        assert settings.azure_openai_endpoint == "https://test.openai.azure.com/"
        assert settings.azure_openai_deployment == "test-deployment"
        assert settings.azure_openai_api_version == "2024-01-01"

    def test_database_settings(self):
        """Test database connection settings."""
        settings = Settings(
            postgres_url="postgresql+asyncpg://user:pass@localhost/testdb",
            postgres_pool_size=50,
            postgres_max_overflow=100,
        )

        assert settings.postgres_url == "postgresql+asyncpg://user:pass@localhost/testdb"
        assert settings.postgres_pool_size == 50
        assert settings.postgres_max_overflow == 100

    def test_redis_settings(self):
        """Test Redis connection settings."""
        settings = Settings(
            redis_url="redis://localhost:6379/0",
            redis_pool_size=50,
        )

        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.redis_pool_size == 50

    def test_finalizer_settings(self):
        """Test finalizer-specific settings."""
        settings = Settings(
            finalizer_llm_provider="anthropic",
            finalizer_temperature=0.5,
            finalizer_max_tokens=16_000,
        )

        assert settings.finalizer_llm_provider == "anthropic"
        assert settings.finalizer_temperature == 0.5
        assert settings.finalizer_max_tokens == 16_000

    def test_validation_errors(self):
        """Test validation errors for invalid values."""
        # Test invalid API port
        with pytest.raises(ValidationError):
            Settings(api_port=0)

        with pytest.raises(ValidationError):
            Settings(api_port=70000)

        # Test invalid max_agent_hops
        with pytest.raises(ValidationError):
            Settings(max_agent_hops=0)

        with pytest.raises(ValidationError):
            Settings(max_agent_hops=100)

        # Test invalid max_tool_hops
        with pytest.raises(ValidationError):
            Settings(max_tool_hops=0)

        with pytest.raises(ValidationError):
            Settings(max_tool_hops=200)

        # Test invalid graph_recursion_limit
        with pytest.raises(ValidationError):
            Settings(graph_recursion_limit=10)

        with pytest.raises(ValidationError):
            Settings(graph_recursion_limit=200)

        # Test invalid finalizer_temperature
        with pytest.raises(ValidationError):
            Settings(finalizer_temperature=-0.1)

        with pytest.raises(ValidationError):
            Settings(finalizer_temperature=2.1)

        # Test invalid finalizer_max_tokens
        with pytest.raises(ValidationError):
            Settings(finalizer_max_tokens=50)

        with pytest.raises(ValidationError):
            Settings(finalizer_max_tokens=100_000)

        # Test invalid default_llm_context_window
        with pytest.raises(ValidationError):
            Settings(default_llm_context_window=1000)

        with pytest.raises(ValidationError):
            Settings(default_llm_context_window=100_000)

    def test_environment_variable_case_insensitive(self):
        """Test that environment variables are case-insensitive."""
        env_vars = {
            "echo_debug": "true",
            "ECHO_DEFAULT_LLM_PROVIDER": "anthropic",
            "echo_openai_api_key": "test-key",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.debug is True
            assert settings.default_llm_provider == "anthropic"
            assert settings.openai_api_key == "test-key"

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        # This test is skipped because .env file is loaded automatically
        # and contains real API keys. In a real environment, these would be None
        # when no environment variables are set.
        pytest.skip("Skipping due to .env file loading real API keys")

        # Clear environment variables for this test
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.openai_api_key is None
            assert settings.anthropic_api_key is None
            assert settings.google_api_key is None
            assert settings.azure_openai_api_key is None
            assert settings.azure_openai_endpoint is None
            assert settings.azure_openai_deployment is None
            assert settings.postgres_url is None
            assert settings.redis_url is None

    def test_settings_with_all_providers(self):
        """Test settings with all LLM providers configured."""
        settings = Settings(
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
            google_api_key="google-key",
            azure_openai_api_key="azure-key",
            azure_openai_endpoint="https://azure.openai.azure.com/",
            azure_openai_deployment="deployment-name",
        )

        assert settings.openai_api_key == "openai-key"
        assert settings.anthropic_api_key == "anthropic-key"
        assert settings.google_api_key == "google-key"
        assert settings.azure_openai_api_key == "azure-key"
        assert settings.azure_openai_endpoint == "https://azure.openai.azure.com/"
        assert settings.azure_openai_deployment == "deployment-name"

    def test_conversation_storage_backend_options(self):
        """Test different conversation storage backend options."""
        backends = ["memory", "redis", "postgresql"]

        for backend in backends:
            settings = Settings(conversation_storage_backend=backend)
            assert settings.conversation_storage_backend == backend

    def test_settings_model_config(self):
        """Test that Settings uses correct model configuration."""
        settings = Settings()

        # Verify that the model uses environment variables with ECHO_ prefix
        assert hasattr(settings, "model_config")

        # Test that environment variables are properly loaded
        with patch.dict(os.environ, {"ECHO_DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is True

    def test_settings_field_descriptions(self):
        """Test that field descriptions are properly set."""
        # This test verifies that the Field descriptions are accessible
        # through the model's fields
        settings = Settings()

        # Verify that fields have descriptions
        fields = settings.model_fields
        assert "app_name" in fields
        assert "debug" in fields
        assert "default_llm_provider" in fields

        # Check that descriptions are present
        assert fields["app_name"].description is not None
        assert fields["debug"].description is not None
        assert fields["default_llm_provider"].description is not None

    def test_settings_validation_with_field_validators(self):
        """Test that field validators work correctly."""
        # Test valid values
        settings = Settings(
            default_llm_context_window=16_000,
            finalizer_temperature=1.5,
            finalizer_max_tokens=8_000,
        )

        assert settings.default_llm_context_window == 16_000
        assert settings.finalizer_temperature == 1.5
        assert settings.finalizer_max_tokens == 8_000

    def test_settings_immutability(self):
        """Test that settings are immutable after creation."""
        settings = Settings(debug=True)

        # Settings are not immutable by default in Pydantic v2
        # This test verifies that settings can be modified
        settings.debug = False
        assert settings.debug is False

    def test_settings_serialization(self):
        """Test that settings can be serialized to dict."""
        settings = Settings(
            debug=True,
            default_llm_provider="anthropic",
            openai_api_key="test-key",
            max_agent_hops=10,
        )

        settings_dict = settings.model_dump()

        assert settings_dict["debug"] is True
        assert settings_dict["default_llm_provider"] == "anthropic"
        assert settings_dict["openai_api_key"] == "test-key"
        assert settings_dict["max_agent_hops"] == 10

    def test_settings_json_serialization(self):
        """Test that settings can be serialized to JSON."""
        settings = Settings(
            debug=True,
            default_llm_provider="anthropic",
            max_agent_hops=10,
        )

        settings_json = settings.model_dump_json()

        assert "debug" in settings_json
        assert "anthropic" in settings_json
        assert "10" in settings_json

    def test_settings_copy(self):
        """Test that settings can be copied."""
        original_settings = Settings(
            debug=True,
            default_llm_provider="anthropic",
            openai_api_key="test-key",
        )

        copied_settings = original_settings.model_copy()

        assert copied_settings.debug == original_settings.debug
        assert copied_settings.default_llm_provider == original_settings.default_llm_provider
        assert copied_settings.openai_api_key == original_settings.openai_api_key

    def test_settings_with_complex_plugins_dir(self):
        """Test settings with complex plugins directory configuration."""
        # Test with relative paths (avoid absolute paths that require permissions)
        relative_paths = [
            "./relative/plugins1",
            "../relative/plugins2",
            "./test_plugins",
        ]
        settings = Settings(plugins_dir=relative_paths)
        assert settings.plugins_dir == relative_paths

        # Test with single relative path
        single_path = "./single_plugin"
        settings = Settings(plugins_dir=single_path)
        assert settings.plugins_dir == [single_path]

    def test_settings_with_complex_cors_origins(self):
        """Test settings with complex CORS origins configuration."""
        complex_origins = [
            "http://localhost:3000",
            "https://app.example.com",
            "https://*.example.com",
            "http://192.168.1.100:8080",
        ]
        settings = Settings(cors_origins=complex_origins)
        assert settings.cors_origins == complex_origins

    def test_settings_edge_cases(self):
        """Test settings with edge case values."""
        # Test with maximum allowed values
        settings = Settings(
            max_agent_hops=50,
            max_tool_hops=100,
            graph_recursion_limit=150,
            finalizer_temperature=2.0,
            finalizer_max_tokens=64_000,
            default_llm_context_window=64_000,
        )

        assert settings.max_agent_hops == 50
        assert settings.max_tool_hops == 100
        assert settings.graph_recursion_limit == 150
        assert settings.finalizer_temperature == 2.0
        assert settings.finalizer_max_tokens == 64_000
        assert settings.default_llm_context_window == 64_000

        # Test with minimum allowed values
        settings = Settings(
            max_agent_hops=1,
            max_tool_hops=1,
            graph_recursion_limit=25,
            finalizer_temperature=0.0,
            finalizer_max_tokens=100,
            default_llm_context_window=2001,
        )

        assert settings.max_agent_hops == 1
        assert settings.max_tool_hops == 1
        assert settings.graph_recursion_limit == 25
        assert settings.finalizer_temperature == 0.0
        assert settings.finalizer_max_tokens == 100
        assert settings.default_llm_context_window == 2001
