"""Test cases for the LLM factory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel

from echo.infrastructure.llm.factory import LLMModelFactory
from echo.infrastructure.llm.providers import ModelConfig


class TestLLMModelFactory:
    """Test cases for LLMModelFactory."""

    @pytest.fixture
    def factory(self):
        """Create a LLMModelFactory instance for testing."""
        return LLMModelFactory()

    def test_factory_initialization(self, factory):
        """Test LLMModelFactory initialization."""
        assert factory is not None
        assert hasattr(factory, "create_base_model")

    @patch("echo.infrastructure.llm.factory.ChatOpenAI")
    def test_create_openai_model(self, mock_openai, factory):
        """Test creating OpenAI model."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_openai.return_value = mock_model

        config = ModelConfig(provider="openai", model_name="gpt-4", temperature=0.1, max_tokens=1000)

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_openai.assert_called_once_with(model="gpt-4", temperature=0.1, max_tokens=1000)

    @patch("echo.infrastructure.llm.factory.ChatAnthropic")
    def test_create_anthropic_model(self, mock_anthropic, factory):
        """Test creating Anthropic model."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_anthropic.return_value = mock_model

        config = ModelConfig(provider="anthropic", model_name="claude-3-sonnet", temperature=0.5, max_tokens=2000)

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_anthropic.assert_called_once_with(model="claude-3-sonnet", temperature=0.5, max_tokens=2000)

    @patch("echo.infrastructure.llm.factory.ChatGoogleGenerativeAI")
    def test_create_gemini_model(self, mock_gemini, factory):
        """Test creating Google Gemini model."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_gemini.return_value = mock_model

        config = ModelConfig(provider="gemini", model_name="gemini-2.0-flash", temperature=0.3, max_tokens=1500)

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_gemini.assert_called_once_with(model="gemini-2.0-flash", temperature=0.3, max_tokens=1500)

    @patch("echo.infrastructure.llm.factory.AzureChatOpenAI")
    def test_create_azure_openai_model(self, mock_azure, factory):
        """Test creating Azure OpenAI model."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_azure.return_value = mock_model

        config = ModelConfig(
            provider="azure_openai",
            model_name="gpt-4",
            temperature=0.2,
            max_tokens=1000,
            azure_deployment="test-deployment",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-15-preview",
        )

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_azure.assert_called_once_with(
            model="gpt-4",
            temperature=0.2,
            max_tokens=1000,
            azure_deployment="test-deployment",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-15-preview",
        )

    def test_create_model_with_unsupported_provider(self, factory):
        """Test creating model with unsupported provider."""
        config = ModelConfig(provider="unsupported_provider", model_name="test-model", temperature=0.1, max_tokens=1000)

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            factory.create_base_model(config)

    def test_create_model_with_missing_required_fields(self, factory):
        """Test creating model with missing required fields."""
        config = ModelConfig(provider="openai", model_name="", temperature=0.1, max_tokens=1000)  # Empty model name

        with pytest.raises(ValueError):
            factory.create_base_model(config)

    @patch("echo.infrastructure.llm.factory.ChatOpenAI")
    def test_create_model_with_optional_fields(self, mock_openai, factory):
        """Test creating model with optional fields."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_openai.return_value = mock_model

        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=1000,
            api_key="test-key",
            base_url="https://custom.openai.com",
        )

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_openai.assert_called_once_with(
            model="gpt-4", temperature=0.1, max_tokens=1000, api_key="test-key", base_url="https://custom.openai.com"
        )

    @patch("echo.infrastructure.llm.factory.ChatAnthropic")
    def test_create_anthropic_model_with_api_key(self, mock_anthropic, factory):
        """Test creating Anthropic model with API key."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_anthropic.return_value = mock_model

        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            temperature=0.5,
            max_tokens=2000,
            api_key="anthropic-test-key",
        )

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_anthropic.assert_called_once_with(
            model="claude-3-sonnet", temperature=0.5, max_tokens=2000, api_key="anthropic-test-key"
        )

    @patch("echo.infrastructure.llm.factory.ChatGoogleGenerativeAI")
    def test_create_gemini_model_with_api_key(self, mock_gemini, factory):
        """Test creating Google Gemini model with API key."""
        mock_model = MagicMock(spec=BaseLanguageModel)
        mock_gemini.return_value = mock_model

        config = ModelConfig(
            provider="gemini",
            model_name="gemini-2.0-flash",
            temperature=0.3,
            max_tokens=1500,
            api_key="google-test-key",
        )

        result = factory.create_base_model(config)

        assert result == mock_model
        mock_gemini.assert_called_once_with(
            model="gemini-2.0-flash", temperature=0.3, max_tokens=1500, api_key="google-test-key"
        )

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Test valid config
        config = ModelConfig(provider="openai", model_name="gpt-4", temperature=0.1, max_tokens=1000)

        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.1
        assert config.max_tokens == 1000

    def test_model_config_with_invalid_temperature(self):
        """Test ModelConfig with invalid temperature."""
        with pytest.raises(ValueError):
            ModelConfig(provider="openai", model_name="gpt-4", temperature=2.5, max_tokens=1000)  # Invalid temperature

    def test_model_config_with_invalid_max_tokens(self):
        """Test ModelConfig with invalid max_tokens."""
        with pytest.raises(ValueError):
            ModelConfig(provider="openai", model_name="gpt-4", temperature=0.1, max_tokens=-1)  # Invalid max_tokens

    @patch("echo.infrastructure.llm.factory.ChatOpenAI")
    def test_factory_reuses_models(self, mock_openai, factory):
        """Test that factory can create multiple models."""
        mock_model1 = MagicMock(spec=BaseLanguageModel)
        mock_model2 = MagicMock(spec=BaseLanguageModel)
        mock_openai.side_effect = [mock_model1, mock_model2]

        config1 = ModelConfig(provider="openai", model_name="gpt-4", temperature=0.1, max_tokens=1000)

        config2 = ModelConfig(provider="openai", model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=500)

        result1 = factory.create_base_model(config1)
        result2 = factory.create_base_model(config2)

        assert result1 == mock_model1
        assert result2 == mock_model2
        assert mock_openai.call_count == 2

    def test_factory_error_handling(self, factory):
        """Test factory error handling."""
        config = ModelConfig(provider="openai", model_name="gpt-4", temperature=0.1, max_tokens=1000)

        with patch("echo.infrastructure.llm.factory.ChatOpenAI") as mock_openai:
            mock_openai.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                factory.create_base_model(config)
