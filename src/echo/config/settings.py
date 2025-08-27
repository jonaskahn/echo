"""Configuration settings for the Echo 🤖 Multi-agents AI Framework.
This module defines a Pydantic ``BaseSettings`` model used to configure the
application via environment variables and a ``.env`` file. Environment
variables are read with the ``ECHO_`` prefix (case-insensitive), and field
descriptions serve as the authoritative documentation for each setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support and validation.
    Notes:
    - Values can be provided via environment variables with prefix ``ECHO_``
      (e.g., ``ECHO_API_PORT=8080``), or from a ``.env`` file.
    - Configuration is case-insensitive and validates assignments at runtime.
    - See ``model_config`` for environment loading behavior.
    """

    app_name: str = Field(default="Echo 🤖 Multi-agents AI Framework", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")

    default_llm_provider: str = Field(default="openai", description="Default LLM provider")
    default_llm_context_window: int = Field(
        default=32_000, le=64_000, gt=2000, description="Default LLM context window size"
    )
    default_llm_temperature: float = Field(default=0.1, description="Default LLM temperature")

    openai_default_model: str = Field(default="gpt-4.1", description="Default OpenAI model")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    anthropic_default_model: str = Field(default="claude-sonnet-4-20250514", description="Default Claude model")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    gemini_default_model: str = Field(default="gemini-2.5-flash", description="Default Google Gemini model")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    azure_openai_default_model: str = Field(default="gpt-4.1", description="Default Azure OpenAI model")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_openai_api_version: Optional[str] = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    azure_openai_deployment: Optional[str] = Field(default=None, description="Azure OpenAI deployment name")

    plugins_dir: Union[str, List[str]] = Field(
        default=["./plugins/src/echo_plugins"],
        description="Plugin directory path(s) - can be a single path or list of paths",
    )

    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port")

    ui_host: str = Field(default="0.0.0.0", description="UI server host")
    ui_port: int = Field(default=8561, ge=1, le=65535, description="UI server port")

    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    postgres_url: Optional[str] = Field(
        default=None, description="PostgreSQL connection URL (e.g., postgresql+asyncpg://user:pass@localhost/echo)"
    )
    postgres_pool_size: int = Field(default=20, description="PostgreSQL connection pool size")
    postgres_max_overflow: int = Field(default=30, description="PostgreSQL max overflow connections")

    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_pool_size: int = Field(default=20, description="Redis connection pool size")

    conversation_storage_backend: str = Field(
        default="memory", description="Backend for conversation storage: memory, redis, postgresql"
    )

    max_agent_hops: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Maximum agent switches before forcing finalization (prevents infinite agent routing loops)",
    )

    max_tool_hops: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum tool calls before forcing finalization (prevents excessive tool usage)",
    )

    graph_recursion_limit: int = Field(
        default=50,
        ge=25,
        le=150,
        description="Maximum number of LangGraph steps per request before halting to prevent infinite loops",
    )

    finalizer_llm_provider: str = Field(default="openai", description="LLM provider for finalizer node")
    finalizer_temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Temperature for finalizer node")
    finalizer_max_tokens: int = Field(
        default=32_000, ge=100, le=64_000, description="Maximum tokens for finalizer responses"
    )

    persistence_type: str = Field(
        default="checkpoint",
        description="Persistence type to use to keep chat context. Support: checkpoint, memory",
    )

    persistence_checkpoint_layer: str = Field(
        default="redis",
        description="Support database to store checkpoint. Support: redis, postgres, sqlite; fallback to in-memory",
    )
    persistence_checkpoint_redis_ttl_minutes: int = Field(
        default=60 * 24,
        ge=5,
        le=7 * 24 * 60,
        description="Default TTL for checkpoints (redis) in minutes",
    )
    persistence_checkpoint_redis_refresh_on_read: bool = Field(
        default=True,
        description="Persistence refresh interval between checkpoint(redis) refresh requests",
    )

    persistence_memory_layer: str = Field(
        default="redis",
        description="Support database to store (active) memory. Support: redis, postgres, sqlite; fallback to in-memory",
    )

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        """Validate that the configured default LLM provider is supported.
        Args:
            value: Provider name supplied via settings/.env (e.g., "openai").
        Returns:
            The validated provider name.
        Raises:
            ValueError: If the provider is not one of the supported options.
        """
        supported_providers = [
            "openai",
            "azure-openai",
            "azure",
            "anthropic",
            "claude",
            "google",
            "gemini",
        ]
        if value not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider: {value}. " f"Supported providers: {', '.join(supported_providers)}"
            )
        return value

    @field_validator("plugins_dir")
    @classmethod
    def validate_plugins_dir(cls, value: Union[str, List[str]]) -> List[str]:
        """Ensure the plugins directory(ies) exist or can be created.
        Args:
            value: Filesystem path(s) to the plugins directory(ies) - can be a single path or list of paths.
        Returns:
            A list of validated paths after ensuring they exist.
        """
        if isinstance(value, str):
            paths = [value]
        else:
            paths = value

        validated_paths = []
        for path_str in paths:
            plugins_path = Path(path_str)
            if not plugins_path.exists():
                plugins_path.mkdir(parents=True, exist_ok=True)
            validated_paths.append(path_str)

        return validated_paths

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Return the API key for the specified provider.
        Provider aliases are supported: "claude" maps to Anthropic, and
        "gemini" maps to Google.
        Args:
            provider: Provider name or alias (e.g., "openai", "anthropic",
                "claude", "google", "gemini").
        Returns:
            The API key string if configured; otherwise, ``None``.
        """
        provider_key_mapping = {
            "openai": self.openai_api_key,
            "azure-openai": self.azure_openai_api_key,
            "azure": self.azure_openai_api_key,
            "anthropic": self.anthropic_api_key,
            "claude": self.anthropic_api_key,
            "google": self.google_api_key,
            "gemini": self.google_api_key,
        }
        return provider_key_mapping.get(provider)

    def get_provider_extra_params(self, provider: str) -> dict | None:
        """Return extra provider params for model initialization.
        Currently used for Azure OpenAI.
        Args:
            provider: Provider name (e.g., "azure-openai", "azure").
        Returns:
            Dictionary of extra parameters for the provider, or None if no extra params needed.
        """
        if provider in {"azure-openai", "azure"}:
            return {
                "azure_endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version,
                "deployment_name": self.azure_openai_deployment,
            }
        return {}

    def validate_provider_credentials(self, provider: str) -> bool:
        """Check whether credentials/config exist for the specified provider.
        Args:
            provider: Provider name to validate.
        Returns:
            True if credentials are valid, False otherwise.
        Notes:
            - For OpenAI/Anthropic/Google: requires non-empty API key.
            - For Azure OpenAI: requires non-empty API key, endpoint, and deployment.
        """
        normalized = {
            "claude": "anthropic",
            "gemini": "google",
            "google-gemini": "google",
            "azure": "azure-openai",
        }.get(provider, provider)

        api_key = self.get_api_key_for_provider(provider)
        if not api_key or not api_key.strip():
            return False

        if normalized == "azure-openai":
            if not self.azure_openai_endpoint or not str(self.azure_openai_endpoint).strip():
                return False
            if not self.azure_openai_deployment or not str(self.azure_openai_deployment).strip():
                return False
        return True

    def _get_model_for_provider(self, provider: str) -> str:
        """Get the default model for a given provider.
        Args:
            provider: Provider name to get model for.
        Returns:
            The default model string for the provider.
        Raises:
            ValueError: If the provider is not supported.
        """
        if provider in {"openai", "azure", "azure-openai"}:
            return self.openai_default_model
        elif provider in {"claude", "anthropic"}:
            return self.anthropic_default_model
        elif provider in {"gemini", "google", "google-gemini"}:
            return self.gemini_default_model
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_default_provider_llm_model(self) -> str:
        """Get the default model for the configured LLM provider."""
        return self._get_model_for_provider(self.default_llm_provider)

    def get_finalizer_provider_llm_model(self) -> str:
        """Get the model for the finalizer LLM provider."""
        return self._get_model_for_provider(self.finalizer_llm_provider)

    model_config = {
        "env_file": ".env",
        "env_prefix": "ECHO_",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore",
    }
