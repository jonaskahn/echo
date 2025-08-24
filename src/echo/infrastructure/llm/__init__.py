"""Echo Framework LLM Infrastructure - Multi-Provider Language Model Management.

This module provides comprehensive language model infrastructure for the Echo multi-agent
AI framework, supporting multiple LLM providers with unified interfaces, intelligent
caching, and provider-specific optimizations.

Architecture Overview:
    The LLM infrastructure implements a multi-provider strategy that enables Echo to:
    - Support multiple LLM providers through unified interfaces
    - Optimize model selection based on task requirements and cost
    - Provide intelligent caching to minimize initialization overhead
    - Handle provider-specific authentication and configuration
    - Enable seamless switching between providers without code changes

Key Features:
    Multi-Provider Support:
        - OpenAI: GPT models with function calling capabilities
        - Anthropic: Claude models with large context windows
        - Google: Gemini models with multimodal capabilities
        - Azure OpenAI: Enterprise-grade hosted OpenAI models

    Performance Optimizations:
        - Model instance caching to prevent repeated initialization
        - Provider-specific connection pooling and rate limiting
        - Lazy loading of provider credentials and configurations
        - Intelligent model selection based on context window requirements

    Configuration Management:
        - Environment-based provider configuration
        - Automatic credential validation and provider health checks
        - Dynamic provider switching based on availability
        - Cost optimization through provider selection strategies

Components:
    LLMModelFactory: Main factory for creating and caching model instances
    ModelConfig: Provider-agnostic model configuration and settings
    BaseLLMProvider: Abstract interface for all LLM provider implementations
    Provider Implementations: Concrete provider classes for each supported service

Design Patterns:
    Factory Pattern: Centralized model creation with caching and optimization
    Strategy Pattern: Pluggable provider implementations with consistent interfaces
    Adapter Pattern: Unified interface over diverse provider APIs
    Singleton Pattern: Shared cache and registry instances across the application

Example Usage:
    Basic model factory usage:

    ```python
    from echo.infrastructure.llm import LLMModelFactory, ModelConfig
    from echo.config import Settings

    settings = Settings()
    settings.default_llm_provider = "openai"
    settings.openai_api_key = "your-api-key"

    llm_factory = LLMModelFactory(settings)

    model = await llm_factory.get_model("gpt-4")

    response = await model.ainvoke("Hello, world!")
    print(response.content)
    ```

    Multi-provider configuration:

    ```python
    settings = Settings()
    settings.openai_api_key = "openai-key"
    settings.anthropic_api_key = "anthropic-key"
    settings.google_api_key = "google-key"

    factory = LLMModelFactory(settings)

    gpt_model = await factory.get_model("gpt-4")
    claude_model = await factory.get_model("claude-3")
    gemini_model = await factory.get_model("gemini-pro")
    ```

    Tool binding for agents:

    ```python
    from langchain_core.tools import Tool

    tools = [
        Tool(name="calculator", func=calculate),
        Tool(name="search", func=search_web)
    ]

    agent_model = await factory.get_model_with_tools("gpt-4", tools)

    response = await agent_model.ainvoke("Calculate 2+2 and search for weather")
    ```

Provider-Specific Features:
    OpenAI:
        - Function calling for tool integration
        - Streaming responses for real-time applications
        - Multiple model variants (GPT-3.5, GPT-4, GPT-4-Turbo)

    Anthropic (Claude):
        - Large context windows (up to 200K tokens)
        - Constitutional AI for safer responses
        - Advanced reasoning capabilities

    Google (Gemini):
        - Multimodal capabilities (text, image, video)
        - Code generation and analysis
        - Multiple model sizes for different use cases

    Azure OpenAI:
        - Enterprise security and compliance
        - Regional data residency
        - Advanced monitoring and logging

The LLM infrastructure provides the foundation for Echo's multi-agent capabilities,
enabling intelligent conversations with optimal provider selection and performance.
"""

from .factory import LLMModelFactory
from .providers import BaseLLMProvider, ModelConfig

__all__ = ["LLMModelFactory", "ModelConfig", "BaseLLMProvider"]
