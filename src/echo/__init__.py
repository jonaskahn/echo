"""Echo Multi-Agent AI Framework - Public Package API.

Echo is a plugin-based multi-agent conversational AI framework built on FastAPI.
It provides a flexible architecture for creating, managing, and orchestrating
multiple AI agents with support for various LLM providers and integration platforms.

Core Features:
    - Multi-agent orchestration with LangGraph
    - Plugin system with SDK support for custom agents
    - Multiple LLM provider support (OpenAI, Anthropic, Google, Azure)
    - Integration handlers for Slack, Discord, and webhooks
    - Scalable storage backends (PostgreSQL, Redis)
    - Real-time conversation management with optimized storage

Public API:
    Settings: Application configuration with environment variable support
    MultiAgentOrchestrator: Core conversation orchestration engine
    SDKPluginManager: Plugin discovery and lifecycle management
    LLMModelFactory: LLM provider abstraction with caching and routing

Example:
    Basic setup and usage:

    ```python
    from echo import Settings, MultiAgentOrchestrator, LLMModelFactory

    settings = Settings()
    settings.default_llm_provider = "openai"
    settings.openai_api_key = "your-api-key"

    llm_factory = LLMModelFactory(settings)
    orchestrator = MultiAgentOrchestrator(settings, llm_factory)

    response = await orchestrator.process_message("Hello, help me with math")
    ```

    Running the complete server:

    ```python
    from echo.main import EchoApplication

    app = EchoApplication()
    app.run()
    ```

Version: 0.1.0 - Production ready with comprehensive plugin system
"""

__version__ = "0.1.0"

from .config.settings import Settings
from .core.orchestrator.coordinator import MultiAgentOrchestrator
from .infrastructure.llm.factory import LLMModelFactory
from .infrastructure.plugins.sdk_manager import SDKPluginManager

__all__ = ["Settings", "MultiAgentOrchestrator", "SDKPluginManager", "LLMModelFactory"]
