"""Echo AI Multi-Agent AI Framework - Public Package API.

Echo AI is a plugin-based multi-agent conversational AI framework built on FastAPI.
It provides intelligent agent orchestration, efficient conversation storage, and
comprehensive multi-provider LLM support.

Quick Start:
    >>> from echo import Settings, MultiAgentOrchestrator, LLMModelFactory
    >>> settings = Settings()
    >>> orchestrator = MultiAgentOrchestrator(settings)
    >>> response = await orchestrator.process_message("Hello, world!")

Key Components:
    - MultiAgentOrchestrator: Core orchestration engine
    - LLMModelFactory: Multi-provider LLM integration
    - Settings: Configuration management
    - Plugin System: Extensible agent architecture

For more information, visit: https://github.com/jonaskahn/echo
"""

from echo.main import EchoApplication

app = EchoApplication()
