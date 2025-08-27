"""Echo Framework Infrastructure Layer - External Services and Data Persistence.

This layer provides data persistence mechanisms and core infrastructure services
for the Echo multi-agent AI framework. It implements the infrastructure concerns
that support the domain and application layers.

Architecture Responsibilities:
    The infrastructure layer handles all external concerns:
    - Data persistence across multiple database backends
    - LLM provider integrations and model management
    - Plugin system with SDK-based discovery and lifecycle
    - Caching, session storage, and performance optimization

Key Components:
    database/: Multi-backend data persistence (PostgreSQL, Redis)
    llm/: LLM provider abstraction and model factory
    plugins/: SDK-based plugin management and discovery

Design Patterns:
    - Repository Pattern: Clean abstractions over data access
    - Factory Pattern: Provider-agnostic service creation
    - Adapter Pattern: Unified interfaces for external services
    - Strategy Pattern: Pluggable backend implementations

Multi-Backend Support:
    Database Backends:
        - PostgreSQL: Primary relational data with ACID guarantees
        - Redis: High-performance session storage and caching

        - In-Memory: Development and testing backends

    LLM Providers:
        - OpenAI: GPT models with function calling
        - Anthropic: Claude models with large context windows
        - Google: Gemini models with multimodal capabilities
        - Azure OpenAI: Enterprise-grade hosted OpenAI models

Storage Optimization:
    - Significant storage reduction through optimized conversation turns
    - Intelligent caching strategies for frequently accessed data
    - Configurable retention policies and data lifecycle management
    - Cost-optimized storage backend selection based on usage patterns

Example Usage:
    Database repository access:

    ```python
    from echo.infrastructure.database import DatabaseFactory
    from echo.config import Settings

    settings = Settings()
    db_factory = DatabaseFactory(settings)

    await db_factory.initialize()
    thread_repo, conv_repo = await db_factory.create_repositories()

    thread = await thread_repo.get_by_id("thread-123")
    ```

    LLM provider usage:

    ```python
    from echo.infrastructure.llm import LLMModelFactory

    llm_factory = LLMModelFactory(settings)
    model = await llm_factory.get_model("gpt-4")

    response = await model.ainvoke("Hello, world!")
    ```

The infrastructure layer ensures that all external dependencies are properly abstracted,
allowing the domain and application layers to remain focused on business logic.
"""
