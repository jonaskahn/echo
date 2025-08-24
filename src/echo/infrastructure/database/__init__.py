"""Echo Framework Database Infrastructure - Multi-Backend Data Persistence.

This module provides comprehensive data persistence infrastructure for the Echo multi-agent
AI framework, supporting multiple database backends with optimized storage strategies
and repository pattern implementations.

Architecture Overview:
    The database layer implements a multi-backend persistence strategy with:
    - Repository pattern for clean data access abstractions
    - Factory pattern for backend selection and configuration
    - Strategy pattern for pluggable storage implementations
    - Connection management with pooling and failover
    - Optimized conversation storage achieving significant space reduction

Supported Backends:
    Production Backends:
        - PostgreSQL: Primary relational database with ACID guarantees
        - Redis: High-performance session storage and caching layer

    Development/Testing Backends:
        - In-Memory: Fast testing and development with no persistence
        - SQLite: Local development with file-based persistence

Storage Optimization Strategy:
    The database layer implements an optimized conversation storage approach:
    - Stores only user input + final AI response (not intermediate LangGraph steps)
    - Achieves significant storage cost reduction vs full message history
    - Maintains conversation continuity for LangGraph context reconstruction
    - Provides comprehensive token usage tracking for cost optimization

Key Components:
    models/: SQLAlchemy ORM models with optimized schemas
    repositories/: Repository pattern implementations for each backend
    connection.py: Database connection management and pooling
    factory.py: Database factory for backend selection and initialization

    schemas/: Database schema definitions and migration scripts

Design Patterns:
    Repository Pattern:
        - Abstract interfaces define data access contracts
        - Concrete implementations provide backend-specific optimizations
        - Clean separation between business logic and data persistence

    Factory Pattern:
        - DatabaseFactory creates appropriate repository instances
        - Backend selection based on configuration
        - Connection management and health monitoring

    Strategy Pattern:
        - Pluggable session stores (Redis, PostgreSQL, in-memory)
        - Configurable conversation storage backends
        - Performance optimizations specific to each backend

Example Usage:
    Initializing database infrastructure:

    ```python
    from echo.infrastructure.database import DatabaseFactory
    from echo.config import Settings

    settings = Settings()
    settings.conversation_storage_backend = "postgresql"
    settings.session_storage_backend = "redis"

    db_factory = DatabaseFactory(settings)
    await db_factory.initialize()

    thread_repo, conversation_repo = await db_factory.create_repositories()

    from echo.domain.models import Thread, ConversationTurn

    thread = Thread(user_id="user-123", org_id="acme-corp")
    await thread_repo.create(thread)

    turn = ConversationTurn(
        thread_id=thread.thread_id,
        user_message="Hello, world!",
        assistant_message="Hi there! How can I help?",
        user_tokens=10,
        assistant_tokens=25
    )
    await conversation_repo.create(turn)
    ```

    Multi-backend configuration:

    ```python
    settings = Settings()
    settings.postgres_url = "postgresql+asyncpg://user:pass@host/echo"
    settings.redis_url = "redis://redis-host:6379/0"
    settings.conversation_storage_backend = "postgresql"
    settings.session_storage_backend = "redis"
    ```

Performance Optimizations:
    - Connection pooling with configurable pool sizes
    - Async/await patterns for non-blocking database operations
    - Batch operations for high-throughput scenarios
    - Intelligent caching strategies for frequently accessed data
    - Index optimization for common query patterns

The database infrastructure ensures data consistency, performance, and scalability
while providing clean abstractions that keep the application layer focused on
business logic rather than persistence concerns.
"""

from .connection import DatabaseConnectionManager, initialize_databases
from .factory import DatabaseFactory
from .models import Base, ConversationModel, OrganizationModel, ThreadModel, UserModel
from .repositories import (
    ConversationRepository,
    InMemoryConversationRepository,
    InMemoryThreadRepository,
    PostgreSQLConversationRepository,
    PostgreSQLThreadRepository,
    ThreadRepository,
)

__all__ = [
    "DatabaseConnectionManager",
    "initialize_databases",
    "DatabaseFactory",
    "Base",
    "ThreadModel",
    "ConversationModel",
    "UserModel",
    "OrganizationModel",
    "ThreadRepository",
    "ConversationRepository",
    "InMemoryThreadRepository",
    "InMemoryConversationRepository",
    "PostgreSQLThreadRepository",
    "PostgreSQLConversationRepository",
]
