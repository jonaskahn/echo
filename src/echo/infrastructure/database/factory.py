"""Echo Framework Database Factory - Repository and Session Store Creation.

This module implements the Factory pattern for creating database repositories and session
stores based on configuration. It provides a clean abstraction that allows the application
to use different storage backends without changing business logic code.

Architecture Benefits:
    The database factory provides several key architectural benefits:
    - Configuration-driven backend selection without code changes
    - Consistent repository interfaces across all backend implementations
    - Easy testing through in-memory implementations
    - Scalability through backend-specific optimizations
    - Clean separation of concerns between configuration and implementation

Factory Pattern Implementation:
    The factory creates appropriate implementations based on settings:
    - Conversation Storage: Memory or PostgreSQL backends
    - Session Storage: Memory, Redis, or PostgreSQL backends
    - Connection Management: Automatic connection pooling and health monitoring
    - Migration Support: Automatic schema management for SQL backends

Supported Backend Combinations:
    Development Configuration:
        - Conversation Storage: In-Memory (fast, no persistence)
        - Session Storage: In-Memory (immediate testing feedback)
        - Benefits: Fast startup, no external dependencies

    Production Configuration:
        - Conversation Storage: PostgreSQL (ACID compliance, complex queries)
        - Session Storage: Redis (high performance, automatic expiration)
        - Benefits: Data durability, optimized performance

    High-Scale Configuration:

        - Session Storage: Redis (sub-millisecond access, clustering)
        - Benefits: Horizontal scaling, multi-datacenter support

Example Usage:
    Basic factory usage:

    ```python
    from echo.infrastructure.database import DatabaseFactory
    from echo.config import Settings

    settings = Settings()
    settings.conversation_storage_backend = "postgresql"
    settings.session_storage_backend = "redis"

    db_factory = DatabaseFactory(settings)
    await db_factory.initialize()

    thread_repo, conversation_repo = await db_factory.create_repositories()

    from echo.domain.models import Thread
    thread = Thread(user_id="user-123")
    created_thread = await thread_repo.create(thread)
    ```

    Testing with in-memory backends:

    ```python
    settings.conversation_storage_backend = "memory"
    settings.session_storage_backend = "memory"

    db_factory = DatabaseFactory(settings)
    await db_factory.initialize()

    thread_repo, conversation_repo = await db_factory.create_repositories()
    ```



Performance Optimizations:
    - Backend-specific optimizations (PostgreSQL indexing)
    - Connection pooling and reuse across repository instances
    - Lazy initialization to avoid unnecessary connections
    - Health monitoring and automatic failover for critical backends

The database factory ensures that backend selection is transparent to the application
layer while providing optimal performance for each deployment scenario.
"""

import logging
from typing import Any, Dict, Tuple

from ...config.settings import Settings
from .connection import DatabaseConnectionManager, initialize_databases
from .repositories import (
    ConversationRepository,
    InMemoryConversationRepository,
    InMemoryThreadRepository,
    PostgreSQLConversationRepository,
    PostgreSQLThreadRepository,
    RedisConversationRepository,
    RedisThreadRepository,
    ThreadRepository,
)

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """Factory for creating optimized database repositories and session stores.

    This factory implements the Factory pattern to provide backend-agnostic access
    to database repositories and session stores. It automatically selects the
    appropriate implementation based on configuration while providing consistent
    interfaces across all backends.

    The factory supports multiple deployment scenarios:
    - Development: Fast in-memory backends for rapid iteration
    - Production: Optimized persistent backends for reliability
    - High-scale: Distributed backends for massive throughput

    Key Features:
        - Configuration-driven backend selection
        - Automatic connection management and pooling
        - Backend-specific performance optimizations
        - Consistent repository interfaces across implementations
        - Health monitoring and automatic failover

    Supported Backends:
        Conversation Storage:
            - memory: In-memory storage for development and testing
            - postgresql: ACID-compliant relational storage with SQLAlchemy


        Session Storage:
            - memory: In-memory session store for testing
            - redis: High-performance session store with TTL support
            - postgresql: Persistent session storage in relational database

    Example:
        ```python
        settings = Settings()
        settings.conversation_storage_backend = "postgresql"
        settings.session_storage_backend = "redis"

        factory = DatabaseFactory(settings)
        await factory.initialize()

        thread_repo, conv_repo = await factory.create_repositories()
        thread = await thread_repo.get_by_id("thread-123")
        ```
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection_manager: DatabaseConnectionManager = None

    async def initialize(self) -> None:
        """Initialize database connections for configured backends only."""
        self.connection_manager = await initialize_databases(self.settings)
        logger.info("Database factory initialized with configured backends only")

    async def create_repositories(self) -> Tuple[ThreadRepository, ConversationRepository]:
        """Create repository instances based on configuration."""

        backend = self.settings.conversation_storage_backend.lower()
        if backend == "postgresql":
            return await self._create_postgresql_repositories()
        elif backend == "redis":
            return await self._create_redis_repositories()
        else:
            return self._create_memory_repositories()

    @staticmethod
    def _create_memory_repositories() -> Tuple[ThreadRepository, ConversationRepository]:
        """Create in-memory repository implementations."""
        thread_repo = InMemoryThreadRepository()
        conversation_repo = InMemoryConversationRepository(thread_repo)

        logger.info("Created in-memory repositories")
        return thread_repo, conversation_repo

    async def _create_postgresql_repositories(self) -> Tuple[ThreadRepository, ConversationRepository]:
        """Create PostgreSQL repository implementations."""
        if not self.connection_manager:
            raise RuntimeError("Database connections not initialized")

        if not self.connection_manager.postgres_session_factory:
            raise RuntimeError("PostgreSQL not configured")

        thread_repo = PostgreSQLThreadRepository(self.connection_manager.postgres_session_factory)
        conversation_repo = PostgreSQLConversationRepository(
            self.connection_manager.postgres_session_factory, thread_repo
        )

        logger.info("Created PostgreSQL repositories with SQLAlchemy")
        return thread_repo, conversation_repo

    async def _create_redis_repositories(self) -> Tuple[ThreadRepository, ConversationRepository]:
        """Create Redis repository implementations."""
        if not self.connection_manager:
            raise RuntimeError("Database connections not initialized")

        if not self.connection_manager.redis_client:
            raise RuntimeError("Redis is not configured")

        thread_repo = RedisThreadRepository(self.connection_manager.redis_client)
        conversation_repo = RedisConversationRepository(self.connection_manager.redis_client, thread_repo)

        logger.info("Created Redis repositories with high-performance storage")
        return thread_repo, conversation_repo

    async def get_connection_manager(self) -> DatabaseConnectionManager:
        """Get the database connection manager."""
        if not self.connection_manager:
            raise RuntimeError("Database factory not initialized")
        return self.connection_manager

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all database backends."""
        if not self.connection_manager:
            return {"status": "not_initialized"}
        db_health = await self.connection_manager.health_check()
        health_info = {
            "database_factory": {
                "status": "healthy",
                "conversation_backend": self.settings.conversation_storage_backend,
                "session_backend": self.settings.session_storage_backend,
                "backends": db_health,
            }
        }

        unhealthy_backends = [name for name, status in db_health.items() if status.get("status") == "unhealthy"]

        if unhealthy_backends:
            health_info["database_factory"]["status"] = "degraded"
            health_info["database_factory"]["unhealthy_backends"] = unhealthy_backends

        return health_info

    async def close(self):
        """Close database connections."""
        if self.connection_manager:
            await self.connection_manager.close_connections()
