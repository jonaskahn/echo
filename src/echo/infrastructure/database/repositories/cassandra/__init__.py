"""Cassandra repository implementations for the Echo framework.

This module provides Cassandra-specific repository implementations.
Currently a placeholder for future Cassandra support.
"""

from .cassandra_repositories import CassandraThreadRepository, CassandraConversationRepository

__all__ = [
    "CassandraThreadRepository",
    "CassandraConversationRepository",
]
