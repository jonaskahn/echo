"""MongoDB repository implementations for the Echo framework.

This module provides MongoDB-specific repository implementations.
Currently a placeholder for future MongoDB support.
"""

from .mongo_repositories import MongoConversationRepository, MongoThreadRepository

__all__ = [
    "MongoThreadRepository",
    "MongoConversationRepository",
]
