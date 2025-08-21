"""MongoDB repository implementations for the Echo framework.

This module provides MongoDB-specific repository implementations.
Currently a placeholder for future MongoDB support.
"""

from .mongo_repositories import MongoThreadRepository, MongoConversationRepository

__all__ = [
    "MongoThreadRepository",
    "MongoConversationRepository",
]
