"""Echo Framework Domain Models - Core Business Entities.

This module exports the primary business entities that form the foundation of the
Echo multi-agent AI framework. These models implement rich domain objects with
business behavior, validation, and clear separation of concerns.

Business Entities:
    User: System user with organization affiliation and activity management
    Organization: Multi-tenant organization with resource limits and settings
    Thread: Conversation container with cost tracking and lifecycle management
    Conversation: Optimized user-assistant exchange storage
    ThreadStatus: Enum for thread lifecycle states

Design Philosophy:
    All domain models follow Domain-Driven Design principles:
    - Rich objects with behavior, not anemic data containers
    - Business rule validation enforced within the model
    - Clear invariants and constraints
    - Technology-agnostic implementation
    - Self-contained with minimal dependencies

Key Features:
    - Comprehensive validation and business rules
    - Token usage tracking for cost optimization
    - Optimized storage strategy for conversation data
    - Multi-tenant organization support
    - Flexible metadata and extensibility

Example Usage:
    ```python
    from echo.domain.models import User, Thread, Conversation, ThreadStatus

    user = User(user_id="alice", org_id="acme-corp")
    if not user.can_create_threads():
        raise ValueError("User cannot create new threads")

    thread = Thread(user_id=user.user_id, org_id=user.org_id)

    conversation = Conversation(
        thread_id=thread.thread_id,
        user_message="Calculate 2+2",
        assistant_message="2+2 equals 4",
        user_tokens=10,
        assistant_tokens=15
    )

    thread.add_conversation_tokens(conversation.user_tokens, conversation.assistant_tokens)

    estimated_cost = thread.get_cost_estimate()
    ```

These models provide the stable foundation for all business operations in Echo.
"""

from .conversation import Conversation
from .organization import Organization
from .thread import Thread, ThreadStatus
from .user import User

__all__ = [
    "Thread",
    "ThreadStatus",
    "Conversation",
    "User",
    "Organization",
]
