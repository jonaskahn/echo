"""Echo Framework Domain Layer - Business Models and Core Logic.

This layer contains the core business models, domain transfer objects (DTOs), and
business logic for the Echo multi-agent AI framework. The domain layer follows
Domain-Driven Design (DDD) principles and is independent of infrastructure concerns.

Architecture:
    The domain layer represents the heart of the business logic with:
    - Pure business models with no external dependencies
    - Rich domain objects with behavior, not just data
    - Business rule validation and enforcement
    - Domain events and business invariants
    - Clear separation from infrastructure and presentation layers

Components:
    models/: Core business entities (User, Thread, ConversationTurn, Organization)
    dtos/: Data transfer objects for cross-boundary communication
    events/: Domain events for business process coordination (future)

Design Principles:
    - Business models contain behavior, not just data
    - Validation and business rules are enforced within domain objects
    - Models are technology-agnostic and testable in isolation
    - Clear boundaries between domain concepts and external concerns

Key Business Concepts:
    User: System users with organization affiliation and activity tracking
    Thread: Conversation containers with cost tracking and lifecycle management
    ConversationTurn: Optimized storage of user-assistant exchanges
    Organization: Multi-tenant organization management with resource limits

Example:
    Working with domain models:

    ```python
    from echo.domain.models import User, Thread, ConversationTurn

    user = User(user_id="user-123", org_id="acme-corp")
    thread = Thread(user_id=user.user_id, org_id=user.org_id)

    if not user.can_create_threads():
        raise ValueError("User cannot create threads")

    turn = ConversationTurn(
        thread_id=thread.thread_id,
        user_message="Hello, how are you?",
        assistant_message="I'm doing well, thank you!",
        user_tokens=15,
        assistant_tokens=25
    )

    thread.add_turn_tokens(turn.user_tokens, turn.assistant_tokens)

    thread_cost = thread.get_cost_estimate()
    turn_cost = turn.get_cost_estimate()
    ```

The domain layer ensures business logic is centralized, testable, and independent
of infrastructure changes, providing a stable foundation for the entire system.
"""
