"""Echo Framework Application Services - High-Level Business Logic Coordination.

This module provides the application service layer for the Echo multi-agent AI framework,
implementing high-level business logic that coordinates between the domain models
and infrastructure services.

Application Service Architecture:
    The service layer implements the Application Service pattern from Domain-Driven Design:
    - Orchestrates complex business workflows involving multiple domain objects
    - Coordinates between infrastructure services (repositories, orchestrator, etc.)
    - Maintains transaction boundaries and ensures data consistency
    - Provides clean interfaces for the API layer to consume
    - Implements cross-cutting concerns like logging, monitoring, and error handling

Key Services:
    ConversationService: Complete conversation lifecycle management
        - Thread creation and management
        - Conversation history optimization (significant storage reduction)
        - Multi-agent orchestration coordination
        - Token usage tracking and cost optimization

    OrchestratorService: LangGraph coordination wrapper
        - Multi-agent routing and execution
        - Tool call management with safety limits
        - Performance monitoring and metrics collection
        - Error handling and recovery mechanisms

Service Layer Responsibilities:
    Business Workflow Orchestration:
        - Coordinate complex multi-step business processes
        - Ensure proper sequencing of operations
        - Handle transaction boundaries and rollback scenarios
        - Implement business rules and validation logic

    Infrastructure Coordination:
        - Repository pattern usage for data access
        - Service discovery and dependency injection
        - External service integration (LLM providers, plugins)
        - Caching and performance optimization strategies

    Cross-Cutting Concerns:
        - Comprehensive logging and monitoring
        - Error handling with proper exception mapping
        - Performance tracking and bottleneck identification
        - Security and authorization enforcement

Example Usage:
    Conversation service integration:

    ```python
    from echo.core.services import ConversationService, OrchestratorService
    from echo.domain.dtos.chat_dtos import ChatRequest, ChatResponse

    conversation_service = ConversationService(
        thread_repository=thread_repo,
        conversation_repository=conv_repo,
        orchestrator=orchestrator
    )

    chat_request = ChatRequest(
        message="Help me solve this math problem: 2x + 5 = 15",
        thread_id="user-123-session",
        user_id="user-123"
    )

    chat_response = await conversation_service.process_message(chat_request)

    print(f"Response: {chat_response.response}")
    print(f"Tokens used: {chat_response.token_usage.total_tokens}")
    ```

    Orchestrator service for custom workflows:

    ```python
    orchestrator_service = OrchestratorService(orchestrator)

    response = await orchestrator_service.process_message(
        message="Continue our math discussion",
        conversation_history=[previous_turns],
        session_id="user-session"
    )

    print(f"Agent hops: {response.agent_hops}")
    print(f"Tools used: {response.tools_used}")
    print(f"Processing time: {response.processing_time}s")
    ```

Design Patterns:
    - Application Service Pattern: High-level workflow coordination
    - Dependency Injection: Clean service boundaries and testability
    - Repository Pattern: Abstracted data access through interfaces
    - Factory Pattern: Service creation and configuration management

The application service layer ensures that complex business logic is properly
encapsulated while providing clean, testable interfaces for the API layer.
"""

from .conversation_service import ConversationService
from .orchestrator_service import OrchestratorResponse, OrchestratorService

__all__ = [
    "ConversationService",
    "OrchestratorService",
    "OrchestratorResponse",
]
