"""Echo Framework Data Transfer Objects - Cross-Layer Communication Contracts.

This module provides comprehensive Data Transfer Objects (DTOs) for the Echo multi-agent
AI framework. DTOs serve as contracts between different architectural layers and external
systems, ensuring data integrity and providing clear API documentation.

Architecture Purpose:
    DTOs act as the boundary between different layers of the system:
    - API Layer ↔ Application Layer: Request/response validation and serialization
    - Application Layer ↔ Infrastructure Layer: Data consistency and mapping
    - Internal Services ↔ External Systems: Integration contract enforcement

Key Features:
    - Pydantic-based validation with comprehensive field constraints
    - Rich OpenAPI documentation generation with examples
    - Type safety across layer boundaries
    - Backward compatibility and versioning support
    - Comprehensive error handling with clear validation messages

DTO Categories:
    Chat DTOs: Conversation processing and message exchange contracts
    Thread DTOs: Conversation thread management and lifecycle operations
    Analytics DTOs: System monitoring, reporting, and metrics collection

Design Principles:
    - Immutable data contracts between system boundaries
    - Clear separation between request and response schemas
    - Rich validation rules that encode business requirements
    - Comprehensive documentation with realistic examples
    - Consistent naming and structure across all DTOs

Example Usage:
    Processing chat requests with validation:

    ```python
    from echo.domain.dtos import ChatRequest, ChatResponse, TokenUsage

    chat_request = ChatRequest(
        message="Help me solve this math problem",
        thread_id="user-123-session",
        user_id="user-123",
        metadata={"context": "homework"}
    )

    response = ChatResponse(
        response="I'd be happy to help with your math problem...",
        thread_id=chat_request.thread_id,
        conversation_id="conv-456",
        token_usage=TokenUsage(
            input_tokens=25,
            output_tokens=85,
            total_tokens=110
        ),
        metadata={
            "agent_used": "math_agent",
            "processing_time": 1.23
        }
    )
    ```

    Analytics and monitoring:

    ```python
    from echo.domain.dtos import AnalyticsRequest, SystemHealthResponse

    analytics_request = AnalyticsRequest(
        start_date="2024-01-01",
        end_date="2024-01-31",
        org_id="acme-corp",
        group_by="day"
    )

    health_response = SystemHealthResponse(
        status="operational",
        uptime=86400.0,
        active_threads=156,
        storage_usage={"efficiency_percentage": 87.5}
    )
    ```

All DTOs provide comprehensive OpenAPI documentation for automatic API documentation
generation and client SDK creation.
"""

from .analytics_dtos import (
    AnalyticsRequest,
    AnalyticsResponse,
    SystemHealthResponse,
    TokenUsageStats,
    TopUser,
    UsageByPeriod,
)
from .chat_dtos import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    TokenUsage,
)
from .thread_dtos import (
    ThreadCreateRequest,
    ThreadListRequest,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdateRequest,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "TokenUsage",
    "ConversationResponse",
    "ThreadCreateRequest",
    "ThreadResponse",
    "ThreadListRequest",
    "ThreadListResponse",
    "ThreadUpdateRequest",
    "AnalyticsRequest",
    "AnalyticsResponse",
    "SystemHealthResponse",
    "TokenUsageStats",
    "UsageByPeriod",
    "TopUser",
]
