"""Echo API Request/Response Schema Models with Validation.

This module defines Pydantic models used throughout the Echo API for request
validation, response serialization, and OpenAPI documentation generation.
All models include comprehensive field validation and documentation.

The schemas follow REST API best practices with:
- Clear field validation and constraints
- Comprehensive documentation for OpenAPI generation
- Type safety with Python typing system
- Consistent naming conventions across the API

Model Categories:
    Chat Models: Request/response schemas for conversation endpoints
    Plugin Models: Schemas for plugin management and discovery
    System Models: Schemas for system monitoring and health checks

Example:
    Using schemas in FastAPI endpoints:

    ```python
    from fastapi import APIRouter
    from .schemas import ChatRequest, ChatResponse

    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            response="Hello!",
            session_id=request.session_id or "new-session"
        )
    ```
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat message processing.

    Represents a user message to be processed by the Echo multi-agent system.
    Supports optional session management and metadata for context preservation.

    Attributes:
        message: User input message to be processed by agents
        session_id: Optional session identifier for conversation threading
        metadata: Optional additional context or configuration data

    Example:
        ```json
        {
          "message": "Help me solve 2x + 5 = 15",
          "session_id": "user-123-math-session",
          "metadata": {
            "preferred_agent": "math_agent",
            "difficulty_level": "intermediate"
          }
        }
        ```
    """

    message: str = Field(
        ..., description="User message to be processed by the multi-agent system", min_length=1, max_length=10000
    )
    session_id: Optional[str] = Field(
        default=None, description="Optional session identifier for conversation threading", max_length=255
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata for context or configuration"
    )


class ChatResponse(BaseModel):
    """Response schema for processed chat messages.

    Contains the agent's response along with session information and optional
    metadata about the processing, such as which agent was used and token usage.

    Attributes:
        response: Agent's response to the user message
        session_id: Session identifier for conversation threading
        metadata: Optional processing metadata (agent used, tokens, etc.)

    Example:
        ```json
        {
          "response": "To solve 2x + 5 = 15, subtract 5 from both sides...",
          "session_id": "user-123-math-session",
          "metadata": {
            "agent_used": "math_agent",
            "tokens_used": 150,
            "processing_time_ms": 245
          }
        }
        ```
    """

    response: str = Field(..., description="Agent's response to the user message")
    session_id: str = Field(..., description="Session identifier for conversation threading")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional processing metadata (agent, tokens, timing)"
    )


class PluginInfo(BaseModel):
    """Schema for plugin information and status.

    Represents metadata and current status for a plugin in the Echo system.
    Used for plugin discovery, monitoring, and management operations.

    Attributes:
        name: Plugin identifier name
        version: Plugin version string
        description: Human-readable plugin description
        capabilities: List of capabilities provided by the plugin
        status: Current plugin health status (healthy/failed)

    Example:
        ```json
        {
          "name": "math_agent",
          "version": "1.2.0",
          "description": "Advanced mathematical computation and problem solving",
          "capabilities": ["arithmetic", "algebra", "calculus", "statistics"],
          "status": "healthy"
        }
        ```
    """

    name: str = Field(..., description="Plugin identifier name")
    version: str = Field(..., description="Plugin version in semantic versioning format")
    description: str = Field(..., description="Human-readable description of plugin functionality")
    capabilities: List[str] = Field(..., description="List of capabilities or features provided by the plugin")
    status: str = Field(..., description="Current plugin health status", pattern="^(healthy|failed)$")


class SystemStatus(BaseModel):
    """Schema for overall system status and health information.

    Provides a comprehensive view of the Echo system's current state including
    plugin status, active sessions, and overall health metrics.

    Attributes:
        status: Overall system status indicator
        available_plugins: List of all discovered plugins
        healthy_plugins: List of currently healthy plugins
        failed_plugins: List of plugins with failures
        total_sessions: Number of active conversation sessions

    Example:
        ```json
        {
          "status": "healthy",
          "available_plugins": ["math_agent", "search_agent", "weather_agent"],
          "healthy_plugins": ["math_agent", "search_agent"],
          "failed_plugins": ["weather_agent"],
          "total_sessions": 42
        }
        ```
    """

    status: str = Field(..., description="Overall system health status", pattern="^(healthy|degraded|failed)$")
    available_plugins: List[str] = Field(..., description="List of all discovered plugin names")
    healthy_plugins: List[str] = Field(..., description="List of currently healthy plugin names")
    failed_plugins: List[str] = Field(..., description="List of plugin names with failures")
    total_sessions: int = Field(..., description="Number of active conversation sessions", ge=0)
