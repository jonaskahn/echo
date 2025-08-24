"""Conversation domain model for optimized conversation storage."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    """User–assistant exchange with token and metadata tracking.

    Represents one final exchange in a thread: the user's input and the
    assistant's final response. This model intentionally omits intermediate
    tool or planner messages to keep storage efficient while still enabling
    context reconstruction when needed.

    Attributes:
        id: Unique conversation identifier. Defaults to a UUID4 string.
        thread_id: Identifier of the parent thread that owns this conversation.
        user_message: The user's input content.
        assistant_message: The assistant's final response content.
        user_tokens: Number of input tokens consumed for this exchange.
        assistant_tokens: Number of output tokens produced for this exchange.
        created_at: Creation timestamp (UTC).
        metadata: Free-form metadata such as processing time, tools used, or
            agent hops.

    Example:
        Create a conversation and compute cost estimate:

        >>> conv = Conversation(
        ...     thread_id="thread-1",
        ...     user_message="What is 2+2?",
        ...     assistant_message="4",
        ...     user_tokens=8,
        ...     assistant_tokens=12,
        ... )
        >>> conv.total_tokens
        20
        >>> round(conv.get_cost_estimate(), 6)
        0.000044
    """

    id: Optional[str | int] = Field(default_factory=lambda: str(uuid.uuid4()), description="Conversation ID")
    thread_id: str | int = Field(description="Thread ID")
    user_message: str = Field(description="User input message")
    assistant_message: Optional[str] = Field(description="Assistant input message")
    user_tokens: Optional[int] = Field(default=0, description="User tokens")
    assistant_tokens: Optional[int] = Field(default=0, description="Assistant tokens")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Created_at")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    @property
    def total_tokens(self) -> int:
        """Return total tokens for this exchange.

        Returns:
            The sum of `user_tokens` and `assistant_tokens`.
        """
        return self.user_tokens + self.assistant_tokens

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata entry.

        Args:
            key: Metadata key to set.
            value: Value to associate with the key. Must be JSON-serializable
                if persisted as-is downstream.
        """
        self.metadata[key] = value

    def get_processing_time(self) -> Optional[float]:
        """Return processing time in seconds, if available.

        Returns:
            Processing time as a float in seconds, or None if not set in
            `metadata["processing_time"]`.
        """
        return self.metadata.get("processing_time")

    def get_tools_used(self) -> list:
        """Return tools used during processing.

        Returns:
            A list of tool names recorded in `metadata["tools_used"]`. Returns
            an empty list when not present.
        """
        return self.metadata.get("tools_used", [])

    def get_agent_hops(self) -> int:
        """Return number of agent-to-agent hops.

        Returns:
            Integer count from `metadata["agent_hops"]`. Defaults to 0 when
            not present.
        """
        return self.metadata.get("agent_hops", 0)

    def get_cost_estimate(self, cost_per_1k_input: float = 0.001, cost_per_1k_output: float = 0.003) -> float:
        """Estimate token cost for this exchange.

        Args:
            cost_per_1k_input: Price per 1,000 input tokens.
            cost_per_1k_output: Price per 1,000 output tokens.

        Returns:
            Estimated USD cost based on `user_tokens` and `assistant_tokens`.
        """
        input_cost = (self.user_tokens / 1000) * cost_per_1k_input
        output_cost = (self.assistant_tokens / 1000) * cost_per_1k_output
        return input_cost + output_cost

    def to_langgraph_messages(self) -> list:
        """Convert to LangGraph message sequence.

        Returns:
            A list containing a HumanMessage (user content) and an AIMessage
            (assistant content), suitable for context reconstruction.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        return [HumanMessage(content=self.user_message), AIMessage(content=self.assistant_message)]

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary.

        Returns:
            A dict with primitive types suitable for JSON serialization. The
            `created_at` field is rendered in ISO 8601 format.
        """
        return {
            "conversation_id": self.id,
            "thread_id": self.thread_id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "user_tokens": self.user_tokens,
            "assistant_tokens": self.assistant_tokens,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Deserialize from a dictionary.

        Args:
            data: Mapping with keys: id, thread_id, user_message,
                assistant_message, user_tokens, assistant_tokens, created_at
                (ISO 8601), and optional metadata.

        Returns:
            A new Conversation instance.

        Raises:
            KeyError: If required keys are missing from the mapping.
            ValueError: If `created_at` is not a valid ISO 8601 timestamp.
        """
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            user_message=data["user_message"],
            assistant_message=data["assistant_message"],
            user_tokens=data["user_tokens"],
            assistant_tokens=data["assistant_tokens"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, thread_id={self.thread_id}, tokens={self.total_tokens})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Conversation):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
