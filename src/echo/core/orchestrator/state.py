"""Echo Framework Agent State Management - Multi-Agent Conversation State.

This module defines the comprehensive state management system for Echo's multi-agent
conversations, providing structured tracking of conversation flow, agent routing,
tool usage, and context preservation across agent switches.

State Management Architecture:
    The agent state system provides several key capabilities:
    - Message History: Complete conversation history with LangChain message integration
    - Agent Tracking: Current agent identification and routing history
    - Hop Counters: Safety limits for agent switches and tool calls
    - Context Preservation: Plugin-specific context and metadata management
    - Session Management: User session identification and grouping

AgentState Schema:
    The AgentState TypedDict provides structured access to all conversation state:
    - messages: LangChain message sequence with automatic aggregation
    - current_agent: Active agent identifier for routing
    - agent_hops: Counter for agent switches (prevents infinite routing)
    - tool_hops: Counter for tool calls (prevents excessive tool usage)
    - session_id: User session identifier for conversation grouping
    - metadata: Flexible key-value storage for additional context

Safety Features:
    - Hop counters prevent infinite loops in agent routing
    - Configurable limits for maximum agent and tool hops
    - Automatic state validation and consistency checking
    - Context size monitoring to prevent memory issues

Example Usage:
    Creating and managing agent state:

    ```python
    from echo.core.orchestrator.state import AgentState
    from langchain_core.messages import HumanMessage, AIMessage

    state = AgentState(
        messages=[HumanMessage("Help me with math problems")],
        current_agent=None,
        agent_hops=0,
        tool_hops=0,
        session_id="user-123-session",
        metadata={"user_preferences": {"math_level": "advanced"}}
    )
    ```

    State utility functions:

    ```python
    last_tool = _last_assistant_tool_call_name(state)
    if last_tool == "calculator":
        pass

    if state["agent_hops"] > max_agent_hops:
        pass
    ```

Integration with LangGraph:
    The AgentState integrates seamlessly with LangGraph's state management:
    - TypedDict compatibility for structured access
    - Message aggregation through add_messages reducer
    - State transitions tracked automatically
    - Immutable state updates for consistency

The state management system ensures reliable, safe, and traceable multi-agent
conversations while providing the flexibility needed for complex agent interactions.
"""

from typing import TYPE_CHECKING, Annotated, Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from typing import TypedDict
else:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        from typing import TypedDict


class AgentState(TypedDict):
    """Comprehensive conversation state schema for multi-agent orchestration.

    This TypedDict defines the complete state structure for Echo's multi-agent
    conversations, providing structured access to conversation history, routing
    information, safety counters, and contextual metadata.

    State Components:
        Message Management:
            - messages: Complete conversation history using LangChain's message system
            - Automatic message aggregation through add_messages reducer
            - Support for all LangChain message types (Human, AI, System, Tool)

        Agent Coordination:
            - current_agent: Identifier of the currently active agent/plugin
            - agent_hops: Counter for agent switches (safety limit enforcement)
            - routing_decision: Most recent routing decision for debugging

        Tool Management:
            - tool_hops: Counter for individual tool calls (resource limit enforcement)
            - last_tool_call: Name of the most recent tool call for routing logic

        Session Context:
            - session_id: User session identifier for conversation grouping
            - metadata: Flexible metadata storage for user preferences and context
            - plugin_context: Plugin-specific ephemeral context preservation

        Parallel Processing:
            - parallel_results: Container for intermediate results from parallel operations

    Safety Features:
        The state schema includes built-in safety mechanisms:
        - agent_hops counter prevents infinite agent routing loops
        - tool_hops counter prevents excessive tool usage
        - State size monitoring for memory management
        - Validation hooks for consistency checking

    Example:
        ```python
        state = AgentState(
            messages=[
                HumanMessage("Calculate 2+2 then search for weather"),
                AIMessage("I'll help you with both calculations and weather info")
            ],
            current_agent="math_agent",
            agent_hops=1,
            tool_hops=2,
            last_tool_call="calculator",
            session_id="user-session-123",
            metadata={
                "user_preferences": {"units": "metric"},
                "conversation_context": "homework_help"
            },
            routing_decision="math_to_weather_transition"
        )
        ```
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: Optional[str]
    agent_hops: int
    tool_hops: int
    last_tool_call: Optional[str]
    session_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

    parallel_results: Optional[Dict[str, Any]]
    routing_decision: Optional[str]

    plugin_context: Optional[Dict[str, Any]]


def _inc_agent_hops(state: AgentState) -> int:
    """Return the next value for the agent hop counter."""
    return state.get("agent_hops", 0) + 1


def _inc_tool_hops(state: AgentState) -> int:
    """Return the next value for the tool hop counter."""
    return state.get("tool_hops", 0) + 1


def _last_assistant_tool_call_name(state: AgentState) -> Optional[str]:
    """Return the name of the most recent assistant tool call, if any."""
    messages = state.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            return message.tool_calls[-1]["name"]
    return None
