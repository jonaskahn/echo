"""Test cases for the AgentState and state utility functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from echo.core.orchestrator.state import (
    AgentState,
    _inc_agent_hops,
    _inc_tool_hops,
    _last_assistant_tool_call_name,
)


class TestAgentState:
    """Test cases for AgentState TypedDict."""

    def test_agent_state_creation(self):
        """Test creating a complete AgentState."""
        state = AgentState(
            messages=[
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            current_agent="math_agent",
            agent_hops=1,
            tool_hops=2,
            last_tool_call="calculator",
            session_id="test-session-123",
            metadata={"user_preferences": {"math_level": "basic"}},
            parallel_results={"intermediate": "data"},
            routing_decision="math_calculation",
            plugin_context={"math_operation": "addition"},
        )

        assert len(state["messages"]) == 2
        assert state["current_agent"] == "math_agent"
        assert state["agent_hops"] == 1
        assert state["tool_hops"] == 2
        assert state["last_tool_call"] == "calculator"
        assert state["session_id"] == "test-session-123"
        assert state["metadata"]["user_preferences"]["math_level"] == "basic"
        assert state["parallel_results"]["intermediate"] == "data"
        assert state["routing_decision"] == "math_calculation"
        assert state["plugin_context"]["math_operation"] == "addition"

    def test_agent_state_minimal(self):
        """Test creating a minimal AgentState with required fields only."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        assert state["messages"] == []
        assert state["current_agent"] is None
        assert state["agent_hops"] == 0
        assert state["tool_hops"] == 0
        assert state["last_tool_call"] is None
        assert state["session_id"] is None
        assert state["metadata"] is None
        assert state["parallel_results"] is None
        assert state["routing_decision"] is None
        assert state["plugin_context"] is None

    def test_agent_state_with_mixed_message_types(self):
        """Test AgentState with different message types."""
        state = AgentState(
            messages=[
                HumanMessage(content="User question"),
                SystemMessage(content="System instruction"),
                AIMessage(content="AI response"),
                AIMessage(content="Another AI response", tool_calls=[{"id": "call_1", "name": "tool1", "args": {}}]),
            ],
            current_agent="test_agent",
            agent_hops=1,
            tool_hops=1,
            last_tool_call="tool1",
            session_id="session-1",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        assert len(state["messages"]) == 4
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], SystemMessage)
        assert isinstance(state["messages"][2], AIMessage)
        assert isinstance(state["messages"][3], AIMessage)
        assert state["messages"][3].tool_calls[0]["name"] == "tool1"

    def test_agent_state_with_complex_metadata(self):
        """Test AgentState with complex metadata structure."""
        complex_metadata = {
            "user_preferences": {
                "language": "en",
                "math_level": "advanced",
                "ui_theme": "dark",
            },
            "conversation_context": {
                "topic": "mathematics",
                "difficulty": "intermediate",
                "session_start": "2024-01-01T00:00:00Z",
            },
            "system_flags": {
                "debug_mode": True,
                "experimental_features": False,
            },
        }

        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id="test-session",
            metadata=complex_metadata,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        assert state["metadata"]["user_preferences"]["language"] == "en"
        assert state["metadata"]["conversation_context"]["topic"] == "mathematics"
        assert state["metadata"]["system_flags"]["debug_mode"] is True

    def test_agent_state_with_plugin_context(self):
        """Test AgentState with plugin-specific context."""
        plugin_context = {
            "math_agent": {
                "last_operation": "addition",
                "variables": {"x": 5, "y": 3},
                "history": ["2+2=4", "5+3=8"],
            },
            "search_agent": {
                "last_query": "weather forecast",
                "results_count": 10,
            },
        }

        state = AgentState(
            messages=[],
            current_agent="math_agent",
            agent_hops=1,
            tool_hops=2,
            last_tool_call="calculator",
            session_id="test-session",
            metadata=None,
            parallel_results=None,
            routing_decision="math_to_search",
            plugin_context=plugin_context,
        )

        assert state["plugin_context"]["math_agent"]["last_operation"] == "addition"
        assert state["plugin_context"]["math_agent"]["variables"]["x"] == 5
        assert state["plugin_context"]["search_agent"]["last_query"] == "weather forecast"

    def test_agent_state_with_parallel_results(self):
        """Test AgentState with parallel processing results."""
        parallel_results = {
            "math_calculation": {
                "result": 42,
                "execution_time": 0.1,
                "status": "completed",
            },
            "web_search": {
                "results": ["result1", "result2"],
                "execution_time": 0.5,
                "status": "completed",
            },
            "pending_task": {
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
        }

        state = AgentState(
            messages=[],
            current_agent="coordinator",
            agent_hops=2,
            tool_hops=3,
            last_tool_call=None,
            session_id="test-session",
            metadata=None,
            parallel_results=parallel_results,
            routing_decision="parallel_execution",
            plugin_context=None,
        )

        assert state["parallel_results"]["math_calculation"]["result"] == 42
        assert state["parallel_results"]["web_search"]["status"] == "completed"
        assert state["parallel_results"]["pending_task"]["status"] == "running"


class TestStateUtilityFunctions:
    """Test cases for state utility functions."""

    def test_inc_agent_hops_with_existing_value(self):
        """Test incrementing agent hops with existing value."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=5,
            tool_hops=0,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _inc_agent_hops(state)

        assert result == 6

    def test_inc_agent_hops_without_existing_value(self):
        """Test incrementing agent hops without existing value."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _inc_agent_hops(state)

        assert result == 1

    def test_inc_agent_hops_with_missing_key(self):
        """Test incrementing agent hops when key is missing."""
        state = {
            "messages": [],
            "current_agent": None,
            "tool_hops": 0,
            "last_tool_call": None,
            "session_id": None,
            "metadata": None,
            "parallel_results": None,
            "routing_decision": None,
            "plugin_context": None,
        }

        result = _inc_agent_hops(state)

        assert result == 1

    def test_inc_tool_hops_with_existing_value(self):
        """Test incrementing tool hops with existing value."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=3,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _inc_tool_hops(state)

        assert result == 4

    def test_inc_tool_hops_without_existing_value(self):
        """Test incrementing tool hops without existing value."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _inc_tool_hops(state)

        assert result == 1

    def test_inc_tool_hops_with_missing_key(self):
        """Test incrementing tool hops when key is missing."""
        state = {
            "messages": [],
            "current_agent": None,
            "agent_hops": 0,
            "last_tool_call": None,
            "session_id": None,
            "metadata": None,
            "parallel_results": None,
            "routing_decision": None,
            "plugin_context": None,
        }

        result = _inc_tool_hops(state)

        assert result == 1

    def test_last_assistant_tool_call_name_with_tool_calls(self):
        """Test getting last assistant tool call name with tool calls."""
        state = AgentState(
            messages=[
                HumanMessage(content="Calculate 2+2"),
                AIMessage(
                    content="I'll calculate that", tool_calls=[{"id": "call_1", "name": "calculator", "args": {}}]
                ),
                AIMessage(content="The result is 4"),
                AIMessage(content="Another response", tool_calls=[{"id": "call_2", "name": "formatter", "args": {}}]),
            ],
            current_agent="math_agent",
            agent_hops=1,
            tool_hops=2,
            last_tool_call="calculator",
            session_id="test-session",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _last_assistant_tool_call_name(state)

        assert result == "formatter"

    def test_last_assistant_tool_call_name_without_tool_calls(self):
        """Test getting last assistant tool call name without tool calls."""
        state = AgentState(
            messages=[
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                SystemMessage(content="System message"),
            ],
            current_agent="test_agent",
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id="test-session",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _last_assistant_tool_call_name(state)

        assert result is None

    def test_last_assistant_tool_call_name_with_empty_messages(self):
        """Test getting last assistant tool call name with empty messages."""
        state = AgentState(
            messages=[],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id=None,
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _last_assistant_tool_call_name(state)

        assert result is None

    def test_last_assistant_tool_call_name_with_mixed_message_types(self):
        """Test getting last assistant tool call name with mixed message types."""
        state = AgentState(
            messages=[
                HumanMessage(content="User input"),
                SystemMessage(content="System instruction"),
                AIMessage(content="AI response without tools"),
                AIMessage(content="AI response with tools", tool_calls=[{"id": "call_1", "name": "tool1", "args": {}}]),
                HumanMessage(content="Another user input"),
                AIMessage(content="Final AI response", tool_calls=[{"id": "call_2", "name": "tool2", "args": {}}]),
            ],
            current_agent="test_agent",
            agent_hops=1,
            tool_hops=2,
            last_tool_call="tool1",
            session_id="test-session",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _last_assistant_tool_call_name(state)

        assert result == "tool2"

    def test_last_assistant_tool_call_name_with_multiple_tool_calls(self):
        """Test getting last assistant tool call name with multiple tool calls in one message."""
        state = AgentState(
            messages=[
                HumanMessage(content="Complex request"),
                AIMessage(
                    content="I'll handle this",
                    tool_calls=[
                        {"id": "call_1", "name": "tool1", "args": {}},
                        {"id": "call_2", "name": "tool2", "args": {}},
                        {"id": "call_3", "name": "tool3", "args": {}},
                    ],
                ),
            ],
            current_agent="test_agent",
            agent_hops=1,
            tool_hops=3,
            last_tool_call="tool1",
            session_id="test-session",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        result = _last_assistant_tool_call_name(state)

        assert result == "tool3"

    def test_last_assistant_tool_call_name_with_missing_key(self):
        """Test getting last assistant tool call name when messages key is missing."""
        state = {
            "current_agent": "test_agent",
            "agent_hops": 1,
            "tool_hops": 0,
            "last_tool_call": None,
            "session_id": "test-session",
            "metadata": None,
            "parallel_results": None,
            "routing_decision": None,
            "plugin_context": None,
        }

        result = _last_assistant_tool_call_name(state)

        assert result is None

    def test_last_assistant_tool_call_name_with_non_list_messages(self):
        """Test getting last assistant tool call name when messages is not a list."""
        state = {
            "messages": "not a list",
            "current_agent": "test_agent",
            "agent_hops": 1,
            "tool_hops": 0,
            "last_tool_call": None,
            "session_id": "test-session",
            "metadata": None,
            "parallel_results": None,
            "routing_decision": None,
            "plugin_context": None,
        }

        result = _last_assistant_tool_call_name(state)

        assert result is None


class TestAgentStateIntegration:
    """Integration tests for AgentState functionality."""

    def test_state_evolution_through_conversation(self):
        """Test how AgentState evolves through a conversation."""
        # Initial state
        state = AgentState(
            messages=[
                HumanMessage(content="What is 2+2?"),
            ],
            current_agent=None,
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id="user-123",
            metadata={"user_preferences": {"math_level": "basic"}},
            parallel_results=None,
            routing_decision=None,
            plugin_context=None,
        )

        # After coordinator decision
        state["current_agent"] = "math_agent"
        state["agent_hops"] = _inc_agent_hops(state)
        state["routing_decision"] = "route_to_math"
        state["messages"].append(
            AIMessage(
                content="I'll help you with math", tool_calls=[{"id": "call_1", "name": "calculator", "args": {}}]
            )
        )

        # After tool execution
        state["tool_hops"] = _inc_tool_hops(state)
        state["last_tool_call"] = _last_assistant_tool_call_name(state)
        state["messages"].append(AIMessage(content="4", tool_call_id="call_123"))

        # After finalization
        state["current_agent"] = None
        state["agent_hops"] = _inc_agent_hops(state)
        state["routing_decision"] = "finalize"
        state["messages"].append(AIMessage(content="The answer is 4."))

        # Verify final state
        assert state["agent_hops"] == 2
        assert state["tool_hops"] == 1
        assert state["last_tool_call"] == "calculator"
        assert len(state["messages"]) == 4
        assert state["routing_decision"] == "finalize"
        assert state["current_agent"] is None

    def test_state_with_complex_plugin_context_evolution(self):
        """Test AgentState with evolving plugin context."""
        state = AgentState(
            messages=[],
            current_agent="math_agent",
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id="user-123",
            metadata=None,
            parallel_results=None,
            routing_decision=None,
            plugin_context={
                "math_agent": {
                    "operations": [],
                    "variables": {},
                }
            },
        )

        # Add math operation
        state["plugin_context"]["math_agent"]["operations"].append("2+2=4")
        state["plugin_context"]["math_agent"]["variables"]["x"] = 2
        state["plugin_context"]["math_agent"]["variables"]["y"] = 2

        # Switch to search agent
        state["current_agent"] = "search_agent"
        state["agent_hops"] = _inc_agent_hops(state)
        state["plugin_context"]["search_agent"] = {
            "queries": [],
            "results": {},
        }

        # Add search query
        state["plugin_context"]["search_agent"]["queries"].append("mathematical operations")

        # Verify state
        assert state["agent_hops"] == 1
        assert state["current_agent"] == "search_agent"
        assert len(state["plugin_context"]["math_agent"]["operations"]) == 1
        assert state["plugin_context"]["math_agent"]["variables"]["x"] == 2
        assert len(state["plugin_context"]["search_agent"]["queries"]) == 1

    def test_state_parallel_processing_simulation(self):
        """Test AgentState with parallel processing simulation."""
        state = AgentState(
            messages=[],
            current_agent="coordinator",
            agent_hops=0,
            tool_hops=0,
            last_tool_call=None,
            session_id="user-123",
            metadata=None,
            parallel_results={
                "math_task": {"status": "running"},
                "search_task": {"status": "pending"},
            },
            routing_decision="parallel_execution",
            plugin_context=None,
        )

        # Update parallel results
        state["parallel_results"]["math_task"]["status"] = "completed"
        state["parallel_results"]["math_task"]["result"] = 42
        state["parallel_results"]["search_task"]["status"] = "running"

        # Add more parallel tasks
        state["parallel_results"]["format_task"] = {"status": "pending"}

        # Verify parallel processing state
        assert state["parallel_results"]["math_task"]["status"] == "completed"
        assert state["parallel_results"]["math_task"]["result"] == 42
        assert state["parallel_results"]["search_task"]["status"] == "running"
        assert state["parallel_results"]["format_task"]["status"] == "pending"
        assert len(state["parallel_results"]) == 3
