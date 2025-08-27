"""Echo Framework Core Layer - Multi-Agent Orchestration and Business Logic.

This module provides the core business logic and orchestration capabilities for the
Echo multi-agent AI framework. It implements intelligent conversation routing,
state management, and multi-agent coordination through LangGraph-based workflows.

Core Architecture:
    The core layer serves as the heart of Echo's multi-agent capabilities:
    - Multi-Agent Orchestration: LangGraph-based conversation routing and coordination
    - State Management: Persistent conversation state across agent switches
    - Business Logic: Application-level services and workflow management
    - Safety Features: Infinite loop prevention and resource limit enforcement

Key Components:
    MultiAgentOrchestrator: LangGraph-based conversation orchestration engine
    AgentState: Comprehensive state management for multi-agent conversations
    Service Container: Dependency injection and service lifecycle management
    Application Services: High-level business logic and workflow coordination

Multi-Agent Orchestration Features:
    - Intelligent agent routing based on conversation content and context
    - Plugin-based agent discovery and dynamic registration
    - Tool routing and execution with safety limits
    - Conversation state persistence across agent switches
    - Infinite loop prevention through hop counter management
    - Comprehensive logging and monitoring of agent interactions

Agent Coordination Patterns:
    Sequential Routing:
        User Input → Coordinator → Agent Selection → Tool Execution → Response

    Multi-Hop Conversations:
        Agent A → Tool Call → Agent B → Tool Call → Final Response

    Plugin Integration:
        Core System ↔ Plugin Agents ↔ Plugin Tools

    State Preservation:
        Conversation context maintained across all agent switches

Example Usage:
    Basic orchestrator setup:

    ```python
    from echo.core import MultiAgentOrchestrator, AgentState
    from echo.infrastructure.llm import LLMModelFactory
    from echo.infrastructure.plugins import SDKPluginManager

    llm_factory = LLMModelFactory(settings)
    plugin_manager = SDKPluginManager(settings)

    orchestrator = MultiAgentOrchestrator(
        settings=settings,
        llm_factory=llm_factory,
        plugin_manager=plugin_manager
    )

    initial_state = AgentState(
        messages=[HumanMessage("Help me with math and then search the web")],
        agent_hops=0,
        tool_hops=0
    )

    final_state = await orchestrator.process_conversation(initial_state)
    response_message = final_state["messages"][-1]
    ```

    Agent state management:

    ```python
    state = AgentState(
        messages=[HumanMessage("Calculate 2+2")],
        current_agent=None,
        agent_hops=0,
        tool_hops=0,
        metadata={"user_id": "user-123"}
    )
    ```

Safety and Performance Features:
    - Maximum agent hop limits to prevent infinite routing loops
    - Maximum tool hop limits to prevent excessive tool usage
    - Conversation state size monitoring and optimization
    - Comprehensive error handling and recovery mechanisms
    - Performance monitoring and bottleneck identification

The core layer provides the intelligent orchestration that makes Echo's multi-agent
system reliable, scalable, and safe for production deployments.
"""

from .orchestrator.coordinator import MultiAgentOrchestrator
from .orchestrator.state import AgentState

__all__ = ["AgentState", "MultiAgentOrchestrator"]
