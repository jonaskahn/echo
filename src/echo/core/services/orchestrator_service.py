"""Echo Framework Orchestrator Service - LangGraph Coordination Wrapper.

This module provides a high-level wrapper around the MultiAgentOrchestrator for application
service integration. It handles the complexities of LangGraph state management, conversation
history preparation, and response processing while providing clean interfaces for the
ConversationService.

Service Architecture:
    The OrchestratorService acts as an adapter between the application service layer and
    the core orchestration engine:
    - State Management: AgentState preparation and result extraction
    - History Integration: Conversation context reconstruction from optimized storage
    - Performance Monitoring: Comprehensive metrics collection and analysis
    - Error Handling: Robust exception management and recovery mechanisms

Key Features:
    LangGraph Integration:
        - AgentState preparation from conversation history
        - Multi-agent routing and tool execution coordination
        - Safety limit enforcement (agent hops, tool hops)
        - Comprehensive state tracking and debugging support

    Performance Optimization:
        - Processing time measurement and reporting
        - Token usage tracking and cost calculation
        - Agent routing efficiency analysis
        - Tool usage pattern monitoring

    Response Processing:
        - Structured response extraction from LangGraph state
        - Metadata enrichment with processing metrics
        - Error handling and user-friendly error messages
        - Debugging information collection for development

The orchestrator service ensures reliable multi-agent coordination while providing
detailed insights into conversation processing performance and behavior.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ...domain.models.conversation import Conversation
from ..orchestrator.coordinator import MultiAgentOrchestrator
from ..orchestrator.state import AgentState

logger = logging.getLogger(__name__)


class OrchestratorResponse:
    """Comprehensive response container for orchestrator processing results.

    This class encapsulates all the information generated during multi-agent conversation
    processing, including the response content, performance metrics, routing information,
    and debugging data.

    Response Components:
        Content Data:
            - response: The final AI response to the user
            - input_tokens: Tokens used for user input and conversation context
            - output_tokens: Tokens generated in the AI response

        Performance Metrics:
            - processing_time: Total time spent in orchestration (seconds)
            - agent_hops: Number of agent switches during processing
            - tool_hops: Number of tool calls executed

        Routing Information:
            - tools_used: List of tools that were called during processing
            - routing_history: Sequence of agent routing decisions
            - error: Any error message if processing failed

    Example:
        ```python
        response = OrchestratorResponse(
            response="The solution is x = 1 or x = -6",
            input_tokens=45,
            output_tokens=78,
            processing_time=2.34,
            agent_hops=2,
            tool_hops=3,
            tools_used=["calculator", "equation_solver"],
            routing_history=["coordinator", "math_agent", "finalizer"]
        )

        print(f"Cost: {response.total_tokens} tokens in {response.processing_time}s")
        print(f"Efficiency: {len(response.tools_used)} tools, {response.agent_hops} agents")
        ```
    """

    def __init__(
        self,
        response: str,
        input_tokens: int,
        output_tokens: int,
        agent_hops: int = 0,
        tool_hops: int = 0,
        processing_time: float = 0.0,
        tools_used: Optional[List[str]] = None,
        routing_history: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.response = response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.agent_hops = agent_hops
        self.tool_hops = tool_hops
        self.processing_time = processing_time
        self.tools_used = tools_used or []
        self.routing_history = routing_history or []
        self.error = error

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "agent_hops": self.agent_hops,
            "tool_hops": self.tool_hops,
            "processing_time": self.processing_time,
            "tools_used": self.tools_used,
            "routing_history": self.routing_history,
            "error": self.error,
        }


class OrchestratorService:
    """Service wrapper for LangGraph orchestration.

    Provides a clean interface for:
    - Context preparation from conversation history
    - Orchestrator execution with monitoring
    - Response extraction and token tracking
    - Error handling and recovery
    """

    def __init__(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator

    async def process_with_context(
        self,
        thread_id: str,
        message: str,
        conversation_history: List[Conversation],
        user_id: str = "anonymous",
        org_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """Process a message with conversation context.

        Args:
            thread_id: Conversation thread identifier
            message: User message to process
            conversation_history: Previous conversation turns for context
            user_id: User identifier
            org_id: Organization identifier
            metadata: Additional metadata

        Returns:
            OrchestratorResponse with processing results
        """
        start_time = time.time()

        try:
            langgraph_context = self._prepare_context(conversation_history, message)

            state: AgentState = {
                "messages": langgraph_context,
                "agent_hops": 0,
                "tool_hops": 0,
                "session_id": thread_id,
                "plugin_context": {},
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "organization_id": org_id,
                    "checkpoint_ns": f"org_{org_id}/user_{user_id}",
                    **(metadata or {}),
                },
            }

            logger.debug(f"Processing message for thread {thread_id} with {len(langgraph_context)} context messages")

            result = await self.orchestrator.ask(state)

            response_text = self._extract_response_text(result)
            processing_time = time.time() - start_time

            input_tokens = self._estimate_input_tokens(langgraph_context)
            output_tokens = self._estimate_output_tokens(response_text)

            routing_history = result.get("plugin_context", {}).get("routing_history", [])
            routing_history = result.get("plugin_context", {}).get("routing_history", [])
            tools_used = self._extract_tools_used(result)

            logger.info(f"Orchestrator completed for thread {thread_id}: {response_text[:100]}...")

            return OrchestratorResponse(
                response=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent_hops=result.get("agent_hops", 0),
                tool_hops=result.get("tool_hops", 0),
                processing_time=processing_time,
                tools_used=tools_used,
                routing_history=routing_history,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = f"Orchestrator error: {str(e)}"

            logger.error(f"Orchestrator failed for thread {thread_id}: {error_message}")

            return OrchestratorResponse(
                response=f"I encountered an error processing your request: {str(e)}",
                input_tokens=self._estimate_tokens(message),
                output_tokens=50,
                processing_time=processing_time,
                error=error_message,
            )

    @staticmethod
    def _prepare_context(history: List[Conversation], current_message: str) -> List[BaseMessage]:
        """Prepare LangGraph message context from conversation history.

        This demonstrates the power of the optimized storage strategy:
        - We stored only user input + final AI response
        - But we can perfectly reconstruct conversation context
        - Much more efficient than storing all intermediate LangGraph messages
        """
        messages = []

        for turn in history:
            messages.extend(turn.to_langgraph_messages())

        messages.append(HumanMessage(content=current_message))

        logger.debug(f"Prepared context: {len(messages)} messages from {len(history)} stored turns")
        return messages

    @staticmethod
    def _extract_response_text(orchestrator_result: Dict[str, Any]) -> str:
        """Extract final response text from orchestrator result."""
        messages = orchestrator_result.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "content", None):
                return msg.content

        return "No response generated"

    @staticmethod
    def _extract_tools_used(orchestrator_result: Dict[str, Any]) -> List[str]:
        """Extract list of tools used during processing."""
        routing_history = orchestrator_result.get("plugin_context", {}).get("routing_history", [])
        routing_history = orchestrator_result.get("plugin_context", {}).get("routing_history", [])
        return list(set(routing_history)) if routing_history else []

    def _estimate_input_tokens(self, messages: List[BaseMessage]) -> int:
        """Estimate input tokens from message context.

        Calculates a rough token count by summing character lengths of message
        contents and dividing by four. This heuristic mirrors common tokenizer
        averages without requiring model-specific tokenizers.
        """
        total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, "content"))
        return max(1, total_chars // 4)

    def _estimate_output_tokens(self, response: str) -> int:
        """Estimate output tokens from response using the same heuristic as input."""
        return max(1, len(response) // 4)

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation for fallback error paths and health checks."""
        return max(1, len(text) // 4)

    async def process_simple_message(
        self, message: str, thread_id: str = "temp_session", user_id: str = "anonymous", org_id: str = "public"
    ) -> OrchestratorResponse:
        """Process a simple message without conversation history.

        Useful for testing or stateless interactions.
        """
        return await self.process_with_context(
            thread_id=thread_id, message=message, conversation_history=[], user_id=user_id, org_id=org_id
        )

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator configuration."""
        try:
            plugin_info = self.orchestrator.plugin_manager.get_plugin_routing_info()
            available_plugins = self.orchestrator.plugin_manager.get_available_plugins()

            return {
                "available_plugins": available_plugins,
                "plugin_info": plugin_info,
                "healthy_plugins": list(self.orchestrator.plugin_manager.healthy_plugins),
                "failed_plugins": list(self.orchestrator.plugin_manager.failed_plugins),
                "max_agent_hops": self.orchestrator.settings.max_agent_hops,
                "max_tool_hops": self.orchestrator.settings.max_tool_hops,
                "graph_recursion_limit": self.orchestrator.settings.graph_recursion_limit,
            }
        except Exception as e:
            logger.warning(f"Error getting orchestrator info: {e}")
            return {"error": str(e), "available_plugins": [], "healthy_plugins": [], "failed_plugins": []}

    async def health_check(self) -> Dict[str, Any]:
        """Perform orchestrator health check."""
        try:
            start_time = time.time()
            test_response = await self.process_simple_message("Health check test")
            response_time = time.time() - start_time

            return {
                "status": "healthy" if test_response.error is None else "unhealthy",
                "response_time": response_time,
                "error": test_response.error,
                "orchestrator_info": self.get_orchestrator_info(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "response_time": None}
