"""Orchestrates multi-agent conversations for the Echo system using LangGraph.

Builds a sequential, tool-routed graph:
  coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator (repeat) -> finalizer -> END

Plugins register their nodes and edges via `PluginManager`. The orchestrator exposes
async entry points and guards against infinite loops using hop counters in `AgentState`.
"""

import traceback
from typing import Any, Dict, List

from echo_sdk.base.loggable import Loggable
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ...config.settings import Settings
from ...infrastructure.llm.factory import LLMModelFactory
from ...infrastructure.plugins.sdk_manager import SDKPluginManager
from .state import AgentState


class GraphNodes:
    COORDINATOR = "coordinator"
    CONTROL_TOOLS = "control_tools"
    SUSPEND = "suspend"
    FINALIZER = "finalizer"


class RoutingResults:
    CONTINUE = "continue"
    SUSPEND = "suspend"
    END = "end"
    FINAL = "final"


class SystemPrompts:
    COORDINATOR_TEMPLATE = """Your goal is to analyze queries and decide go to next agent in **AVAILABLE AGENTS**.
**AVAILABLE AGENTS**
{plugin_descriptions}
- finalize: Call when you think the answer for the user query/question is ready or no suitable agents.
**DECISION OUTPUT**
- Choose ONE of: {tool_options} | finalize"""

    SUSPENSION_TEMPLATE = """You have reached maximum agent call ({current}/{maximum}) allowed by the system.
**What this means:**
- The system cannot process any more agent switches
- You must provide a final answer based on the information gathered so far
- Further processing is not possible

**What you should do:**
1. Acknowledge that you've hit the system limit. Explain it friendly to users, do not use term system limit or agent stuff
2. Explain what you were able to accomplish base on results.
3. Provide the best possible answer with the available information
4. If the answer is incomplete, explain why and suggest the user continue the chat

**IMPORTANT**, never makeup the answer if provided information by agents not enough
Please provide a helpful response that addresses the user's query while explaining the hop limit situation."""

    FINALIZATION = """You are the Finalizer, responsible for creating the final response for a multi-agent conversation.

CRITICAL REQUIREMENTS:
1. Be comprehensive but concise - include all relevant information from agent work
2. Maintain the language and tone used in the chat
3. Connect all the work done by different agents into a coherent, flowing answer
4. Be creative and engaging in your synthesis
5. Address the user's original query directly and completely
6. If multiple agents contributed, explain how their work fits together
7. Use a conversational, helpful tone that feels natural and human-like"""


class ToolLoggingHandler(BaseCallbackHandler):
    """Handles tool execution logging and hop counting."""

    def __init__(self, logger, state_updater=None):
        self.logger = logger
        self.state_updater = state_updater

    def on_tool_start(self, serialized=None, input_str=None, **kwargs):
        try:
            name = serialized.get("name") if isinstance(serialized, dict) else None
            self.logger.debug(f"Tool start: name={name or 'unknown'} input={input_str}")
            if name.startswith("goto_") or name == "finalize":
                self.logger.debug(f"Tool name={name or 'unknown'} is skipped from counting")
            elif self.state_updater:
                self.logger.debug(f"Updating tool_hops: +1 (tool: {name})")
                self.state_updater("tool_hops", 1)
            else:
                self.logger.warning("No state_updater available for tool_hops tracking")
        except Exception as e:
            self.logger.error(f"Error in on_tool_start: {e}")
            pass

    def on_tool_end(self, output=None, **kwargs):
        try:
            preview = str(output)[:200] if output else None
            self.logger.debug(f"Tool end: output={preview}")
        except Exception:
            pass


class MultiAgentOrchestrator(Loggable):
    """Coordinates the sequential, tool-routed multi-agent workflow."""

    def __init__(
        self,
        plugin_manager: SDKPluginManager,
        llm_factory: LLMModelFactory,
        settings: Settings,
        checkpointer: Any | None = None,
    ) -> None:
        super().__init__()
        self.plugin_manager = plugin_manager
        self.llm_factory = llm_factory
        self.settings = settings
        self.checkpointer = checkpointer

        self.coordinator_model = self._create_coordinator_model()
        self.finalizer_model = self._create_finalizer_model()
        self.graph = self._build_graph()

    def _create_coordinator_model(self):
        """Creates and configures the LLM model for the coordinator with bound routing tools.

        The coordinator model is responsible for making routing decisions between different
        plugin agents. It uses a low temperature setting for deterministic routing and
        binds the coordinator tools with parallel_tool_calls=False to ensure one tool
        per decision step.
        """
        from ...infrastructure.llm.providers import ModelConfig

        control_tools = self.plugin_manager.get_coordinator_tools()
        config = ModelConfig(
            provider=self.settings.default_llm_provider,
            model_name=self.settings.get_default_provider_llm_model(),
            temperature=self.settings.default_llm_temperature,
            max_tokens=self.settings.default_llm_context_window,
        )

        base_model = self.llm_factory.create_base_model(config)
        return base_model.bind_tools(control_tools, parallel_tool_calls=True)

    def _create_finalizer_model(self):
        """Creates the finalizer model with different configuration than coordinator."""
        from ...infrastructure.llm.providers import ModelConfig

        config = ModelConfig(
            provider=self.settings.finalizer_llm_provider,
            model_name=self.settings.get_finalizer_provider_llm_model(),
            temperature=self.settings.finalizer_temperature,
            max_tokens=self.settings.finalizer_max_tokens,
        )

        return self.llm_factory.create_base_model(config)

    def _build_graph(self) -> StateGraph:
        """Constructs the complete LangGraph workflow for multi-agent orchestration.

        This method assembles the entire conversation flow graph including:
        - Core orchestration nodes (coordinator, control_tools, suspend, finalizer)
        - Dynamic plugin nodes and their connections
        - Routing logic and conditional edges
        - Entry point configuration

        The resulting graph implements the workflow:
        coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator (repeat) -> finalizer -> END
        """
        graph = StateGraph(AgentState)

        self._add_core_nodes(graph)
        self._add_plugin_nodes_and_edges(graph)

        graph.set_entry_point(GraphNodes.COORDINATOR)
        self._add_routing_edges(graph)

        compilation_options = {"checkpointer": self.checkpointer} if self.checkpointer else {}
        compiled_graph = graph.compile(**compilation_options)

        self.logger.debug(f"Graph built with \n{compiled_graph.get_graph().draw_mermaid()}")
        return compiled_graph

    def rebuild_graph(self) -> None:
        try:
            self.logger.info("Rebuilding orchestrator graph after plugin changes...")
            self.coordinator_model = self._create_coordinator_model()
            self.graph = self._build_graph()
            self.logger.info("Orchestrator graph rebuild complete")
        except Exception as e:
            self.logger.error(f"Failed to rebuild orchestrator graph: {e}")
            raise

    def _add_core_nodes(self, graph: StateGraph) -> None:
        """Registers the four core orchestration nodes that form the backbone of the conversation flow.

        These nodes handle the essential workflow steps:
        - coordinator: Makes routing decisions between agents
        - control_tools: Executes the coordinator's tool calls
        - suspend: Handles hop limit scenarios gracefully
        - finalizer: Produces the final user-facing response
        """
        graph.add_node(GraphNodes.COORDINATOR, self._coordinator_node)
        graph.add_node(GraphNodes.CONTROL_TOOLS, ToolNode(tools=self.plugin_manager.get_coordinator_tools()))
        graph.add_node(GraphNodes.SUSPEND, self._suspend_node)
        graph.add_node(GraphNodes.FINALIZER, self._finalizer_node)

    def _add_plugin_nodes_and_edges(self, graph: StateGraph) -> None:
        """Integrates all registered plugin nodes and their routing logic into the main graph.

        This method dynamically adds plugin-specific functionality to the orchestration:
        - Plugin agent nodes that handle specialized tasks
        - Direct edges for linear plugin workflows
        - Conditional edges for complex plugin routing decisions

        Each plugin bundle provides its own nodes and edge definitions that are
        seamlessly integrated into the main conversation flow.
        """
        for plugin_name, bundle in self.plugin_manager.plugin_bundles.items():
            nodes = bundle.get_graph_nodes()
            edges = bundle.get_graph_edges()

            for node_name, node_func in nodes.items():
                graph.add_node(node_name, node_func)

            for edge_def in edges["direct_edges"]:
                graph.add_edge(edge_def[0], edge_def[1])

            for node_name, edge_info in edges["conditional_edges"].items():
                graph.add_conditional_edges(node_name, edge_info["condition"], edge_info["mapping"])

    def _add_routing_edges(self, graph: StateGraph) -> None:
        """Establishes the complete routing network that determines conversation flow between nodes.

        This method creates the decision tree that guides the conversation:
        - Coordinator routing: Decides whether to continue, suspend, or end based on hop limits and tool calls
        - Control tools routing: Maps tool execution results to specific plugin agents or finalization
        - Final edges: Connects suspend and finalizer nodes to complete the workflow

        The routing logic ensures proper conversation flow while respecting hop limits and
        enabling dynamic plugin selection based on user needs.
        """
        graph.add_conditional_edges(
            GraphNodes.COORDINATOR,
            self._coordinator_routing_logic,
            {
                RoutingResults.CONTINUE: GraphNodes.CONTROL_TOOLS,
                RoutingResults.SUSPEND: GraphNodes.SUSPEND,
                RoutingResults.END: END,
            },
        )

        route_mapping = {RoutingResults.END: GraphNodes.FINALIZER}
        for plugin_name in self.plugin_manager.get_available_plugins():
            route_mapping[f"{plugin_name}_agent"] = f"{plugin_name}_agent"

        graph.add_conditional_edges(GraphNodes.CONTROL_TOOLS, self._route_after_control_tools, route_mapping)

        graph.add_edge(GraphNodes.SUSPEND, GraphNodes.FINALIZER)
        graph.add_edge(GraphNodes.FINALIZER, END)

    def _coordinator_routing_logic(self, state: AgentState) -> str:
        """Evaluates the current conversation state to determine the next routing decision.

        This method implements the core decision logic for conversation flow:
        - If hop limits are exceeded, route to suspend for graceful handling
        - If the last message contains tool calls, continue processing
        - Otherwise, end the conversation and proceed to finalization

        The routing ensures conversation safety by preventing infinite loops while
        maintaining the flexibility to handle complex multi-agent workflows.
        """
        if self._is_hop_limit_reached(state):
            return RoutingResults.SUSPEND
        elif self._has_tool_calls(state):
            return RoutingResults.CONTINUE
        else:
            return RoutingResults.END

    def _is_hop_limit_reached(self, state: AgentState) -> bool:
        """Determines whether the conversation has exceeded configured hop limits.

        Hop limits prevent infinite loops and excessive resource consumption:
        - agent_hops: Number of agent switches in the conversation
        - tool_hops: Number of tool executions performed

        When either limit is reached, the conversation should be suspended to
        allow for graceful completion with available information.
        """
        return (
            state.get("agent_hops", 0) >= self.settings.max_agent_hops
            or state.get("tool_hops", 0) >= self.settings.max_tool_hops
        )

    @staticmethod
    def _has_tool_calls(state: AgentState) -> bool:
        """Checks whether the most recent message in the conversation contains pending tool calls.

        Tool calls indicate that the coordinator has made a decision to route to a specific
        plugin agent or tool. The presence of tool calls means the conversation should
        continue processing rather than ending or suspending.
        """
        messages = state.get("messages", [])
        if not messages:
            return False
        return bool(getattr(messages[-1], "tool_calls", None))

    def _coordinator_node(self, state: AgentState) -> AgentState:
        """Executes the main decision-making step that determines conversation routing.

        This node analyzes the current conversation context and available plugins to
        make intelligent routing decisions. It constructs a system prompt that lists
        all available agents and their capabilities, then uses the coordinator model
        to choose the most appropriate next step or agent for the user's request.
        """
        plugin_routing_info = self.plugin_manager.get_plugin_routing_info()

        system_content = self._build_coordinator_prompt(plugin_routing_info)
        system_message = SystemMessage(content=system_content)

        coordinator_response = self.coordinator_model.invoke([system_message] + state["messages"])

        return self._create_state_update(coordinator_response, state.get("agent_hops", 0), state)

    @staticmethod
    def _build_coordinator_prompt(plugin_routing_info: Dict[str, str]) -> str:
        """Constructs the system prompt that guides the coordinator's routing decisions.

        This method creates a comprehensive prompt that:
        - Lists all available plugin agents with their descriptions
        - Provides clear instructions for the routing decision
        - Formats the available tool options for the coordinator model

        The prompt ensures the coordinator has complete information about available
        capabilities to make optimal routing decisions.
        """
        plugin_descriptions = [f"- {name}: {desc}" for name, desc in plugin_routing_info.items()]
        tool_options = " | ".join(f"goto_{name}" for name in plugin_routing_info.keys())

        return SystemPrompts.COORDINATOR_TEMPLATE.format(
            plugin_descriptions="\n".join(plugin_descriptions), tool_options=tool_options
        )

    def _suspend_node(self, state: AgentState) -> AgentState:
        """Handles graceful conversation termination when hop limits are exceeded.

        When the conversation reaches maximum allowed hops, this node provides a
        user-friendly explanation of the situation and generates a final response
        based on the information gathered so far. It ensures users understand why
        the conversation is ending and what was accomplished.
        """
        current_hops = state.get("agent_hops", 0)
        max_hops = self.settings.max_agent_hops

        suspension_message = SystemMessage(
            content=SystemPrompts.SUSPENSION_TEMPLATE.format(current=current_hops, maximum=max_hops)
        )

        suspension_response = self._invoke_model_with_prompt(suspension_message, state["messages"])
        return self._create_state_update(suspension_response, current_hops, state)

    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Synthesizes the complete conversation into a coherent final response."""
        finalization_prompt = SystemMessage(content=SystemPrompts.FINALIZATION)
        safe_messages = self._filter_safe_messages(state["messages"])
        final_response = self.finalizer_model.invoke([finalization_prompt] + safe_messages)
        return self._create_state_update(final_response, state.get("agent_hops", 0), state)

    @staticmethod
    def _create_state_update(message: AIMessage, agent_hops: int, state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Creates a standardized state update structure for graph node responses.

        This method ensures consistency in how all nodes update the conversation state
        by providing a uniform structure that includes the new message and preserves
        the current hop count and other state fields for proper conversation tracking.
        """
        update = {
            "messages": [message],
            "agent_hops": agent_hops,
        }

        if state:
            for key in ["tool_hops", "current_agent", "plugin_context", "session_id"]:
                if key in state:
                    update[key] = state[key]

        return update

    def _filter_safe_messages(self, messages: List) -> List:
        """Removes messages with incomplete tool call sequences to prevent validation errors.

        This method ensures message safety by filtering out assistant messages that
        contain tool calls without corresponding tool response messages. Incomplete
        tool call sequences can cause issues with LLM providers and must be removed
        before sending messages to the model.
        """
        if not messages:
            return []

        filtered_messages = []
        for message_index, message in enumerate(messages):
            if self._is_incomplete_tool_call_sequence(message, messages, message_index):
                self.logger.warning(f"Skipping incomplete tool call sequence in message {message_index}")
                continue
            filtered_messages.append(message)

        return filtered_messages

    def _is_incomplete_tool_call_sequence(self, message: Any, messages: List, message_index: int) -> bool:
        """Determines if an assistant message contains tool calls without proper responses.

        This method validates that all tool calls in an assistant message have
        corresponding tool response messages in the subsequent conversation history.
        Incomplete sequences are identified by checking if all tool call IDs have
        matching tool response messages within a reasonable lookahead window.
        """
        if not (hasattr(message, "tool_calls") and message.tool_calls and isinstance(message, AIMessage)):
            return False

        tool_call_ids = {tc.get("id") for tc in message.tool_calls if tc.get("id")}
        found_tool_responses = self._find_tool_responses(messages, message_index)

        return not tool_call_ids.issubset(found_tool_responses)

    @staticmethod
    def _find_tool_responses(messages: List, message_index: int) -> set:
        """Searches for tool response messages that correspond to tool calls.

        This method examines subsequent messages in the conversation to find
        tool response messages that match the tool call IDs from an assistant message.
        It uses a limited lookahead window to efficiently search for responses
        without examining the entire conversation history.
        """
        found_tool_responses = set()
        look_ahead_limit = min(message_index + 10, len(messages))

        for next_message in messages[message_index + 1 : look_ahead_limit]:
            if hasattr(next_message, "tool_call_id") and next_message.tool_call_id:
                found_tool_responses.add(next_message.tool_call_id)

        return found_tool_responses

    def _merge_updated_state(self, result: Dict[str, Any], input_state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge callback-updated state values into the graph execution result."""
        if "tool_hops" in input_state:
            old_value = result.get("tool_hops", "NOT_FOUND")
            new_value = input_state["tool_hops"]
            result["tool_hops"] = new_value
            self.logger.debug(f"Merged tool_hops: {old_value} -> {new_value}")

        if "agent_hops" in result:
            self.logger.debug(f"Preserved agent_hops: {result['agent_hops']}")

        return result

    def _make_state_updater(self, input_state: Dict[str, Any]):
        """Creates a callback function for updating conversation state counters.

        This static method returns a closure that can be used by callback handlers
        to increment hop counters without directly mutating the external state.
        The closure provides a clean interface for updating tool_hops and other
        conversation metrics during execution.
        """

        def state_updater(field: str, increment: int):
            if field == "tool_hops":
                old_value = input_state.get("tool_hops", 0)
                new_value = old_value + increment
                input_state["tool_hops"] = new_value
                self.logger.debug(f"State updater: tool_hops {old_value} -> {new_value} (+{increment})")
            else:
                self.logger.debug(f"State updater: {field} +{increment}")

        return state_updater

    def _build_run_config(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assembles the complete configuration for LangGraph execution.

        This method creates the configuration dictionary that controls how the
        graph executes, including recursion limits, callback handlers for logging
        and state tracking, and any configurable parameters passed in the input state.
        The configuration ensures proper execution monitoring and resource management.
        """
        config = {"recursion_limit": self.settings.graph_recursion_limit}

        if BaseCallbackHandler is not object:
            callbacks = [ToolLoggingHandler(self.logger, self._make_state_updater(input_state))]
            config["callbacks"] = callbacks

        configurable = input_state.get("configurable") or {}
        if configurable:
            config["configurable"] = configurable

        return config

    def _invoke_model_with_prompt(self, system_message: SystemMessage, messages: List) -> AIMessage:
        """Executes the coordinator model with a system prompt and conversation history.

        This method safely invokes the coordinator model by first filtering the message
        history to remove any incomplete tool call sequences that could cause validation
        errors. It combines the system prompt with the filtered conversation history
        to generate appropriate responses for different conversation stages.
        """
        safe_messages = self._filter_safe_messages(messages)
        return self.coordinator_model.invoke([system_message] + safe_messages)

    def _route_after_control_tools(self, state: AgentState) -> str:
        """Determines the next node in the graph based on tool execution results.

        This method analyzes the output of coordinator tool executions to determine
        the appropriate next step in the conversation flow. It validates the tool
        message format and routes to either a specific plugin agent or the finalizer
        based on the tool execution result.
        """
        last_message = state.get("messages", [])[-1] if state.get("messages") else None

        if not self._is_valid_tool_message(last_message):
            self.logger.warning("No valid tool message found in routing")
            return RoutingResults.END

        tool_result = last_message.content
        self.logger.debug(f"Routing decision: tool_result='{tool_result}'")

        return self._determine_route(tool_result, state)

    @staticmethod
    def _is_valid_tool_message(message: Any) -> bool:
        """Validates that a message has the required structure for tool routing.

        This method ensures that a message object exists and contains a content
        attribute, which is necessary for extracting the tool execution result
        that determines the next routing decision.
        """
        return message and hasattr(message, "content")

    def _determine_route(self, tool_result: str, state: AgentState) -> str:
        """Analyzes tool execution results to determine the appropriate routing path.

        This method implements the routing logic that maps tool execution outcomes
        to specific graph nodes. It handles three scenarios:
        - Final results route to conversation end
        - Valid plugin names route to their respective agents
        - Unknown results trigger a warning and route to end

        The routing ensures proper conversation flow and graceful handling of
        unexpected tool results.
        """
        if tool_result == RoutingResults.FINAL:
            return RoutingResults.END
        elif tool_result in self.plugin_manager.plugin_bundles:
            return self._handle_plugin_route(tool_result, state)
        else:
            self.logger.warning(f"Unknown routing result: '{tool_result}', ending conversation")
            return RoutingResults.END

    def _handle_plugin_route(self, tool_result: str, state: AgentState) -> str:
        """Processes routing decisions to specific plugin agents.

        This method handles the transition to plugin agents by constructing the
        target agent node name and logging the routing decision. It tracks agent
        switches to provide visibility into conversation flow and ensures proper
        routing to the intended plugin agent for specialized task processing.
        """
        target_agent = f"{tool_result}_agent"
        self.logger.debug(f"Routing to agent: {target_agent}")

        current_agent = state.get("current_agent")
        if current_agent != tool_result:
            self.logger.debug(f"Agent switch detected: {current_agent} -> {tool_result}")
        else:
            self.logger.debug(f"Continuing with same agent: {tool_result}")

        return target_agent

    async def ask(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the complete multi-agent conversation workflow asynchronously.

        This is the main entry point for the orchestrator that processes user queries
        through the entire conversation graph. It handles hop limit validation,
        graph execution with proper configuration, and graceful error handling.
        The method returns the final conversation state with accumulated messages
        and execution metadata.
        """
        if self._should_warn_about_initial_hops(input_data):
            self._log_initial_hops_warning(input_data)

        try:
            config = self._build_run_config(input_data)
            result = await self.graph.ainvoke(input_data, config=config)
            return self._merge_updated_state(result, input_data)
        except Exception as e:
            return self._handle_orchestrator_error(e, input_data)

    def _should_warn_about_initial_hops(self, input_data: Dict[str, Any]) -> bool:
        """Determines if the conversation should warn about excessive initial hop count.

        This method checks if the conversation is starting with a hop count that
        already exceeds the configured maximum. Such situations may indicate
        potential issues with conversation state management or hop counting.
        """
        return input_data.get("agent_hops", 0) >= self.settings.max_agent_hops

    def _log_initial_hops_warning(self, input_data: Dict[str, Any]) -> None:
        """Logs a warning message when conversation starts with excessive hop count.

        This method provides visibility into potential conversation state issues
        by logging when a conversation begins with a hop count that exceeds the
        configured maximum. The warning helps with debugging and monitoring
        conversation state management.
        """
        initial_hops = input_data.get("agent_hops", 0)
        self.logger.warning(
            f"Initial agent hops ({initial_hops}) already exceed MAX_AGENT_HOPS "
            f"({self.settings.max_agent_hops}), routing to suspend"
        )

    def _handle_orchestrator_error(self, error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provides graceful error handling for orchestrator execution failures.

        This method ensures that any errors during graph execution are properly
        logged with full traceback information and returns a user-friendly error
        response. It maintains conversation state consistency by incrementing hop
        count and including error details for debugging purposes.
        """
        error_trace = traceback.format_exc()
        self.logger.error(f"Orchestrator AI invoke error: {error}\nTraceback:\n{error_trace}")

        return {
            "messages": [AIMessage(content=f"I encountered an error processing your request. Error: {str(error)}")],
            "agent_hops": input_data.get("agent_hops", 0) + 1,
            "tool_hops": input_data.get("tool_hops", 0),  # Preserve tool_hops
            "error": str(error),
        }
