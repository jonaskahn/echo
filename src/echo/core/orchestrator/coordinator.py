"""Orchestrates multi-agent conversations for the Echo system using LangGraph.

Builds a sequential, tool-routed graph:
  coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator (repeat) -> finalizer -> END

Plugins register their nodes and edges via `PluginManager`. The orchestrator exposes
sync/async entry points and guards against infinite loops using a hop counter in
`AgentState`.
"""

from typing import Dict, Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from echo_sdk.base.loggable import Loggable
from .state import AgentState
from ...config.settings import Settings
from ...infrastructure.llm.factory import LLMModelFactory
from ...infrastructure.plugins.sdk_manager import SDKPluginManager


class ToolLoggingHandler(BaseCallbackHandler):

    def __init__(self, logger, state_updater=None):
        self.logger = logger
        self.state_updater = state_updater

    def on_tool_start(self, serialized=None, input_str=None, **kwargs):
        try:
            name = serialized.get("name") if isinstance(serialized, dict) else None
            self.logger.info(f"Tool start: name={name or 'unknown'} input={input_str}")

            if self.state_updater:
                self.state_updater("tool_hops", 1)
        except Exception:
            pass

    def on_tool_end(self, output=None, **kwargs):
        try:
            preview = str(output)[:200] if output else None
            self.logger.info(f"Tool end: output={preview}")
        except Exception:
            pass


class MultiAgentOrchestrator(Loggable):
    """Coordinates the sequential, tool-routed multi-agent workflow.

    Graph shape:
      coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator ... -> finalizer -> END

    State:
      Uses `AgentState` with a `messages` history and a `hops` counter to prevent loops.

    Attributes:
      - plugin_manager: Provides coordinator tools and plugin bundles (nodes/edges).
      - llm_factory: Builds the base LLM used by the coordinator.
      - settings: Global runtime configuration.
      - coordinator_model: LLM bound with coordinator routing tools.
      - graph: Compiled LangGraph ready to invoke.
    """

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
        self.graph = self._build_graph()

    def _create_coordinator_model(self):
        """Build the coordinator LLM and bind routing tools.

        - Chooses the model from `settings.default_llm_provider`.
        - Uses low temperature for deterministic routing.
        - Binds coordinator tools with `parallel_tool_calls=False` (one tool per step).

        Returns:
            Any: Runnable LLM with `invoke()` that may emit tool calls.
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
        return base_model.bind_tools(control_tools, parallel_tool_calls=False)

    def _build_graph(self) -> StateGraph:
        """Assemble and compile the LangGraph with dynamic plugin nodes/edges.

        Nodes:
          - coordinator: chooses the next agent or finalization via tools
          - control_tools: executes the coordinator's tool call
          - suspend: handles hop limit reached scenarios
          - finalizer: produces the final answer

        Edges:
          - coordinator -> control_tools when last message has tool_calls, else -> END
          - control_tools -> mapping: 'end' -> 'finalizer'; '{plugin}_agent' -> '{plugin}_agent'
          - suspend -> finalizer (after AI responds to limit warning)
          - finalizer -> END

        Also registers all plugin-provided nodes and direct/conditional edges.

        Returns:
            StateGraph: Compiled graph runnable.
        """
        graph = StateGraph(AgentState)

        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("control_tools", ToolNode(tools=self.plugin_manager.get_coordinator_tools()))
        graph.add_node("suspend", self._suspend_node)
        graph.add_node("finalizer", self._finalizer_node)

        for plugin_name, bundle in self.plugin_manager.plugin_bundles.items():
            nodes = bundle.get_graph_nodes()
            edges = bundle.get_graph_edges()

            for node_name, node_func in nodes.items():
                graph.add_node(node_name, node_func)

            for edge_def in edges["direct_edges"]:
                graph.add_edge(edge_def[0], edge_def[1])

            for node_name, edge_info in edges["conditional_edges"].items():
                graph.add_conditional_edges(node_name, edge_info["condition"], edge_info["mapping"])

        graph.set_entry_point("coordinator")

        def coordinator_routing_logic(s):
            if (
                s.get("agent_hops", 0) >= self.settings.max_agent_hops
                or s.get("tool_hops", 0) >= self.settings.max_tool_hops
            ):
                return "suspend"
            elif getattr(s["messages"][-1], "tool_calls", None):
                return "continue"
            else:
                return "end"

        graph.add_conditional_edges(
            "coordinator",
            coordinator_routing_logic,
            {"continue": "control_tools", "suspend": "suspend", "end": END},
        )

        route_mapping = {"end": "finalizer"}
        for plugin_name in self.plugin_manager.get_available_plugins():
            route_mapping[f"{plugin_name}_agent"] = f"{plugin_name}_agent"

        graph.add_conditional_edges("control_tools", self._route_after_control_tools, route_mapping)

        graph.add_edge("suspend", "finalizer")
        graph.add_edge("finalizer", END)

        compilation_options = {}
        if self.checkpointer is not None:
            compilation_options["checkpointer"] = self.checkpointer
        compiled_graph = graph.compile(**compilation_options)
        self.logger.debug(f"Graph built with \n{compiled_graph.get_graph().draw_mermaid()}")
        return compiled_graph

    def _coordinator_node(self, state: AgentState) -> AgentState:
        """Coordinator decision step.

        - Stops early when `MAX_HOPS` is reached.
        - Builds a `SystemMessage` that lists available `goto_{plugin}` tools and `finalize`.
        - Invokes the coordinator model to produce a single tool call or a normal reply.

        Args:
            state: Current `AgentState`.

        Returns:
            AgentState: State updates with the coordinator's `AIMessage` and incremented hops.
        """

        plugin_routing_info = self.plugin_manager.get_plugin_routing_info()
        available_plugin_descriptions = []

        for plugin_name, description in plugin_routing_info.items():
            available_plugin_descriptions.append(f"- {plugin_name}: {description}")

        goto_tool_options = " | ".join(f"goto_{name}" for name in plugin_routing_info.keys())

        system_content = f"""Your goal is analyze queries and decide to appropriate **AVAILABLE AGENTS**.
**AVAILABLE AGENTS**
{chr(10).join(available_plugin_descriptions)}
- finalize: Call when you think the answer for the user query/question is finish processing.
**CURRENT DECISION**
- Choose ONE of: {goto_tool_options} | finalize"""

        system_message = SystemMessage(content=system_content)
        coordinator_response = self.coordinator_model.invoke([system_message] + state["messages"])
        return {"messages": [coordinator_response], "agent_hops": state.get("agent_hops", 0)}

    def _suspend_node(self, state: AgentState) -> AgentState:
        """Handle hop limit reached scenarios by informing the AI and allowing a response.

        This node intercepts when max_agent_hops is reached and gives the AI a chance
        to communicate with the user about the situation before finalization.

        Args:
            state: Current `AgentState`.

        Returns:
            AgentState: State with AI's response to the hop limit warning.
        """
        current_agent_hops = state.get("agent_hops", 0)
        maximum_agent_hops = self.settings.max_agent_hops

        suspension_message = SystemMessage(
            content=f"""You have reached maximum agent call ({current_agent_hops}/{maximum_agent_hops}) allowed by the system.

**What this means:**
- The system cannot process any more agent switches, explain to user this friendly that you can't unable to complete answer due system limit
- You must provide a final answer based on the information gathered so far
- Further processing is not possible

**What you should do:**
1. Acknowledge that you've hit the system limit
2. Explain what you were able to accomplish
3. Provide the best possible answer with the available information
4. If the answer is incomplete, explain why and suggest user continue the chat to see the complete answer

Please provide a helpful response that addresses the user's query while explaining the hop limit situation."""
        )

        suspension_response = self._invoke_model_with_prompt(suspension_message, state["messages"])

        return {
            "messages": [suspension_response],
            "agent_hops": current_agent_hops,
        }

    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Produce the final user-facing answer.

        Uses the coordinator model to summarize the accumulated `messages` into
        a concise, coherent response while preserving the user's language.

        Args:
            state: Current `AgentState`.

        Returns:
            AgentState: State with a final `AIMessage` appended.
        """
        conversation_messages = state.get("messages", [])

        finalization_prompt = SystemMessage(
            content="""You're goal is to help creating the final response for a multi-agent conversation.
CRITICAL REQUIREMENTS:
1. Be comprehensive but concise.
2. Maintain the language used in the chat.
3. Connect all the work done by different agents into a coherent answer following up last user query."""
        )
        final_response = self._invoke_model_with_prompt(finalization_prompt, conversation_messages)

        return {
            "messages": [final_response],
            "agent_hops": state.get("agent_hops", 0),
        }

    def _filter_safe_messages(self, messages: list) -> list:
        """Filter out messages that could cause OpenAI validation errors.

        Specifically removes incomplete tool call sequences where an assistant message
        with tool_calls is not followed by corresponding tool response messages.

        Args:
            messages: List of messages to filter

        Returns:
            List of safe messages that can be sent to OpenAI
        """
        if not messages:
            return []

        filtered_messages = []

        for message_index, message in enumerate(messages):
            if hasattr(message, "tool_calls") and message.tool_calls and isinstance(message, AIMessage):

                tool_call_ids = {tc.get("id") for tc in message.tool_calls if tc.get("id")}
                found_tool_responses = set()

                look_ahead_limit = min(message_index + 10, len(messages))
                for next_message in messages[message_index + 1 : look_ahead_limit]:
                    if hasattr(next_message, "tool_call_id") and next_message.tool_call_id:
                        found_tool_responses.add(next_message.tool_call_id)

                if not tool_call_ids.issubset(found_tool_responses):
                    self.logger.warning(f"Skipping incomplete tool call sequence in message {message_index}")
                    continue

            filtered_messages.append(message)

        return filtered_messages

    @staticmethod
    def _make_state_updater(input_state: Dict[str, Any]):
        """Create a closure that increments counters in the input state.

        This helper returns a lightweight function used by callback handlers to
        update execution counters without mutating external scope directly.

        Args:
            input_state: The state dictionary passed into the graph run.

        Returns:
            Callable that accepts a field name and increment value and applies
            the update to the corresponding counter in ``input_state``.
        """

        def state_updater(field: str, increment: int):
            if field == "tool_hops":
                input_state["tool_hops"] = input_state.get("tool_hops", 0) + increment

        return state_updater

    def _build_run_config(self, input_state: Dict[str, Any]):
        """Construct the LangGraph run configuration.

        Assembles the recursion limit, optional callback handlers (for tool
        logging and hop counting), and forwards any ``configurable`` values
        embedded in the provided state.

        Args:
            input_state: State dict containing optional ``configurable`` values.

        Returns:
            Dict with configuration parameters for graph invocation.
        """
        callbacks = (
            [ToolLoggingHandler(self.logger, self._make_state_updater(input_state))]
            if BaseCallbackHandler is not object
            else None
        )
        base = {"recursion_limit": self.settings.graph_recursion_limit}
        if callbacks:
            base["callbacks"] = callbacks

        configurable = input_state.get("configurable") or {}
        if configurable:
            base["configurable"] = configurable
        return base

    def _invoke_model_with_prompt(self, systemMessage: SystemMessage, messages: list):
        """Invoke the coordinator model with a system prompt and filtered messages.

        Applies message safety filtering to avoid invalid tool call sequences
        before passing the conversation to the coordinator model.

        Args:
            systemMessage: System message that sets behavior or instructions.
            messages: Conversation history to provide as context.

        Returns:
            AIMessage produced by the coordinator model.
        """
        safe_messages = self._filter_safe_messages(messages)
        return self.coordinator_model.invoke([systemMessage] + safe_messages)

    def _route_after_control_tools(self, state: AgentState) -> str:
        """Map the output of coordinator tools to the next graph node.

        Reads the last tool message content and routes as follows:
          - 'final' or unknown value -> 'end' (mapped to 'finalizer')
          - exact plugin name or 'goto_{plugin}' -> '{plugin}_agent'

        Args:
            state: AgentState: Current `AgentState`.

        Returns:
            str: Next route label understood by the conditional edges.
        """

        last_message = state["messages"][-1] if state["messages"] else None

        if not last_message or not hasattr(last_message, "content"):
            self.logger.warning("No valid tool message found in routing")
            return "end"

        tool_result = last_message.content
        self.logger.info(f"Routing decision: tool_result='{tool_result}'")

        if tool_result == "final":
            return "end"
        elif tool_result in self.plugin_manager.plugin_bundles:
            target_agent = f"{tool_result}_agent"
            self.logger.info(f"Routing to agent: {target_agent}")

            current_agent = state.get("current_agent")
            if current_agent != tool_result:
                self.logger.info(f"Agent switch detected: {current_agent} -> {tool_result}")
            else:
                self.logger.info(f"Continuing with same agent: {tool_result}")

            return target_agent

        self.logger.warning(f"Unknown routing result: '{tool_result}', ending conversation")
        return "end"

    async def ask(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestrator asynchronously."""
        initial_agent_hops = input_data.get("agent_hops", 0)
        if initial_agent_hops >= self.settings.max_agent_hops:
            self.logger.warning(
                f"Initial agent hops ({initial_agent_hops}) already exceed MAX_AGENT_HOPS ({self.settings.max_agent_hops}), routing to suspend"
            )

        try:
            config = self._build_run_config(input_data)
            return await self.graph.ainvoke(input_data, config=config)
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            self.logger.error(f"Orchestrator AI invoke error: {e}\nTraceback:\n{error_trace}")
            return {
                "messages": [AIMessage(content=f"I encountered an error processing your request. Error: {str(e)}")],
                "agent_hops": input_data.get("agent_hops", 0) + 1,
                "error": str(e),
            }
