"""Test cases for the MultiAgentOrchestrator coordinator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph

from echo.core.orchestrator.coordinator import (
    GraphNodes,
    MultiAgentOrchestrator,
    RoutingResults,
    SystemPrompts,
    ToolLoggingHandler,
)


class TestToolLoggingHandler:
    """Test cases for ToolLoggingHandler."""

    def test_init(self):
        """Test ToolLoggingHandler initialization."""
        logger = MagicMock()
        state_updater = MagicMock()
        handler = ToolLoggingHandler(logger, state_updater)

        assert handler.logger == logger
        assert handler.state_updater == state_updater

    def test_on_tool_start_with_valid_tool(self):
        """Test on_tool_start with valid tool call."""
        logger = MagicMock()
        state_updater = MagicMock()
        handler = ToolLoggingHandler(logger, state_updater)

        serialized = {"name": "calculator"}
        input_str = "2 + 2"

        handler.on_tool_start(serialized=serialized, input_str=input_str)

        logger.debug.assert_called()
        state_updater.assert_called_with("tool_hops", 1)

    def test_on_tool_start_with_goto_tool(self):
        """Test on_tool_start with goto tool (should not count)."""
        logger = MagicMock()
        state_updater = MagicMock()
        handler = ToolLoggingHandler(logger, state_updater)

        serialized = {"name": "goto_math_agent"}

        handler.on_tool_start(serialized=serialized)

        logger.debug.assert_called()
        state_updater.assert_not_called()

    def test_on_tool_start_with_finalize_tool(self):
        """Test on_tool_start with finalize tool (should not count)."""
        logger = MagicMock()
        state_updater = MagicMock()
        handler = ToolLoggingHandler(logger, state_updater)

        serialized = {"name": "finalize"}

        handler.on_tool_start(serialized=serialized)

        logger.debug.assert_called()
        state_updater.assert_not_called()

    def test_on_tool_start_without_state_updater(self):
        """Test on_tool_start without state_updater."""
        logger = MagicMock()
        handler = ToolLoggingHandler(logger, None)

        serialized = {"name": "calculator"}

        handler.on_tool_start(serialized=serialized)

        logger.warning.assert_called()

    def test_on_tool_start_with_exception(self):
        """Test on_tool_start with exception handling."""
        logger = MagicMock()
        state_updater = MagicMock(side_effect=Exception("Test error"))
        handler = ToolLoggingHandler(logger, state_updater)

        serialized = {"name": "calculator"}

        # Should not raise exception
        handler.on_tool_start(serialized=serialized)

        logger.error.assert_called()

    def test_on_tool_end(self):
        """Test on_tool_end method."""
        logger = MagicMock()
        handler = ToolLoggingHandler(logger, None)

        output = "The result is 4"

        handler.on_tool_end(output=output)

        logger.debug.assert_called()

    def test_on_tool_end_with_long_output(self):
        """Test on_tool_end with long output (should truncate)."""
        logger = MagicMock()
        handler = ToolLoggingHandler(logger, None)

        output = "x" * 300  # Longer than 200 chars

        handler.on_tool_end(output=output)

        # Should truncate to 200 chars
        call_args = logger.debug.call_args[0][0]
        assert "output=" in call_args
        assert len(call_args.split("output=")[1]) <= 203  # "output=" + 200 chars + quotes

    def test_on_tool_end_with_exception(self):
        """Test on_tool_end with exception handling."""
        logger = MagicMock(side_effect=Exception("Test error"))
        handler = ToolLoggingHandler(logger, None)

        # Should not raise exception
        handler.on_tool_end(output="test")


class TestMultiAgentOrchestrator:
    """Test cases for MultiAgentOrchestrator."""

    @pytest.fixture
    def orchestrator(self, settings, mock_llm_factory, mock_plugin_manager):
        """Create a MultiAgentOrchestrator instance for testing."""
        return MultiAgentOrchestrator(
            plugin_manager=mock_plugin_manager,
            llm_factory=mock_llm_factory,
            settings=settings,
        )

    def test_init(self, orchestrator, settings, mock_llm_factory, mock_plugin_manager):
        """Test MultiAgentOrchestrator initialization."""
        assert orchestrator.plugin_manager == mock_plugin_manager
        assert orchestrator.llm_factory == mock_llm_factory
        assert orchestrator.settings == settings
        assert orchestrator.checkpointer is None
        assert orchestrator.coordinator_model is not None
        assert orchestrator.finalizer_model is not None
        assert orchestrator.graph is not None

    def test_init_with_checkpointer(self, settings, mock_llm_factory, mock_plugin_manager, mock_checkpointer):
        """Test initialization with checkpointer."""
        orchestrator = MultiAgentOrchestrator(
            plugin_manager=mock_plugin_manager,
            llm_factory=mock_llm_factory,
            settings=settings,
            checkpointer=mock_checkpointer,
        )

        assert orchestrator.checkpointer == mock_checkpointer

    @patch("echo.core.orchestrator.coordinator.ModelConfig")
    def test_create_coordinator_model(self, mock_model_config, orchestrator, mock_llm_factory):
        """Test coordinator model creation."""
        mock_plugin_manager = orchestrator.plugin_manager
        mock_plugin_manager.get_coordinator_tools.return_value = []

        # Trigger model creation
        orchestrator._create_coordinator_model()

        mock_llm_factory.create_base_model.assert_called_once()
        mock_plugin_manager.get_coordinator_tools.assert_called_once()

    @patch("echo.core.orchestrator.coordinator.ModelConfig")
    def test_create_finalizer_model(self, mock_model_config, orchestrator, mock_llm_factory):
        """Test finalizer model creation."""
        orchestrator._create_finalizer_model()

        mock_llm_factory.create_base_model.assert_called_once()

    def test_build_graph(self, orchestrator):
        """Test graph building."""
        graph = orchestrator._build_graph()

        assert graph is not None
        # Verify graph has expected structure
        assert hasattr(graph, "get_graph")

    def test_add_core_nodes(self, orchestrator):
        """Test adding core nodes to graph."""
        graph = StateGraph(MagicMock())

        orchestrator._add_core_nodes(graph)

        # Verify all core nodes are added
        nodes = graph.nodes
        assert GraphNodes.COORDINATOR in nodes
        assert GraphNodes.CONTROL_TOOLS in nodes
        assert GraphNodes.SUSPEND in nodes
        assert GraphNodes.FINALIZER in nodes

    def test_add_plugin_nodes_and_edges(self, orchestrator):
        """Test adding plugin nodes and edges."""
        graph = StateGraph(MagicMock())
        mock_bundle = MagicMock()
        mock_bundle.get_graph_nodes.return_value = {"test_agent": MagicMock()}
        mock_bundle.get_graph_edges.return_value = {
            "direct_edges": [("node1", "node2")],
            "conditional_edges": {"node1": {"condition": MagicMock(), "mapping": {}}},
        }

        orchestrator.plugin_manager.plugin_bundles = {"test_plugin": mock_bundle}

        orchestrator._add_plugin_nodes_and_edges(graph)

        mock_bundle.get_graph_nodes.assert_called_once()
        mock_bundle.get_graph_edges.assert_called_once()

    def test_add_routing_edges(self, orchestrator):
        """Test adding routing edges."""
        graph = StateGraph(MagicMock())

        orchestrator._add_routing_edges(graph)

        # Verify edges are added (this is mostly testing that no exceptions occur)
        assert True

    def test_coordinator_routing_logic_hop_limit_reached(self, orchestrator, sample_agent_state):
        """Test coordinator routing when hop limit is reached."""
        sample_agent_state["agent_hops"] = orchestrator.settings.max_agent_hops

        result = orchestrator._coordinator_routing_logic(sample_agent_state)

        assert result == RoutingResults.SUSPEND

    def test_coordinator_routing_logic_tool_hops_limit_reached(self, orchestrator, sample_agent_state):
        """Test coordinator routing when tool hops limit is reached."""
        sample_agent_state["tool_hops"] = orchestrator.settings.max_tool_hops

        result = orchestrator._coordinator_routing_logic(sample_agent_state)

        assert result == RoutingResults.SUSPEND

    def test_coordinator_routing_logic_with_tool_calls(self, orchestrator, sample_agent_state):
        """Test coordinator routing with tool calls present."""
        # Add a message with tool calls
        message_with_tools = AIMessage(content="test", tool_calls=[{"name": "test_tool"}])
        sample_agent_state["messages"].append(message_with_tools)

        result = orchestrator._coordinator_routing_logic(sample_agent_state)

        assert result == RoutingResults.CONTINUE

    def test_coordinator_routing_logic_no_tool_calls(self, orchestrator, sample_agent_state):
        """Test coordinator routing without tool calls."""
        result = orchestrator._coordinator_routing_logic(sample_agent_state)

        assert result == RoutingResults.END

    def test_is_hop_limit_reached_agent_hops(self, orchestrator, sample_agent_state):
        """Test hop limit detection for agent hops."""
        sample_agent_state["agent_hops"] = orchestrator.settings.max_agent_hops

        result = orchestrator._is_hop_limit_reached(sample_agent_state)

        assert result is True

    def test_is_hop_limit_reached_tool_hops(self, orchestrator, sample_agent_state):
        """Test hop limit detection for tool hops."""
        sample_agent_state["tool_hops"] = orchestrator.settings.max_tool_hops

        result = orchestrator._is_hop_limit_reached(sample_agent_state)

        assert result is True

    def test_is_hop_limit_reached_not_reached(self, orchestrator, sample_agent_state):
        """Test hop limit detection when limits not reached."""
        sample_agent_state["agent_hops"] = 1
        sample_agent_state["tool_hops"] = 1

        result = orchestrator._is_hop_limit_reached(sample_agent_state)

        assert result is False

    def test_has_tool_calls_with_tools(self, orchestrator, sample_agent_state):
        """Test tool call detection with tools present."""
        message_with_tools = AIMessage(content="test", tool_calls=[{"name": "test_tool"}])
        sample_agent_state["messages"].append(message_with_tools)

        result = orchestrator._has_tool_calls(sample_agent_state)

        assert result is True

    def test_has_tool_calls_without_tools(self, orchestrator, sample_agent_state):
        """Test tool call detection without tools."""
        result = orchestrator._has_tool_calls(sample_agent_state)

        assert result is False

    def test_has_tool_calls_empty_messages(self, orchestrator, empty_agent_state):
        """Test tool call detection with empty messages."""
        result = orchestrator._has_tool_calls(empty_agent_state)

        assert result is False

    @pytest.mark.asyncio
    async def test_coordinator_node(self, orchestrator, sample_agent_state):
        """Test coordinator node execution."""
        result = orchestrator._coordinator_node(sample_agent_state)

        assert "messages" in result
        assert "agent_hops" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_build_coordinator_prompt(self, orchestrator):
        """Test coordinator prompt building."""
        plugin_info = {"math_agent": "Handles math calculations", "search_agent": "Performs web searches"}

        prompt = orchestrator._build_coordinator_prompt(plugin_info)

        assert "math_agent" in prompt
        assert "search_agent" in prompt
        assert "goto_math_agent" in prompt
        assert "goto_search_agent" in prompt

    @pytest.mark.asyncio
    async def test_suspend_node(self, orchestrator, sample_agent_state):
        """Test suspend node execution."""
        result = orchestrator._suspend_node(sample_agent_state)

        assert "messages" in result
        assert "agent_hops" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_finalizer_node(self, orchestrator, sample_agent_state):
        """Test finalizer node execution."""
        result = orchestrator._finalizer_node(sample_agent_state)

        assert "messages" in result
        assert "agent_hops" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_create_state_update(self, orchestrator, sample_agent_state):
        """Test state update creation."""
        message = AIMessage(content="Test response")
        agent_hops = 5

        result = orchestrator._create_state_update(message, agent_hops, sample_agent_state)

        assert result["messages"] == [message]
        assert result["agent_hops"] == agent_hops
        assert "tool_hops" in result
        assert "session_id" in result

    def test_create_state_update_without_state(self, orchestrator):
        """Test state update creation without existing state."""
        message = AIMessage(content="Test response")
        agent_hops = 5

        result = orchestrator._create_state_update(message, agent_hops)

        assert result["messages"] == [message]
        assert result["agent_hops"] == agent_hops

    def test_filter_safe_messages_empty(self, orchestrator):
        """Test filtering empty messages."""
        result = orchestrator._filter_safe_messages([])

        assert result == []

    def test_filter_safe_messages_no_tool_calls(self, orchestrator, sample_messages):
        """Test filtering messages without tool calls."""
        result = orchestrator._filter_safe_messages(sample_messages)

        assert result == sample_messages

    def test_filter_safe_messages_with_complete_tool_calls(self, orchestrator, mock_tool_calls, mock_tool_response):
        """Test filtering messages with complete tool call sequences."""
        messages = [
            HumanMessage(content="Calculate 2+2"),
            AIMessage(content="I'll calculate that", tool_calls=mock_tool_calls),
            mock_tool_response,
        ]

        result = orchestrator._filter_safe_messages(messages)

        assert len(result) == 3
        assert result == messages

    def test_filter_safe_messages_with_incomplete_tool_calls(self, orchestrator, mock_tool_calls):
        """Test filtering messages with incomplete tool call sequences."""
        messages = [
            HumanMessage(content="Calculate 2+2"),
            AIMessage(content="I'll calculate that", tool_calls=mock_tool_calls),
            # Missing tool response
        ]

        result = orchestrator._filter_safe_messages(messages)

        assert len(result) == 1  # Only HumanMessage should remain
        assert isinstance(result[0], HumanMessage)

    def test_is_incomplete_tool_call_sequence_with_tool_calls(self, orchestrator, mock_tool_calls):
        """Test incomplete tool call sequence detection with tool calls."""
        message = AIMessage(content="test", tool_calls=mock_tool_calls)
        messages = [message]

        result = orchestrator._is_incomplete_tool_call_sequence(message, messages, 0)

        assert result is True  # No tool response in messages

    def test_is_incomplete_tool_call_sequence_without_tool_calls(self, orchestrator):
        """Test incomplete tool call sequence detection without tool calls."""
        message = AIMessage(content="test")
        messages = [message]

        result = orchestrator._is_incomplete_tool_call_sequence(message, messages, 0)

        assert result is False

    def test_find_tool_responses(self, orchestrator, mock_tool_response):
        """Test finding tool responses."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="test", tool_calls=[{"id": "call_123"}]),
            mock_tool_response,
        ]

        result = orchestrator._find_tool_responses(messages, 1)

        assert "call_123" in result

    def test_find_tool_responses_no_responses(self, orchestrator):
        """Test finding tool responses when none exist."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="test", tool_calls=[{"id": "call_123"}]),
        ]

        result = orchestrator._find_tool_responses(messages, 1)

        assert result == set()

    def test_merge_updated_state(self, orchestrator):
        """Test merging updated state."""
        result = {"messages": [], "agent_hops": 5}
        input_state = {"tool_hops": 10}

        merged = orchestrator._merge_updated_state(result, input_state)

        assert merged["tool_hops"] == 10
        assert merged["agent_hops"] == 5

    def test_make_state_updater(self, orchestrator):
        """Test state updater creation."""
        input_state = {"tool_hops": 5}

        updater = orchestrator._make_state_updater(input_state)
        updater("tool_hops", 3)

        assert input_state["tool_hops"] == 8

    def test_build_run_config(self, orchestrator, sample_agent_state):
        """Test run configuration building."""
        config = orchestrator._build_run_config(sample_agent_state)

        assert "recursion_limit" in config
        assert config["recursion_limit"] == orchestrator.settings.graph_recursion_limit

    def test_build_run_config_with_configurable(self, orchestrator, sample_agent_state):
        """Test run configuration building with configurable options."""
        sample_agent_state["configurable"] = {"test_option": "value"}

        config = orchestrator._build_run_config(sample_agent_state)

        assert "configurable" in config
        assert config["configurable"]["test_option"] == "value"

    @pytest.mark.asyncio
    async def test_invoke_model_with_prompt(self, orchestrator, sample_messages):
        """Test model invocation with prompt."""
        system_message = SystemMessage(content="You are a helpful assistant")

        result = orchestrator._invoke_model_with_prompt(system_message, sample_messages)

        assert isinstance(result, AIMessage)

    def test_route_after_control_tools_with_valid_message(self, orchestrator, sample_agent_state):
        """Test routing after control tools with valid message."""
        message = AIMessage(content="math_agent")
        sample_agent_state["messages"].append(message)

        result = orchestrator._route_after_control_tools(sample_agent_state)

        assert result == "math_agent_agent"

    def test_route_after_control_tools_with_final_result(self, orchestrator, sample_agent_state):
        """Test routing after control tools with final result."""
        message = AIMessage(content="final")
        sample_agent_state["messages"].append(message)

        result = orchestrator._route_after_control_tools(sample_agent_state)

        assert result == RoutingResults.END

    def test_route_after_control_tools_with_invalid_message(self, orchestrator, sample_agent_state):
        """Test routing after control tools with invalid message."""
        result = orchestrator._route_after_control_tools(sample_agent_state)

        assert result == RoutingResults.END

    def test_is_valid_tool_message_valid(self, orchestrator):
        """Test valid tool message detection."""
        message = AIMessage(content="test")

        result = orchestrator._is_valid_tool_message(message)

        assert result is True

    def test_is_valid_tool_message_invalid(self, orchestrator):
        """Test invalid tool message detection."""
        result = orchestrator._is_valid_tool_message(None)

        assert result is False

    def test_determine_route_final(self, orchestrator, sample_agent_state):
        """Test route determination with final result."""
        result = orchestrator._determine_route("final", sample_agent_state)

        assert result == RoutingResults.END

    def test_determine_route_valid_plugin(self, orchestrator, sample_agent_state):
        """Test route determination with valid plugin."""
        result = orchestrator._determine_route("math_agent", sample_agent_state)

        assert result == "math_agent_agent"

    def test_determine_route_unknown_plugin(self, orchestrator, sample_agent_state):
        """Test route determination with unknown plugin."""
        result = orchestrator._determine_route("unknown_plugin", sample_agent_state)

        assert result == RoutingResults.END

    def test_handle_plugin_route(self, orchestrator, sample_agent_state):
        """Test plugin route handling."""
        result = orchestrator._handle_plugin_route("math_agent", sample_agent_state)

        assert result == "math_agent_agent"

    def test_handle_plugin_route_agent_switch(self, orchestrator, sample_agent_state):
        """Test plugin route handling with agent switch."""
        sample_agent_state["current_agent"] = "search_agent"

        result = orchestrator._handle_plugin_route("math_agent", sample_agent_state)

        assert result == "math_agent_agent"

    @pytest.mark.asyncio
    async def test_ask_success(self, orchestrator, sample_agent_state):
        """Test successful ask execution."""
        result = await orchestrator.ask(sample_agent_state)

        assert "messages" in result
        assert "agent_hops" in result

    @pytest.mark.asyncio
    async def test_ask_with_initial_hops_warning(self, orchestrator, sample_agent_state):
        """Test ask with initial hops warning."""
        sample_agent_state["agent_hops"] = orchestrator.settings.max_agent_hops

        result = await orchestrator.ask(sample_agent_state)

        assert "messages" in result
        assert "agent_hops" in result

    @pytest.mark.asyncio
    async def test_ask_with_exception(self, orchestrator, sample_agent_state):
        """Test ask with exception handling."""
        # Mock graph to raise exception
        orchestrator.graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))

        result = await orchestrator.ask(sample_agent_state)

        assert "messages" in result
        assert "error" in result
        assert "Test error" in result["error"]

    def test_should_warn_about_initial_hops_true(self, orchestrator):
        """Test initial hops warning detection when true."""
        input_data = {"agent_hops": orchestrator.settings.max_agent_hops}

        result = orchestrator._should_warn_about_initial_hops(input_data)

        assert result is True

    def test_should_warn_about_initial_hops_false(self, orchestrator):
        """Test initial hops warning detection when false."""
        input_data = {"agent_hops": 1}

        result = orchestrator._should_warn_about_initial_hops(input_data)

        assert result is False

    def test_log_initial_hops_warning(self, orchestrator):
        """Test initial hops warning logging."""
        input_data = {"agent_hops": orchestrator.settings.max_agent_hops}

        # Should not raise exception
        orchestrator._log_initial_hops_warning(input_data)

    def test_handle_orchestrator_error(self, orchestrator, sample_agent_state):
        """Test orchestrator error handling."""
        error = Exception("Test error")

        result = orchestrator._handle_orchestrator_error(error, sample_agent_state)

        assert "messages" in result
        assert "error" in result
        assert "agent_hops" in result
        assert "tool_hops" in result
        assert "Test error" in result["error"]


class TestSystemPrompts:
    """Test cases for SystemPrompts constants."""

    def test_coordinator_template(self):
        """Test coordinator template formatting."""
        template = SystemPrompts.COORDINATOR_TEMPLATE
        assert "AVAILABLE AGENTS" in template
        assert "DECISION OUTPUT" in template
        assert "{plugin_descriptions}" in template
        assert "{tool_options}" in template

    def test_suspension_template(self):
        """Test suspension template formatting."""
        template = SystemPrompts.SUSPENSION_TEMPLATE
        assert "maximum agent call" in template
        assert "{current}" in template
        assert "{maximum}" in template

    def test_finalization(self):
        """Test finalization prompt."""
        prompt = SystemPrompts.FINALIZATION
        assert "Finalizer" in prompt
        assert "comprehensive" in prompt
        assert "conversational" in prompt


class TestGraphNodes:
    """Test cases for GraphNodes constants."""

    def test_graph_nodes_constants(self):
        """Test GraphNodes constants are defined."""
        assert GraphNodes.COORDINATOR == "coordinator"
        assert GraphNodes.CONTROL_TOOLS == "control_tools"
        assert GraphNodes.SUSPEND == "suspend"
        assert GraphNodes.FINALIZER == "finalizer"


class TestRoutingResults:
    """Test cases for RoutingResults constants."""

    def test_routing_results_constants(self):
        """Test RoutingResults constants are defined."""
        assert RoutingResults.CONTINUE == "continue"
        assert RoutingResults.SUSPEND == "suspend"
        assert RoutingResults.END == "end"
        assert RoutingResults.FINAL == "final"
