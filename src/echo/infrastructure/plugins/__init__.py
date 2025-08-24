"""Echo Framework Plugin Infrastructure - SDK-Based Agent Discovery and Management.

This module provides comprehensive plugin infrastructure for the Echo multi-agent AI framework,
enabling dynamic discovery, loading, and lifecycle management of specialized agents through
the Echo SDK system.

Plugin System Architecture:
    The plugin infrastructure implements a sophisticated agent extension system:
    - SDK-Based Discovery: Automatic discovery of plugins following Echo SDK contracts
    - Dynamic Loading: Runtime loading and validation of plugin agents and tools
    - LangGraph Integration: Seamless integration with Echo's orchestration system
    - Health Monitoring: Continuous health checking and failure isolation
    - Hot Reload: Development-friendly plugin reloading without server restart

Key Components:
    SDKPluginManager: Central plugin discovery and lifecycle management
    SDKPluginBundle: Container for plugin contract, agent, model, and tools
    Plugin Discovery: Automatic scanning of configured plugin directories
    Validation System: Comprehensive plugin contract and dependency validation

Plugin Development Benefits:
    External Development:
        - Plugins can be developed independently of the core Echo framework
        - Full access to Echo SDK for agent and tool development
        - Standard plugin contract ensures compatibility and reliability
        - Automatic discovery and registration without manual configuration

    Runtime Management:
        - Dynamic plugin loading and unloading
        - Health monitoring with automatic failure isolation
        - Hot reload capabilities for development iteration
        - Comprehensive error reporting and debugging support

    LangGraph Integration:
        - Automatic LangGraph node and edge generation
        - Tool routing and execution coordination
        - State management integration with core orchestrator
        - Performance monitoring and optimization

Supported Plugin Types:
    Specialized Agents:
        - Math Agent: Mathematical computation and problem solving
        - Search Agent: Web search and information retrieval
        - Code Agent: Programming assistance and code analysis
        - Custom Agents: Domain-specific specialized assistants

    Tool Providers:
        - Calculation tools for mathematical operations
        - Search tools for information retrieval
        - API integration tools for external services
        - Custom tools for specialized functionality

Example Usage:
    Plugin manager initialization:

    ```python
    from echo.infrastructure.plugins import SDKPluginManager
    from echo.infrastructure.llm import LLMModelFactory
    from echo.config import Settings

    settings = Settings()
    settings.plugins_dir = ["./plugins/src/echo_plugins", "./custom_plugins"]

    llm_factory = LLMModelFactory(settings)
    plugin_manager = SDKPluginManager(
        plugins_dirs=settings.plugins_dir,
        llm_factory=llm_factory
    )

    await plugin_manager.discover_and_load()

    available_plugins = plugin_manager.get_available_plugins()
    math_bundle = plugin_manager.get_plugin_bundle("math_agent")
    ```

    Plugin integration with orchestrator:

    ```python
    for plugin_name in plugin_manager.get_available_plugins():
        bundle = plugin_manager.get_plugin_bundle(plugin_name)
        if bundle:
            nodes = bundle.get_graph_nodes()
            edges = bundle.get_graph_edges()
    ```

    Plugin health monitoring:

    ```python
    healthy_plugins = plugin_manager.healthy_plugins
    failed_plugins = plugin_manager.failed_plugins

    await plugin_manager.reload_plugin("failed_plugin_name")
    ```

Plugin Development Workflow:
    1. Create plugin using Echo SDK templates and contracts
    2. Place plugin in configured plugin directory
    3. Plugin is automatically discovered on next scan/reload
    4. Validation ensures plugin meets Echo SDK requirements
    5. Successful plugins are loaded and integrated into orchestration
    6. Health monitoring provides continuous availability checking

The plugin infrastructure enables Echo to be extended with specialized capabilities
while maintaining system stability and performance through proper isolation and
management of external components.
"""

from .sdk_manager import SDKPluginBundle, SDKPluginManager

__all__ = [
    "SDKPluginManager",
    "SDKPluginBundle",
]
