"""Echo AI CLI - Command Line Interface for the Echo AI Multi-Agent AI Framework."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
import threading
from typing import List

from .config.settings import Settings
from .main import EchoApplication


def print_banner() -> None:
    """Display the Echo AI framework banner."""
    print("🤖 Echo AI Multi-Agent AI Framework")
    print("=" * 50)


def get_version() -> str:
    """Get the current Echo AI package version."""
    try:
        return importlib.metadata.version("echo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def print_status(message: str) -> None:
    """Print a status message."""
    print(f"ℹ️  {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")

    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")

    if missing_deps:
        print_error(f"Missing dependencies: {', '.join(missing_deps)}")
        print_status("Please install missing dependencies with: pip install echo-app[dev]")
        return False

    return True


def start_api_server(settings: Settings) -> None:
    """Start the Echo AI API server."""
    print_status("Starting Echo AI API server...")
    print_status(f"API Server: http://{settings.api_host}:{settings.api_port}")
    print_status(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    echo_app = EchoApplication(settings)
    echo_app.run()


def start_ui_server(settings: Settings) -> None:
    """Start the Echo AI Streamlit UI server."""
    print_status("Starting Echo AI Streamlit UI server...")
    print_status(f"UI Server: http://{settings.ui_host}:{settings.ui_port}")
    os.environ["ECHO_API_BASE_URL"] = f"http://{settings.api_host}:{settings.api_port}"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/echo/ui/app.py",
                "--server.port",
                str(settings.ui_port),
                "--server.address",
                settings.ui_host,
                "--server.headless",
                "true",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start UI server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_warning("\nShutting down Echo AI services...")
        print_success("Echo AI services stopped.")


def start_services(services: List[str], settings: Settings, debug: bool = False) -> None:
    """Start Echo AI services based on command line arguments."""
    if not check_dependencies():
        sys.exit(1)

    print_banner()
    print_status(f"Echo AI Version: {get_version()}")

    if debug:
        print_status("Debug mode enabled")
        settings.debug = True

    if "all" in services:
        ui_thread = threading.Thread(target=start_ui_server, args=(settings,))
        ui_thread.start()
        start_api_server(settings)
    elif "api" in services:
        start_api_server(settings)
    elif "ui" in services:
        start_ui_server(settings)
    else:
        print_success("\n✅ Echo AI services started successfully!\n")


def show_system_info() -> None:
    """Display system information."""
    print_banner()
    print(f"Version: {get_version()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    # Show Echo AI environment variables
    echo_vars = {k: v for k, v in os.environ.items() if k.startswith("ECHO_")}

    print("\nEcho AI Environment Variables:")
    if echo_vars:
        for var, value in echo_vars.items():
            print(f"  {var}: {value}")
    else:
        print("  ECHO_*: No Echo AI-specific environment variables found")


def list_plugins() -> None:
    """List available plugins."""
    try:
        from echo_sdk.utils import (
            get_directory_discovery_summary,
            get_environment_discovery_summary,
            get_plugin_registry,
        )

        print_banner()
        print("Available Plugins:")

        # Get plugin registry
        registry = get_plugin_registry()
        if registry:
            for plugin_name, plugin_info in registry.items():
                print(f"  📦 {plugin_name}: {plugin_info.get('description', 'No description')}")
        else:
            print("  No plugins found in registry")

        # Show discovery summaries
        env_summary = get_environment_discovery_summary()
        dir_summary = get_directory_discovery_summary()

        if env_summary:
            print(f"\nEnvironment Discovery: {env_summary}")
        if dir_summary:
            print(f"Directory Discovery: {dir_summary}")

    except ImportError:
        print_error("echo_sdk not available. Cannot list plugins.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Echo AI 🤖 Multi-Agent AI Framework - Plugin-based multi-agent chatbot system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  echo start all                     # Start both API and UI
  echo start api                     # Start only API server
  echo start ui                      # Start only UI
  echo start all --debug             # Start both with debug mode
  echo start api --api-port 8080     # Start API on custom port
  echo version                       # Show version information
  echo info                          # Show system information
  echo plugins                       # List available plugins
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Echo AI services")
    start_parser.add_argument(
        "services", nargs="+", choices=["api", "ui", "all"], help="Services to start (api, ui, or all)"
    )
    start_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    start_parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    start_parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    start_parser.add_argument("--ui-host", default="0.0.0.0", help="UI server host")
    start_parser.add_argument("--ui-port", type=int, default=8501, help="UI server port")
    start_parser.add_argument("--no-check-deps", action="store_true", help="Skip dependency checking")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Plugins command
    subparsers.add_parser("plugins", help="List available plugins")

    args = parser.parse_args()

    if args.command == "start":
        settings = Settings(
            api_host=args.api_host, api_port=args.api_port, ui_host=args.ui_host, ui_port=args.ui_port, debug=args.debug
        )
        start_services(args.services, settings, args.debug)
    elif args.command == "version":
        print_banner()
        print(f"Echo AI Version: {get_version()}")
    elif args.command == "info":
        show_system_info()
    elif args.command == "plugins":
        list_plugins()
    else:
        parser.print_help()


# Create CLI app instance for entry point
app = main

if __name__ == "__main__":
    main()
