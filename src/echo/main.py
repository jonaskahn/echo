"""Echo Multi-Agent AI Framework Application Entry Point.

This module provides the main application factory for the Echo framework, which is a
plugin-based multi-agent conversational AI system. The application is built on FastAPI
and provides REST API endpoints for managing conversations and plugins.

Key Components:
    EchoApplication: Main application factory that configures and runs the FastAPI server

Architecture:
    The application follows a layered architecture with proper dependency injection:
    - API Layer: FastAPI routers for REST endpoints
    - Application Layer: Business logic services (conversation, orchestrator)
    - Domain Layer: Core business models and DTOs
    - Infrastructure Layer: External services (database, LLM providers, plugins)
    - Integration Layer: Platform handlers (Slack, Discord, webhooks)

Example:
    Basic usage to run the Echo application:

    ```python
    from echo.main import EchoApplication
    from echo.config.settings import Settings

    settings = Settings()
    app = EchoApplication(settings)
    app.run()
    ```

    Using the pre-configured instance:

    ```python
    from echo.main import app
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import initialize_container, router
from .config.settings import Settings


class EchoApplication:
    """Echo FastAPI application factory with complete lifecycle management.

    This class orchestrates the creation, configuration, and execution of the Echo
    multi-agent AI framework. It handles service initialization, middleware setup,
    route registration, and server startup/shutdown procedures.

    The application supports both development and production configurations with
    automatic service discovery and dependency injection.

    Attributes:
        settings: Application configuration from environment variables
        app: FastAPI application instance (created when needed)
        logger: Application logger for lifecycle events

    Example:
        Creating and running the application:

        ```python
        from echo.config.settings import Settings

        settings = Settings()
        settings.debug = True
        settings.api_port = 8080

        echo_app = EchoApplication(settings)
        echo_app.run()
        ```

        Creating just the FastAPI instance:

        ```python
        app_instance = echo_app.create_app()
        pass
        ```
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize the Echo application with configuration and logging.

        Args:
            settings: Application configuration. If None, loads from environment
                     variables and .env file using default Settings()
        """
        self.settings = settings or Settings()
        self.app: FastAPI | None = None
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _setup_logging() -> None:
        """Configure structured logging for the entire application.

        Sets up a consistent log format across all Echo components with
        timestamp, module name, level, and message information.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _create_lifespan_manager(self):
        """Create FastAPI lifespan manager for service initialization and cleanup.

        The lifespan manager handles startup and shutdown procedures:
        - Startup: Initialize LLM factory, plugin manager, and orchestrator services
        - Shutdown: Clean up resources and close connections gracefully

        Returns:
            AsyncContextManager: Lifespan context manager for FastAPI application
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.logger.info("Starting Echo 🤖 Multi-agents AI Framework...")
            await initialize_container(self.settings)
            self.logger.info("Echo 🤖 Multi-agents AI Framework started successfully")
            yield
            self.logger.info("Shutting down Echo 🤖 Multi-agents AI Framework...")

        return lifespan

    def _configure_middleware(self) -> None:
        """Configure FastAPI middleware stack for production and development.

        Adds Cross-Origin Resource Sharing (CORS) middleware with configurable
        origins from settings. In development, typically allows all origins (*).
        In production, should be restricted to specific domains.
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self) -> None:
        """Register API routes and health endpoints for the application.

        Includes:
        - Main API routes under /api/v1 prefix (chat, plugins, system)
        - Root health check endpoint for service monitoring
        """
        self.app.include_router(router)

        @self.app.get("/")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint for monitoring and load balancers.

            Returns:
                Dict containing service status, version, and basic info
            """
            return {
                "message": "Echo 🤖 Multi-agents AI Framework",
                "version": "1.0.1",
                "status": "running",
            }

    def create_app(self) -> FastAPI:
        """Create and fully configure the FastAPI application instance.

        Creates a production-ready FastAPI application with all middleware,
        routes, and lifecycle management configured. The app is ready to
        be served by any ASGI-compatible server.

        Returns:
            FastAPI: Complete configured application ready for deployment

        Example:
            ```python
            echo_app = EchoApplication()
            fastapi_instance = echo_app.create_app()

            import uvicorn
            uvicorn.run(fastapi_instance, host="0.0.0.0", port=8000)
            ```
        """
        self.app = FastAPI(
            title="Echo 🤖 Multi-agents AI Framework",
            description="Plugin-based multi-agent chatbot system",
            version="1.0.1",
            lifespan=self._create_lifespan_manager(),
        )

        self._configure_middleware()
        self._register_routes()

        return self.app

    def run(self) -> None:
        """Start the Echo server using Uvicorn with environment-appropriate configuration.

        Automatically selects the appropriate server configuration:
        - Debug mode: Hot reload enabled, debug logging, module string import
        - Production mode: No reload, info logging, direct app instance

        Server configuration is loaded from Settings (host, port, debug flag).

        Raises:
            RuntimeError: If FastAPI application creation fails in production mode
        """
        if self.settings.debug:
            uvicorn.run(
                "echo.main:app",
                host=self.settings.api_host,
                port=self.settings.api_port,
                reload=True,
                log_level="debug",
            )
        else:
            if not self.app:
                self.create_app()

            if self.app is None:
                raise RuntimeError("Failed to create FastAPI application")

            uvicorn.run(
                self.app,
                host=self.settings.api_host,
                port=self.settings.api_port,
                reload=False,
                log_level="info",
            )


echo_application = EchoApplication()
app = echo_application.create_app()


def main() -> None:
    """Main entry point for running Echo as a standalone application.

    Loads environment variables from .env file and starts the server
    with configuration from Settings. This function is called when
    running the module directly with `python -m echo`.
    """
    echo_application.run()
