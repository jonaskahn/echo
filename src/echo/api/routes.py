"""Echo API Routes Aggregation Module.

This module serves as the central aggregation point for all Echo framework API endpoints.
It combines individual feature routers into a unified API under the /api/v1 prefix
and provides service initialization capabilities for the FastAPI application.

Architecture:
    The API follows RESTful conventions and is organized into logical feature areas:
    - /api/v1/chat: Conversation management and message processing
    - /api/v1/plugins: Plugin management and discovery
    - /api/v1/system: System monitoring and health checks

Components:
    router: Main FastAPI router with all endpoints configured
    initialize_api: Service initialization function for application startup

Example:
    Using the router in a FastAPI application:

    ```python
    from fastapi import FastAPI
    from echo.api.routes import router, initialize_api

    app = FastAPI()
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        await initialize_api(settings)
    ```
"""

from fastapi import APIRouter

from .routers.chat import router as chat_router
from .routers.plugins import router as plugins_router
from .routers.system import router as system_router
from .services import initialize_container

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(chat_router)
api_v1_router.include_router(plugins_router)
api_v1_router.include_router(system_router)

router = api_v1_router

__all__ = ["router", "initialize_container"]
