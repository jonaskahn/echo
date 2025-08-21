"""API package exports.

Exposes:
- `router`: FastAPI APIRouter with all endpoints
- `initialize_api()`: wiring for orchestrator, plugin manager, and services
"""

from .routes import router, initialize_container

__all__ = ["router", "initialize_container"]
