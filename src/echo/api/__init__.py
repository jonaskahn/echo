"""API package exports.

Exposes:
- `router`: FastAPI APIRouter with all endpoints
- `initialize_api()`: wiring for orchestrator, plugin manager, and services
"""

from .routes import initialize_container, router

__all__ = ["router", "initialize_container"]
