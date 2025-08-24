"""System monitoring endpoints for health and status.

Provides two endpoints:
- GET /status: Returns overall system status and plugin health.
- GET /health: Lightweight liveness probe for load balancers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...infrastructure.plugins.sdk_manager import SDKPluginManager
from ..schemas import SystemStatus
from ..services import global_service_container

system_api_router = APIRouter()


def get_plugin_manager() -> SDKPluginManager:
    """Return the SDK plugin manager from the global service container.

    Raises RuntimeError if the container has not been initialized.
    """
    return global_service_container.get_plugin_manager()


@system_api_router.get("/status", response_model=SystemStatus)
async def get_comprehensive_system_status(
    plugin_manager: SDKPluginManager = Depends(get_plugin_manager),
) -> SystemStatus:
    """Return overall system status and plugin health.

    The response includes available, healthy, and failed plugins plus a
    current session count.
    """
    return SystemStatus(
        status="operational",
        available_plugins=plugin_manager.get_available_plugins(),
        healthy_plugins=list(plugin_manager.healthy_plugins),
        failed_plugins=list(plugin_manager.failed_plugins),
        total_sessions=0,
    )


@system_api_router.get("/health")
async def simple_health_check() -> dict:
    """Return a lightweight liveness status for load balancers and probes."""
    return {"status": "healthy"}


router = system_api_router
