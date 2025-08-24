"""API service initialization and container wiring utilities."""

from __future__ import annotations

from ..config.settings import Settings
from ..core.services.service_container import global_service_container, initialize_container


async def initialize_api(application_settings: Settings) -> None:
    """Initialize the global service container using provided application settings.

    Sets up infrastructure, LLM providers, plugin manager, and application services.
    Raises RuntimeError/ValueError if initialization or validation fails.
    """
    await initialize_container(application_settings)


__all__ = ["initialize_api", "global_service_container"]
