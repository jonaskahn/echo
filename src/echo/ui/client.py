from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ChatResult:
    """Result of a chat message exchange with the Echo backend."""

    response: str
    thread_id: str
    conversation_id: str
    metadata: Dict[str, Any]


@dataclass
class PluginInfo:
    """Information about a plugin including metadata and health status."""

    name: str
    version: str
    description: str
    capabilities: List[str]
    status: str


@dataclass
class SystemStatus:
    """Overall system health and status information."""

    status: str
    available_plugins: List[str]
    healthy_plugins: List[str]
    failed_plugins: List[str]
    total_sessions: int


class EchoApiClient:
    """HTTP client for communicating with the Echo FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an HTTP request to the backend and return the JSON response."""
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            response = client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()

    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        user_id: str = "anonymous",
        org_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a chat message and return the assistant's response."""
        payload = {
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id,
            "org_id": org_id,
            "metadata": metadata or {},
        }

        data = self._make_request("POST", "/api/v1/chat", json=payload)
        return ChatResult(
            response=data.get("response", ""),
            thread_id=data.get("thread_id", thread_id or ""),
            conversation_id=data.get("conversation_id", ""),
            metadata=data.get("metadata", {}),
        )

    def get_plugins(self) -> List[PluginInfo]:
        """Fetch all available plugins with their metadata and status."""
        data = self._make_request("GET", "/api/v1/plugins")
        return [PluginInfo(**plugin) for plugin in data]

    def get_system_status(self) -> SystemStatus:
        """Fetch system status and health information."""
        data = self._make_request("GET", "/api/v1/status")
        return SystemStatus(**data)

    def reload_plugins(self) -> Dict[str, Any]:
        """Reload all plugins and return the result."""
        return self._make_request("POST", "/api/v1/plugins/reload")
