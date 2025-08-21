from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class ChatResult:
    response: str
    thread_id: str
    conversation_id: str
    metadata: Dict[str, Any]


class EchoApiClient:
    """Lightweight client for the Echo FastAPI backend.

    This client wraps synchronous HTTP calls to the server and returns
    structured results for use by the Streamlit UI.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Create a client bound to ``base_url``.

        Trailing slashes are removed to keep path joins consistent.
        """
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        user_id: str = "anonymous",
        org_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a chat message and return the server's response.

        Parameters:
        - ``message``: user input text
        - ``thread_id``: existing thread to continue; if ``None`` a new thread
          may be created by the server
        - ``user_id`` and ``org_id``: identifiers forwarded to the backend
        - ``metadata``: arbitrary request metadata
        """
        payload = {
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id,
            "org_id": org_id,
            "metadata": metadata or {},
        }
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            resp = client.post("/api/v1/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return ChatResult(
                response=data.get("response", ""),
                thread_id=data.get("thread_id", thread_id or ""),
                conversation_id=data.get("conversation_id", ""),
                metadata=data.get("metadata", {}),
            )
