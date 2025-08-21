from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st

try:
    from .client import EchoApiClient
except Exception:  # When executed as a script (no package context)
    import sys
    from pathlib import Path

    current_file = Path(__file__).resolve()
    src_dir = current_file.parents[2]  # .../src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from echo.ui.client import EchoApiClient


def _get_backend_url() -> str:
    """Return the Echo API base URL.

    Reads the ``ECHO_API_BASE_URL`` environment variable and falls back to
    ``http://localhost:8000`` when unset.
    """
    return os.environ.get("ECHO_API_BASE_URL", "http://localhost:8000")


def _ensure_session_state() -> None:
    """Initialize Streamlit session state used by the chat UI.

    Ensures the following keys exist and are safe to use across reruns:
    - ``messages``: list of message dictionaries shown in the transcript
    - ``thread_id``: server-issued thread identifier or ``None``
    - ``client``: initialized ``EchoApiClient`` bound to the backend URL
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None
    if "client" not in st.session_state:
        st.session_state["client"] = EchoApiClient(base_url=_get_backend_url())
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = os.environ.get("ECHO_DEFAULT_USER_ID", "anonymous")
    if "org_id" not in st.session_state:
        st.session_state["org_id"] = os.environ.get("ECHO_DEFAULT_ORG_ID", "public")
    if "recent_threads" not in st.session_state:
        st.session_state["recent_threads"] = []
    if "show_settings" not in st.session_state:
        st.session_state["show_settings"] = False


def _send_message(user_text: str) -> Dict[str, Any]:
    """Send a user message to the backend and return the assistant reply.

    Persists ``thread_id`` in the session on first response so subsequent
    messages are associated with the same conversation.

    Returns a dictionary compatible with the UI transcript structure
    (``{"role": "assistant", "content": str, "metadata": dict}``).
    """
    client: EchoApiClient = st.session_state["client"]
    thread_id = st.session_state.get("thread_id")
    user_id = st.session_state.get("user_id", "anonymous")
    org_id = st.session_state.get("org_id", "public")
    result = client.chat(message=user_text, thread_id=thread_id, user_id=user_id, org_id=org_id)
    if not thread_id and result.thread_id:
        st.session_state["thread_id"] = result.thread_id
    # Track recent thread ids for quick loading
    if result.thread_id:
        recent: list[str] = st.session_state.get("recent_threads", [])
        if result.thread_id in recent:
            recent.remove(result.thread_id)
        recent.insert(0, result.thread_id)
        st.session_state["recent_threads"] = recent[:20]
    return {
        "role": "assistant",
        "content": result.response,
        "metadata": result.metadata,
    }


def main() -> None:
    """Render and run the Streamlit chat interface for Echo."""
    st.set_page_config(page_title="Echo Debug Chat", page_icon="🤖", layout="wide")
    st.title("Echo 🤖 Debug Chat UI")

    _ensure_session_state()

    top_left, top_right = st.columns([6, 1])
    with top_right:
        st.toggle("Settings", key="show_settings")

    if st.session_state.get("show_settings"):
        col_main, col_right = st.columns([4, 1], gap="large")

        with col_right:
            st.subheader("Settings")
            backend = _get_backend_url()
            st.text_input("Backend URL", value=backend, key="backend_url", disabled=True)
            st.text_input("User ID", key="user_id")
            st.text_input("Org ID", key="org_id")

        with col_main:
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    else:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pass

    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                assistant_msg = _send_message(user_text)
                st.markdown(assistant_msg["content"])
                st.session_state["messages"].append(assistant_msg)


if __name__ == "__main__":
    main()
