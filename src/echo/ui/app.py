from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

try:
    from .client import EchoApiClient, PluginInfo, SystemStatus
except Exception:
    import sys
    from pathlib import Path

    current_file = Path(__file__).resolve()
    src_dir = current_file.parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from echo.ui.client import EchoApiClient, PluginInfo, SystemStatus


def get_backend_url() -> str:
    """Return the Echo API base URL from environment or default."""
    return os.environ.get("ECHO_API_BASE_URL", "http://localhost:8000")


def initialize_session_state() -> None:
    """Initialize Streamlit session state for the application."""
    defaults = {
        "messages": [],
        "thread_id": None,
        "client": EchoApiClient(base_url=get_backend_url()),
        "user_id": os.environ.get("ECHO_DEFAULT_USER_ID", "anonymous"),
        "org_id": os.environ.get("ECHO_DEFAULT_ORG_ID", "public"),
        "recent_threads": [],
        "sidebar_section": "Chat",
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def send_chat_message(user_text: str) -> Dict[str, Any]:
    """Send a user message and return the assistant's response."""
    client: EchoApiClient = st.session_state["client"]
    thread_id = st.session_state.get("thread_id")
    user_id = st.session_state.get("user_id", "anonymous")
    org_id = st.session_state.get("org_id", "public")

    result = client.chat(message=user_text, thread_id=thread_id, user_id=user_id, org_id=org_id)

    if not thread_id and result.thread_id:
        st.session_state["thread_id"] = result.thread_id

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


def display_chat_message(message: Dict[str, Any]) -> None:
    """Display a single chat message with optional metadata."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("metadata"):
            with st.expander("🔍 Response Metadata", expanded=False):
                st.json(message["metadata"])


def render_chat_section() -> None:
    """Render the main chat interface."""
    st.header("💬 Chat")

    for message in st.session_state["messages"]:
        display_chat_message(message)

    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                assistant_msg = send_chat_message(user_text)
                st.markdown(assistant_msg["content"])

                # Display metadata with icon if available
                if assistant_msg.get("metadata"):
                    st.markdown("---")
                    with st.expander("🔍 Response Metadata", expanded=False):
                        st.json(assistant_msg["metadata"])

                st.session_state["messages"].append(assistant_msg)


def display_plugin_info(plugin: PluginInfo) -> None:
    """Display information for a single plugin."""
    with st.expander(f"**{plugin.name}** v{plugin.version}"):
        st.write(f"**Description:** {plugin.description}")
        st.write(f"**Capabilities:** {', '.join(plugin.capabilities)}")
        if plugin.status == "healthy":
            st.success("Status: Healthy")
        else:
            st.error("Status: Failed")


def render_plugin_manager_section() -> None:
    """Render the plugin manager section."""
    st.header("🔌 Plugin Manager")

    client: EchoApiClient = st.session_state["client"]

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reload Plugins"):
            with st.spinner("Reloading plugins..."):
                try:
                    result = client.reload_plugins()
                    st.success("Plugins reloaded successfully!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Failed to reload plugins: {str(e)}")

    try:
        plugins = client.get_plugins()

        if not plugins:
            st.info("No plugins found.")
            return

        healthy_plugins = [p for p in plugins if p.status == "healthy"]
        failed_plugins = [p for p in plugins if p.status == "failed"]

        if healthy_plugins:
            st.subheader("✅ Healthy Plugins")
            for plugin in healthy_plugins:
                display_plugin_info(plugin)

        if failed_plugins:
            st.subheader("❌ Failed Plugins")
            for plugin in failed_plugins:
                display_plugin_info(plugin)

    except Exception as e:
        st.error(f"Failed to fetch plugins: {str(e)}")


def render_system_info_section() -> None:
    """Render the system info section."""
    st.header("ℹ️ System Info")

    client: EchoApiClient = st.session_state["client"]

    try:
        status = client.get_system_status()

        st.metric("System Status", status.status)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Plugins", len(status.available_plugins))
        with col2:
            st.metric("Healthy", len(status.healthy_plugins))
        with col3:
            st.metric("Failed", len(status.failed_plugins))

        st.metric("Active Sessions", status.total_sessions)

        with st.expander("Available Plugins"):
            if status.available_plugins:
                for plugin in status.available_plugins:
                    if plugin in status.healthy_plugins:
                        st.write(f"✅ {plugin}")
                    else:
                        st.write(f"❌ {plugin}")
            else:
                st.write("No plugins available")

        with st.expander("System Details"):
            st.json(
                {
                    "status": status.status,
                    "available_plugins": status.available_plugins,
                    "healthy_plugins": status.healthy_plugins,
                    "failed_plugins": status.failed_plugins,
                    "total_sessions": status.total_sessions,
                }
            )

    except Exception as e:
        st.error(f"Failed to fetch system status: {str(e)}")


def render_sidebar() -> None:
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("Echo 🤖 Debug Chat")
        st.divider()

        if st.button("💬 Chat", use_container_width=True):
            st.session_state["sidebar_section"] = "Chat"
        if st.button("🔌 Plugin Manager", use_container_width=True):
            st.session_state["sidebar_section"] = "Plugin Manager"
        if st.button("ℹ️ System Info", use_container_width=True):
            st.session_state["sidebar_section"] = "System Info"

        st.divider()
        st.write(f"**Current:** {st.session_state['sidebar_section']}")
        st.caption(f"Backend: {get_backend_url()}")


def render_main_content() -> None:
    """Render the main content based on the selected sidebar section."""
    section_renderers = {
        "Chat": render_chat_section,
        "Plugin Manager": render_plugin_manager_section,
        "System Info": render_system_info_section,
    }

    current_section = st.session_state["sidebar_section"]
    renderer = section_renderers.get(current_section)

    if renderer:
        renderer()


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Echo Debug Chat", page_icon="🤖")

    initialize_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    load_dotenv()
    main()
