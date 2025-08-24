"""Test cases for the chat API router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from echo.api.routers.chat import router


class TestChatRouter:
    """Test cases for the chat router."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with the chat router."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client for the app."""
        return TestClient(app)

    def test_chat_router_includes_correct_paths(self, app):
        """Test that the chat router includes the correct paths."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/chat" in routes
        assert "/api/v1/chat/{conversation_id}" in routes
        assert "/api/v1/chat/{conversation_id}/messages" in routes

    @pytest.mark.asyncio
    async def test_create_chat_conversation_success(self, client):
        """Test successful chat conversation creation."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.create_conversation.return_value = {
                "conversation_id": "conv-123",
                "status": "created",
                "created_at": "2024-01-01T00:00:00Z",
            }

            response = client.post(
                "/api/v1/chat",
                json={"user_id": "user-123", "initial_message": "Hello, how are you?", "metadata": {"source": "web"}},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["conversation_id"] == "conv-123"
            assert data["status"] == "created"

            mock_service.create_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_chat_conversation_without_initial_message(self, client):
        """Test chat conversation creation without initial message."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.create_conversation.return_value = {
                "conversation_id": "conv-123",
                "status": "created",
                "created_at": "2024-01-01T00:00:00Z",
            }

            response = client.post("/api/v1/chat", json={"user_id": "user-123", "metadata": {"source": "web"}})

            assert response.status_code == 201
            data = response.json()
            assert data["conversation_id"] == "conv-123"

    @pytest.mark.asyncio
    async def test_create_chat_conversation_validation_error(self, client):
        """Test chat conversation creation with validation error."""
        response = client.post(
            "/api/v1/chat", json={"user_id": "", "initial_message": "Hello"}  # Invalid empty user_id
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_chat_conversation_service_error(self, client):
        """Test chat conversation creation with service error."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.create_conversation.side_effect = Exception("Service error")

            response = client.post("/api/v1/chat", json={"user_id": "user-123", "initial_message": "Hello"})

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_send_message_success(self, client):
        """Test successful message sending."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.return_value = {
                "message_id": "msg-123",
                "conversation_id": "conv-123",
                "content": "I'm doing well, thank you!",
                "agent_hops": 1,
                "tool_hops": 2,
                "created_at": "2024-01-01T00:00:00Z",
            }

            response = client.post(
                "/api/v1/chat/conv-123", json={"message": "How are you?", "metadata": {"source": "web"}}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message_id"] == "msg-123"
            assert data["conversation_id"] == "conv-123"
            assert data["content"] == "I'm doing well, thank you!"
            assert data["agent_hops"] == 1
            assert data["tool_hops"] == 2

            mock_service.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_validation_error(self, client):
        """Test message sending with validation error."""
        response = client.post(
            "/api/v1/chat/conv-123", json={"message": "", "metadata": {"source": "web"}}  # Invalid empty message
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_send_message_conversation_not_found(self, client):
        """Test message sending with non-existent conversation."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.side_effect = ValueError("Conversation not found")

            response = client.post(
                "/api/v1/chat/non-existent", json={"message": "Hello", "metadata": {"source": "web"}}
            )

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_service_error(self, client):
        """Test message sending with service error."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.side_effect = Exception("Service error")

            response = client.post("/api/v1/chat/conv-123", json={"message": "Hello", "metadata": {"source": "web"}})

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_conversation_messages_success(self, client):
        """Test successful retrieval of conversation messages."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.get_conversation_messages.return_value = {
                "conversation_id": "conv-123",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello", "created_at": "2024-01-01T00:00:00Z"},
                    {"id": "msg-2", "role": "assistant", "content": "Hi there!", "created_at": "2024-01-01T00:01:00Z"},
                ],
                "total_count": 2,
            }

            response = client.get("/api/v1/chat/conv-123/messages")

            assert response.status_code == 200
            data = response.json()
            assert data["conversation_id"] == "conv-123"
            assert len(data["messages"]) == 2
            assert data["total_count"] == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"

            mock_service.get_conversation_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_messages_with_pagination(self, client):
        """Test retrieval of conversation messages with pagination."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.get_conversation_messages.return_value = {
                "conversation_id": "conv-123",
                "messages": [
                    {"id": "msg-2", "role": "assistant", "content": "Hi there!", "created_at": "2024-01-01T00:01:00Z"}
                ],
                "total_count": 2,
            }

            response = client.get("/api/v1/chat/conv-123/messages?limit=1&offset=1")

            assert response.status_code == 200
            data = response.json()
            assert len(data["messages"]) == 1
            assert data["total_count"] == 2

            # Verify pagination parameters were passed
            call_args = mock_service.get_conversation_messages.call_args
            assert call_args[1]["limit"] == 1
            assert call_args[1]["offset"] == 1

    @pytest.mark.asyncio
    async def test_get_conversation_messages_conversation_not_found(self, client):
        """Test message retrieval with non-existent conversation."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.get_conversation_messages.side_effect = ValueError("Conversation not found")

            response = client.get("/api/v1/chat/non-existent/messages")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_conversation_messages_service_error(self, client):
        """Test message retrieval with service error."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.get_conversation_messages.side_effect = Exception("Service error")

            response = client.get("/api/v1/chat/conv-123/messages")

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_conversation_messages_invalid_pagination(self, client):
        """Test message retrieval with invalid pagination parameters."""
        response = client.get("/api/v1/chat/conv-123/messages?limit=-1")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_conversation_messages_empty_conversation(self, client):
        """Test message retrieval from empty conversation."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.get_conversation_messages.return_value = {
                "conversation_id": "conv-123",
                "messages": [],
                "total_count": 0,
            }

            response = client.get("/api/v1/chat/conv-123/messages")

            assert response.status_code == 200
            data = response.json()
            assert data["conversation_id"] == "conv-123"
            assert len(data["messages"]) == 0
            assert data["total_count"] == 0

    @pytest.mark.asyncio
    async def test_send_message_with_complex_metadata(self, client):
        """Test message sending with complex metadata."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.return_value = {
                "message_id": "msg-123",
                "conversation_id": "conv-123",
                "content": "Response",
                "agent_hops": 1,
                "tool_hops": 0,
                "created_at": "2024-01-01T00:00:00Z",
            }

            complex_metadata = {
                "source": "web",
                "user_agent": "Mozilla/5.0...",
                "ip_address": "192.168.1.1",
                "session_data": {"session_id": "sess-123", "user_preferences": {"language": "en"}},
                "request_id": "req-456",
            }

            response = client.post(
                "/api/v1/chat/conv-123", json={"message": "Hello with complex metadata", "metadata": complex_metadata}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message_id"] == "msg-123"

            # Verify metadata was passed to service
            call_args = mock_service.send_message.call_args
            assert call_args[1]["metadata"] == complex_metadata

    @pytest.mark.asyncio
    async def test_create_conversation_with_complex_metadata(self, client):
        """Test conversation creation with complex metadata."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.create_conversation.return_value = {
                "conversation_id": "conv-123",
                "status": "created",
                "created_at": "2024-01-01T00:00:00Z",
            }

            complex_metadata = {
                "source": "mobile_app",
                "app_version": "1.2.3",
                "device_info": {"platform": "iOS", "version": "15.0"},
                "user_preferences": {"language": "es", "timezone": "America/New_York"},
            }

            response = client.post(
                "/api/v1/chat",
                json={"user_id": "user-123", "initial_message": "Hello from mobile", "metadata": complex_metadata},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["conversation_id"] == "conv-123"

            # Verify metadata was passed to service
            call_args = mock_service.create_conversation.call_args
            assert call_args[1]["metadata"] == complex_metadata

    @pytest.mark.asyncio
    async def test_send_message_with_long_content(self, client):
        """Test message sending with long content."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.return_value = {
                "message_id": "msg-123",
                "conversation_id": "conv-123",
                "content": "Response to long message",
                "agent_hops": 1,
                "tool_hops": 0,
                "created_at": "2024-01-01T00:00:00Z",
            }

            long_message = "This is a very long message " * 100  # 2500 characters

            response = client.post(
                "/api/v1/chat/conv-123", json={"message": long_message, "metadata": {"source": "web"}}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message_id"] == "msg-123"

    @pytest.mark.asyncio
    async def test_send_message_with_special_characters(self, client):
        """Test message sending with special characters."""
        with patch("echo.api.routers.chat.get_orchestrator_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            mock_service.send_message.return_value = {
                "message_id": "msg-123",
                "conversation_id": "conv-123",
                "content": "Response with special chars",
                "agent_hops": 1,
                "tool_hops": 0,
                "created_at": "2024-01-01T00:00:00Z",
            }

            special_message = "Hello! How are you? 😊 This has emojis and special chars: áéíóú ñ"

            response = client.post(
                "/api/v1/chat/conv-123", json={"message": special_message, "metadata": {"source": "web"}}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message_id"] == "msg-123"
