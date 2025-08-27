"""Chat endpoints for conversation processing via the multi-agent orchestrator.

Provides POST /chat to process a user message within an existing or new thread.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.services import ConversationService
from ...domain.dtos.chat_dtos import ChatRequest, ChatResponse
from ..services import global_service_container

chat_api_router = APIRouter()


def get_conversation_service() -> ConversationService:
    """Return the conversation service from the global container.

    Raises RuntimeError if the container has not been initialized.
    """
    return global_service_container.get_conversation_service()


@chat_api_router.post("/chat", response_model=ChatResponse)
async def process_chat_message(
    chat_request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    """Process a user message and return the agent response.

    Uses the orchestrator to route, persist, and enrich the response with
    conversation metadata.
    """
    return await conversation_service.process_message(chat_request)


router = chat_api_router
