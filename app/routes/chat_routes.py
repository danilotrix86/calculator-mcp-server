import json
import logging
from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.chat_service import chat_with_openai_streaming
from app.middleware.rate_limit import limiter, _get_solve_key
from app.config import rate_limit_config


router = APIRouter(tags=["chat"])


async def _sse_generator(request: ChatRequest, api_key_override: str | None):
    async for chunk in chat_with_openai_streaming(request, api_key_override=api_key_override):
        yield f"data: {chunk}\n\n"


@router.post("/chat/session")
@limiter.limit(rate_limit_config.CHAT, key_func=_get_solve_key)
async def chat_session_endpoint(
    request: Request,
    payload: ChatRequest,
    x_user_id: str | None = Header(default=None),
    x_openai_api_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream a chat tutor response given conversation history and problem context."""
    return StreamingResponse(
        _sse_generator(payload, api_key_override=x_openai_api_key),
        media_type="text/event-stream",
    )
