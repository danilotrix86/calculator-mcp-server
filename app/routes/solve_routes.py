from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

from app.schemas.solve import SolveRequest, SolveResponse
from app.services.openai_service import solve_with_openai, solve_with_openai_streaming


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> SolveResponse:
    result = await solve_with_openai(query_text=payload.text, api_key_override=x_openai_api_key)
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return result


@router.post("/solve/stream")
async def solve_stream_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    async def stream_generator() -> AsyncIterator[str]:
        async for chunk in solve_with_openai_streaming(query_text=payload.text, api_key_override=x_openai_api_key, force_tool=True):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


