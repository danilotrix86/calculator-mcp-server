from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import logging

from app.schemas.solve import SolveRequest, SolveResponse
from app.services.openai_service import solve_with_openai, solve_with_openai_streaming


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> SolveResponse:
    logging.info("Processing solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    result = await solve_with_openai(query_text=payload.text, api_key_override=x_openai_api_key)
    if result.error:
        logging.error("Solve request error: %s", result.error)
        raise HTTPException(status_code=400, detail=result.error)
    logging.info("Solve request completed successfully. Tool used: %s", result.tool_called if result.used_tool else "None")
    return result


@router.post("/solve/stream")
async def solve_stream_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    logging.info("Processing streaming solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    
    async def stream_generator() -> AsyncIterator[str]:
        async for chunk in solve_with_openai_streaming(query_text=payload.text, api_key_override=x_openai_api_key, force_tool=True):
            yield f"data: {chunk}\n\n"
    
    logging.info("Starting streaming response")
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


