from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
import logging

from app.schemas.solve import SolveRequest, SolveResponse, SolveImageRequest
from app.services.solve_service import SolveService, get_solve_service


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(
    payload: SolveRequest, 
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> SolveResponse:
    return await solve_service.solve(payload=payload, api_key_override=x_openai_api_key)


@router.post("/solve/stream")
async def solve_stream_endpoint(
    payload: SolveRequest, 
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    logging.info("Starting streaming response")
    return StreamingResponse(
        solve_service.solve_stream(payload=payload, api_key_override=x_openai_api_key),
        media_type="text/event-stream"
    )


@router.post("/solve/image")
async def solve_image_endpoint(
    payload: SolveImageRequest, 
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> StreamingResponse:
    """Extract formula from image and stream the solution"""
    logging.info("Starting image processing and streaming response")
    return StreamingResponse(
        solve_service.solve_image(payload=payload, api_key_override=x_openai_api_key),
        media_type="text/event-stream"
    )
