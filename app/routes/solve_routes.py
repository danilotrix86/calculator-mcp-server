from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
import logging

from app.schemas.solve import SolveRequest, SolveResponse, SolveImageRequest
from app.services.solve_service import SolveService, get_solve_service
from app.middleware.rate_limit import limiter, _get_solve_key
from app.config import rate_limit_config


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
@limiter.limit(rate_limit_config.SOLVE, key_func=_get_solve_key)
async def solve_endpoint(
    request: Request,
    payload: SolveRequest,
    x_user_id: str | None = Header(default=None),
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> SolveResponse:
    return await solve_service.solve(payload=payload, user_id=x_user_id, api_key_override=x_openai_api_key)


@router.post("/solve/stream")
@limiter.limit(rate_limit_config.SOLVE, key_func=_get_solve_key)
async def solve_stream_endpoint(
    request: Request,
    payload: SolveRequest,
    x_user_id: str | None = Header(default=None),
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    logging.info("Starting streaming response")
    return StreamingResponse(
        solve_service.solve_stream(payload=payload, user_id=x_user_id, api_key_override=x_openai_api_key),
        media_type="text/event-stream"
    )


@router.post("/solve/image")
@limiter.limit(rate_limit_config.SOLVE, key_func=_get_solve_key)
async def solve_image_endpoint(
    request: Request,
    payload: SolveImageRequest,
    x_user_id: str | None = Header(default=None),
    x_openai_api_key: str | None = Header(default=None),
    solve_service: SolveService = Depends(get_solve_service)
) -> StreamingResponse:
    """Extract formula from image and stream the solution"""
    logging.info("Starting image processing and streaming response")
    return StreamingResponse(
        solve_service.solve_image(payload=payload, user_id=x_user_id, api_key_override=x_openai_api_key),
        media_type="text/event-stream"
    )
