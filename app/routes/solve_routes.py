from fastapi import APIRouter, Depends, HTTPException, Header

from app.schemas.solve import SolveRequest, SolveResponse
from app.services.openai_service import solve_with_openai


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> SolveResponse:
    result = await solve_with_openai(query_text=payload.text, api_key_override=x_openai_api_key)
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return result


