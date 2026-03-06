import math
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.services.supabase_service import SupabaseService, get_supabase_service

router = APIRouter(tags=["history"])

MAX_PREVIEW_LENGTH = 200


def _truncate(text: str, max_len: int = MAX_PREVIEW_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


@router.get("/history")
async def get_history(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=50),
    x_user_id: Optional[str] = Header(default=None),
    supabase_service: SupabaseService = Depends(get_supabase_service),
):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing user id")

    items, total = await supabase_service.get_user_history(
        user_id=x_user_id,
        page=page,
        limit=limit,
    )

    total_pages = max(1, math.ceil(total / limit))

    data = []
    for row in items:
        data.append({
            "id": row["id"],
            "question": row.get("question", ""),
            "responsePreview": _truncate(row.get("response", "")),
            "toolUsed": row.get("tool_used"),
            "queryType": row.get("query_type"),
            "createdAt": row.get("created_at"),
        })

    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "totalPages": total_pages,
        "hasMore": page < total_pages,
    }


@router.get("/history/{query_id}")
async def get_history_item(
    query_id: str,
    x_user_id: Optional[str] = Header(default=None),
    supabase_service: SupabaseService = Depends(get_supabase_service),
):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing user id")

    item = await supabase_service.get_query_by_id(query_id, x_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Query not found")

    return {
        "id": item["id"],
        "question": item.get("question", ""),
        "response": item.get("response", ""),
        "toolUsed": item.get("tool_used"),
        "queryType": item.get("query_type"),
        "createdAt": item.get("created_at"),
    }
