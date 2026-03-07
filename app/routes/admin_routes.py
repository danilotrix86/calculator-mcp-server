from fastapi import APIRouter, Query, Depends, HTTPException, Request
from typing import Optional
from pydantic import BaseModel
from app.middleware.auth import verify_admin
from app.middleware.rate_limit import limiter
from app.config import rate_limit_config
from app.services import blog_admin_service
from app.services.supabase_service import get_supabase_service
from app.services import retention_service

router = APIRouter(prefix="/admin", tags=["admin"])


# ============== SCHEMAS ==============

class AuthorCreate(BaseModel):
    name: str
    slug: str
    bio: Optional[str] = None
    avatar: Optional[str] = None
    twitter: Optional[str] = None
    linkedin: Optional[str] = None


class CategoryCreate(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    color: str = "#6366f1"
    icon: Optional[str] = None


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    slug: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None


class PostCreate(BaseModel):
    title: str
    slug: str
    excerpt: str
    content: str
    featured_image: Optional[str] = None
    category_id: str
    author_id: str
    published: bool = False
    featured: bool = False
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    reading_time: int = 5


class PostUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    excerpt: Optional[str] = None
    content: Optional[str] = None
    featured_image: Optional[str] = None
    category_id: Optional[str] = None
    author_id: Optional[str] = None
    published: Optional[bool] = None
    featured: Optional[bool] = None
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    reading_time: Optional[int] = None


# ============== AUTH CHECK ==============

@router.get("/check")
@limiter.limit(rate_limit_config.ADMIN)
async def check_auth(request: Request, username: str = Depends(verify_admin)):
    """Check if admin credentials are valid."""
    return {"authenticated": True, "username": username}


# ============== AUTHORS ==============

@router.get("/authors")
@limiter.limit(rate_limit_config.ADMIN)
async def get_authors(request: Request, username: str = Depends(verify_admin)):
    """Get all authors."""
    authors = await blog_admin_service.get_all_authors()
    return {"authors": authors}


@router.post("/authors")
@limiter.limit(rate_limit_config.ADMIN)
async def create_author(request: Request, data: AuthorCreate, username: str = Depends(verify_admin)):
    """Create a new author."""
    author = await blog_admin_service.create_author(data.model_dump())
    if not author:
        raise HTTPException(status_code=500, detail="Failed to create author")
    return {"author": author}


# ============== CATEGORIES ==============

@router.get("/categories")
@limiter.limit(rate_limit_config.ADMIN)
async def get_categories(request: Request, username: str = Depends(verify_admin)):
    """Get all categories."""
    categories = await blog_admin_service.get_all_categories_admin()
    return {"categories": categories}


@router.post("/categories")
@limiter.limit(rate_limit_config.ADMIN)
async def create_category(request: Request, data: CategoryCreate, username: str = Depends(verify_admin)):
    """Create a new category."""
    category = await blog_admin_service.create_category(data.model_dump())
    if not category:
        raise HTTPException(status_code=500, detail="Failed to create category")
    return {"category": category}


@router.put("/categories/{category_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def update_category(request: Request, category_id: str, data: CategoryUpdate, username: str = Depends(verify_admin)):
    """Update a category."""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    category = await blog_admin_service.update_category(category_id, update_data)
    if not category:
        raise HTTPException(status_code=500, detail="Failed to update category")
    return {"category": category}


@router.delete("/categories/{category_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def delete_category(request: Request, category_id: str, username: str = Depends(verify_admin)):
    """Delete a category."""
    success = await blog_admin_service.delete_category(category_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete category")
    return {"success": True}


# ============== POSTS ==============

@router.get("/posts")
@limiter.limit(rate_limit_config.ADMIN)
async def get_posts(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    username: str = Depends(verify_admin)
):
    """Get all posts (including unpublished)."""
    return await blog_admin_service.get_all_posts_admin(page, limit)


@router.get("/posts/{post_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def get_post(request: Request, post_id: str, username: str = Depends(verify_admin)):
    """Get a single post by ID."""
    post = await blog_admin_service.get_post_by_id(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"post": post}


@router.post("/posts")
@limiter.limit(rate_limit_config.ADMIN)
async def create_post(request: Request, data: PostCreate, username: str = Depends(verify_admin)):
    """Create a new post."""
    try:
        post = await blog_admin_service.create_post(data.model_dump())
        return {"post": post}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create post: {str(e)}")


@router.put("/posts/{post_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def update_post(request: Request, post_id: str, data: PostUpdate, username: str = Depends(verify_admin)):
    """Update a post."""
    try:
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}
        post = await blog_admin_service.update_post(post_id, update_data)
        return {"post": post}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update post: {str(e)}")


@router.delete("/posts/{post_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def delete_post(request: Request, post_id: str, username: str = Depends(verify_admin)):
    """Delete a post."""
    success = await blog_admin_service.delete_post(post_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete post")
    return {"success": True}


# ============== REQUESTS ==============

@router.get("/requests")
@limiter.limit(rate_limit_config.ADMIN)
async def get_requests(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    username: str = Depends(verify_admin)
):
    """Get all user requests (admin view)."""
    supabase_service = get_supabase_service()
    requests_data, total = await supabase_service.get_all_requests_admin(page, limit)
    
    items = []
    for req in requests_data:
        response_text = req.get("response") or ""
        response_preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
        
        user_info = req.get("app_users")
        user_name = user_info.get("name") if user_info else None
        user_email = user_info.get("email") if user_info else None
        
        query_type = req.get("query_type")
        query_text = req.get("query_text") or ""
        question = req.get("question") or ""
        # For text queries, prefer query_text (original with spaces);
        # for image queries, prefer question (updated with extracted formula).
        if query_type == "image" or query_text.startswith("Image upload"):
            display_question = question
        else:
            display_question = query_text or question
        
        items.append({
            "id": req.get("id"),
            "userId": req.get("user_id"),
            "userName": user_name,
            "userEmail": user_email,
            "question": display_question,
            "responsePreview": response_preview,
            "queryType": req.get("query_type"),
            "toolUsed": req.get("tool_used"),
            "createdAt": req.get("created_at"),
        })
    
    return {"requests": items, "total": total}


@router.get("/requests/{request_id}")
@limiter.limit(rate_limit_config.ADMIN)
async def get_request(request: Request, request_id: str, username: str = Depends(verify_admin)):
    """Get a single request by ID (admin view)."""
    supabase_service = get_supabase_service()
    req = await supabase_service.get_request_by_id_admin(request_id)
    
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    
    user_info = req.get("app_users")
    user_name = user_info.get("name") if user_info else None
    user_email = user_info.get("email") if user_info else None
    
    query_type = req.get("query_type")
    query_text = req.get("query_text") or ""
    question = req.get("question") or ""
    if query_type == "image" or query_text.startswith("Image upload"):
        display_question = question
    else:
        display_question = query_text or question
    
    return {
        "request": {
            "id": req.get("id"),
            "userId": req.get("user_id"),
            "userName": user_name,
            "userEmail": user_email,
            "question": display_question,
            "response": req.get("response"),
            "queryType": req.get("query_type"),
            "toolUsed": req.get("tool_used"),
            "createdAt": req.get("created_at"),
        }
    }

# ============== RETENTION ==============

@router.get("/retention")
@limiter.limit(rate_limit_config.ADMIN)
async def get_retention(request: Request, username: str = Depends(verify_admin)):
    """Get user retention analytics."""
    data = await retention_service.get_retention_metrics()
    return data




