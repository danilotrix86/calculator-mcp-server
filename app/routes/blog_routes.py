from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime
from app.services import blog_service

router = APIRouter(prefix="/blog", tags=["blog"])


@router.get("/warmup")
async def warmup():
    """
    Warmup endpoint to keep the server alive.
    Call this endpoint every 10 minutes via a cron service to prevent cold starts.
    This makes actual Supabase queries to warm up the database connection.
    """
    warmup_results = {
        "status": "warm",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Query 1: Fetch recent posts (warms up blog_posts, blog_categories, blog_authors tables)
    try:
        posts = await blog_service.get_recent_posts(1)
        warmup_results["checks"]["recent_posts"] = {
            "success": True,
            "count": len(posts)
        }
    except Exception as e:
        warmup_results["checks"]["recent_posts"] = {
            "success": False,
            "error": str(e)
        }
    
    # Query 2: Fetch categories (warms up blog_categories table separately)
    try:
        categories = await blog_service.get_all_categories()
        warmup_results["checks"]["categories"] = {
            "success": True,
            "count": len(categories)
        }
    except Exception as e:
        warmup_results["checks"]["categories"] = {
            "success": False,
            "error": str(e)
        }
    
    # Determine overall status
    all_success = all(
        check.get("success", False) 
        for check in warmup_results["checks"].values()
    )
    warmup_results["supabase_warm"] = all_success
    
    return warmup_results


@router.get("/posts")
async def get_posts(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Get all published blog posts with pagination."""
    return await blog_service.get_all_posts(page, limit)


@router.get("/posts/featured")
async def get_featured_posts(limit: int = Query(3, ge=1, le=10)):
    """Get featured blog posts."""
    posts = await blog_service.get_featured_posts(limit)
    return {"posts": posts}


@router.get("/posts/recent")
async def get_recent_posts(limit: int = Query(5, ge=1, le=20)):
    """Get recent blog posts."""
    posts = await blog_service.get_recent_posts(limit)
    return {"posts": posts}


@router.get("/posts/search")
async def search_posts(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50)
):
    """Search blog posts."""
    posts = await blog_service.search_posts(q, limit)
    return {"posts": posts}


@router.get("/posts/related/{post_id}")
async def get_related_posts(
    post_id: str,
    category_id: str = Query(...),
    limit: int = Query(3, ge=1, le=10)
):
    """Get related posts from the same category."""
    posts = await blog_service.get_related_posts(post_id, category_id, limit)
    return {"posts": posts}


@router.get("/posts/{slug}")
async def get_post_by_slug(slug: str):
    """Get a single blog post by slug."""
    post = await blog_service.get_post_by_slug(slug)
    if not post:
        return {"error": "Post not found", "post": None}
    return {"post": post}


@router.get("/categories")
async def get_categories():
    """Get all blog categories with post counts."""
    categories = await blog_service.get_all_categories()
    return {"categories": categories}


@router.get("/categories/{slug}")
async def get_posts_by_category(
    slug: str,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Get posts by category slug."""
    return await blog_service.get_posts_by_category(slug, page, limit)




