import logging
from typing import Optional, List, Dict, Any
from app.services.supabase_service import get_async_supabase_client


async def get_all_posts(page: int = 1, limit: int = 10) -> Dict[str, Any]:
    """
    Get all published blog posts with pagination.
    """
    client = get_async_supabase_client()
    if not client:
        return {"posts": [], "total": 0}
    
    try:
        start = (page - 1) * limit
        end = start + limit - 1
        
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)", count="exact") \
            .eq("published", True) \
            .order("created_at", desc=True) \
            .range(start, end) \
            .execute()
        
        return {
            "posts": result.data or [],
            "total": result.count or 0
        }
    except Exception as e:
        logging.error(f"Error fetching posts: {str(e)}")
        return {"posts": [], "total": 0}


async def get_featured_posts(limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get featured published blog posts.
    """
    client = get_async_supabase_client()
    if not client:
        return []
    
    try:
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("published", True) \
            .eq("featured", True) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching featured posts: {str(e)}")
        return []


async def get_post_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """
    Get a single blog post by slug.
    """
    client = get_async_supabase_client()
    if not client:
        return None
    
    try:
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("slug", slug) \
            .eq("published", True) \
            .single() \
            .execute()
        
        return result.data
    except Exception as e:
        logging.error(f"Error fetching post by slug: {str(e)}")
        return None


async def get_posts_by_category(category_slug: str, page: int = 1, limit: int = 10) -> Dict[str, Any]:
    """
    Get posts by category slug.
    """
    client = get_async_supabase_client()
    if not client:
        return {"posts": [], "category": None, "total": 0}
    
    try:
        cat_result = await client.table("blog_categories") \
            .select("*") \
            .eq("slug", category_slug) \
            .single() \
            .execute()
        
        if not cat_result.data:
            return {"posts": [], "category": None, "total": 0}
        
        category = cat_result.data
        
        start = (page - 1) * limit
        end = start + limit - 1
        
        posts_result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)", count="exact") \
            .eq("category_id", category["id"]) \
            .eq("published", True) \
            .order("created_at", desc=True) \
            .range(start, end) \
            .execute()
        
        return {
            "posts": posts_result.data or [],
            "category": category,
            "total": posts_result.count or 0
        }
    except Exception as e:
        logging.error(f"Error fetching posts by category: {str(e)}")
        return {"posts": [], "category": None, "total": 0}


async def get_all_categories() -> List[Dict[str, Any]]:
    """
    Get all categories with post counts.
    """
    client = get_async_supabase_client()
    if not client:
        return []
    
    try:
        cat_result = await client.table("blog_categories") \
            .select("*") \
            .order("name") \
            .execute()
        
        categories = cat_result.data or []
        
        posts_result = await client.table("blog_posts") \
            .select("category_id") \
            .eq("published", True) \
            .execute()
        
        count_map = {}
        for post in posts_result.data or []:
            cat_id = post.get("category_id")
            if cat_id:
                count_map[cat_id] = count_map.get(cat_id, 0) + 1
        
        for cat in categories:
            cat["post_count"] = count_map.get(cat["id"], 0)
        
        return categories
    except Exception as e:
        logging.error(f"Error fetching categories: {str(e)}")
        return []


async def get_recent_posts(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent published posts.
    """
    client = get_async_supabase_client()
    if not client:
        return []
    
    try:
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("published", True) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching recent posts: {str(e)}")
        return []


async def get_related_posts(post_id: str, category_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get related posts from the same category.
    """
    client = get_async_supabase_client()
    if not client:
        return []
    
    try:
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("category_id", category_id) \
            .eq("published", True) \
            .neq("id", post_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching related posts: {str(e)}")
        return []


async def search_posts(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search posts by title, excerpt, or content.
    """
    client = get_async_supabase_client()
    if not client:
        return []
    
    try:
        result = await client.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("published", True) \
            .or_(f"title.ilike.%{query}%,excerpt.ilike.%{query}%,content.ilike.%{query}%") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    except Exception as e:
        logging.error(f"Error searching posts: {str(e)}")
        return []
