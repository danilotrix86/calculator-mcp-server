import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from app.services.supabase_service import supabase


# ============== AUTHORS ==============

async def get_all_authors() -> List[Dict[str, Any]]:
    """Get all authors."""
    if not supabase:
        return []
    
    try:
        result = supabase.table("blog_authors") \
            .select("*") \
            .order("name") \
            .execute()
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching authors: {str(e)}")
        return []


async def create_author(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new author."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("blog_authors") \
            .insert(data) \
            .execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logging.error(f"Error creating author: {str(e)}")
        return None


# ============== CATEGORIES ==============

async def get_all_categories_admin() -> List[Dict[str, Any]]:
    """Get all categories for admin."""
    if not supabase:
        return []
    
    try:
        result = supabase.table("blog_categories") \
            .select("*") \
            .order("name") \
            .execute()
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching categories: {str(e)}")
        return []


async def create_category(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new category."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("blog_categories") \
            .insert(data) \
            .execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logging.error(f"Error creating category: {str(e)}")
        return None


async def update_category(category_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a category."""
    if not supabase:
        return None
    
    try:
        # Update the category
        supabase.table("blog_categories") \
            .update(data) \
            .eq("id", category_id) \
            .execute()
        
        # Fetch and return the updated category
        result = supabase.table("blog_categories") \
            .select("*") \
            .eq("id", category_id) \
            .single() \
            .execute()
        return result.data
    except Exception as e:
        logging.error(f"Error updating category: {str(e)}")
        return None


async def delete_category(category_id: str) -> bool:
    """Delete a category."""
    if not supabase:
        return False
    
    try:
        supabase.table("blog_categories") \
            .delete() \
            .eq("id", category_id) \
            .execute()
        return True
    except Exception as e:
        logging.error(f"Error deleting category: {str(e)}")
        return False


# ============== POSTS ==============

async def get_all_posts_admin(page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """Get all posts (including unpublished) for admin."""
    if not supabase:
        return {"posts": [], "total": 0}
    
    try:
        start = (page - 1) * limit
        end = start + limit - 1
        
        result = supabase.table("blog_posts") \
            .select("*, category:blog_categories(id, name, slug), author:blog_authors(id, name)", count="exact") \
            .order("created_at", desc=True) \
            .range(start, end) \
            .execute()
        
        return {
            "posts": result.data or [],
            "total": result.count or 0
        }
    except Exception as e:
        logging.error(f"Error fetching posts for admin: {str(e)}")
        return {"posts": [], "total": 0}


async def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """Get a post by ID for editing."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("id", post_id) \
            .single() \
            .execute()
        return result.data
    except Exception as e:
        logging.error(f"Error fetching post by ID: {str(e)}")
        return None


async def create_post(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new post."""
    if not supabase:
        raise ValueError("Supabase client not initialized")
    
    try:
        # Handle published_at field based on published status
        if data.get("published"):
            data["published_at"] = datetime.now(timezone.utc).isoformat()
        else:
            data["published_at"] = None
        
        logging.info(f"Creating post with data: {data}")
        result = supabase.table("blog_posts") \
            .insert(data) \
            .execute()
        if not result.data:
            raise ValueError("No data returned from insert")
        return result.data[0]
    except Exception as e:
        logging.error(f"Error creating post: {str(e)}")
        raise


async def update_post(post_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Update a post."""
    if not supabase:
        raise ValueError("Supabase client not initialized")
    
    try:
        # Handle published_at field based on published status
        if "published" in data:
            if data["published"]:
                # Only set published_at if it's being published for the first time
                # Check if already has published_at
                existing = supabase.table("blog_posts") \
                    .select("published_at") \
                    .eq("id", post_id) \
                    .single() \
                    .execute()
                if existing.data and not existing.data.get("published_at"):
                    data["published_at"] = datetime.now(timezone.utc).isoformat()
            else:
                data["published_at"] = None
        
        logging.info(f"Updating post {post_id} with data: {data}")
        # Update the post
        supabase.table("blog_posts") \
            .update(data) \
            .eq("id", post_id) \
            .execute()
        
        # Fetch and return the updated post
        result = supabase.table("blog_posts") \
            .select("*") \
            .eq("id", post_id) \
            .single() \
            .execute()
        if not result.data:
            raise ValueError("Post not found after update")
        return result.data
    except Exception as e:
        logging.error(f"Error updating post: {str(e)}")
        raise


async def delete_post(post_id: str) -> bool:
    """Delete a post."""
    if not supabase:
        return False
    
    try:
        supabase.table("blog_posts") \
            .delete() \
            .eq("id", post_id) \
            .execute()
        return True
    except Exception as e:
        logging.error(f"Error deleting post: {str(e)}")
        return False

