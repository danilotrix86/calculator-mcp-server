import logging
from typing import Optional, List, Dict, Any
from app.services.supabase_service import supabase
from fastapi.concurrency import run_in_threadpool


# ============== AUTHORS ==============

async def get_all_authors() -> List[Dict[str, Any]]:
    """Get all authors."""
    if not supabase:
        return []
    
    try:
        query = supabase.table("blog_authors") \
            .select("*") \
            .order("name") \
            
        result = await run_in_threadpool(query.execute)
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching authors: {str(e)}")
        return []


async def create_author(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new author."""
    if not supabase:
        return None
    
    try:
        query = supabase.table("blog_authors") \
            .insert(data) \
            
        result = await run_in_threadpool(query.execute)
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
        query = supabase.table("blog_categories") \
            .select("*") \
            .order("name") \
            
        result = await run_in_threadpool(query.execute)
        return result.data or []
    except Exception as e:
        logging.error(f"Error fetching categories: {str(e)}")
        return []


async def create_category(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new category."""
    if not supabase:
        return None
    
    try:
        query = supabase.table("blog_categories") \
            .insert(data) \
            
        result = await run_in_threadpool(query.execute)
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
        query = supabase.table("blog_categories") \
            .update(data) \
            .eq("id", category_id) \
            
        await run_in_threadpool(query.execute)
        
        # Fetch and return the updated category
        query = supabase.table("blog_categories") \
            .select("*") \
            .eq("id", category_id) \
            .single() \
            
        result = await run_in_threadpool(query.execute)
        return result.data
    except Exception as e:
        logging.error(f"Error updating category: {str(e)}")
        return None


async def delete_category(category_id: str) -> bool:
    """Delete a category."""
    if not supabase:
        return False
    
    try:
        query = supabase.table("blog_categories") \
            .delete() \
            .eq("id", category_id) \
            
        await run_in_threadpool(query.execute)
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
        
        query = supabase.table("blog_posts") \
            .select("*, category:blog_categories(id, name, slug), author:blog_authors(id, name)", count="exact") \
            .order("created_at", desc=True) \
            .range(start, end) \
            
        
        result = await run_in_threadpool(query.execute)
        
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
        query = supabase.table("blog_posts") \
            .select("*, category:blog_categories(*), author:blog_authors(*)") \
            .eq("id", post_id) \
            .single() \
            
        result = await run_in_threadpool(query.execute)
        return result.data
    except Exception as e:
        logging.error(f"Error fetching post by ID: {str(e)}")
        return None


async def create_post(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new post."""
    if not supabase:
        raise ValueError("Supabase client not initialized")
    
    try:
        logging.info(f"Creating post with data: {data}")
        query = supabase.table("blog_posts") \
            .insert(data) \
            
        result = await run_in_threadpool(query.execute)
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
        logging.info(f"Updating post {post_id} with data: {data}")
        # Update the post
        query = supabase.table("blog_posts") \
            .update(data) \
            .eq("id", post_id) \
            
        await run_in_threadpool(query.execute)
        
        # Fetch and return the updated post
        query = supabase.table("blog_posts") \
            .select("*") \
            .eq("id", post_id) \
            .single() \
            
        result = await run_in_threadpool(query.execute)
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
        query = supabase.table("blog_posts") \
            .delete() \
            .eq("id", post_id) \
            
        await run_in_threadpool(query.execute)
        return True
    except Exception as e:
        logging.error(f"Error deleting post: {str(e)}")
        return False

