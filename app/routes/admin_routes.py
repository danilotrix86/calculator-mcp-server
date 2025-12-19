from fastapi import APIRouter, Query, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.middleware.auth import verify_admin
from app.services import blog_admin_service

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
async def check_auth(username: str = Depends(verify_admin)):
    """Check if admin credentials are valid."""
    return {"authenticated": True, "username": username}


# ============== AUTHORS ==============

@router.get("/authors")
async def get_authors(username: str = Depends(verify_admin)):
    """Get all authors."""
    authors = await blog_admin_service.get_all_authors()
    return {"authors": authors}


@router.post("/authors")
async def create_author(data: AuthorCreate, username: str = Depends(verify_admin)):
    """Create a new author."""
    author = await blog_admin_service.create_author(data.model_dump())
    if not author:
        raise HTTPException(status_code=500, detail="Failed to create author")
    return {"author": author}


# ============== CATEGORIES ==============

@router.get("/categories")
async def get_categories(username: str = Depends(verify_admin)):
    """Get all categories."""
    categories = await blog_admin_service.get_all_categories_admin()
    return {"categories": categories}


@router.post("/categories")
async def create_category(data: CategoryCreate, username: str = Depends(verify_admin)):
    """Create a new category."""
    category = await blog_admin_service.create_category(data.model_dump())
    if not category:
        raise HTTPException(status_code=500, detail="Failed to create category")
    return {"category": category}


@router.put("/categories/{category_id}")
async def update_category(category_id: str, data: CategoryUpdate, username: str = Depends(verify_admin)):
    """Update a category."""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    category = await blog_admin_service.update_category(category_id, update_data)
    if not category:
        raise HTTPException(status_code=500, detail="Failed to update category")
    return {"category": category}


@router.delete("/categories/{category_id}")
async def delete_category(category_id: str, username: str = Depends(verify_admin)):
    """Delete a category."""
    success = await blog_admin_service.delete_category(category_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete category")
    return {"success": True}


# ============== POSTS ==============

@router.get("/posts")
async def get_posts(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    username: str = Depends(verify_admin)
):
    """Get all posts (including unpublished)."""
    return await blog_admin_service.get_all_posts_admin(page, limit)


@router.get("/posts/{post_id}")
async def get_post(post_id: str, username: str = Depends(verify_admin)):
    """Get a single post by ID."""
    post = await blog_admin_service.get_post_by_id(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"post": post}


@router.post("/posts")
async def create_post(data: PostCreate, username: str = Depends(verify_admin)):
    """Create a new post."""
    post = await blog_admin_service.create_post(data.model_dump())
    if not post:
        raise HTTPException(status_code=500, detail="Failed to create post")
    return {"post": post}


@router.put("/posts/{post_id}")
async def update_post(post_id: str, data: PostUpdate, username: str = Depends(verify_admin)):
    """Update a post."""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    post = await blog_admin_service.update_post(post_id, update_data)
    if not post:
        raise HTTPException(status_code=500, detail="Failed to update post")
    return {"post": post}


@router.delete("/posts/{post_id}")
async def delete_post(post_id: str, username: str = Depends(verify_admin)):
    """Delete a post."""
    success = await blog_admin_service.delete_post(post_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete post")
    return {"success": True}




