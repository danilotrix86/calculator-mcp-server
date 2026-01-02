from fastapi import APIRouter
from fastapi.responses import Response
from datetime import datetime
from app.services.supabase_service import supabase

router = APIRouter(tags=["sitemap"])

# Static pages configuration
STATIC_PAGES = [
    {"loc": "/", "changefreq": "daily", "priority": "1.0"},
    {"loc": "/calcolatore-scientifico", "changefreq": "monthly", "priority": "0.8"},
    {"loc": "/calcolatore-matrici", "changefreq": "monthly", "priority": "0.8"},
    {"loc": "/grafici-funzioni", "changefreq": "monthly", "priority": "0.8"},
    {"loc": "/blog", "changefreq": "daily", "priority": "0.9"},
    {"loc": "/supportaci", "changefreq": "monthly", "priority": "0.5"},
]

BASE_URL = "https://www.risolutorematematico.it"


def generate_sitemap_xml(static_pages: list, blog_posts: list, categories: list) -> str:
    """Generate sitemap XML string."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Add static pages
    for page in static_pages:
        xml_parts.append("  <url>")
        xml_parts.append(f"    <loc>{BASE_URL}{page['loc']}</loc>")
        xml_parts.append(f"    <lastmod>{today}</lastmod>")
        xml_parts.append(f"    <changefreq>{page['changefreq']}</changefreq>")
        xml_parts.append(f"    <priority>{page['priority']}</priority>")
        xml_parts.append("  </url>")
    
    # Add blog categories
    for category in categories:
        xml_parts.append("  <url>")
        xml_parts.append(f"    <loc>{BASE_URL}/blog/categoria/{category['slug']}</loc>")
        xml_parts.append(f"    <lastmod>{today}</lastmod>")
        xml_parts.append("    <changefreq>weekly</changefreq>")
        xml_parts.append("    <priority>0.7</priority>")
        xml_parts.append("  </url>")
    
    # Add blog posts
    for post in blog_posts:
        # Parse the updated_at or created_at date
        post_date = post.get('updated_at') or post.get('created_at')
        if post_date:
            # Handle ISO format date
            try:
                if 'T' in post_date:
                    post_date = post_date.split('T')[0]
                else:
                    post_date = post_date[:10]
            except:
                post_date = today
        else:
            post_date = today
            
        xml_parts.append("  <url>")
        xml_parts.append(f"    <loc>{BASE_URL}/blog/{post['slug']}</loc>")
        xml_parts.append(f"    <lastmod>{post_date}</lastmod>")
        xml_parts.append("    <changefreq>monthly</changefreq>")
        xml_parts.append("    <priority>0.8</priority>")
        xml_parts.append("  </url>")
    
    xml_parts.append("</urlset>")
    
    return "\n".join(xml_parts)


@router.get("/sitemap.xml")
async def get_sitemap():
    """Generate dynamic sitemap XML."""
    blog_posts = []
    categories = []
    
    if supabase:
        try:
            # Fetch all published blog posts
            posts_result = supabase.table("blog_posts") \
                .select("slug, updated_at, created_at") \
                .eq("published", True) \
                .order("created_at", desc=True) \
                .execute()
            
            blog_posts = posts_result.data or []
            
            # Fetch all categories
            cats_result = supabase.table("blog_categories") \
                .select("slug") \
                .execute()
            
            categories = cats_result.data or []
            
        except Exception as e:
            print(f"Error fetching blog data for sitemap: {e}")
    
    # Generate XML
    sitemap_xml = generate_sitemap_xml(STATIC_PAGES, blog_posts, categories)
    
    return Response(
        content=sitemap_xml,
        media_type="application/xml",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )


@router.get("/robots.txt")
async def get_robots():
    """Serve robots.txt with sitemap reference."""
    robots_content = f"""User-agent: *
Allow: /

# Sitemap
Sitemap: {BASE_URL}/sitemap.xml

# Disallow admin pages
Disallow: /admin/
"""
    
    return Response(
        content=robots_content,
        media_type="text/plain"
    )












