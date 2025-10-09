from fastapi import APIRouter
import logging
from app.services import supabase_service

router = APIRouter(tags=["health"])


@router.get("/health/supabase")
async def supabase_health():
    """Check if the Supabase connection is working"""
    if not supabase_service.supabase:
        return {
            "status": "error", 
            "message": "Supabase client not initialized. Check environment variables SUPABASE_URL and SUPABASE_KEY."
        }
    
    try:
        # Try to query the supabase service
        test_result = supabase_service.supabase.table("user_queries").select("id").limit(1).execute()
        return {
            "status": "ok", 
            "message": "Supabase connection successful", 
            "config": {
                "url_set": bool(supabase_service.supabase_url),
                "key_set": bool(supabase_service.supabase_key)
            }
        }
    except Exception as e:
        logging.error(f"Supabase connection test failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Supabase connection failed: {str(e)}"
        }
