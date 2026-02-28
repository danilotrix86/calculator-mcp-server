from datetime import datetime, timedelta
import os
import logging
from typing import Optional, Dict, Any
from supabase import create_client, Client
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SupabaseService:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.client: Optional[Client] = None

        if self.supabase_url:
            logging.info("SUPABASE_URL found: %s...", self.supabase_url[:12])
        else:
            logging.warning("SUPABASE_URL not found in environment variables")
            
        if self.supabase_key:
            logging.info("SUPABASE_KEY found")
        else:
            logging.warning("SUPABASE_KEY not found in environment variables")

        if self.supabase_url and self.supabase_key:
            self.client = create_client(self.supabase_url, self.supabase_key)
            logging.info("Supabase client initialized")
        else:
            logging.warning("Supabase credentials not found in environment variables")

    async def save_query(self, question: str) -> Optional[str]:
        """Save a new query to Supabase."""
        if not self.client:
            return None
        
        try:
            query = self.client.table("user_queries").insert({"question": question})
            result = await run_in_threadpool(query.execute)
            query_id = result.data[0]["id"] if result.data else None
            if query_id:
                logging.info("Question saved to Supabase with ID: %s", query_id)
            return query_id
        except Exception as e:
            logging.error("Error saving question to Supabase: %s", str(e))
            return None

    async def update_query_response(self, query_id: str, response: str, tool_used: Optional[str] = None) -> bool:
        """Update an existing query with a response."""
        if not self.client or not query_id:
            return False
        
        try:
            query = self.client.table("user_queries").update({"response": response, "tool_used": tool_used}).eq("id", query_id)
            await run_in_threadpool(query.execute)
            logging.info("Response saved to Supabase for query ID: %s", query_id)
            return True
        except Exception as e:
            logging.error("Error updating Supabase with response: %s", str(e))
            return False

    async def update_query_question(self, query_id: str, question: str) -> bool:
        """Update the question for an existing query."""
        if not self.client or not query_id:
            return False
        
        try:
            query = self.client.table("user_queries").update({"question": question}).eq("id", query_id)
            await run_in_threadpool(query.execute)
            return True
        except Exception as e:
            logging.error("Error updating Supabase with question: %s", str(e))
            return False

    async def update_query_error(self, query_id: str, error_message: str) -> bool:
        """Update an existing query with an error message."""
        if not self.client or not query_id:
            return False
        
        try:
            query = self.client.table("user_queries").update({"response": f"Error: {error_message}", "tool_used": None}).eq("id", query_id)
            await run_in_threadpool(query.execute)
            return True
        except Exception as e:
            logging.error("Error updating Supabase with error response: %s", str(e))
            return False

    async def find_cached_response(self, question: str, ttl_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Find a cached response for the given question."""
        if not self.client:
            return None
        
        try:
            logging.info("Checking cache for query: %s...", question[:50])
            
            query = self.client.table("user_queries").select("*").eq("question", question).order("created_at", desc=True).limit(5)
            result = await run_in_threadpool(query.execute)
            
            if result.data:
                logging.info("Found %d potential cache matches", len(result.data))
                now = datetime.utcnow()
                
                for row in result.data:
                    # Check TTL
                    created_at_str = row.get("created_at")
                    if created_at_str:
                        try:
                            # Handle typical Supabase ISO string (e.g. 2023-10-25T12:34:56.789+00:00)
                            if "." in created_at_str:
                                dt_str = created_at_str.split(".")[0]
                            else:
                                dt_str = created_at_str.split("+")[0].replace("Z", "")
                                
                            created_dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
                            if now - created_dt > timedelta(hours=ttl_hours):
                                logging.info("Cache entry %s expired (older than %d hours)", row['id'], ttl_hours)
                                continue
                        except Exception as dt_err:
                            logging.warning("Could not parse date %s: %s", created_at_str, dt_err)
                            
                    response = row.get("response")
                    if response and not str(response).strip() == "" and not str(response).startswith("Error:"):
                        logging.info("Cache hit! Using query ID: %s", row['id'])
                        return row
                
                logging.info("No valid/unexpired successful response found in recent matches")
                return None
                
            logging.info("No cache matches found")
            return None
        except Exception as e:
            logging.error("Error finding cached response: %s", str(e))
            return None

# Singleton instance for backward compatibility if needed, 
# but we prefer dependency injection
_supabase_service_instance = None

def get_supabase_service() -> SupabaseService:
    global _supabase_service_instance
    if _supabase_service_instance is None:
        _supabase_service_instance = SupabaseService()
    return _supabase_service_instance

# For backward compatibility with modules that haven't been refactored yet
supabase = get_supabase_service().client
