import os
import logging
from typing import Optional, Dict, Any, Union
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

# Log the status of environment variables
if supabase_url:
    logging.info(f"SUPABASE_URL found: {supabase_url[:12]}...")
else:
    logging.warning("SUPABASE_URL not found in environment variables")
    
if supabase_key:
    logging.info("SUPABASE_KEY found")
else:
    logging.warning("SUPABASE_KEY not found in environment variables")

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized")
else:
    logging.warning("Supabase credentials not found in environment variables")


async def save_query(question: str) -> Optional[str]:
    """
    Save a new query to Supabase.
    
    Args:
        question: The user's question/query
        
    Returns:
        Optional query ID if successful, None otherwise
    """
    if not supabase:
        return None
    
    try:
        result = supabase.table("user_queries").insert({
            "question": question
        }).execute()
        query_id = result.data[0]["id"] if result.data else None
        if query_id:
            logging.info(f"Question saved to Supabase with ID: {query_id}")
        return query_id
    except Exception as e:
        logging.error(f"Error saving question to Supabase: {str(e)}")
        return None


async def update_query_response(query_id: str, response: str, tool_used: Optional[str] = None) -> bool:
    """
    Update an existing query with a response.
    
    Args:
        query_id: The ID of the query to update
        response: The response to save
        tool_used: The name of the tool used, if any
        
    Returns:
        True if successful, False otherwise
    """
    if not supabase or not query_id:
        return False
    
    try:
        supabase.table("user_queries").update({
            "response": response,
            "tool_used": tool_used
        }).eq("id", query_id).execute()
        logging.info(f"Response saved to Supabase for query ID: {query_id}")
        return True
    except Exception as e:
        logging.error(f"Error updating Supabase with response: {str(e)}")
        return False


async def update_query_question(query_id: str, question: str) -> bool:
    """
    Update the question for an existing query.
    
    Args:
        query_id: The ID of the query to update
        question: The new question text
        
    Returns:
        True if successful, False otherwise
    """
    if not supabase or not query_id:
        return False
    
    try:
        supabase.table("user_queries").update({
            "question": question
        }).eq("id", query_id).execute()
        return True
    except Exception as e:
        logging.error(f"Error updating Supabase with question: {str(e)}")
        return False


async def update_query_error(query_id: str, error_message: str) -> bool:
    """
    Update an existing query with an error message.
    
    Args:
        query_id: The ID of the query to update
        error_message: The error message
        
    Returns:
        True if successful, False otherwise
    """
    if not supabase or not query_id:
        return False
    
    try:
        supabase.table("user_queries").update({
            "response": f"Error: {error_message}",
            "tool_used": None
        }).eq("id", query_id).execute()
        return True
    except Exception as e:
        logging.error(f"Error updating Supabase with error response: {str(e)}")
        return False


async def find_cached_response(question: str) -> Optional[Dict[str, Any]]:
    """
    Find a cached response for the given question.
    
    Args:
        question: The user's question/query
        
    Returns:
        The cached row data if found, None otherwise
    """
    if not supabase:
        return None
    
    try:
        logging.info(f"Checking cache for query: {question[:50]}...")
        
        # Get recent queries matching the question
        # Fetch a few to ensure we find a successful one if the latest was an error
        result = supabase.table("user_queries").select("*").eq("question", question).order("created_at", desc=True).limit(5).execute()
        
        if result.data:
            logging.info(f"Found {len(result.data)} potential cache matches")
            for row in result.data:
                response = row.get("response")
                # Check if response exists and is not an error
                if response and not str(response).strip() == "" and not str(response).startswith("Error:"):
                    logging.info(f"Cache hit! Using query ID: {row['id']}")
                    return row
            
            logging.info("No valid successful response found in recent matches")
            return None
            
        logging.info("No cache matches found")
        return None
    except Exception as e:
        logging.error(f"Error finding cached response: {str(e)}")
        return None
