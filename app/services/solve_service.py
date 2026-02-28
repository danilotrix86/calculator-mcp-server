import logging
import json
import hashlib
from typing import AsyncIterator, Optional

from fastapi import HTTPException, Depends
from app.schemas.solve import SolveRequest, SolveResponse, SolveImageRequest
from app.services.openai_service import solve_with_openai, solve_with_openai_streaming, extract_formula_from_image
from app.services.supabase_service import SupabaseService, get_supabase_service
from app.services.math_service import normalize_expression


class SolveService:
    def __init__(self, supabase_service: SupabaseService):
        self.supabase_service = supabase_service

    def _build_image_hash(self, image_data: str) -> str:
        cleaned = (image_data or "").strip()
        if cleaned.startswith("data:") and "," in cleaned:
            cleaned = cleaned.split(",", 1)[1]
        # Normalize accidental base64 whitespace/newlines from clients.
        cleaned = "".join(cleaned.split())
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    def _image_hash_cache_key(self, image_hash: str) -> str:
        return f"Image hash: {image_hash}"

    def _encode_image_hash_cache_response(self, formula: str, response: str, tool_used: Optional[str]) -> str:
        return json.dumps(
            {
                "cache_type": "image_hash_v1",
                "formula": formula,
                "response": response,
                "tool_used": tool_used,
            }
        )

    def _decode_image_hash_cache_response(self, response_text: str) -> Optional[dict]:
        try:
            payload = json.loads(response_text or "")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_type") != "image_hash_v1":
            return None
        if "response" not in payload:
            return None
        return payload

    async def _save_image_hash_cache(self, image_hash_key: str, formula: str, response: str, tool_used: Optional[str]) -> None:
        if not response or not response.strip():
            return
        cache_query_id = await self.supabase_service.save_query(image_hash_key)
        if not cache_query_id:
            return
        cache_payload = self._encode_image_hash_cache_response(formula=formula, response=response, tool_used=tool_used)
        await self.supabase_service.update_query_response(cache_query_id, cache_payload, tool_used)

    def _parse_stream_chunk(self, chunk: str, complete_response: list, tool_used_ref: dict) -> None:
        try:
            chunk_data = json.loads(chunk)
            chunk_type = chunk_data.get("type", "")
            
            if chunk_type == "tool_call":
                tool_used_ref["name"] = chunk_data.get("tool")
            
            if chunk_type in ("content_chunk", "content"):
                content = chunk_data.get("content", "")
                if content:
                    complete_response.append(content)
            elif chunk_type == "content_complete":
                content = chunk_data.get("content", "")
                if content and not complete_response:
                    complete_response.append(content)
        except (json.JSONDecodeError, TypeError):
            pass

    async def solve(self, payload: SolveRequest, api_key_override: Optional[str] = None) -> SolveResponse:
        logging.info("Processing solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
        
        try:
            normalized_text = normalize_expression(payload.text)
        except Exception:
            normalized_text = payload.text.strip()
            
        # Check for cached response
        cached = await self.supabase_service.find_cached_response(normalized_text)
        if not cached:
            # Fallback: check if it was previously cached from an image upload
            cached = await self.supabase_service.find_cached_response(f"Image upload: {normalized_text}")
            
        if cached:
            logging.info("Found cached response for query: %s", payload.text[:50])
            tool_used = cached.get("tool_used")
            response_text = cached.get("response", "")
            return SolveResponse(
                answer=response_text,
                used_tool=bool(tool_used),
                tool_called=tool_used
            )
            
        # Store the question in Supabase
        query_id = await self.supabase_service.save_query(normalized_text)
        
        # Process the request with OpenAI
        result = await solve_with_openai(query_text=payload.text, api_key_override=api_key_override)
        
        if result.error:
            logging.error("Solve request error: %s", result.error)
            
            # Update Supabase with error
            if query_id:
                await self.supabase_service.update_query_error(query_id, result.error)
            
            raise HTTPException(status_code=400, detail=result.error)
        
        logging.info("Solve request completed successfully. Tool used: %s", result.tool_called if result.used_tool else "None")
        
        # Update Supabase with response
        if query_id:
            await self.supabase_service.update_query_response(
                query_id, 
                result.answer, 
                result.tool_called if result.used_tool else None
            )
        
        return result

    async def solve_stream(self, payload: SolveRequest, api_key_override: Optional[str] = None) -> AsyncIterator[str]:
        logging.info("Processing streaming solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
        
        try:
            normalized_text = normalize_expression(payload.text)
        except Exception:
            normalized_text = payload.text.strip()
            
        # Check for cached response
        cached = await self.supabase_service.find_cached_response(normalized_text)
        if not cached:
            # Fallback: check if it was previously cached from an image upload
            cached = await self.supabase_service.find_cached_response(f"Image upload: {normalized_text}")
            
        if cached:
            logging.info("Found cached response for query: %s", payload.text[:50])
            
            tool_used = cached.get("tool_used")
            response_text = cached.get("response", "")
            
            # Notify about tool usage if applicable (informative)
            if tool_used:
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_used, 'tool_id': 'cached', 'args': {}})}\n\n"
                # Small delay to ensure order?
                yield f"data: {json.dumps({'type': 'tool_result', 'tool_id': 'cached', 'result': 'Restored from database'})}\n\n"
            
            # Send the full content
            yield f"data: {json.dumps({'type': 'content', 'content': response_text})}\n\n"
            yield f"data: {json.dumps({'type': 'content_complete'})}\n\n"
            return

        # Store the question in Supabase
        query_id = await self.supabase_service.save_query(normalized_text)
        
        # Store the complete response to update Supabase later
        complete_response = []
        tool_used_ref = {"name": None}
        
        try:
            async for chunk in solve_with_openai_streaming(query_text=payload.text, api_key_override=api_key_override, force_tool=True):
                self._parse_stream_chunk(chunk, complete_response, tool_used_ref)
                yield f"data: {chunk}\n\n"
                
            # After streaming is complete, update Supabase with the complete response
            if query_id:
                final_response = "".join(complete_response)
                await self.supabase_service.update_query_response(query_id, final_response, tool_used_ref["name"])
        except Exception as e:
            error_msg = "An unexpected error occurred during processing."
            logging.error(f"Error in streaming: {str(e)}", exc_info=True)
            
            # Update Supabase with error
            if query_id:
                await self.supabase_service.update_query_error(query_id, str(e))
            
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    async def solve_image(self, payload: SolveImageRequest, api_key_override: Optional[str] = None) -> AsyncIterator[str]:
        logging.info("Processing image solve request")
        
        # Store the question in Supabase (initially as "Image upload")
        query_id = await self.supabase_service.save_query("Image upload (formula extraction)")
        image_hash = self._build_image_hash(payload.image_data)
        image_hash_key = self._image_hash_cache_key(image_hash)
        
        # Store the complete response to update Supabase later
        complete_response = []
        tool_used_ref = {"name": None}
        
        try:
            # First try a true image-hash cache, so identical images skip extraction.
            cached_by_hash = await self.supabase_service.find_cached_response(image_hash_key)
            if cached_by_hash:
                decoded = self._decode_image_hash_cache_response(cached_by_hash.get("response", ""))
                if decoded:
                    cached_formula = decoded.get("formula", "")
                    cached_response = decoded.get("response", "")
                    cached_tool_used = decoded.get("tool_used")

                    if query_id:
                        if cached_formula:
                            await self.supabase_service.update_query_question(query_id, f"Image upload: {cached_formula}")
                        await self.supabase_service.update_query_response(query_id, cached_response, cached_tool_used)

                    if cached_formula:
                        yield f"data: {json.dumps({'type': 'extracted_formula', 'formula': cached_formula})}\n\n"
                    if cached_tool_used:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': cached_tool_used, 'tool_id': 'cached', 'args': {}})}\n\n"
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_id': 'cached', 'result': 'Restored from database'})}\n\n"
                    yield f"data: {json.dumps({'type': 'content', 'content': cached_response})}\n\n"
                    yield f"data: {json.dumps({'type': 'content_complete'})}\n\n"
                    return

            # First extract the formula from the image
            formula = await extract_formula_from_image(image_data=payload.image_data, api_key_override=api_key_override)
            
            try:
                normalized_formula = normalize_expression(formula)
            except Exception:
                normalized_formula = formula.strip()
                
            # Update Supabase with the extracted formula
            if query_id:
                await self.supabase_service.update_query_question(query_id, f"Image upload: {normalized_formula}")
            
            # Check if formula extraction had an error
            if formula.startswith("Error"):
                error_data = json.dumps({'type': 'error', 'error': formula})
                yield f"data: {error_data}\n\n"
                
                # Update Supabase with error
                if query_id:
                    await self.supabase_service.update_query_error(query_id, formula)
                return
            
            # Notify client that formula was extracted
            yield f"data: {json.dumps({'type': 'extracted_formula', 'formula': formula})}\n\n"

            # Reuse cached solve result for repeated image formulas.
            # We key image cache entries by normalized extracted formula.
            cached = await self.supabase_service.find_cached_response(f"Image upload: {normalized_formula}")
            if cached:
                logging.info("Found cached image solve response for formula: %s", normalized_formula[:50])
                tool_used = cached.get("tool_used")
                response_text = cached.get("response", "")

                if query_id and response_text:
                    await self.supabase_service.update_query_response(query_id, response_text, tool_used)
                    await self._save_image_hash_cache(
                        image_hash_key=image_hash_key,
                        formula=formula,
                        response=response_text,
                        tool_used=tool_used,
                    )

                if tool_used:
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_used, 'tool_id': 'cached', 'args': {}})}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool_id': 'cached', 'result': 'Restored from database'})}\n\n"

                yield f"data: {json.dumps({'type': 'content', 'content': response_text})}\n\n"
                yield f"data: {json.dumps({'type': 'content_complete'})}\n\n"
                return
            
            # If multiple formulas were extracted (separated by semicolons), use the first one
            # or combine them into a system of equations
            if ";" in formula:
                formulas = [f.strip() for f in formula.split(";")]
                # If we have exactly two equations, treat them as a system
                if len(formulas) == 2 and all("=" in f for f in formulas):
                    # Format as a system of equations
                    query = f"Solve the system of equations: {formulas[0]} and {formulas[1]}"
                    logging.info(f"Processing as a system of equations: {query}")
                else:
                    # Otherwise just use the first formula
                    query = formulas[0]
                    logging.info(f"Multiple formulas detected, using the first one: {query}")
            else:
                query = formula
            
            # Then solve the formula using the existing streaming endpoint
            try:
                generator = solve_with_openai_streaming(query_text=query, api_key_override=api_key_override, force_tool=True)
                async for chunk in generator:
                    self._parse_stream_chunk(chunk, complete_response, tool_used_ref)
                    yield f"data: {chunk}\n\n"
                
                # After streaming is complete, update Supabase with the complete response
                if query_id:
                    final_response = "".join(complete_response)
                    await self.supabase_service.update_query_response(query_id, final_response, tool_used_ref["name"])
                    await self._save_image_hash_cache(
                        image_hash_key=image_hash_key,
                        formula=formula,
                        response=final_response,
                        tool_used=tool_used_ref["name"],
                    )
                        
            except Exception as e:
                error_msg = "An unexpected error occurred while solving the formula."
                logging.error(f"Error solving formula: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                
                # Update Supabase with error
                if query_id:
                    await self.supabase_service.update_query_error(query_id, f"Error solving formula: {str(e)}")
        except Exception as e:
            error_msg = "An unexpected error occurred while processing the image."
            logging.error(f"Error processing image: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
            
            # Update Supabase with error
            if query_id:
                await self.supabase_service.update_query_error(query_id, f"Error processing image: {str(e)}")

def get_solve_service(supabase_service: SupabaseService = Depends(get_supabase_service)) -> SolveService:
    return SolveService(supabase_service)
