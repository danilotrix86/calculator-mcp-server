from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import logging
import json

from app.schemas.solve import SolveRequest, SolveResponse, SolveImageRequest
from app.services.openai_service import solve_with_openai, solve_with_openai_streaming, extract_formula_from_image
from app.services import supabase_service


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> SolveResponse:
    logging.info("Processing solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    
    # Store the question in Supabase
    query_id = await supabase_service.save_query(payload.text)
    
    # Process the request with OpenAI
    result = await solve_with_openai(query_text=payload.text, api_key_override=x_openai_api_key)
    
    if result.error:
        logging.error("Solve request error: %s", result.error)
        
        # Update Supabase with error
        if query_id:
            await supabase_service.update_query_error(query_id, result.error)
        
        raise HTTPException(status_code=400, detail=result.error)
    
    logging.info("Solve request completed successfully. Tool used: %s", result.tool_called if result.used_tool else "None")
    
    # Update Supabase with response
    if query_id:
        await supabase_service.update_query_response(
            query_id, 
            result.answer, 
            result.tool_called if result.used_tool else None
        )
    
    return result


@router.post("/solve/stream")
async def solve_stream_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    logging.info("Processing streaming solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    
    # Store the question in Supabase
    query_id = await supabase_service.save_query(payload.text)
    
    # Store the complete response to update Supabase later
    complete_response = []
    tool_used = None
    
    async def stream_generator() -> AsyncIterator[str]:
        nonlocal complete_response, tool_used
        try:
            async for chunk in solve_with_openai_streaming(query_text=payload.text, api_key_override=x_openai_api_key, force_tool=True):
                # Parse the chunk to extract tool information if available
                try:
                    chunk_data = json.loads(chunk)
                    if "tool_called" in chunk_data:
                        tool_used = chunk_data["tool_called"]
                    if "content" in chunk_data:
                        complete_response.append(chunk_data.get("content", ""))
                except (json.JSONDecodeError, TypeError):
                    # If not JSON or doesn't have content, store the raw chunk
                    complete_response.append(chunk)
                
                yield f"data: {chunk}\n\n"
                
            # After streaming is complete, update Supabase with the complete response
            if query_id:
                final_response = "".join([str(chunk) for chunk in complete_response if chunk])
                await supabase_service.update_query_response(query_id, final_response, tool_used)
        except Exception as e:
            error_msg = f"Error in streaming: {str(e)}"
            logging.error(error_msg)
            
            # Update Supabase with error
            if query_id:
                await supabase_service.update_query_error(query_id, error_msg)
            
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    logging.info("Starting streaming response")
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


@router.post("/solve/image")
async def solve_image_endpoint(payload: SolveImageRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Extract formula from image and stream the solution"""
    logging.info("Processing image solve request")
    
    # Store the question in Supabase (initially as "Image upload")
    query_id = await supabase_service.save_query("Image upload (formula extraction)")
    
    # Store the complete response to update Supabase later
    extracted_formula = None
    complete_response = []
    tool_used = None
    
    async def stream_generator() -> AsyncIterator[str]:
        nonlocal complete_response, tool_used, extracted_formula
        try:
            # First extract the formula from the image
            formula = await extract_formula_from_image(image_data=payload.image_data, api_key_override=x_openai_api_key)
            extracted_formula = formula
            
            # Update Supabase with the extracted formula
            if query_id:
                await supabase_service.update_query_question(query_id, f"Image upload: {formula}")
            
            # Check if formula extraction had an error
            if formula.startswith("Error"):
                error_data = json.dumps({'type': 'error', 'error': formula})
                yield f"data: {error_data}\n\n"
                
                # Update Supabase with error
                if query_id:
                    await supabase_service.update_query_error(query_id, formula)
                return
            
            # Notify client that formula was extracted
            yield f"data: {json.dumps({'type': 'extracted_formula', 'formula': formula})}\n\n"
            
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
                generator = solve_with_openai_streaming(query_text=query, api_key_override=x_openai_api_key, force_tool=True)
                async for chunk in generator:
                    # Parse the chunk to extract tool information if available
                    try:
                        chunk_data = json.loads(chunk)
                        if "tool_called" in chunk_data:
                            tool_used = chunk_data["tool_called"]
                        if "content" in chunk_data:
                            complete_response.append(chunk_data.get("content", ""))
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON or doesn't have content, store the raw chunk
                        complete_response.append(chunk)
                    
                    yield f"data: {chunk}\n\n"
                
                # After streaming is complete, update Supabase with the complete response
                if query_id:
                    final_response = "".join([str(chunk) for chunk in complete_response if chunk])
                    await supabase_service.update_query_response(query_id, final_response, tool_used)
                        
            except Exception as e:
                error_msg = f"Error solving formula: {str(e)}"
                logging.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                
                # Update Supabase with error
                if query_id:
                    await supabase_service.update_query_error(query_id, f"Error solving formula: {str(e)}")
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logging.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
            
            # Update Supabase with error
            if query_id:
                await supabase_service.update_query_error(query_id, f"Error processing image: {str(e)}")
    
    logging.info("Starting image processing and streaming response")
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


