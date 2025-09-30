from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import logging
import json

from app.schemas.solve import SolveRequest, SolveResponse, SolveImageRequest
from app.services.openai_service import solve_with_openai, solve_with_openai_streaming, extract_formula_from_image


router = APIRouter(tags=["solve"])


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> SolveResponse:
    logging.info("Processing solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    result = await solve_with_openai(query_text=payload.text, api_key_override=x_openai_api_key)
    if result.error:
        logging.error("Solve request error: %s", result.error)
        raise HTTPException(status_code=400, detail=result.error)
    logging.info("Solve request completed successfully. Tool used: %s", result.tool_called if result.used_tool else "None")
    return result


@router.post("/solve/stream")
async def solve_stream_endpoint(payload: SolveRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Stream the solution as it's being generated"""
    logging.info("Processing streaming solve request: %s", payload.text[:50] + "..." if len(payload.text) > 50 else payload.text)
    
    async def stream_generator() -> AsyncIterator[str]:
        async for chunk in solve_with_openai_streaming(query_text=payload.text, api_key_override=x_openai_api_key, force_tool=True):
            yield f"data: {chunk}\n\n"
    
    logging.info("Starting streaming response")
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


@router.post("/solve/image")
async def solve_image_endpoint(payload: SolveImageRequest, x_openai_api_key: str | None = Header(default=None)) -> StreamingResponse:
    """Extract formula from image and stream the solution"""
    logging.info("Processing image solve request")
    
    async def stream_generator() -> AsyncIterator[str]:
        try:
            # First extract the formula from the image
            formula = await extract_formula_from_image(image_data=payload.image_data, api_key_override=x_openai_api_key)
            
            # Check if formula extraction had an error
            if formula.startswith("Error"):
                yield f"data: {json.dumps({'type': 'error', 'error': formula})}\n\n"
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
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                error_msg = f"Error solving formula: {str(e)}"
                logging.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logging.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    logging.info("Starting image processing and streaming response")
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


