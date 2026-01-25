import os
import json
import logging
import base64
from typing import Any, Dict, List, Optional, AsyncGenerator
from pathlib import Path

from pydantic import BaseModel

from app.schemas.solve import SolveResponse
from app.services.tool_registry import get_tools_for_openai, execute_tool_call
from app.prompts import MATH_SOLVER_SYSTEM_PROMPT, MATH_VISION_SYSTEM_PROMPT
from dotenv import load_dotenv


# Lazy module import to avoid hard dependency at import time
_openai_client: Optional[Any] = None


class OpenAIMessage(BaseModel):
    role: str
    content: str


async def init_openai_client() -> None:
    global _openai_client
    if _openai_client is not None:
        return
    # Load environment from .env if present
    try:
        # 1) Load project root .env
        load_dotenv()
        
    except Exception:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Defer error until first call
        _openai_client = None
        return
    try:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=api_key)
    except Exception:
        _openai_client = None


async def shutdown_openai_client() -> None:
    # AsyncOpenAI has no close method as of now; keep placeholder for future
    pass


def _build_tool_spec() -> List[Dict[str, Any]]:
    return get_tools_for_openai()


async def _chat_once(messages: List[Dict[str, str]], api_key_override: Optional[str] = None, stream: bool = False, force_tool: bool = False) -> Dict[str, Any]:
    client = _openai_client
    if api_key_override:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key_override)
        except Exception as exc:
            return {"error": f"Failed to initialize OpenAI client with override: {exc}"}
    if client is None:
        return {"error": "OPENAI_API_KEY is not configured or client unavailable"}
    tools = _build_tool_spec()
    try:
        tool_choice = "required" if force_tool else "auto"
        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.2,
            stream=stream,
        )
        if not stream:
            return resp.model_dump()
        else:
            # For streaming, return the response object directly
            return resp
    except Exception as exc:
        return {"error": str(exc)}


async def extract_formula_from_image(image_data: str, api_key_override: Optional[str] = None) -> str:
    """Extract mathematical formula from an image using OpenAI Vision API.
    
    Args:
        image_data: Base64 encoded image data
        api_key_override: Optional API key to use instead of the configured one
        
    Returns:
        Extracted formula text or error message
    """
    client = _openai_client
    if api_key_override:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key_override)
        except Exception as exc:
            logging.error(f"Error initializing OpenAI client: {exc}")
            return f"Error initializing OpenAI client: {exc}"
    
    if client is None:
        logging.error("OPENAI_API_KEY is not configured or client unavailable")
        return "Error: OPENAI_API_KEY is not configured or client unavailable"
    
    try:
        # Validate base64 data
        if not image_data or len(image_data) < 100:  # Basic check for valid base64 data
            logging.error("Invalid image data provided")
            return "Error: Invalid image data provided"
            
        response = await client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o which has vision capabilities
            messages=[
                {
                    "content": MATH_VISION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the mathematical formula from this image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        extracted_text = response.choices[0].message.content
        result = extracted_text.strip()
        logging.info(f"Successfully extracted formula: {result[:50]}{'...' if len(result) > 50 else ''}")
        return result
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logging.error(error_msg)
        return error_msg


async def solve_with_openai(query_text: str, api_key_override: Optional[str] = None) -> SolveResponse:
    if not query_text.strip():
        return SolveResponse(answer="", error="Empty query")

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": MATH_SOLVER_SYSTEM_PROMPT
        },
        {"role": "user", "content": query_text},
    ]

    first = await _chat_once(messages, api_key_override=api_key_override)
    if "error" in first:
        return SolveResponse(answer="", error=first["error"]) 

    choice = (first.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        content = message.get("content") or ""
        return SolveResponse(answer=content, used_tool=False)

    # Handle a single tool call for simplicity; extend to multiple if needed
    tool = tool_calls[0]
    tool_name = tool.get("function", {}).get("name")
    tool_args_str = tool.get("function", {}).get("arguments", "{}")

    # Execute tool
    exec_result = await execute_tool_call(tool_name, tool_args_str)
    
    try:
        parsed_result = json.loads(exec_result)
        logging.info("Tool result: %s", parsed_result)
    except Exception as e:
        logging.info("Tool result (raw): %s", exec_result)

    # Provide tool result back to the model
    messages.append({
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool.get("id", "tool_0"),
        "name": tool_name or "",
        "content": exec_result,
    })

    second = await _chat_once(messages, api_key_override=api_key_override)
    if "error" in second:
        return SolveResponse(answer="", error=second["error"], tool_called=tool_name)

    final_choice = (second.get("choices") or [{}])[0]
    final_msg = final_choice.get("message", {})
    final_content = final_msg.get("content") or ""

    # Fallback: if model produced no final text, surface the tool result directly
    if not final_content:
        try:
            parsed_tool = json.loads(exec_result or "{}")
            # Prefer common keys
            for key in ("result", "solutions", "confidence_interval"):
                if key in parsed_tool and parsed_tool[key] is not None:
                    final_content = str(parsed_tool[key])
                    break
            if not final_content and "error" in parsed_tool:
                final_content = f"Tool error: {parsed_tool['error']}"
        except Exception:
            # Keep empty if parsing fails
            final_content = ""

    return SolveResponse(
        answer=final_content,
        used_tool=True,
        tool_called=tool_name,
        tool_args=None,
    )


async def solve_with_openai_streaming(query_text: str, api_key_override: Optional[str] = None, force_tool: bool = False) -> AsyncGenerator[str, None]:
    """
    Stream the solution as it's being generated, with support for forcing tool usage.
    
    Args:
        query_text: The math problem to solve
        api_key_override: Optional API key to use instead of the configured one
        force_tool: If True, force the model to use a tool
        
    Yields:
        JSON chunks with partial responses
    """
    if not query_text.strip():
        yield json.dumps({"error": "Empty query"})
        return

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": MATH_SOLVER_SYSTEM_PROMPT
        },
        {"role": "user", "content": query_text},
    ]

    # First call to get the tool call
    first_response = await _chat_once(messages, api_key_override=api_key_override, force_tool=force_tool)
    if isinstance(first_response, dict) and "error" in first_response:
        yield json.dumps({"error": first_response["error"]})
        return

    choice = (first_response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        # If no tool calls were made (and not streaming), return the content directly
        content = message.get("content") or ""
        yield json.dumps({"type": "content", "content": content})
        return

    # Add assistant message with tool calls
    messages.append({
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    })
    
    # Handle all tool calls
    for tool in tool_calls:
        tool_id = tool.get("id", "tool_0")
        tool_name = tool.get("function", {}).get("name")
        tool_args_str = tool.get("function", {}).get("arguments", "{}")
        
        # Notify that we're calling a tool
        yield json.dumps({
            "type": "tool_call", 
            "tool": tool_name, 
            "tool_id": tool_id,
            "args": json.loads(tool_args_str)
        })

        # Execute tool
        exec_result = await execute_tool_call(tool_name, tool_args_str)
        
        # Notify of tool result
        try:
            parsed_result = json.loads(exec_result)
            logging.info(f"Streaming tool result for {tool_name} ({tool_id}): {parsed_result}")
            yield json.dumps({"type": "tool_result", "tool_id": tool_id, "result": parsed_result})
        except Exception as e:
            logging.info(f"Streaming tool result (raw) for {tool_name} ({tool_id}): {exec_result}")
            yield json.dumps({"type": "tool_result", "tool_id": tool_id, "result": exec_result})

        # Add tool response message for this specific tool call
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name or "",
            "content": exec_result,
        })

    # Stream the final response
    second_response = await _chat_once(
        messages, 
        api_key_override=api_key_override, 
        stream=True
    )
    
    # Handle streaming response
    buffer = ""
    try:
        # Check if we got a streaming response or a dict with error
        if isinstance(second_response, dict):
            if "error" in second_response:
                yield json.dumps({"type": "error", "error": second_response["error"]})
            else:
                # If it's a dict but not an error, try to get content from it
                content = ""
                if "choices" in second_response and len(second_response["choices"]) > 0:
                    if "message" in second_response["choices"][0]:
                        content = second_response["choices"][0]["message"].get("content", "")
                yield json.dumps({"type": "content", "content": content})
        else:
            # Normal streaming response
            async for chunk in second_response:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    buffer += content_chunk
                    yield json.dumps({"type": "content_chunk", "content": content_chunk})
            
            # Send final complete message
            yield json.dumps({"type": "content_complete", "content": buffer})
    except Exception as e:
        logging.error(f"Error in streaming response: {str(e)}")
        yield json.dumps({"type": "error", "error": str(e)})


