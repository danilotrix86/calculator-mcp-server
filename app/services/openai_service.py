import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.schemas.solve import SolveResponse
from app.services.tool_registry import get_tools_for_openai, execute_tool_call
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


async def _chat_once(messages: List[Dict[str, str]], api_key_override: Optional[str] = None) -> Dict[str, Any]:
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
        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        return resp.model_dump()
    except Exception as exc:
        return {"error": str(exc)}


async def solve_with_openai(query_text: str, api_key_override: Optional[str] = None) -> SolveResponse:
    if not query_text.strip():
        return SolveResponse(answer="", error="Empty query")

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a precise math assistant. When needed, use tools to compute. "
                "Explain succinctly and return final numeric/symbolic results.")
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
    return SolveResponse(
        answer=final_msg.get("content") or "",
        used_tool=True,
        tool_called=tool_name,
        tool_args=None,
    )


