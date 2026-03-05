import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any

from app.schemas.chat import ChatRequest
from app.prompts import MATH_TUTOR_SYSTEM_PROMPT, MATH_TUTOR_STANDALONE_PROMPT
from app.services.openai_service import _chat_once
from app.services.tool_registry import execute_tool_call


def _build_chat_messages(request: ChatRequest) -> List[Dict[str, Any]]:
    if request.initial_problem and request.initial_solution:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": MATH_TUTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Problema da risolvere:\n{request.initial_problem}",
            },
            {
                "role": "assistant",
                "content": f"Ho risolto il problema. Ecco la soluzione completa:\n\n{request.initial_solution}",
            },
        ]
    else:
        messages = [
            {"role": "system", "content": MATH_TUTOR_STANDALONE_PROMPT},
        ]

    history = request.messages[-request.max_turns:]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    return messages


async def chat_with_openai_streaming(
    request: ChatRequest,
    api_key_override: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a tutor chat response.

    Yields SSE-compatible JSON strings:
      {"type": "tool_call",       "tool": str, "tool_id": str, "args": dict}
      {"type": "tool_result",     "tool_id": str, "result": any}
      {"type": "content_chunk",   "content": str}
      {"type": "content_complete","content": str}
      {"type": "error",           "error": str}
    """
    messages = _build_chat_messages(request)

    # First call — non-streaming so we can detect tool calls.
    # If no tools are needed we immediately do a second streaming call.
    first_response = await _chat_once(
        messages,
        api_key_override=api_key_override,
        stream=False,
        force_tool=False,
    )

    if isinstance(first_response, dict) and "error" in first_response:
        yield json.dumps({"type": "error", "error": first_response["error"]})
        return

    choice = (first_response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        # No tools needed — do a real streaming call for the answer
        streaming_response = await _chat_once(
            messages,
            api_key_override=api_key_override,
            stream=True,
            force_tool=False,
        )

        buffer = ""
        chunk_count = 0
        try:
            if isinstance(streaming_response, dict):
                content = ""
                if "choices" in streaming_response and streaming_response["choices"]:
                    content = streaming_response["choices"][0].get("message", {}).get("content", "")
                if content:
                    yield json.dumps({"type": "content_chunk", "content": content})
                yield json.dumps({"type": "content_complete", "content": content})
            else:
                async for chunk in streaming_response:
                    if chunk.choices and hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        piece = chunk.choices[0].delta.content
                        chunk_count += 1
                        buffer += piece
                        yield json.dumps({"type": "content_chunk", "content": piece})

                yield json.dumps({"type": "content_complete", "content": buffer})
        except Exception as exc:
            logging.error(f"[chat] Streaming error (no-tool path): {exc}")
            yield json.dumps({"type": "error", "error": str(exc)})
        return

    # There are tool calls — execute them all, then stream the final explanation
    messages.append({
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    })

    for tool in tool_calls:
        tool_id = tool.get("id", "tool_0")
        tool_name = tool.get("function", {}).get("name")
        tool_args_str = tool.get("function", {}).get("arguments", "{}")

        yield json.dumps({
            "type": "tool_call",
            "tool": tool_name,
            "tool_id": tool_id,
            "args": json.loads(tool_args_str),
        })

        exec_result = await execute_tool_call(tool_name, tool_args_str)

        try:
            parsed_result = json.loads(exec_result)
            yield json.dumps({"type": "tool_result", "tool_id": tool_id, "result": parsed_result})
        except Exception:
            yield json.dumps({"type": "tool_result", "tool_id": tool_id, "result": exec_result})

        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name or "",
            "content": exec_result,
        })

    # Stream the final explanation after tool execution
    second_response = await _chat_once(
        messages,
        api_key_override=api_key_override,
        stream=True,
        force_tool=False,
    )

    buffer = ""
    chunk_count = 0
    try:
        if isinstance(second_response, dict):
            if "error" in second_response:
                yield json.dumps({"type": "error", "error": second_response["error"]})
            else:
                content = ""
                if "choices" in second_response and second_response["choices"]:
                    content = second_response["choices"][0].get("message", {}).get("content", "")
                if content:
                    yield json.dumps({"type": "content_chunk", "content": content})
                yield json.dumps({"type": "content_complete", "content": content})
        else:
            async for chunk in second_response:
                if chunk.choices and hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    chunk_count += 1
                    buffer += piece
                    yield json.dumps({"type": "content_chunk", "content": piece})

            yield json.dumps({"type": "content_complete", "content": buffer})
    except Exception as exc:
        logging.error(f"[chat] Streaming error (tool path): {exc}")
        yield json.dumps({"type": "error", "error": str(exc)})
