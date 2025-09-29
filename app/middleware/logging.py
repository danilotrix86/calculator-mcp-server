import time
import json
from typing import Callable

from fastapi import FastAPI, Request


def add_logging_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        return response


def format_tool_log(name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json)
    except Exception:
        args = {"raw": arguments_json}
    return json.dumps({"tool": name, "args": args}, ensure_ascii=False)



