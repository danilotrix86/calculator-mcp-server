import time
import json
import logging
import sys
from typing import Callable

from fastapi import FastAPI, Request
from pythonjsonlogger import jsonlogger


def setup_json_logging():
    logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logHandler = logging.StreamHandler(sys.stdout)
    
    # Only use JSON logging if explicitly enabled via env var,
    # otherwise stick to human-readable format for local dev
    import os
    if os.environ.get("USE_JSON_LOGGING") == "true":
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s',
            rename_fields={
                "levelname": "severity",
                "asctime": "timestamp"
            }
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
    logHandler.setFormatter(formatter)
    
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)


def add_logging_middleware(app: FastAPI) -> None:
    # Set up basic logging configuration
    setup_json_logging()

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable):
        start = time.perf_counter()
        
        # Log the incoming request
        logging.info("Incoming request: %s %s", request.method, request.url.path, extra={
            "request_method": request.method,
            "request_path": request.url.path
        })
        
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
            
            # Log the outgoing response status
            logging.info("Completed request: %s %s - Status: %s - Time: %.2fms", 
                request.method, request.url.path, response.status_code, elapsed_ms,
                extra={
                    "request_method": request.method,
                    "request_path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms
                }
            )
            
            return response
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logging.error("Failed request: %s %s - Error: %s - Time: %.2fms",
                request.method, request.url.path, str(e), elapsed_ms,
                extra={
                    "request_method": request.method,
                    "request_path": request.url.path,
                    "error": str(e),
                    "elapsed_ms": elapsed_ms
                }, exc_info=True)
            raise


def format_tool_log(name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json)
    except Exception:
        args = {"raw": arguments_json}
    return json.dumps({"tool": name, "args": args}, ensure_ascii=False)



