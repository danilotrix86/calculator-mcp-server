import logging
from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from app.config import rate_limit_config


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind proxies (Cloud Run, nginx, etc.)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_solve_key(request: Request) -> str:
    """Composite key: prefer X-User-Id (set by the Next.js proxy) so per-user limits
    apply even when multiple users share an IP. Falls back to IP."""
    user_id = request.headers.get("x-user-id")
    if user_id:
        return f"user:{user_id}"
    return _get_client_ip(request)


limiter = Limiter(
    key_func=_get_client_ip,
    default_limits=[rate_limit_config.DEFAULT],
    storage_uri="memory://",
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    logging.warning(
        "Rate limit exceeded: %s %s from %s — limit: %s",
        request.method,
        request.url.path,
        _get_client_ip(request),
        exc.detail,
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too many requests",
            "detail": str(exc.detail),
            "retry_after": exc.detail,
        },
    )
