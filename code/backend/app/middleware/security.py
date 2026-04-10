"""
Security middleware for Optionix backend.
Provides comprehensive security features including headers, rate limiting, and validation.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..config import settings

logger = logging.getLogger(__name__)

redis_client: Optional[Any] = None
try:
    import redis

    _client = redis.from_url(
        settings.redis_url, decode_responses=True, socket_timeout=1
    )
    _client.ping()
    redis_client = _client
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available for rate limiting: {e}")

# In-memory fallback store: key -> {window_key: count}
_memory_store: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.security_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
                "font-src 'self' data:; connect-src 'self' https:; frame-ancestors 'none';"
            ),
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Server": "Optionix-API",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in self.security_headers.items():
            response.headers[header] = value
        if "server" in response.headers:
            del response.headers["server"]
        return response


class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with Redis and in-memory fallback"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.default_limit = settings.rate_limit_requests_per_minute
        self.default_window = settings.rate_limit_window_minutes * 60
        self.endpoint_limits: Dict[str, Dict[str, int]] = {
            "/auth/login": {"requests": 5, "window": 300},
            "/auth/register": {"requests": 3, "window": 3600},
            "/trades": {"requests": 100, "window": 60},
            "/market/volatility": {"requests": 10, "window": 60},
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if settings.testing or settings.environment == "testing":
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = "999999"
            response.headers["X-RateLimit-Remaining"] = "999998"
            response.headers["X-RateLimit-Reset"] = "9999999999"
            return response
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        rate_limit_result = await self._check_rate_limits(client_ip, endpoint)

        if rate_limit_result["exceeded"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "retry_after": rate_limit_result["retry_after"],
                },
                headers={
                    "Retry-After": str(rate_limit_result["retry_after"]),
                    "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_limit_result["reset_time"]),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])
        return response

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    async def _check_rate_limits(self, client_ip: str, endpoint: str) -> Dict[str, Any]:
        current_time = int(time.time())

        if endpoint in self.endpoint_limits:
            limit_config = self.endpoint_limits[endpoint]
            requests_limit = limit_config["requests"]
            window_seconds = limit_config["window"]
        else:
            requests_limit = self.default_limit
            window_seconds = self.default_window

        window_key = str(current_time // window_seconds)
        reset_time = (current_time // window_seconds + 1) * window_seconds

        if redis_client is not None:
            try:
                rate_key = f"rl:{client_ip}:{endpoint}:{window_key}"
                current_requests = int(redis_client.get(rate_key) or 0)
                if current_requests >= requests_limit:
                    return {
                        "exceeded": True,
                        "limit": requests_limit,
                        "remaining": 0,
                        "retry_after": reset_time - current_time,
                        "reset_time": reset_time,
                    }
                pipe = redis_client.pipeline()
                pipe.incr(rate_key)
                pipe.expire(rate_key, window_seconds)
                pipe.execute()
                return {
                    "exceeded": False,
                    "limit": requests_limit,
                    "remaining": requests_limit - current_requests - 1,
                    "retry_after": 0,
                    "reset_time": reset_time,
                }
            except Exception:
                pass  # Fall through to in-memory

        # In-memory fallback
        store_key = f"{client_ip}:{endpoint}"
        current_requests = _memory_store[store_key].get(window_key, 0)
        if current_requests >= requests_limit:
            return {
                "exceeded": True,
                "limit": requests_limit,
                "remaining": 0,
                "retry_after": reset_time - current_time,
                "reset_time": reset_time,
            }
        _memory_store[store_key][window_key] = current_requests + 1
        # Clean old windows
        for old_key in list(_memory_store[store_key].keys()):
            if old_key != window_key:
                del _memory_store[store_key][old_key]

        return {
            "exceeded": False,
            "limit": requests_limit,
            "remaining": requests_limit - current_requests - 1,
            "retry_after": 0,
            "reset_time": reset_time,
        }


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"union\s+select",
            r"drop\s+table",
            r"exec\s*\(",
            r"eval\s*\(",
            r"javascript:",
            r"vbscript:",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "request_too_large", "message": "Request too large"},
            )

        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "invalid_headers", "message": "User-Agent too long"},
            )

        if self._contains_suspicious_patterns(str(request.url)):
            logger.warning(f"Suspicious URL pattern detected: {request.url}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "suspicious_request",
                    "message": "Request contains suspicious patterns",
                },
            )

        return await call_next(request)

    def _contains_suspicious_patterns(self, text: str) -> bool:
        import re

        text_lower = text.lower()
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive audit logging"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.sensitive_endpoints = [
            "/auth/login",
            "/auth/register",
            "/trades",
            "/positions",
            "/accounts",
            "/market/volatility",
        ]
        self.log_all_requests = settings.debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        if self._should_log_request(request):
            try:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                    "client_ip": (request.client.host if request.client else "unknown"),
                }
                if response.status_code >= 400:
                    logger.warning(
                        f"HTTP {response.status_code} {request.url}", extra=log_entry
                    )
                else:
                    logger.debug(
                        f"HTTP {response.status_code} {request.url}", extra=log_entry
                    )
            except Exception as e:
                logger.error(f"Failed to log request: {e}")

        response.headers["X-Process-Time"] = str(round(process_time, 4))
        return response

    def _should_log_request(self, request: Request) -> bool:
        if self.log_all_requests:
            return True
        for endpoint in self.sensitive_endpoints:
            if request.url.path.startswith(endpoint):
                return True
        return False
