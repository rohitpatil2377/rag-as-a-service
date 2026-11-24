"""FastAPI wiring: request context middleware + error handlers + health router.

Applied once by the modular-monolith app (``apps/api``) and reusable by any service
that is later split out into its own process.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from rag_common.errors import PlatformError
from rag_common.logging import get_logger

_log = get_logger(__name__)
Handler = Callable[[Request], Awaitable[Response]]

# HTTP status at/above which an PlatformError is logged at error level (server faults).
_SERVER_ERROR_FLOOR = 500


async def _context_middleware(request: Request, call_next: Handler) -> Response:
    """Bind request_id (and later tenant_id) into the logging context."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["x-request-id"] = request_id
    return response


def _apply_security_headers(response: Response) -> None:
    """Set the HTTP security headers a VAPT scan expects on every API response.

    The API serves JSON only, so a maximally strict CSP is correct (nothing to source or
    frame). These defend against clickjacking, MIME sniffing, referrer leakage, and stop
    version/tech disclosure via the Server banner. Values are configurable (``Settings``).
    """
    from rag_common.config import get_settings  # noqa: PLC0415 — avoid import cycle at module load

    settings = get_settings()
    if not settings.security_headers_enabled:
        return
    h = response.headers
    h["X-Content-Type-Options"] = "nosniff"
    h["X-Frame-Options"] = "DENY"
    h["Referrer-Policy"] = "strict-origin-when-cross-origin"
    h["Content-Security-Policy"] = settings.csp_api
    h["Permissions-Policy"] = settings.permissions_policy
    if settings.hsts_max_age > 0:
        h["Strict-Transport-Security"] = f"max-age={settings.hsts_max_age}; includeSubDomains"
    # Overwrite uvicorn's version-disclosing Server banner (also stripped at the server via
    # ``server_header=False``; this is defence-in-depth in case a proxy re-adds one).
    h["Server"] = settings.server_banner


async def _security_headers_middleware(request: Request, call_next: Handler) -> Response:
    """Stamp security headers on every response, including error responses."""
    response = await call_next(request)
    _apply_security_headers(response)
    return response


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PlatformError)
    async def _rag_error_handler(_: Request, exc: PlatformError) -> JSONResponse:
        # Abstain is a normal 200 outcome, not an error to alarm on.
        if exc.http_status >= _SERVER_ERROR_FLOOR:
            _log.error("rag_error", code=exc.code, message=exc.message, details=exc.details)
        return JSONResponse(status_code=exc.http_status, content=exc.to_problem())


def install(app: FastAPI) -> None:
    """Attach context middleware + error handlers to an app."""
    # Registered last ⇒ runs outermost, so it stamps headers on EVERY response, including
    # the ones produced by the error handlers below.
    app.middleware("http")(_context_middleware)
    app.middleware("http")(_security_headers_middleware)
    _install_error_handlers(app)
