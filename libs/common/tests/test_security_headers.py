"""VAPT hardening: every API response carries the expected HTTP security headers.

Closes the 42 "Missing Header" findings from the WebScanner report by asserting the
middleware stamps HSTS / CSP / X-Frame-Options / X-Content-Type-Options / Referrer-Policy
/ Permissions-Policy — on success responses AND on error responses — and replaces the
version-disclosing Server banner.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_common.config import get_settings
from rag_common.errors import UnauthorizedError
from rag_common.middleware import install

_EXPECTED = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
}


def _app() -> FastAPI:
    get_settings.cache_clear()
    app = FastAPI()
    install(app)

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise UnauthorizedError("missing bearer token")

    return app


def test_security_headers_on_success() -> None:
    client = TestClient(_app())
    r = client.get("/ok")
    assert r.status_code == 200
    for header, value in _EXPECTED.items():
        assert r.headers.get(header) == value, header
    assert "default-src" in r.headers.get("content-security-policy", "")
    assert r.headers.get("strict-transport-security", "").startswith("max-age=")
    assert "geolocation=()" in r.headers.get("permissions-policy", "")
    # Server banner must not disclose uvicorn/version.
    assert "uvicorn" not in r.headers.get("server", "").lower()


def test_security_headers_on_error_response() -> None:
    # A 401 rejection must ALSO carry the headers (error responses were an unprotected path).
    client = TestClient(_app())
    r = client.get("/boom")
    assert r.status_code == 401
    assert r.headers.get("x-frame-options") == "DENY"
    assert "default-src" in r.headers.get("content-security-policy", "")


def test_headers_can_be_disabled() -> None:
    import os

    os.environ["RAG_SECURITY_HEADERS_ENABLED"] = "false"
    try:
        get_settings.cache_clear()
        client = TestClient(_app())
        r = client.get("/ok")
        assert "x-frame-options" not in {k.lower() for k in r.headers}
    finally:
        os.environ.pop("RAG_SECURITY_HEADERS_ENABLED", None)
        get_settings.cache_clear()
