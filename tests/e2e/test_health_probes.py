"""Liveness + readiness probes (K8s).

Liveness is dependency-free (process up). Readiness reports store reachability; the in-memory
profile has no external stores, so it is always ready.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _client() -> httpx.AsyncClient:
    from rag_api.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t")


async def test_liveness_is_dependency_free(inmemory: None) -> None:
    async with _client() as c:
        r = await c.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


async def test_readiness_is_true_on_the_inmemory_profile(inmemory: None) -> None:
    async with _client() as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["checks"] == {"profile": True}


async def test_readiness_503_when_a_store_is_unreachable(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the readiness probe at real-store mode with dead URLs — it must report NOT ready
    (503) with a per-store breakdown, so K8s sheds traffic instead of serving broken answers."""
    from rag_api import health
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_INMEMORY", "false")
    monkeypatch.setenv("RAG_QDRANT_URL", "http://127.0.0.1:1")  # nothing listens here
    monkeypatch.setenv("RAG_OPENSEARCH_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("RAG_TEI_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("RAG_POSTGRES_DSN", "postgresql+psycopg://x:x@127.0.0.1:1/x")
    get_settings.cache_clear()
    try:
        ready, checks = await health.readiness()
        assert ready is False
        assert checks["qdrant"] is False and checks["postgres"] is False
    finally:
        await health.aclose()
        get_settings.cache_clear()
