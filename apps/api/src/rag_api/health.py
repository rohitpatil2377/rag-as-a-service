"""Liveness + readiness probes (K8s).

Two distinct signals, as Kubernetes expects:

* **liveness** — is the PROCESS alive? Cheap, dependency-free; a failing liveness probe makes
  K8s RESTART the pod, so it must never depend on a downstream store (a store blip would
  restart-loop every replica).
* **readiness** — can this replica actually SERVE? It checks the stores a query needs (vector,
  BM25, embeddings, metadata catalog). A failing readiness probe pulls the pod OUT of the
  Service load-balancer WITHOUT restarting it, so a store blip sheds traffic and recovers on
  its own.

In-memory profile has no external stores, so it is always ready — the same twin the tests run.
"""

from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings

_log = get_logger(__name__)
_PROBE_TIMEOUT = 3.0
_HTTP_SERVER_ERROR = 500  # >=500 => the store itself is failing (4xx still means it is UP)

# One tiny pooled engine reused across readiness probes — created lazily so an import never
# opens a connection, and never per-probe (K8s calls readiness every few seconds).
_engine: AsyncEngine | None = None


def _pg_engine() -> AsyncEngine:
    global _engine  # noqa: PLW0603 — module-level lazy singleton, guarded
    if _engine is None:
        _engine = create_async_engine(get_settings().postgres_dsn, pool_size=1, max_overflow=0, pool_pre_ping=True)
    return _engine


async def _http_ok(client: httpx.AsyncClient, base_url: str, path: str) -> bool:
    """A store is reachable if it answers below 500 (200/redirect/4xx all mean 'up')."""
    try:
        resp = await client.get(f"{base_url.rstrip('/')}{path}")
    except (httpx.HTTPError, OSError):
        return False
    return resp.status_code < _HTTP_SERVER_ERROR


async def _pg_ok() -> bool:
    try:
        async with _pg_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # any driver/connection error means "not ready"
        _log.warning("readiness_postgres_unreachable", error=str(exc))
        return False
    return True


async def readiness() -> tuple[bool, dict[str, bool]]:
    """(ready, per-store checks). Ready only when EVERY store a query needs is reachable.

    In-memory profile: always ready (no external stores)."""
    settings = get_settings()
    if settings.inmemory:
        return True, {"profile": True}
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
        qdrant, opensearch, tei, postgres = await asyncio.gather(
            _http_ok(client, settings.qdrant_url, "/healthz"),
            _http_ok(client, settings.opensearch_url, "/_cluster/health"),
            _http_ok(client, settings.tei_url, "/health"),
            _pg_ok(),
        )
    checks = {"qdrant": qdrant, "opensearch": opensearch, "tei": tei, "postgres": postgres}
    return all(checks.values()), checks


async def aclose() -> None:
    """Dispose the readiness engine (app shutdown)."""
    global _engine  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
