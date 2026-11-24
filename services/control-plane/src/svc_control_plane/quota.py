"""Per-tenant quotas & rate limits (FR-H1 — noisy-neighbour isolation).

Rate limits are enforced per ORGANISATION over a sliding 60-second window, so one busy
tenant cannot starve the others; the document cap is per knowledge base. Limits are
operator-configurable (``RAG_QUOTA_*``); a breach surfaces as 429 with a stable code
(``rate_limited`` / ``quota_exceeded``) and a retry hint — never a silent drop.

The limiter runs on the shared :mod:`rag_common.coordination` seam: an in-process sliding
window by default (one API process), or a Redis-backed shared window when ``RAG_REDIS_URL``
is set, so N replicas enforce ONE per-org limit — a busy tenant can't beat the cap by
spraying requests across replicas. No call-site change between the two.
"""

from __future__ import annotations

from rag_clients.factory import get_catalog
from rag_common import QuotaExceededError, RateLimitError, get_coordinator, get_logger, get_settings

_log = get_logger(__name__)

_WINDOW_S = 60.0


def _check(org_id: str, action: str, limit: int) -> None:
    """Record one event on the org's sliding window; raise 429 if it is over ``limit``."""
    allowed, retry_in = get_coordinator().allow(f"{org_id}:{action}", limit, _WINDOW_S)
    if not allowed:
        _log.warning("rate_limited", tenant_id=org_id, action=action, limit=limit)
        raise RateLimitError(
            f"rate limit reached: {limit} {action} requests per minute — retry in ~{retry_in}s",
            details={"action": action, "limit_per_minute": limit, "retry_after_seconds": retry_in},
        )


def check_query_rate(org_id: str) -> None:
    """Gate a query/ask request against the org's per-minute limit."""
    settings = get_settings()
    if settings.quota_enabled:
        _check(org_id, "query", settings.quota_queries_per_minute)


def check_ingest_rate(org_id: str) -> None:
    """Gate an ingest request against the org's per-minute limit."""
    settings = get_settings()
    if settings.quota_enabled:
        _check(org_id, "ingest", settings.quota_ingests_per_minute)


async def check_document_cap(kb_id: str) -> None:
    """Refuse ingest when the KB is at its document cap (hard resource quota)."""
    settings = get_settings()
    if not settings.quota_enabled or settings.quota_max_documents_per_kb <= 0:
        return
    count = len(await get_catalog().list_documents(kb_id))
    if count >= settings.quota_max_documents_per_kb:
        _log.warning("quota_exceeded", kb_id=kb_id, documents=count)
        raise QuotaExceededError(
            f"this knowledge base is at its document limit ({settings.quota_max_documents_per_kb}) — "
            "delete documents or ask your operator to raise the quota",
            details={"kb_id": kb_id, "documents": count, "limit": settings.quota_max_documents_per_kb},
        )


def reset_limiter() -> None:
    """Clear limiter state (tests)."""
    from rag_common import reset_coordinator  # noqa: PLC0415

    reset_coordinator()
