"""E2E: adaptive query router (FR-D5) — deterministic, in-memory.

With the flag ON, a complex query routes to the 'heavy' path (deeper top_k) and the
decision is recorded on the QUERY audit event; a simple query stays 'standard'. Grounding
contracts are unchanged. No model needed (extractive_dev + heuristic router).
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, SourceKind

pytestmark = pytest.mark.e2e

_TEXT = "Uptime target is 99.5 percent baseline. Latency P95 is 3 seconds. Backups run nightly."


async def _seed(tenant: str) -> None:
    from svc_control_plane import pipeline

    from rag_common import tenant_scope

    with tenant_scope(tenant):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=tenant,
                doc_id="brief",
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
            ),
            actor="loader",
        )


async def test_router_records_path_on_query_audit(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTER_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink

    tenant = "cag"
    await _seed(tenant)
    sink = get_audit_sink()

    # Complex query -> heavy path.
    await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Compare the uptime and latency targets and reconcile them"),
        actor="officer",
    )
    events = await sink.list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert q.payload["route_path"] == "heavy"
    assert q.payload["route_top_k"] > 5

    # Simple query -> standard path.
    await pipeline.answer_query(RetrievalQuery(tenant_id=tenant, text="uptime target?"), actor="officer")
    events = await sink.list_events(tenant, None, 100)
    q2 = next(e for e in events if e.action is AuditAction.QUERY)  # newest-first
    assert q2.payload["route_path"] == "standard"
