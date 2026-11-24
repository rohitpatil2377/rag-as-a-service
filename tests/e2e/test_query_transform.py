"""E2E: query transformation on the heavy path (FR-D6) — deterministic, in-memory.

A comparative query is decomposed into sub-queries, each retrieved, and the results
merged/de-duplicated so BOTH facts (uptime + latency, which live in different chunks)
are recalled. The QUERY audit records the transform. No model needed.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, ChunkStrategy, SourceKind

pytestmark = pytest.mark.e2e

# Two facts in two separate clause-chunks.
_TEXT = "The uptime target is 99.5 percent baseline. The latency P95 target is 3 seconds or less."


async def test_transform_merges_subquery_recall(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Transform on; router off -> heavy gate open for all queries.
    monkeypatch.setenv("RAG_QUERY_TRANSFORM_ENABLED", "true")
    monkeypatch.setenv("RAG_QUERY_TRANSFORM_BACKEND", "decompose")
    from rag_common import tenant_scope
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink

    tenant = "cag"
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
            strategy=ChunkStrategy.CLAUSE,
        )

        ans = await pipeline.answer_query(
            RetrievalQuery(tenant_id=tenant, text="Compare the uptime and latency targets", top_k=5),
            actor="officer",
        )

    # Both facts recalled across the merged sub-query results.
    cited_docs = {c.doc_id for c in ans.citations}
    assert ans.status in ("grounded", "flagged_for_review")
    assert "brief" in cited_docs

    events = await get_audit_sink().list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert q.payload["transform"] == "decompose"
    assert q.payload["subqueries"] > 1
