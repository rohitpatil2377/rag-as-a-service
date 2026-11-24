"""E2E: agentic retrieval retry (FR-F2, P2 CRAG-lite) — deterministic, in-memory.

A conjunctive question retrieves at 0.71 against a 0.75 abstain gate — without the
retry the platform abstains. With ``RAG_AGENTIC_RETRY_ENABLED`` the pipeline
transforms the query into sub-queries, retrieves each, and the merged result (0.80)
clears the gate: the SAME question now answers, and the decision is audited
(``agentic_retry: improved`` on a RETRIEVE event). One bounded retry, no loops.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, SourceKind

pytestmark = pytest.mark.e2e

_TEXT = "The uptime target is 99.5 percent. The latency target is 3 seconds."
_QUESTION = "What is the uptime target and what is the latency target?"
_GATE = "0.75"  # between direct retrieval (~0.71) and sub-query retrieval (0.80)


async def _arrange(tenant: str) -> None:
    from svc_control_plane import pipeline

    from rag_clients.factory import get_catalog

    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id="targets",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(_TEXT.encode()).decode(),
            ingested_at=datetime(2026, 7, 3, tzinfo=UTC),
        ),
        actor="loader",
    )
    await get_catalog().set_tenant_config(tenant, "abstain_threshold", _GATE)


async def test_weak_retrieval_abstains_without_retry(inmemory: None) -> None:
    from svc_control_plane import pipeline

    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _arrange(tenant)
        ans = await pipeline.answer_query(RetrievalQuery(tenant_id=tenant, text=_QUESTION, top_k=3), actor="officer")
    assert ans.status.value == "abstained"
    assert ans.abstain_reason == "retrieval_confidence_below_threshold"


async def test_retry_transforms_and_answers(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AGENTIC_RETRY_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _arrange(tenant)
        ans = await pipeline.answer_query(RetrievalQuery(tenant_id=tenant, text=_QUESTION, top_k=3), actor="officer")

    # The same question that abstained above now answers, with citations.
    assert ans.status.value in ("grounded", "flagged_for_review")
    assert ans.citations

    events = await get_audit_sink().list_events(tenant, None, 200)
    retries = [e for e in events if e.action is AuditAction.RETRIEVE and "agentic_retry" in e.payload]
    assert len(retries) == 1
    assert retries[0].payload["agentic_retry"] == "improved"
    assert retries[0].payload["retry_to"] > retries[0].payload["retry_from"]


async def test_strong_retrieval_never_retries(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry only fires when the first retrieval would abstain — confident
    retrievals pay zero extra latency or audit noise."""
    monkeypatch.setenv("RAG_AGENTIC_RETRY_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _arrange(tenant)
        ans = await pipeline.answer_query(
            RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=3), actor="officer"
        )

    assert ans.status.value in ("grounded", "flagged_for_review")
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert not [e for e in events if "agentic_retry" in e.payload]
