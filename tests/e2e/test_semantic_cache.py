"""E2E: semantic answer cache (FR-E5, P2) — deterministic, in-memory.

Proves the four safety-critical behaviours, through the pipeline (not the class):
* a repeat question answers from cache — no second retrieval/generation, audited
  as a QUERY with ``cache_hit: true``;
* any ingest into the KB invalidates its cache (no stale answers after new knowledge);
* a caller with different ACL groups NEVER gets another caller's cached answer;
* abstains are not cached (a later answerable state isn't masked).
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, SourceKind

pytestmark = pytest.mark.e2e

_TEXT = "The uptime target is 99.5 percent."


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _ingest(tenant: str, doc_id: str = "sla", text: str = _TEXT) -> None:
    from svc_control_plane import pipeline

    from rag_clients.factory import get_catalog

    # Only GROUNDED answers cache; relax this KB's faithful gate so the small lexical
    # judge doesn't flag the short extractive answer — the judge is not under test here.
    await get_catalog().set_tenant_config(tenant, "faithful_threshold", "0.1")

    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(text.encode()).decode(),
            ingested_at=datetime(2026, 7, 3, tzinfo=UTC),
        ),
        actor="loader",
    )


async def test_repeat_question_hits_cache_and_is_audited(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _ingest(tenant)
        q = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=3)

        first = await pipeline.answer_query(q, actor="officer")
        assert first.status.value == "grounded"

        second = await pipeline.answer_query(q, actor="officer")

    # Identical cited answer, replayed.
    assert second.answer == first.answer
    assert [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations]

    events = await get_audit_sink().list_events(tenant, None, 200)
    hits = [e for e in events if e.action is AuditAction.QUERY and e.payload.get("cache_hit") is True]
    assert len(hits) == 1
    # The cached path never re-generates: exactly one GENERATE event for two answers.
    assert sum(1 for e in events if e.action is AuditAction.GENERATE) == 1


async def test_ingest_invalidates_cache(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _ingest(tenant)
        q = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=3)
        await pipeline.answer_query(q, actor="officer")

        # New knowledge lands in the KB -> every cached answer may be stale.
        await _ingest(tenant, doc_id="sla-v2", text="The uptime target was revised to 99.9 percent.")

        again = await pipeline.answer_query(q, actor="officer")

    assert again.status.value in ("grounded", "flagged_for_review")
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert not [e for e in events if e.payload.get("cache_hit") is True]


async def test_cache_never_crosses_acl_boundaries(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-A6: an entitled caller's cached answer must not replay to a caller whose
    document visibility differs — the ACL groups are part of the cache key."""
    _enable(monkeypatch)
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _ingest(tenant)
        text = "What is the uptime target?"
        entitled = RetrievalQuery(tenant_id=tenant, text=text, top_k=3, acl=["finance"])
        public = RetrievalQuery(tenant_id=tenant, text=text, top_k=3, acl=[])

        await pipeline.answer_query(entitled, actor="cfo")
        await pipeline.answer_query(public, actor="intern")  # same words, different visibility

    events = await get_audit_sink().list_events(tenant, None, 200)
    assert not [e for e in events if e.payload.get("cache_hit") is True]


async def test_abstains_are_not_cached(inmemory: None, monkeypatch: pytest.MonkeyPatch, lexical_gate: None) -> None:
    _enable(monkeypatch)
    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_common import tenant_scope

    tenant = "cag"
    with tenant_scope(tenant):
        await _ingest(tenant)
        q = RetrievalQuery(tenant_id=tenant, text="What colour is the moon made of cheese?", top_k=3)
        first = await pipeline.answer_query(q, actor="officer")
        assert first.status.value == "abstained"
        await pipeline.answer_query(q, actor="officer")

    events = await get_audit_sink().list_events(tenant, None, 200)
    assert not [e for e in events if e.payload.get("cache_hit") is True]


async def test_raising_the_abstain_floor_stops_the_cache_serving_the_old_answer(
    inmemory: None, monkeypatch: pytest.MonkeyPatch, lexical_gate: None
) -> None:
    """A hit returns a stored answer verbatim, so it skips the gate that produced it. Raising the
    KB's abstain threshold is the one control an operator has for "stop answering marginal
    questions"; without re-checking, the marginal answers keep coming until the entry expires."""
    from svc_control_plane import pipeline

    from rag_clients.factory import get_catalog

    _enable(monkeypatch)
    tenant = "cachegate"
    await _ingest(tenant)
    q = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=3)

    first = await pipeline.answer_query(q, actor="u")
    assert first.status == "grounded", first
    cached = await pipeline.answer_query(q, actor="u")
    assert cached.status == "grounded"

    # Now demand far more confidence than that answer ever had.
    await get_catalog().set_tenant_config(tenant, "abstain_threshold", str(first.confidence + 0.5))
    after = await pipeline.answer_query(q, actor="u")
    assert after.status == "abstained", "the cache served an answer today's gate would refuse"
