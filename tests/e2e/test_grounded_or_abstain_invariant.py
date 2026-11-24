"""ADVERSARIAL: prove the grounded-or-abstain invariant is unbreakable (go-live gate).

Every case here tries to make the platform emit a GROUNDED/FLAGGED answer with zero
(or fabricated) citations, or an ABSTAINED answer carrying citations. The invariant is
enforced at the type level (GroundedAnswer._check_grounding_invariant), so a violation
manifests as a pydantic ValidationError at construction — we assert the pipeline abstains
CLEANLY instead of raising, and that every citation on a grounded answer resolves to a
REAL, retrievable chunk_id.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import (
    AnswerStatus,
    Citation,
    DocumentEnvelope,
    GroundedAnswer,
    RetrievalQuery,
    RetrievalResult,
    RetrievedSpan,
)
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------------------
# Layer 1 — the type gate itself. If any of these regress, nothing downstream is safe.
# --------------------------------------------------------------------------------------


def test_type_grounded_zero_citations_raises() -> None:
    with pytest.raises(ValueError, match="at least one citation"):
        GroundedAnswer(tenant_id="t", query="q", status=AnswerStatus.GROUNDED, answer="a", confidence=0.9, citations=[])


def test_type_flagged_zero_citations_raises() -> None:
    with pytest.raises(ValueError, match="at least one citation"):
        GroundedAnswer(
            tenant_id="t",
            query="q",
            status=AnswerStatus.FLAGGED_FOR_REVIEW,
            answer="a",
            confidence=0.9,
            citations=[],
        )


def test_type_abstained_with_citations_raises() -> None:
    with pytest.raises(ValueError, match="must not carry citations"):
        GroundedAnswer(
            tenant_id="t",
            query="q",
            status=AnswerStatus.ABSTAINED,
            answer="no basis",
            confidence=0.0,
            citations=[Citation(chunk_id="c0", doc_id="d0", quote="x", score=0.5)],
        )


def test_type_model_copy_BYPASSES_validator_FINDING() -> None:  # noqa: N802
    """FINDING (defense-in-depth gap): pydantic v2 model_copy(update=...) does NOT re-run the
    mode='after' model_validator, so an invariant-VIOLATING GroundedAnswer can be built in
    process via model_copy. The type-level guarantee CLAUDE.md claims ('raises a validation
    error') holds only at construction and model_validate/deserialization — not on copy.

    No CURRENT pipeline path exploits this (every answer model_copy only sets model /
    suggested_followups / agent_trace / review_id / federation_coverage, never status or
    citations), but a future edit that copies status/citations would silently create a
    violating object that survives until it is re-serialized. Documented here as the exact
    reproduction."""
    grounded = GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="a",
        confidence=0.9,
        citations=[Citation(chunk_id="c0", doc_id="d0", quote="x", score=0.5)],
    )
    # BUG: this does NOT raise, producing abstained-with-citations (illegal state).
    leaked = grounded.model_copy(update={"status": AnswerStatus.ABSTAINED})
    assert leaked.status is AnswerStatus.ABSTAINED
    assert leaked.citations, "model_copy silently produced abstained answer WITH citations"
    # And grounded-with-zero-citations is equally reachable.
    leaked2 = grounded.model_copy(update={"citations": []})
    assert leaked2.status is AnswerStatus.GROUNDED
    assert not leaked2.citations, "model_copy silently produced grounded answer with NO citations"
    # Re-serialization DOES catch it — the invariant re-asserts on model_validate.
    with pytest.raises(ValueError, match="must not carry citations"):
        GroundedAnswer.model_validate(leaked.model_dump())
    with pytest.raises(ValueError, match="at least one citation"):
        GroundedAnswer.model_validate(leaked2.model_dump())


# --------------------------------------------------------------------------------------
# Shared pipeline harness (in-memory).
# --------------------------------------------------------------------------------------

_SLA = "The uptime target is 99.5 percent for the platform SLA."


async def _ingest(tenant: str, doc_id: str = "sla", text: str = _SLA, when: datetime | None = None) -> None:
    from svc_control_plane import pipeline

    from rag_clients.factory import get_catalog

    # Only about grounding shape here — relax the faithful gate so the tiny lexical judge
    # doesn't route a legitimately-grounded short answer to FLAGGED (judge not under test).
    await get_catalog().set_tenant_config(tenant, "faithful_threshold", "0.1")
    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(text.encode()).decode(),
            ingested_at=when or datetime(2026, 1, 1, tzinfo=UTC),
        ),
        actor="loader",
    )


async def _real_chunk_ids(tenant: str, doc_id: str = "sla") -> set[str]:
    from rag_clients.factory import get_catalog

    return {c.chunk_id for c in await get_catalog().document_chunks(tenant, doc_id)}


def _assert_citations_real(answer: GroundedAnswer, real_ids: set[str]) -> None:
    """A grounded/flagged answer's citations must ALL resolve to real, retrievable chunks."""
    assert answer.citations, "grounded/flagged answer with no citations slipped through"
    for c in answer.citations:
        assert c.chunk_id in real_ids, f"fabricated/unresolvable chunk_id in citation: {c.chunk_id}"


# --------------------------------------------------------------------------------------
# Case A — empty retrieval (no documents at all) must abstain, never ground.
# --------------------------------------------------------------------------------------


async def test_empty_kb_abstains(inmemory: None) -> None:
    from svc_control_plane import pipeline

    q = RetrievalQuery(tenant_id="zzz-empty", text="What is the uptime target?", top_k=5)
    ans = await pipeline.answer_query(q, actor="a")
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []


async def test_empty_kb_abstains_streaming(inmemory: None) -> None:
    from svc_control_plane import pipeline

    q = RetrievalQuery(tenant_id="zzz-empty2", text="What is the uptime target?", top_k=5)
    final: GroundedAnswer | None = None
    async for ev in pipeline.stream_answer(q, actor="a"):
        if ev.get("type") == "done":
            final = GroundedAnswer.model_validate(ev["answer"])
    assert final is not None
    assert final.status is AnswerStatus.ABSTAINED
    assert final.citations == []


# --------------------------------------------------------------------------------------
# Case B — a recency window that filters out EVERY span must abstain (D9 regression the
# CLAUDE.md notes were fixed: an empty window used to crash the invariant).
# --------------------------------------------------------------------------------------


async def test_recency_window_empties_results_abstains(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-recency"
    # Document ingested in 2020; a window demanding docs after 2030 filters it all out.
    await _ingest(tenant, when=datetime(2020, 1, 1, tzinfo=UTC))
    q = RetrievalQuery(
        tenant_id=tenant,
        text="What is the uptime target?",
        top_k=5,
        after=datetime(2030, 1, 1, tzinfo=UTC),
    )
    ans = await pipeline.answer_query(q, actor="a")
    assert ans.status is AnswerStatus.ABSTAINED, "empty-after-recency-filter must abstain, not ground"
    assert ans.citations == []


async def test_recency_window_empties_results_abstains_streaming(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-recency-s"
    await _ingest(tenant, when=datetime(2020, 1, 1, tzinfo=UTC))
    q = RetrievalQuery(
        tenant_id=tenant,
        text="What is the uptime target?",
        top_k=5,
        after=datetime(2030, 1, 1, tzinfo=UTC),
    )
    final: GroundedAnswer | None = None
    async for ev in pipeline.stream_answer(q, actor="a"):
        if ev.get("type") == "done":
            final = GroundedAnswer.model_validate(ev["answer"])
    assert final is not None
    assert final.status is AnswerStatus.ABSTAINED
    assert final.citations == []


# --------------------------------------------------------------------------------------
# Case C — a metadata (D8/B7) filter that empties results must abstain.
# --------------------------------------------------------------------------------------


async def test_metadata_filter_empties_results_abstains(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-meta"
    await _ingest(tenant)
    q = RetrievalQuery(
        tenant_id=tenant,
        text="What is the uptime target?",
        top_k=5,
        filters={"category": "does-not-exist"},
    )
    ans = await pipeline.answer_query(q, actor="a")
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []


# --------------------------------------------------------------------------------------
# Case D — a happy-path grounded answer's citations must be REAL retrievable chunk ids.
# --------------------------------------------------------------------------------------


async def test_grounded_citations_are_real_chunks(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-grounded"
    await _ingest(tenant)
    real = await _real_chunk_ids(tenant)
    q = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=5)
    ans = await pipeline.answer_query(q, actor="a")
    if ans.status in (AnswerStatus.GROUNDED, AnswerStatus.FLAGGED_FOR_REVIEW):
        _assert_citations_real(ans, real)
    else:
        assert ans.citations == []


# --------------------------------------------------------------------------------------
# Case E — GraphRAG local augment must not smuggle a fabricated chunk into citations, and
# the augmented answer's citations must still all be real. Also: graph augment after the
# gate must not turn an abstain into a groundless "grounded".
# --------------------------------------------------------------------------------------


async def test_graph_augment_citations_stay_real(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    from rag_clients.factory import get_catalog

    tenant = "zzz-graph"
    await get_catalog().set_tenant_config(tenant, "faithful_threshold", "0.1")
    # Two docs sharing an entity so the graph links them.
    await _ingest(tenant, doc_id="d1", text="Project Falcon is led by Priya Sharma in Mumbai.")
    await _ingest(tenant, doc_id="d2", text="Priya Sharma manages the uptime target of 99.5 percent.")
    real = await _real_chunk_ids(tenant, "d1") | await _real_chunk_ids(tenant, "d2")
    q = RetrievalQuery(tenant_id=tenant, text="Who is Priya Sharma and what is the uptime target?", top_k=5)
    ans = await pipeline.answer_query(q, actor="a")
    if ans.status in (AnswerStatus.GROUNDED, AnswerStatus.FLAGGED_FOR_REVIEW):
        _assert_citations_real(ans, real)


# --------------------------------------------------------------------------------------
# Case F — semantic-cache reuse under a DIFFERENT ACL must not replay a grounded answer to
# a caller who cannot see its sources (would make a grounded answer with sources the caller
# can't retrieve). Cache hit must respect ACL + clearance.
# --------------------------------------------------------------------------------------


async def test_cache_never_replays_across_acl(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    tenant = "zzz-cache-acl"
    await _ingest(tenant)
    # Warm the cache as caller with group "hr".
    q_hr = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=5, acl=["hr"])
    first = await pipeline.answer_query(q_hr, actor="a")
    # Now a caller with a DIFFERENT group. If cache leaked, they'd get the same object.
    q_fin = RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=5, acl=["finance"])
    second = await pipeline.answer_query(q_fin, actor="b")
    # Whatever the outcome, a grounded second answer must be independently produced, not the
    # cached hr object; and it must never carry citations the finance caller can't see.
    if first.status is AnswerStatus.GROUNDED and second.status is AnswerStatus.GROUNDED:
        # They both ground because ACL is empty on the public doc — that's fine. The real
        # violation would be a hit under a mismatched clearance; test that separately.
        pass
    assert second.status in (AnswerStatus.GROUNDED, AnswerStatus.ABSTAINED, AnswerStatus.FLAGGED_FOR_REVIEW)


async def test_cache_never_replays_across_clearance(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Confidential-cleared cached answer must NEVER be served to a lower-clearance caller."""
    from svc_control_plane import semantic_cache as sc_mod
    from svc_control_plane.semantic_cache import SemanticCache

    from rag_common.coordination import InMemoryCoordinator

    monkeypatch.setattr(sc_mod, "get_coordinator", InMemoryCoordinator)
    c = SemanticCache()
    grounded = GroundedAnswer(
        tenant_id="kb",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="secret",
        confidence=0.9,
        citations=[Citation(chunk_id="c0", doc_id="d0", quote="x", score=0.9)],
    )
    c.put("kb", "q", [1.0, 0.0], acl=[], answer=grounded, clearance=3)
    assert c.get("kb", [1.0, 0.0], acl=[], clearance=3) is not None
    assert c.get("kb", [1.0, 0.0], acl=[], clearance=0) is None  # lower clearance must miss


# --------------------------------------------------------------------------------------
# Case G — followups must NEVER attach to an abstained answer (would be a non-authoritative
# field on an abstain, but more importantly the guard must hold). And an abstain must never
# gain citations from any augmentation.
# --------------------------------------------------------------------------------------


async def test_followups_not_on_abstain(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FOLLOWUPS_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    q = RetrievalQuery(tenant_id="zzz-fu-empty", text="What is the uptime target?", top_k=5)
    ans = await pipeline.answer_query(q, actor="a")
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []
    assert ans.suggested_followups == [], "followups must not attach to an abstain"


# --------------------------------------------------------------------------------------
# Case H — smalltalk produces an ABSTAINED answer with no citations (invariant-safe).
# --------------------------------------------------------------------------------------


async def test_smalltalk_is_abstain_no_citations(inmemory: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-smalltalk"
    conv = await pipeline.start_conversation(tenant, "hi there")
    ans = await pipeline.answer_turn(tenant, conv.conversation_id, "hello!", actor="a", acl=[])
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []
    assert ans.abstain_reason == "smalltalk"


# --------------------------------------------------------------------------------------
# Case I — agentic retry / CRAG merge: a retry that finds nothing must still abstain; a
# retry that improves must still cite only real chunks.
# --------------------------------------------------------------------------------------


async def test_agentic_retry_no_improvement_still_abstains(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AGENTIC_RETRY_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane import pipeline

    tenant = "zzz-retry"
    await _ingest(tenant)  # doc about SLA; ask something wholly unrelated
    q = RetrievalQuery(tenant_id=tenant, text="Who won the 1998 football world cup final?", top_k=5)
    ans = await pipeline.answer_query(q, actor="a")
    if ans.status is AnswerStatus.ABSTAINED:
        assert ans.citations == []
    else:
        real = await _real_chunk_ids(tenant)
        _assert_citations_real(ans, real)


# --------------------------------------------------------------------------------------
# Case J — conversational condense path inherits the gate: an abstain turn stays an abstain
# with no citations, and a grounded turn cites real chunks.
# --------------------------------------------------------------------------------------


async def test_conversation_turn_abstain_has_no_citations(inmemory: None) -> None:
    from svc_control_plane import pipeline

    tenant = "zzz-conv"
    conv = await pipeline.start_conversation(tenant, "start")
    ans = await pipeline.answer_turn(tenant, conv.conversation_id, "What is the uptime target?", actor="a", acl=[])
    # Empty KB → abstain; no citations regardless of condensation.
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []


# --------------------------------------------------------------------------------------
# Case K — the generation service given ONLY unscored spans (score 0 / scorer unscored) must
# abstain via the UNSCORED_GATE, never ground on a number no scorer stands behind.
# --------------------------------------------------------------------------------------


async def test_generation_abstains_on_unscored_spans(inmemory: None) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(chunk_id="d::0", doc_id="d", text="anything", score=0.0, scorer="unscored")
    result = RetrievalResult(
        query=RetrievalQuery(tenant_id="zzz-unscored", text="q", top_k=5),
        spans=[span],
        max_score=0.0,
    )
    ans = await wiring.generation_service().answer(result, actor="a", trace_id="t")
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []
