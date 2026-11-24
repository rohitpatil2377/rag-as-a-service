"""`_parent_expand` — query-time parent CONTEXT attachment (FR-C2, Option B).

Pins: attaches deduped parent text to child spans without changing max_score (the abstain gate
scores CHILDREN only), no-ops on legacy chunks (no parent_id) and when the flag is off, and applies
the ACL/clearance guard so a parent the caller could not retrieve is never attached.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from svc_control_plane import pipeline

from rag_common import get_settings
from rag_schemas import Chunk, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.chunk import Span


@pytest.fixture
def _pc_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_PARENT_CHILD_ENABLED", "true")
    get_settings.cache_clear()

    async def _overrides(_kb: str) -> dict[str, float]:
        return {}

    monkeypatch.setattr(pipeline, "kb_threshold_overrides", _overrides)
    yield
    get_settings.cache_clear()


def _child(cid: str, *, parent_id: str | None, score: float) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id=cid,
        doc_id="d",
        text=f"child {cid}",
        span=Span(start=0, end=5),
        score=score,
        scorer="cross_encoder",
        parent_id=parent_id,
    )


def _parent_chunk(acl: list[str] | None = None, classification: int = 0) -> Chunk:
    return Chunk(
        chunk_id="d::p",
        doc_id="d",
        tenant_id="kb1",
        text="The parent section context.",
        acl=acl or [],
        classification=classification,
        metadata={"chunk_role": "parent"},
    )


def _seed_catalog(monkeypatch: pytest.MonkeyPatch, parent: Chunk | None) -> None:
    class _Catalog:
        async def get_chunks(self, _tenant: str, ids: list[str]) -> dict[str, Chunk]:
            return {parent.chunk_id: parent} if parent and parent.chunk_id in ids else {}

    catalog = _Catalog()

    def _get_catalog() -> _Catalog:
        return catalog

    monkeypatch.setattr(pipeline, "get_catalog", _get_catalog)


def _result(spans: list[RetrievedSpan]) -> RetrievalResult:
    q = RetrievalQuery(tenant_id="kb1", text="q", top_k=5)
    return RetrievalResult(query=q, spans=spans, max_score=max((s.score for s in spans), default=0.0))


async def test_attaches_parent_text_without_changing_max_score(_pc_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch, _parent_chunk())
    retrieval = _result([_child("c1", parent_id="d::p", score=0.8), _child("c2", parent_id="d::p", score=0.6)])
    out, added = await pipeline._parent_expand(retrieval.query, retrieval)
    assert added == 1  # one distinct parent
    assert all(s.parent_text == "The parent section context." for s in out.spans)
    assert out.max_score == retrieval.max_score  # abstain gate value unchanged by expansion


async def test_no_op_on_legacy_chunks_without_parent_id(_pc_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch, _parent_chunk())
    retrieval = _result([_child("c1", parent_id=None, score=0.8)])
    out, added = await pipeline._parent_expand(retrieval.query, retrieval)
    assert added == 0
    assert out is retrieval  # untouched


async def test_no_op_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PARENT_CHILD_ENABLED", "false")
    get_settings.cache_clear()
    try:
        _seed_catalog(monkeypatch, _parent_chunk())
        retrieval = _result([_child("c1", parent_id="d::p", score=0.8)])
        out, added = await pipeline._parent_expand(retrieval.query, retrieval)
        assert added == 0
        assert out is retrieval
    finally:
        get_settings.cache_clear()


async def test_aggregate_context_never_exceeds_the_char_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct parents behind distinct children are kept best-first only until the AGGREGATE parent
    context would exceed the per-KB char budget; the rest are dropped."""
    monkeypatch.setenv("RAG_PARENT_CHILD_ENABLED", "true")
    get_settings.cache_clear()

    budget = 1000

    async def _overrides(_kb: str) -> dict[str, float]:
        return {"parent_child_context_max_chars": float(budget)}

    monkeypatch.setattr(pipeline, "kb_threshold_overrides", _overrides)

    # Four distinct parents of 400 chars each (distinct text so the dedup set stays distinct): the
    # first two fit under 1000, a third would reach 1200 so it (and the fourth) are dropped.
    parents = {
        f"d::p{i}": Chunk(chunk_id=f"d::p{i}", doc_id="d", tenant_id="kb1", text=chr(97 + i) * 400) for i in range(4)
    }

    class _Catalog:
        async def get_chunks(self, _tenant: str, ids: list[str]) -> dict[str, Chunk]:
            return {pid: parents[pid] for pid in ids if pid in parents}

    catalog = _Catalog()

    def _get_catalog() -> _Catalog:
        return catalog

    monkeypatch.setattr(pipeline, "get_catalog", _get_catalog)
    try:
        spans = [_child(f"c{i}", parent_id=f"d::p{i}", score=1.0 - i * 0.1) for i in range(4)]
        retrieval = _result(spans)
        out, added = await pipeline._parent_expand(retrieval.query, retrieval)
        attached = {s.parent_text for s in out.spans if s.parent_text}
        assert added == len(attached)
        assert sum(len(t) for t in attached) <= budget, "aggregate parent context exceeded the budget"
        assert added == 2  # two 400-char parents fit under 1000; a third would reach 1200
    finally:
        get_settings.cache_clear()


async def test_parent_the_caller_cannot_see_is_not_attached(_pc_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Parent restricted to a group the caller does not hold (query.acl empty) -> not attached.
    _seed_catalog(monkeypatch, _parent_chunk(acl=["secret-group"]))
    retrieval = _result([_child("c1", parent_id="d::p", score=0.8)])
    out, added = await pipeline._parent_expand(retrieval.query, retrieval)
    assert added == 0
    assert all(s.parent_text is None for s in out.spans)
