"""Tier 3 — CONVERGENCE-BASED coverage-verification for the whole-document read (in-memory e2e).

Completeness is driven by whether the grounded refine loop reaches a FIXED POINT, not by any
boilerplate/substantive heuristic. Gated by RAG_COVERAGE_VERIFY_ENABLED:

* flag ON, omitted-then-converged — the first read omits an extractable point; the loop re-reads the
  uncited chunks through the same grounded gen.answer seam, cites it, then a further pass adds nothing
  new → converged → complete=true, the point is cited;
* flag ON, non-groundable uncited — the grounded model grounds nothing new on the uncited chunks →
  the loop converges immediately → complete=true (nothing missed) with those chunks left uncited;
* flag ON, forced non-convergence — max_loops=1 while a second point remains extractable → the loop
  stops still adding → complete=false with a "did not converge" reason;
* flag OFF — byte-identical to today (completeness = distinct cited / all chunks, no loop).

Uses the zero-model extractive gateway (cites the top source [1] of each grounded call); a small
selective gateway models a non-groundable chunk (grounds only sources carrying a sentinel).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from svc_control_plane import pipeline, wiring

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.llm_gateway import _split_sources
from rag_core.query_plan import PlanContext, QueryPlan, QueryPlanner
from rag_schemas import Chunk, DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AnswerStatus, AuditAction, SourceKind


class _FixedPlanner:
    def __init__(self, plan: QueryPlan) -> None:
        self._plan = plan

    async def plan(self, query: str, ctx: PlanContext) -> QueryPlan:
        return self._plan


class _SelectiveGateway:
    """Extractive twin that grounds ONLY sources carrying the sentinel ``GROUNDABLE`` — otherwise it
    emits the designed refusal, so its batch abstains. Models a non-groundable (footer/signature) chunk
    the grounded model can extract no new cited point from."""

    model_name = "selective"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        blocks = _split_sources(prompt)
        if not blocks:
            return "I don't know."
        top = blocks[0][1].strip().replace("\n", " ")
        if "GROUNDABLE" in top:
            return f"{top[:400]} [1]"
        return "I don't know."

    async def complete_with_usage(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> tuple[str, dict[str, int | bool | str]]:
        return await self.complete(prompt), {}

    async def stream(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield await self.complete(prompt)

    async def list_models(self) -> list[str]:
        return []


def _use_planner(monkeypatch: pytest.MonkeyPatch, planner: QueryPlanner) -> None:
    async def _resolve(tenant_id: str) -> QueryPlanner:
        return planner

    monkeypatch.setattr(wiring, "resolve_tenant_query_planner", _resolve)


def _whole_doc_plan(doc_id: str = "sop") -> QueryPlan:
    return QueryPlan(
        strategy="whole_document",
        coverage="exhaustive",
        target_doc=doc_id,
        answer_shape="summary",
        rationale="comprehensive coverage of one named document",
        confidence=0.9,
    )


async def _seed_doc(tenant: str, doc_id: str, texts: list[str]) -> None:
    """Register a doc + its chunks directly in the catalog (whole-doc reads via document_chunks)."""
    catalog = get_catalog()
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            uri=f"kb://{doc_id}",
            metadata={"title": "Cleanroom SOP"},
            ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
    )
    await catalog.record_chunks(
        tenant,
        [
            Chunk(chunk_id=f"{doc_id}-c{i}", doc_id=doc_id, tenant_id=tenant, text=t, page=1)
            for i, t in enumerate(texts)
        ],
    )


def _reset_settings() -> None:
    from rag_common.config import get_settings

    get_settings.cache_clear()


# Padding so a small doc's chars/4 estimate exceeds the (min) full-read budget → forces map-reduce,
# giving several independent batches to run concurrently. One chunk per batch (RAG_WHOLE_DOC_SUMMARY_
# BATCH_CHUNKS=1) is set per test.
_FILLER = " This clause is part of the standard operating procedure and remains binding throughout." * 20


def _long(text: str) -> str:
    return (text + _FILLER)[:1400]


_OBLIGATION_A = "Operators must gown fully before entering the sterile clean room area."
_OBLIGATION_B = "Deviations must be recorded and investigated within twenty-four hours of discovery."
_OBLIGATION_C = "Waste must be segregated into labelled bins and disposed of within seven days."


async def test_flag_on_converges_after_extracting_omitted_point(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitted-then-converged: the loop extracts+cites the point the first read missed, then a further
    pass adds nothing → converged → complete=true."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    monkeypatch.setenv("RAG_COVERAGE_VERIFY_ENABLED", "true")
    _reset_settings()

    tenant = "acme"
    await _seed_doc(tenant, "sop", [_OBLIGATION_A, _OBLIGATION_B])
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED
    cited = {c.chunk_id for c in answer.citations}
    assert cited == {"sop-c0", "sop-c1"}  # the map cited c0; the loop extracted + cited c1
    assert _OBLIGATION_B.rstrip(".") in answer.answer
    assert answer.completeness is not None
    assert answer.completeness.complete is True
    assert (answer.completeness.covered, answer.completeness.expected) == (2, 2)  # converged ⇒ expected=covered
    assert any(s.step == "coverage" and "converged" in s.detail for s in (answer.agent_trace or []))
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert any(e.action is AuditAction.QUERY and e.payload.get("coverage_stop_reason") == "converged" for e in events)


async def test_flag_on_non_groundable_uncited_converges_immediately(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grounded model grounds nothing new on the uncited chunk → the loop converges immediately →
    complete=true (nothing missed), with that chunk left uncited. No boilerplate heuristic involved."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    monkeypatch.setenv("RAG_COVERAGE_VERIFY_ENABLED", "true")
    _reset_settings()

    tenant = "acme"
    # c0 is groundable (carries the sentinel); c1 is not — the selective gateway refuses on it.
    await _seed_doc(
        tenant,
        "sop",
        ["GROUNDABLE: operators must gown before entry.", "Confidential footer. 144 College Street. Page 1 of 1."],
    )
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    gen = wiring.generation_service()
    monkeypatch.setattr(gen, "_llm", _SelectiveGateway())
    monkeypatch.setattr(gen, "_resolve_gateway", None)

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED
    assert {c.chunk_id for c in answer.citations} == {"sop-c0"}  # c1 grounded nothing → stays uncited
    assert answer.completeness is not None
    assert answer.completeness.complete is True  # converged: nothing more was extractable
    assert (answer.completeness.covered, answer.completeness.expected) == (1, 1)
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert any(e.action is AuditAction.QUERY and e.payload.get("coverage_stop_reason") == "converged" for e in events)


async def test_flag_on_non_convergence_reports_incomplete(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_loops=1 while a second extractable point remains → the loop stops still adding →
    complete=false with a 'did not converge' reason and expected showing more may be extractable."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    monkeypatch.setenv("RAG_COVERAGE_VERIFY_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_COVERAGE_MAX_LOOPS", "1")
    _reset_settings()

    tenant = "acme"
    await _seed_doc(tenant, "sop", [_OBLIGATION_A, _OBLIGATION_B, _OBLIGATION_C])
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED
    # Map cited c0; the single allowed refine pass cited c1; c2 still uncited when the loop stopped.
    assert {c.chunk_id for c in answer.citations} == {"sop-c0", "sop-c1"}
    assert answer.completeness is not None
    assert answer.completeness.complete is False
    assert (answer.completeness.covered, answer.completeness.expected) == (2, 3)  # 2 cited + 1 still uncited
    assert answer.completeness.reason is not None and "did not converge" in answer.completeness.reason
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert any(
        e.action is AuditAction.QUERY and e.payload.get("coverage_stop_reason") == "stopped_on_max_loops"
        for e in events
    )


async def test_flag_off_is_todays_behaviour(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag OFF → byte-identical: completeness = distinct cited / ALL chunks, no loop, no coverage trace."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    # RAG_COVERAGE_VERIFY_ENABLED deliberately unset → default OFF.
    _reset_settings()

    tenant = "acme"
    await _seed_doc(tenant, "sop", [_OBLIGATION_A, _OBLIGATION_B, _OBLIGATION_C])
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED
    assert {c.chunk_id for c in answer.citations} == {"sop-c0"}  # only the first chunk cited; no refine loop
    assert _OBLIGATION_B.rstrip(".") not in answer.answer
    assert answer.completeness is not None
    assert (answer.completeness.covered, answer.completeness.expected) == (1, 3)  # today's measure
    assert answer.completeness.complete is False
    assert answer.completeness.reason == "2 of 3 passage(s) not reflected in the summary"
    assert not any(s.step == "coverage" and "Coverage loop" in s.detail for s in (answer.agent_trace or []))
    events = await get_audit_sink().list_events(tenant, None, 200)
    assert not any(e.action is AuditAction.QUERY and "coverage_stop_reason" in e.payload for e in events)


class _ConcurrencyGateway:
    """Extractive twin that records how many `complete` calls are in flight at once — so a test can
    prove the MAP batches actually run CONCURRENTLY (not one-by-one)."""

    model_name = "concurrency"

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.02)  # hold the slot so overlapping batches are observable
            blocks = _split_sources(prompt)
            if not blocks:
                return "I don't know."
            top = blocks[0][1].strip().replace("\n", " ")
            return f"{top[:400]} [1]"
        finally:
            self.inflight -= 1

    async def complete_with_usage(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> tuple[str, dict[str, int | bool | str]]:
        return await self.complete(prompt), {}

    async def stream(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield await self.complete(prompt)

    async def list_models(self) -> list[str]:
        return []


class _BoomGateway:
    """Extractive twin that RAISES on any batch whose top source carries the sentinel ``BOOM`` — models
    one batch failing mid-fan-out while the others succeed."""

    model_name = "boom"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        blocks = _split_sources(prompt)
        if not blocks:
            return "I don't know."
        top = blocks[0][1].strip().replace("\n", " ")
        if "BOOM" in top:
            raise RuntimeError("batch boom")
        return f"{top[:400]} [1]"

    async def complete_with_usage(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> tuple[str, dict[str, int | bool | str]]:
        return await self.complete(prompt), {}

    async def stream(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield await self.complete(prompt)

    async def list_models(self) -> list[str]:
        return []


async def test_map_batches_run_concurrently_and_reduce_is_order_stable(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The independent MAP batches run CONCURRENTLY (>1 in flight), and the combined answer's citations
    are assembled in DOCUMENT ORDER regardless of completion order."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    # Force map-reduce with one chunk per batch so there are several independent batches to overlap.
    monkeypatch.setenv("RAG_WHOLE_DOC_FULL_READ_TOKEN_BUDGET", "1000")
    monkeypatch.setenv("RAG_WHOLE_DOC_SUMMARY_BATCH_CHUNKS", "1")
    monkeypatch.setenv("RAG_WHOLE_DOC_MAP_CONCURRENCY", "4")
    _reset_settings()

    tenant = "acme"
    texts = [_long(f"Obligation {i}: operators must complete step {i} before proceeding.") for i in range(5)]
    await _seed_doc(tenant, "sop", texts)
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    gw = _ConcurrencyGateway()
    gen = wiring.generation_service()
    monkeypatch.setattr(gen, "_llm", gw)
    monkeypatch.setattr(gen, "_resolve_gateway", None)

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("read_mode") == "map_reduce"
    assert gw.max_inflight >= 2  # batches genuinely overlapped, not awaited one-by-one
    # Order-stable reduce: citations in DOCUMENT order, one per single-chunk batch.
    assert [c.chunk_id for c in answer.citations] == [f"sop-c{i}" for i in range(5)]


async def test_one_map_batch_raising_still_yields_grounded_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single batch raising contributes nothing (per-batch fail-safe) — the other grounded batches
    still produce a grounded, cited answer; never a 500."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    monkeypatch.setenv("RAG_WHOLE_DOC_FULL_READ_TOKEN_BUDGET", "1000")
    monkeypatch.setenv("RAG_WHOLE_DOC_SUMMARY_BATCH_CHUNKS", "1")
    _reset_settings()

    tenant = "acme"
    await _seed_doc(
        tenant,
        "sop",
        [
            _long("Operators must gown before entering the clean room."),
            _long("BOOM this batch will raise inside the gateway."),
            _long("Waste must be segregated and disposed of within seven days."),
        ],
    )
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    gen = wiring.generation_service()
    monkeypatch.setattr(gen, "_llm", _BoomGateway())
    monkeypatch.setattr(gen, "_resolve_gateway", None)

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"), actor="officer"
    )

    assert answer.status is AnswerStatus.GROUNDED  # the failing batch did not sink the whole answer
    assert {c.chunk_id for c in answer.citations} == {"sop-c0", "sop-c2"}  # c1's batch raised → omitted
