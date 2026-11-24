"""Agentic query planner — pipeline integration (in-memory, no external model).

Covers the routing contract the planner introduces:

* planner ON + a ``whole_document`` plan → the whole-document map-reduce reads ALL of the target
  doc's chunks (not a top-k sample), grounds on MULTIPLE chunks, reports HONEST full-coverage
  completeness, and cites real chunk ids;
* flag OFF → today's behaviour (no plan on the audit, no whole-document answer);
* planner ON + a ``passages`` plan → falls through to the UNCHANGED normal RAG path;
* a planner FAILURE (raising gateway) falls back to the heuristic plan and still answers — never 500s.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest
from svc_control_plane import pipeline, wiring

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.query_planner_llm import LLMQueryPlanner
from rag_core.query_plan import PlanContext, QueryPlan, QueryPlanner
from rag_schemas import Chunk, DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AnswerStatus, AuditAction, SourceKind


class _FixedPlanner:
    """A planner that returns a preset plan, standing in for the LLM backend."""

    def __init__(self, plan: QueryPlan) -> None:
        self._plan = plan

    async def plan(self, query: str, ctx: PlanContext) -> QueryPlan:
        return self._plan


class _BrokenGateway:
    model_name = "broken"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        raise RuntimeError("gateway down")

    async def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0):  # type: ignore[no-untyped-def]
        raise NotImplementedError
        yield ""  # pragma: no cover

    async def list_models(self) -> list[str]:
        return []


def _use_planner(monkeypatch: pytest.MonkeyPatch, planner: QueryPlanner) -> None:
    """Force the pipeline to route through ``planner`` (mirrors resolving the tenant planner)."""

    async def _resolve(tenant_id: str) -> QueryPlanner:
        return planner

    monkeypatch.setattr(wiring, "resolve_tenant_query_planner", _resolve)


async def _seed_multichunk_doc(tenant: str, doc_id: str, obligations: list[str], title: str | None = None) -> None:
    """Register a document + its chunks directly in the catalog (one obligation per chunk).

    The whole-document summary reads from the catalog (document_chunks), so no vector/sparse index
    is needed — this gives deterministic control over the chunk count the map-reduce must cover."""
    catalog = get_catalog()
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            uri=f"kb://{doc_id}",
            metadata={"title": title} if title else {},
            ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
    )
    chunks = [
        Chunk(chunk_id=f"{doc_id}-c{i}", doc_id=doc_id, tenant_id=tenant, text=text, page=i + 1)
        for i, text in enumerate(obligations)
    ]
    await catalog.record_chunks(tenant, chunks)


async def _ingest_text(tenant: str, doc_id: str, text: str) -> None:
    """Full-pipeline ingest (indexes for normal retrieval), used by the passages/flag-off controls."""
    from rag_common import tenant_scope

    with tenant_scope(tenant):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=tenant,
                doc_id=doc_id,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(text.encode()).decode(),
                ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
            ),
            actor="loader",
        )


# Each obligation is padded to ~900 chars so the six chunks together (~5.4k chars ≈ 1.3k tokens)
# exceed a 1000-token budget in the map-reduce test, while still fitting the default full-read budget.
def _obligation(n: str, clause: str) -> str:
    filler = (
        " This clause forms part of the master services agreement between the parties and is binding "
        "for the full term of the engagement and any renewal thereof, and it survives termination."
    )
    return (f"Obligation {n}: {clause}" + filler * 5)[:900]


_OBLIGATIONS = [
    _obligation("one", "the vendor must encrypt all data at rest using AES-256."),
    _obligation("two", "the vendor must notify the client of any breach within twenty-four hours."),
    _obligation("three", "the vendor must retain audit logs for seven years."),
    _obligation("four", "the vendor must permit an annual third-party security audit."),
    _obligation("five", "the vendor must delete all client data within thirty days of termination."),
    _obligation("six", "the vendor must maintain uptime of at least ninety-nine point nine percent."),
]


def _whole_doc_plan() -> QueryPlan:
    return QueryPlan(
        strategy="whole_document",
        coverage="exhaustive",
        target_doc="contract",
        answer_shape="summary",
        rationale="comprehensive coverage of one named document",
        confidence=0.9,
    )


async def test_planner_routes_whole_document_map_reduce_reads_every_chunk(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    # Force the batched map-reduce path (tiny full-read budget) with one chunk per batch → every
    # chunk grounds its own extract, so coverage is provably whole-doc across DISTINCT chunk ids.
    monkeypatch.setenv("RAG_WHOLE_DOC_FULL_READ_TOKEN_BUDGET", "1000")
    monkeypatch.setenv("RAG_WHOLE_DOC_SUMMARY_BATCH_CHUNKS", "1")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    await _seed_multichunk_doc(tenant, "contract", _OBLIGATIONS)
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"),
        actor="officer",
    )

    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("intent") == "whole_document"
    assert (answer.structured or {}).get("read_mode") == "map_reduce"
    # Coverage spans MULTIPLE distinct chunks — well beyond the ~5 a top-k sample would reach.
    cited = {c.chunk_id for c in answer.citations}
    assert cited == {f"contract-c{i}" for i in range(len(_OBLIGATIONS))}
    # Late chunks (past a top-5 window) are cited → the whole document was read, not sampled.
    assert "contract-c5" in cited and "contract-c4" in cited
    # Completeness reflects the WHOLE document and passes HONESTLY on full coverage.
    assert answer.completeness is not None
    assert answer.completeness.complete is True
    assert (answer.completeness.covered, answer.completeness.expected) == (len(_OBLIGATIONS), len(_OBLIGATIONS))

    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY and e.payload.get("intent") == "whole_document")
    assert q.payload["target_doc"] == "contract"
    assert q.payload["strategy"] == "whole_document"
    assert q.payload["read_mode"] == "map_reduce"


async def test_planner_whole_document_single_full_read_when_it_fits(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that fits the budget is read in ONE full-context pass, not sampled or map-reduced."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
    # Default full-read budget is ample for this small doc → the single full-context read path.
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    await _seed_multichunk_doc(tenant, "contract", _OBLIGATIONS)
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_plan()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in this document"),
        actor="officer",
    )

    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("read_mode") == "full_read"
    # One grounded pass over the ENTIRE document → real citation(s) to actual chunk ids.
    assert answer.citations and all(c.chunk_id.startswith("contract-c") for c in answer.citations)
    # FIX 4: completeness reflects DISTINCT cited chunks vs chunks READ — a single full read that
    # only cited some chunks does NOT falsely report the whole doc complete. `expected` is chunks
    # read; `covered` is the count of distinct cited chunk ids; complete iff every read chunk cited.
    assert answer.completeness is not None
    assert answer.completeness.expected == len(_OBLIGATIONS)
    assert answer.completeness.covered == len({c.chunk_id for c in answer.citations})
    assert answer.completeness.complete is (answer.completeness.covered >= len(_OBLIGATIONS))

    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY and e.payload.get("intent") == "whole_document")
    assert q.payload["read_mode"] == "full_read"


async def test_planner_multi_target_summarises_each_matched_doc(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 1 (new): an ambiguous target within the cap summarises EACH matched doc and combines them
    into one grounded answer — hiding nothing, citations attributed per document."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    await _seed_multichunk_doc(tenant, "va-2023", _OBLIGATIONS[:2], title="Vendor Agreement 2023")
    await _seed_multichunk_doc(tenant, "va-2024", _OBLIGATIONS[2:4], title="Vendor Agreement 2024")
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_target_plan("Vendor Agreement")))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="Summarise the key obligations in the vendor agreement"),
        actor="officer",
    )
    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("intent") == "whole_document_multi"
    assert set((answer.structured or {}).get("targets") or []) == {"va-2023", "va-2024"}
    # Citations are attributed across BOTH documents (real chunk ids from each).
    cited_docs = {c.doc_id for c in answer.citations}
    assert cited_docs == {"va-2023", "va-2024"}
    # Each document has its own labelled section.
    assert "Vendor Agreement 2023" in answer.answer and "Vendor Agreement 2024" in answer.answer

    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY and e.payload.get("intent") == "whole_document_multi")
    assert set(q.payload["targets"]) == {"va-2023", "va-2024"}


async def test_planner_multi_target_over_cap_falls_through_with_clarifications(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 1 (new): matches EXCEEDING the cap do not guess or dump — normal RAG answers and the
    matched titles are surfaced as non-authoritative clarification options."""
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_WHOLE_DOC_MULTI_TARGET_CAP", "3")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    # Four docs whose ids carry the "vendor agreement" tokens → all fuzzy-match; ingested (indexed)
    # so the normal RAG fall-through can actually answer.
    ids = ["vendor-agreement-2021", "vendor-agreement-2022", "vendor-agreement-2023", "vendor-agreement-2024"]
    for did in ids:
        await _ingest_text(tenant, did, f"The vendor agreement {did[-4:]} requires audit logs for seven years.")
    _use_planner(monkeypatch, _FixedPlanner(_whole_doc_target_plan("Vendor Agreement")))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="vendor agreement audit logs"), actor="officer"
    )
    # Answered from normal RAG (NOT the whole-doc mechanism), with the ambiguity surfaced.
    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("intent") != "whole_document"
    assert (answer.structured or {}).get("intent") != "whole_document_multi"
    assert set(answer.clarification_options) == set(ids)  # labels == doc_ids (no title metadata)

    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY and "plan_ambiguous_targets" in e.payload)
    assert set(q.payload["plan_ambiguous_targets"]) == set(ids)


async def test_flag_off_is_todays_behaviour(inmemory: None, lexical_gate: None) -> None:
    tenant = "acme"
    await _ingest_text(tenant, "leave", "The annual leave entitlement is twenty days per calendar year.")

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="annual leave entitlement"), actor="officer"
    )
    assert answer.status is AnswerStatus.GROUNDED
    # No plan was produced, and the whole-document mechanism never ran.
    assert (answer.structured or {}).get("intent") != "whole_document"
    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert "plan_strategy" not in q.payload


async def test_planner_passages_falls_through_to_normal_rag(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    await _ingest_text(tenant, "leave", "The annual leave entitlement is twenty days per calendar year.")
    _use_planner(
        monkeypatch,
        _FixedPlanner(
            QueryPlan(
                strategy="passages",
                coverage="targeted",
                target_doc=None,
                answer_shape="direct",
                rationale="a pointed fact",
                confidence=0.8,
            )
        ),
    )

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="annual leave entitlement"), actor="officer"
    )
    # Normal RAG path — grounded, and NOT the whole-document mechanism.
    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("intent") != "whole_document"
    events = await get_audit_sink().list_events(tenant, None, 200)
    q = next(e for e in events if e.action is AuditAction.QUERY and e.payload.get("plan_strategy") == "passages")
    assert q.payload["plan_coverage"] == "targeted"


def _doc(doc_id: str, title: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id=doc_id,
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        metadata={"title": title},
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


def _whole_doc_target_plan(target: str) -> QueryPlan:
    return QueryPlan(
        strategy="whole_document",
        coverage="exhaustive",
        target_doc=target,
        answer_shape="summary",
        rationale="r",
        confidence=0.9,
    )


def test_match_plan_targets_returns_all_ambiguous_matches() -> None:
    """FIX 1 (new): an ambiguous fuzzy target returns EVERY matched doc (the caller summarises each,
    never guesses one)."""
    from svc_control_plane.pipeline import _match_plan_targets

    visible = [_doc("va-2023", "Vendor Agreement 2023"), _doc("va-2024", "Vendor Agreement 2024")]
    q = RetrievalQuery(tenant_id="acme", text="summarise the vendor agreement")
    matched = _match_plan_targets(_whole_doc_target_plan("Vendor Agreement"), q, visible, visible)
    assert set(matched) == {"va-2023", "va-2024"}


def test_match_plan_targets_exact_title_wins() -> None:
    """FIX 1: an exact title equality resolves to that single doc even amid fuzzy neighbours."""
    from svc_control_plane.pipeline import _match_plan_targets

    visible = [_doc("va-2023", "Vendor Agreement 2023"), _doc("hr", "HR Handbook")]
    q = RetrievalQuery(tenant_id="acme", text="x")
    assert _match_plan_targets(_whole_doc_target_plan("HR Handbook"), q, visible, visible) == ["hr"]


async def test_planner_failure_falls_back_and_still_answers(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    await _ingest_text(tenant, "leave", "The annual leave entitlement is twenty days per calendar year.")
    # A raising gateway inside the real LLM planner: it must degrade to the heuristic plan, never raise.
    _use_planner(monkeypatch, LLMQueryPlanner(_BrokenGateway()))

    answer = await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="annual leave entitlement"), actor="officer"
    )
    # The query is still answered (heuristic fallback → passages → normal RAG); no 500.
    assert answer.status is AnswerStatus.GROUNDED
    assert (answer.structured or {}).get("intent") != "whole_document"
