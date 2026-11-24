"""The Phase-8 strategies through the real query pipeline.

``libs/rag-core/tests/test_strategies.py`` carries the selection contract on injected scores.
This file pins the WIRING, and the claims that have to hold whatever the strategy decides:

1. **Flag off is byte-identical.** The section pass reads the catalog and calls the reranker, and
   the grain narrows both retrieval legs, so an operator must be able to decline it completely.
2. **It must not add an audit event.** Same reason as Phases 7, 10 and 12: ``count_actions``
   meters QUERY events for FinOps showback.
3. **A failure costs the section pass, never the answer.** The existing passage retrieval runs
   either way.

Plus the one thing specific to this phase: the selection reaches ``query_log.selected_sections``,
the Phase-1 column Phase 12 gave a writer and left NULL with a ``TODO(Phase-8)``.

What this file does NOT claim is that a real cross-encoder scores a real section above the
threshold — the reranker here is the deterministic lexical one. That measurement is the phase's
acceptance and is run against the AEGCL corrigendum itself; see ``docs/rag_v2/phase8_aegcl.md``.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, tenant_scope
from rag_core.doc_profile import DocProfile, ProfileFacet
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import SectionAmends, SectionEnrichment
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, ChunkLevel, SourceKind

_TENANT = "strat8kb"
_DOC = "corrigendum-v.txt"
_TEXT = (
    "Amended clauses. The bidder shall furnish an Earnest Money Deposit of Rs. 5 lakh with the bid. "
    "Switches. The switches shall comply with UL 62368 and FCC Part 15 Class A certifications."
)
# The two sections the card tree below describes. Ids are what `selected_sections` stores.
_AMEND = "sec-amend"
_SWITCH = "sec-switch"


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool, **env: str) -> None:
    """The Phase-8 flag is ALSO gated on the Phase-7 router: a strategy is chosen by an intent."""
    monkeypatch.setenv("RAG_V2_ROUTER", "true" if on else "false")
    monkeypatch.setenv("RAG_V2_RETRIEVAL", "true" if on else "false")
    monkeypatch.setenv("RAG_QUERY_LOG_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _tree() -> LayoutTree:
    def section(sid: str, crumb: str, ordinal: int) -> LayoutSection:
        return LayoutSection(
            section_id=sid,
            breadcrumb=crumb,
            title=crumb,
            depth=0,
            ordinal=ordinal,
            text="body",
            text_hash="h",
            blocks=(LayoutBlock(block_id=f"{sid}-b0", block_type="paragraph", text="body", page=1, ordinal=0),),
        )

    return LayoutTree(
        doc_id=_DOC,
        sections=(section(_AMEND, "Amended Clauses", 0), section(_SWITCH, "Attachment-1 > Switches", 1)),
        source="markup",
    )


async def _seed(*, amendment: bool = True, facets: bool = True) -> None:
    """Ingest, then write the Phase-2/3/4 artifacts directly: they are produced only on the v2
    INGEST path, which is a separate flag and is off here — this file is about the QUERY path."""
    with tenant_scope(_TENANT):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=_DOC,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 10, 22, tzinfo=UTC),
            ),
            actor="loader",
        )
    catalog = _twin()
    await catalog.replace_document_profile(
        _TENANT,
        _DOC,
        DocProfile(
            doc_type="corrigendum",
            purpose="Amends the bidding document.",
            title=_DOC,
            facets=(ProfileFacet("eligibility", "Eligibility", "d"), ProfileFacet("certifications", "Certs", "d")),
        ),
    )
    await catalog.replace_document_structure(_TENANT, _tree())
    await catalog.replace_section_enrichment(
        _TENANT,
        _DOC,
        {
            _AMEND: SectionEnrichment(
                section_id=_AMEND,
                summary="Amends the eligibility criteria and the earnest money deposit.",
                facet_ids=("eligibility",) if facets else (),
                amends=(
                    SectionAmends(target_ref="Appendix to ITB-2", old="Rs. 35 Cr", new="Rs. 19.44 Cr", quote="amended")
                    if amendment
                    else None
                ),
            ),
            _SWITCH: SectionEnrichment(
                section_id=_SWITCH,
                summary="Switch certifications: UL 62368, FCC Part 15 Class A.",
                facet_ids=("certifications",) if facets else (),
            ),
        },
    )


async def _ask(question: str, **kwargs: Any) -> Any:
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5, **kwargs), actor="asker"
        )


async def _logged() -> dict[str, Any]:
    rows = await _twin().query_log(_TENANT)
    return rows[-1] if rows else {}


# --- the switch ----------------------------------------------------------------------------------


async def test_flag_off_selects_no_sections(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    assert (await _logged())["selected_sections"] is None, "NULL means no section pass ran"


async def test_the_selection_reaches_the_query_log(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan §1.1's ``selected_sections``. Phase 12 gave the column its writer and left it NULL
    with a TODO(Phase-8); this is what fills it."""
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    stored = (await _logged())["selected_sections"]
    assert stored is not None and set(stored) <= {_AMEND, _SWITCH}, "the pass ran and named real sections"


async def test_a_lookup_question_runs_no_section_pass(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the aggregate class reads every section. A lookup searches for passages, and a NULL
    here is the honest record of that — not an empty selection."""
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert (await _logged())["selected_sections"] is None


async def test_an_unscoped_aggregate_runs_no_section_pass(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strategy reads EVERY section of everything in scope, so an unscoped aggregate would
    load the whole KB onto the latency path — and answer about "this document" from documents
    the asker never named."""
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the compliance details")

    assert (await _logged())["selected_sections"] is None


async def test_a_document_with_no_sections_runs_no_section_pass(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every document ingested before the v2 flag is in this state."""
    _enable(monkeypatch, on=True)
    await _seed()
    await _twin().replace_document_structure(_TENANT, LayoutTree(doc_id=_DOC, sections=(), source="markup"))

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    assert (await _logged())["selected_sections"] is None


# --- what the strategy does ------------------------------------------------------------------------


# The amendment view needs a cue the Phase-7 classifier does NOT read as relational first. Its
# ordering is deliberate and correct — "which clauses were amended" IS a question about the
# amendment relation — so an AGGREGATE that wants the amendment view asks for it the other way:
# "currently applicable", "in force", "latest". Worth pinning, because the obvious phrasing
# routes somewhere else and a reader of this test would otherwise assume a bug.
_AMENDMENT_QUERY = "list all the currently applicable compliance details from this document"


async def test_the_obvious_amendment_phrasing_is_a_relational_question(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the amended compliance details from this document", doc_ids=(_DOC,))

    row = await _logged()
    assert json.loads(row["router_json"])["intent"] == "relational"
    assert row["selected_sections"] is None, "relational has no section pass"


async def test_the_amendment_view_pulls_in_the_amending_section(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance document's own shape: a corrigendum's amendment table is the single most
    important thing in it, and no passage score is what puts it in scope.

    The relative floor is turned OFF and the sections are seeded with NO facets, so the amendment
    union is the ONLY thing that can select one. With either in play the test would pass whether
    step 5 existed or not — the floor always selects the best-scoring section, and an exhaustive
    intent unions in every facetted section."""
    _enable(monkeypatch, on=True, RAG_V2_SECTION_RELATIVE="0")
    await _seed(amendment=True, facets=False)

    await _ask(_AMENDMENT_QUERY, doc_ids=(_DOC,))
    with_amendment = (await _logged())["selected_sections"]

    await _seed(amendment=False, facets=False)  # same question, nothing in the document amends
    await _ask(_AMENDMENT_QUERY, doc_ids=(_DOC,))
    without = (await _logged())["selected_sections"]

    assert with_amendment == [_AMEND], "selected on the amendment view alone, scoring nothing"
    assert without == [], "no amends column, no amendment-view union — and the pass still ran"


async def test_the_strategy_rides_the_query_audit_payload(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    events = await get_audit_sink().list_events(_TENANT, None, 50)
    payload = next(e.payload for e in reversed(events) if e.action is AuditAction.QUERY)
    assert payload["rag_v2_strategy"] == "aggregate"
    assert payload["rag_v2_sections_considered"] == 2


async def test_a_lookup_narrows_retrieval_to_the_v2_grains(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan's `hybrid(q, filters, levels, n)` primitive, reaching the stores. Asserted on the
    QUERY the retriever is handed, because the corpus here is v1 chunks — which is exactly why
    the grain must be opt-in: asking for it on a v1 KB returns nothing."""
    seen: list[tuple[ChunkLevel, ...]] = []
    original = pipeline._cross_lingual_retrieve

    async def _spy(query: RetrievalQuery, *a: Any, **k: Any) -> Any:
        seen.append(query.levels)
        return await original(query, *a, **k)

    monkeypatch.setattr(pipeline, "_cross_lingual_retrieve", _spy)
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert seen and seen[0] == (ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION)


async def test_a_grain_that_matches_nothing_falls_back_to_no_grain(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this flag would otherwise ship. A v2 level exists only on chunks written
    by the v2 INGEST path — a separate flag — so every document in every deployment today is
    level-less, and a lookup narrowed to the three v2 grains retrieves zero spans and ABSTAINS
    on a question that was answerable a moment before the flag went on.

    The corpus in this file is exactly that case: ingested with v2 ingest OFF."""
    seen: list[tuple[ChunkLevel, ...]] = []
    original = pipeline._cross_lingual_retrieve

    async def _spy(query: RetrievalQuery, *a: Any, **k: Any) -> Any:
        seen.append(query.levels)
        return await original(query, *a, **k)

    monkeypatch.setattr(pipeline, "_cross_lingual_retrieve", _spy)
    _enable(monkeypatch, on=True)
    await _seed()

    answer = await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert seen == [(ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION), ()], "narrowed, then retried without"
    assert answer.citations, "and the question is still answered from the v1 corpus"


async def test_the_grain_is_not_retried_when_it_found_something(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is paid only in the broken case — a second retrieval on every lookup would
    double the cost of the cheapest path in the pipeline."""
    calls: list[tuple[ChunkLevel, ...]] = []

    async def _one_hit(query: RetrievalQuery, *a: Any, **k: Any) -> Any:
        calls.append(query.levels)
        from rag_schemas import RetrievalResult, RetrievedSpan

        span = RetrievedSpan(chunk_id="c1", doc_id=_DOC, text="Rs. 5 lakh", score=0.9, scorer="cross_encoder")
        return RetrievalResult(query=query, spans=[span], max_score=0.9, mode_used=query.mode)

    monkeypatch.setattr(pipeline, "_cross_lingual_retrieve", _one_hit)
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert calls == [(ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION)], "one call, not two"


async def test_an_aggregate_does_not_narrow_the_grain(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It does not run a passage search at all, so a level filter would narrow a retrieval it
    does not perform — and would strip the v1 chunks the fallback answer is built from."""
    seen: list[tuple[ChunkLevel, ...]] = []
    original = pipeline._cross_lingual_retrieve

    async def _spy(query: RetrievalQuery, *a: Any, **k: Any) -> Any:
        seen.append(query.levels)
        return await original(query, *a, **k)

    monkeypatch.setattr(pipeline, "_cross_lingual_retrieve", _spy)
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    assert all(levels == () for levels in seen)


# --- what the sections are scored against --------------------------------------------------------


def test_the_probes_are_the_query_the_sub_queries_and_each_facets_own_words() -> None:
    """The facet's own words are the probe that measured best against the real corrigendum — and
    they are deliberately NOT the router's ``sub_queries``, which have a second consumer (the
    decomposition block uses each as a RETRIEVAL query, where a bare keyword is a poor dense
    query). Two consumers, two shapes; see ``pipeline._section_probes``."""
    from rag_core.router import RouterDecision

    decision = RouterDecision(
        intent="aggregate",
        facets=((_DOC, "certifications"), (_DOC, "supply_chain")),
        sub_queries=("What are the certifications requirements?",),
    )

    probes = pipeline._section_probes("give me the compliance details", decision)

    assert probes == (
        "give me the compliance details",
        "What are the certifications requirements?",
        "certifications",
        "supply chain",
    )


def test_a_probe_is_never_blank_and_never_repeated() -> None:
    """Two documents sharing a facet id is the ordinary case for `compare`, and scoring the same
    string twice is a wasted rerank call over the whole candidate set."""
    from rag_core.router import RouterDecision

    decision = RouterDecision(intent="compare", facets=((_DOC, "eligibility"), ("other.pdf", "eligibility")))

    assert pipeline._section_probes("   ", decision) == ("eligibility",)


async def test_a_section_past_the_candidate_cap_is_not_scored_but_is_still_considered(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latency bound. Every probe scores every candidate, so a large document is the one
    place the section pass's cost grows without limit — but a capped-out section must stay
    SELECTABLE by the unions, which need no score, and must still be counted as considered."""
    _enable(monkeypatch, on=True, RAG_V2_MAX_SECTION_CANDIDATES="1", RAG_V2_SECTION_RELATIVE="0")
    await _seed(amendment=True, facets=False)

    await _ask(_AMENDMENT_QUERY, doc_ids=(_DOC,))

    stored = (await _logged())["selected_sections"]
    assert stored == [_AMEND], "selected on the amendment union, with no score of its own"
    events = await get_audit_sink().list_events(_TENANT, None, 50)
    payload = next(e.payload for e in reversed(events) if e.action is AuditAction.QUERY)
    assert payload["rag_v2_sections_considered"] == 2, "both sections were weighed"


async def test_an_aggregate_scopes_to_the_whole_facet_taxonomy(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aggregate deliberately names no facet, so matching facets by word overlap — which is
    right for a lookup — selects almost nothing for it. Measured on the real AEGCL corrigendum:
    the acceptance query matched 1 of the document's 8 facets on one run and 0 on another, and
    probing the whole taxonomy instead took the acceptance from 6/14 to 11/14."""
    _enable(monkeypatch, on=True, RAG_V2_SECTION_RELATIVE="0")
    await _seed()

    # Nothing in this query says "certifications"; the switches section carries that facet.
    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    stored = (await _logged())["selected_sections"]
    assert stored is not None and _SWITCH in stored, "selected on a facet the query never named"


async def test_a_lookup_does_not_widen_to_the_whole_taxonomy(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counterpart. A lookup asks about one thing, and unioning in every facetted section
    would make the facet vocabulary meaningless — it runs no section pass at all."""
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert (await _logged())["selected_sections"] is None


# --- what this feature must not do ---------------------------------------------------------------------


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()
    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    without = sum(1 for e in events if e.action is AuditAction.QUERY)

    _enable(monkeypatch, on=True)
    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    with_strategy = sum(1 for e in events if e.action is AuditAction.QUERY)

    assert with_strategy - without == without, "the second question added exactly as many events as the first"


async def test_a_failing_section_pass_does_not_fail_the_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()

    async def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(pipeline, "document_sections", _boom)
    answer = await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    assert answer.query, "the answer was still returned"
    assert (await _logged())["selected_sections"] is None, "and the failure reads as no pass, not an empty one"


async def test_the_router_decision_and_the_selection_are_logged_together(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored selection is only interpretable beside the decision that produced it — the
    intent chose the strategy and the facets drove step 4."""
    _enable(monkeypatch, on=True)
    await _seed()

    await _ask("list all the compliance details from this document", doc_ids=(_DOC,))

    row = await _logged()
    assert json.loads(row["router_json"])["intent"] == "aggregate"
    assert row["selected_sections"] is not None
