"""The Phase-7 router through the real query pipeline.

``libs/rag-core/tests/test_router.py`` carries the contract and the intent measurement. This file
pins the WIRING, and the two claims that have to hold whatever the router decides:

1. **Flag off is byte-identical.** The router's `doc_ids` scope retrieval and its `sub_queries`
   drive decomposition, so it changes what the pipeline does and an operator must be able to
   decline it completely.
2. **It must not add an audit event.** Same reason as the Phase-12 timing slice and the Phase-10
   verifier: ``count_actions`` meters QUERY events for FinOps showback.

Plus the one thing that is specific to this phase: the decision reaches ``query_log.router_json``,
the Phase-1 column Phase 12 gave a writer and left NULL with a ``TODO(Phase-7)``.
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
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AuditAction, SourceKind

_TENANT = "router7kb"
_DOC = "corrigendum-v.txt"
_OTHER = "unrelated.txt"
_TEXT = (
    "Financial qualification. The bidder shall have an average annual turnover of Rs. 19.44 Crores. "
    "Eligibility. Every bidder shall furnish an Earnest Money Deposit of Rs. 5 lakh with the bid."
)
_OTHER_TEXT = "Canteen policy. Meals are served between 12:00 and 14:00 on working days."


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool, backend: str = "heuristic", **env: str) -> None:
    monkeypatch.setenv("RAG_V2_ROUTER", "true" if on else "false")
    monkeypatch.setenv("RAG_V2_ROUTER_BACKEND", backend)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


async def _ingest(doc_id: str, text: str) -> None:
    with tenant_scope(_TENANT):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=doc_id,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(text.encode()).decode(),
                ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
            ),
            actor="loader",
        )


async def _profile(doc_id: str, *, doc_type: str, facets: tuple[str, ...]) -> None:
    """Give a document a Phase-3 card. The router's whole input is the card, and the card is
    written only on the v2 ingest path — which is off here, so the test writes it directly."""
    await _twin().replace_document_profile(
        _TENANT,
        doc_id,
        DocProfile(
            doc_type=doc_type,
            purpose="Amends the bidding document." if doc_type == "corrigendum" else "Staff canteen hours.",
            title=doc_id,
            facets=tuple(ProfileFacet(f, f.replace("_", " ").title(), "d") for f in facets),
        ),
    )


async def _seed() -> None:
    await _ingest(_DOC, _TEXT)
    await _ingest(_OTHER, _OTHER_TEXT)
    await _profile(_DOC, doc_type="corrigendum", facets=("financial_qualification", "eligibility"))
    await _profile(_OTHER, doc_type="policy", facets=("canteen_hours",))


async def _ask(question: str, **kwargs: Any) -> Any:
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5, **kwargs), actor="asker"
        )


async def _router_json() -> dict[str, Any] | None:
    rows = await _twin().query_log(_TENANT)
    raw = rows[-1]["router_json"] if rows else None
    return json.loads(raw) if raw else None


# --- the switch ----------------------------------------------------------------------------------


async def test_flag_off_records_no_decision(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False, RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    await _ask("list all the financial qualification requirements")

    assert await _router_json() is None, "NULL means the router did not run"


async def test_the_decision_reaches_the_query_log(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan §1.1's `router_json`. Phase 12 gave the column its writer and left it NULL with a
    TODO(Phase-7); this is what fills it."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    await _ask("list all the financial qualification requirements")

    stored = await _router_json()
    assert stored is not None
    assert stored["intent"] == "aggregate"
    assert stored["router_version"] >= 1
    assert {"doc_ids", "facets", "sub_queries", "needs_amendment_view"} <= set(stored)


async def test_the_router_reads_the_cards_and_matches_a_facet(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of routing on cards rather than titles: `financial_qualification` is a facet of
    the corrigendum's own content, and nothing in its filename says so."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    await _ask("what are the financial qualification rules?")

    stored = await _router_json()
    assert stored is not None
    assert {"doc_id": _DOC, "facet_id": "financial_qualification"} in stored["facets"]


async def test_an_unprofiled_document_is_not_in_scope(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing on a title alone is what the existing planner already does; a cardless document in
    scope would let the router believe it had been shown a profile it had not."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _ingest(_DOC, _TEXT)  # ingested, never profiled

    await _ask("what are the financial qualification rules?", doc_ids=(_DOC,))

    stored = await _router_json()
    assert stored is not None and stored["facets"] == []


# --- what the decision does ------------------------------------------------------------------------


async def test_a_named_document_becomes_the_decisions_scope(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    stored = await _router_json()
    assert stored is not None and stored["doc_ids"] == [_DOC]


async def test_a_sampled_scope_does_not_become_doc_ids(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Echoing a sample of the KB back as a scope would narrow retrieval to an arbitrary sample —
    the one way this path could be worse than no router at all."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    await _ask("what is the earnest money deposit?")

    stored = await _router_json()
    assert stored is not None and stored["doc_ids"] == []


async def test_the_answer_still_comes_back_with_the_router_on(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fall-through-to-RAG fail-safe discovery §8.4 asks for: a decision selects an EXISTING
    mechanism, so routing cannot leave a question unanswered."""
    _enable(monkeypatch, on=True)
    await _seed()

    answer = await _ask("what is the earnest money deposit?", doc_ids=(_DOC,))

    assert answer.query and answer.status is not None


# --- what this feature must not do ---------------------------------------------------------------------


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()
    await _ask("list all the requirements")
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    without = sum(1 for e in events if e.action is AuditAction.QUERY)

    _enable(monkeypatch, on=True)
    await _ask("list all the requirements")
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    with_router = sum(1 for e in events if e.action is AuditAction.QUERY)

    assert with_router - without == without, "the second question added exactly as many events as the first"


async def test_a_failing_router_does_not_fail_the_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()

    async def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("router exploded")

    monkeypatch.setattr(pipeline, "_router_scope", _boom)
    answer = await _ask("list all the requirements")

    assert answer.query, "the answer was still returned"


async def test_a_broken_gateway_falls_back_to_the_heuristic(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LLM backend's failure mode is the DETERMINISTIC backend's decision, not no decision.

    Only the ROUTER's gateway is broken, by constructing the real adapter over a raising one.
    Replacing ``wiring.resolve_tenant_gateway`` instead would break generation too, and the test
    would then pass or fail on whether the ANSWER survived — a different claim."""
    _enable(monkeypatch, on=True, backend="llm", RAG_QUERY_LOG_ENABLED="true")
    await _seed()

    class _Broken:
        model_name = "broken"

        async def complete(self, prompt: str, **_: Any) -> str:
            raise RuntimeError("model unavailable")

        def stream(self, prompt: str, **_: Any) -> Any:
            raise NotImplementedError

        async def list_models(self) -> list[str]:
            return []

    from rag_clients.query_router_v2 import LLMQueryRouter

    def _over_broken_gateway(_gateway: object, **kwargs: Any) -> LLMQueryRouter:
        return LLMQueryRouter(_Broken(), **kwargs)

    monkeypatch.setattr("svc_control_plane.pipeline.LLMQueryRouter", _over_broken_gateway)
    await _ask("list all the financial qualification requirements")

    stored = await _router_json()
    assert stored is not None and stored["intent"] == "aggregate"
    assert stored["rationale"].startswith("heuristic")
