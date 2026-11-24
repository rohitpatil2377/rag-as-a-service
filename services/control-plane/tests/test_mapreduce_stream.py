"""The STREAMED map-reduce answer (RAG v2 Phase 12's "stream the reduce output").

The claim everything else rests on is **the streamed answer and the non-streamed answer are the
same text**. The reduce calls run concurrently, so a later facet routinely finishes first, and
emitting it first would make the same question answer differently depending on which transport
the caller's console happened to use. So the first test here is an equality test between the two
paths, and the rest pin the pieces that make it hold: answer order, incremental citation
numbering, and the heading not appearing without a body under it.

Also pinned: Phases 7-9 now run on the streaming path at all. They did not before this slice,
which meant a tenant with the v2 flags on got the whole v2 query path through the API and none
of it through the console — the same question, answered two ways, with nothing saying so.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline, wiring

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, tenant_scope
from rag_core.doc_profile import DocProfile, ProfileFacet
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import SectionEnrichment
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AnswerStatus, AuditAction, SourceKind

_TENANT = "mrstreamkb"
_DOC = "corrigendum-v.txt"
_SWITCH_TEXT = "Switches. The switches shall comply with UL 62368 and FCC Part 15 Class A certifications."
_FIRE_TEXT = "Firewalls. Each firewall solution shall be supplied with an HBOM and an SBOM."
_TEXT = f"{_SWITCH_TEXT}\n\n{_FIRE_TEXT}"
_QUERY = "list all the compliance details from this document"


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool = True, **env: str) -> None:
    for flag in ("RAG_V2_ROUTER", "RAG_V2_RETRIEVAL", "RAG_V2_GENERATE"):
        monkeypatch.setenv(flag, "true" if on else "false")
    monkeypatch.setenv("RAG_QUERY_LOG_ENABLED", "true")
    # The faithfulness checker is a separate concern and its own tests; off here so the status
    # these tests assert is the one `ground` produced.
    monkeypatch.setenv("RAG_FAITHFULNESS_ENABLED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


_SWITCH_SECTION = "- The switches shall comply with UL 62368 [m1]"
_FIRE_SECTION = "- Each firewall is supplied with an HBOM and an SBOM [m2]"


class _Gateway:
    """Answers by PROMPT CONTENT, and can stream. The reduce calls run concurrently, so a double
    scripted by call order would hand one facet's reply to another under a different schedule."""

    def __init__(self, *, reduce_delay_first: float = 0.0, fail_first: bool = False) -> None:
        self.prompts: list[str] = []
        self.streamed: list[str] = []
        self._delay_first = reduce_delay_first
        self._fail_first = fail_first

    def _reduce_reply(self, prompt: str) -> str:
        return _SWITCH_SECTION if "certifications" in prompt.rsplit("## Topic", maxsplit=1)[-1] else _FIRE_SECTION

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        if "## Extracted items for this topic" in prompt:
            return self._reduce_reply(prompt)
        return _map_reply(prompt)

    async def stream(self, prompt: str, **_: Any) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        self.streamed.append(prompt)
        if self._fail_first:
            raise RuntimeError("stream unavailable")
        if self._delay_first:
            import asyncio

            await asyncio.sleep(self._delay_first)
        for token in self._reduce_reply(prompt).split(" "):
            yield token + " "

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _map_reply(prompt: str) -> str:
    import json

    quote = "UL 62368" if "Switches" in prompt else "HBOM and an SBOM"
    facet = "certifications" if "Switches" in prompt else "supply_chain"
    return json.dumps([{"facet_id": facet, "statement": f"The document requires {quote}.", "quote": quote}])


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
        sections=(section("sec-switch", "21. Switches", 0), section("sec-fire", "22. Firewalls", 1)),
        source="markup",
    )


async def _seed() -> None:
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
            facets=(ProfileFacet("certifications", "Certifications", "d"), ProfileFacet("supply_chain", "SC", "d")),
        ),
    )
    await catalog.replace_document_structure(_TENANT, _tree())
    await catalog.replace_section_enrichment(
        _TENANT,
        _DOC,
        {
            "sec-switch": SectionEnrichment(
                section_id="sec-switch", summary="Switch certifications.", facet_ids=("certifications",)
            ),
            "sec-fire": SectionEnrichment(
                section_id="sec-fire", summary="Firewall supply chain.", facet_ids=("supply_chain",)
            ),
        },
    )
    existing = await catalog.document_chunks(_TENANT, _DOC)
    base = existing[0]
    await catalog.record_chunks(
        _TENANT,
        [
            base.model_copy(update={"chunk_id": "ck-switch", "section_id": "sec-switch", "text": _SWITCH_TEXT}),
            base.model_copy(update={"chunk_id": "ck-fire", "section_id": "sec-fire", "text": _FIRE_TEXT}),
        ],
    )


def _patch(monkeypatch: pytest.MonkeyPatch, gw: _Gateway) -> None:
    async def _resolve(_tenant: str, model: str | None = None) -> Any:
        return gw

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)


async def _stream(monkeypatch: pytest.MonkeyPatch, gw: _Gateway, question: str = _QUERY) -> tuple[str, Any]:
    """Returns the concatenated token text and the final answer."""
    _patch(monkeypatch, gw)
    text = ""
    final = None
    with tenant_scope(_TENANT):
        async for event in pipeline.stream_answer(
            RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5, doc_ids=(_DOC,)), actor="asker"
        ):
            if event["type"] == "token":
                text += event["text"]
            elif event["type"] == "done":
                final = event["answer"]
    return text, final


async def _ask(monkeypatch: pytest.MonkeyPatch, gw: _Gateway, question: str = _QUERY) -> Any:
    _patch(monkeypatch, gw)
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5, doc_ids=(_DOC,)), actor="asker"
        )


# --- the claim everything rests on ------------------------------------------------------------------


async def test_the_streamed_text_equals_the_non_streamed_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reduce calls run concurrently and a later facet routinely finishes first. If the
    stream emitted completion order, the same question would answer differently depending on
    which transport the caller's console used."""
    _enable(monkeypatch)
    await _seed()

    streamed, final = await _stream(monkeypatch, _Gateway())
    direct = await _ask(monkeypatch, _Gateway())

    assert streamed.strip() == direct.answer.strip()
    assert final is not None and final["answer"].strip() == direct.answer.strip()


async def test_the_answer_order_is_not_completion_order(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first batch is made slow on purpose, so the second finishes first. It still comes
    second in the text."""
    _enable(monkeypatch)
    await _seed()

    streamed, _final = await _stream(monkeypatch, _Gateway(reduce_delay_first=0.05))

    assert streamed.index("certifications") < streamed.index("supply chain")


async def test_the_citations_are_numbered_identically_on_both_paths(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming path rewrites the markers piece by piece and the non-streaming path in one
    call, through the same resolver. First appearance and streaming order are the same order,
    which is what makes the numbering agree."""
    _enable(monkeypatch)
    await _seed()

    _streamed, final = await _stream(monkeypatch, _Gateway())
    direct = await _ask(monkeypatch, _Gateway())

    assert final is not None
    assert [c["chunk_id"] for c in final["citations"]] == [c.chunk_id for c in direct.citations]
    assert "[1]" in final["answer"] and "[m1]" not in final["answer"]


# --- the shape of the stream --------------------------------------------------------------------------


async def test_the_opening_line_goes_out_before_any_model_answers(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is a fact the caller already holds, and it is most of what makes this feel streamed."""
    _enable(monkeypatch)
    await _seed()
    gw = _Gateway()
    _patch(monkeypatch, gw)

    first = ""
    with tenant_scope(_TENANT):
        async for event in pipeline.stream_answer(
            RetrievalQuery(tenant_id=_TENANT, text=_QUERY, top_k=5, doc_ids=(_DOC,)), actor="asker"
        ):
            if event["type"] == "token":
                first = event["text"]
                break

    assert _DOC in first, "the profile line, not a model token"


async def test_the_first_batch_is_streamed_and_the_rest_are_not(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time-to-first-token is the only thing streaming is for, and the first batch is the one
    nothing can block. Streaming the rest would buy nothing — their tokens could not be emitted
    until their turn came anyway."""
    _enable(monkeypatch)
    await _seed()
    gw = _Gateway()

    await _stream(monkeypatch, gw)

    reduce_prompts = [p for p in gw.prompts if "## Extracted items for this topic" in p]
    assert len(reduce_prompts) == 2, "two facets, two reduce calls"
    assert len(gw.streamed) == 1, "exactly one of them was streamed"


async def test_the_coverage_line_closes_the_stream(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    await _seed()

    streamed, _final = await _stream(monkeypatch, _Gateway())

    assert streamed.strip().endswith("sections in scope.")


async def test_a_failing_first_stream_loses_its_section_not_the_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the heading it would have carried does not appear with the next facet's bullets
    underneath it."""
    _enable(monkeypatch)
    await _seed()

    streamed, final = await _stream(monkeypatch, _Gateway(fail_first=True))

    assert "supply chain" in streamed, "the second facet still wrote"
    assert "### certifications\n### supply chain" not in streamed
    assert final is not None and final["status"] == AnswerStatus.GROUNDED.value


# --- the switch ---------------------------------------------------------------------------------------


async def test_flag_off_streams_the_ordinary_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()

    streamed, final = await _stream(monkeypatch, _Gateway())

    assert "sections in scope." not in streamed, "no coverage line, so no map-reduce answer"
    assert final is not None


async def test_a_lookup_streams_the_ordinary_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the aggregate class gets a section selection, and without one there is nothing to
    map over — so the ordinary single-completion stream runs."""
    _enable(monkeypatch)
    await _seed()

    streamed, _final = await _stream(monkeypatch, _Gateway(), "what certification must the switches have?")

    assert "sections in scope." not in streamed


# --- what it must not do ------------------------------------------------------------------------------


async def test_the_router_decision_and_selection_reach_the_query_log(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is written by the WRAPPER, which cannot see the impl's locals — so without the
    internal meta event the streaming path would log NULL for both, leaving the Phase-1 columns
    blind to exactly the path the console uses."""
    import json

    _enable(monkeypatch)
    await _seed()

    await _stream(monkeypatch, _Gateway())

    rows = await _twin().query_log(_TENANT)
    assert rows, "a row was written"
    row = rows[-1]
    assert json.loads(row["router_json"])["intent"] == "aggregate"
    assert row["selected_sections"] is not None


async def test_the_internal_meta_event_never_reaches_the_client(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is not part of the client-facing protocol (`phase`, `token`, `done`)."""
    _enable(monkeypatch)
    await _seed()
    _patch(monkeypatch, _Gateway())

    seen: set[str] = set()
    with tenant_scope(_TENANT):
        async for event in pipeline.stream_answer(
            RetrievalQuery(tenant_id=_TENANT, text=_QUERY, top_k=5, doc_ids=(_DOC,)), actor="asker"
        ):
            seen.add(str(event["type"]))

    assert seen <= {"phase", "token", "done"}, f"unexpected event types: {seen}"


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()
    await _stream(monkeypatch, _Gateway())
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    without = sum(1 for e in events if e.action is AuditAction.QUERY)

    _enable(monkeypatch, on=True)
    await _stream(monkeypatch, _Gateway())
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    with_stream = sum(1 for e in events if e.action is AuditAction.QUERY)

    assert with_stream - without == without, "the second question added exactly as many events as the first"
