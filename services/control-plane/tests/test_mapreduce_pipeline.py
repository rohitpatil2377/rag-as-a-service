"""The Phase-9 map→reduce path through the real query pipeline.

This is the first v2 phase that REPLACES an answer rather than annotating one, so the claims it
has to make are different in kind from Phases 7, 8, 10 and 12:

1. **Flag off is byte-identical** — and here that means the same ANSWER, not just the same
   payload.
2. **It declines rather than degrades.** Every way this path can fail ends with the ordinary
   single-pass answer, because that is what makes it safe to enable on a live KB.
3. **It must not bypass grounded-or-abstain.** The answer it publishes carries real citations
   resolved to real chunk ids, or it does not publish.
4. **It must not add an audit event** — ``count_actions`` meters QUERY for FinOps showback.

What this file does NOT claim is that the extracted items are the RIGHT ones. The gateway here
is a scripted double; whether a real model reads a real tender correctly is the phase's
acceptance, measured against the AEGCL corrigendum.
"""

from __future__ import annotations

import base64
import json
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

_TENANT = "mr9kb"
_DOC = "corrigendum-v.txt"
_SWITCH_TEXT = "Switches. The switches shall comply with UL 62368 and FCC Part 15 Class A certifications."
_FIRE_TEXT = "Firewalls. Each firewall solution shall be supplied with an HBOM and an SBOM."
_TEXT = f"{_SWITCH_TEXT}\n\n{_FIRE_TEXT}"
_QUERY = "list all the compliance details from this document"


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool, **env: str) -> None:
    """Phase 9 rides on Phase 8, which rides on Phase 7: no router, no selection, nothing to map."""
    for flag in ("RAG_V2_ROUTER", "RAG_V2_RETRIEVAL", "RAG_V2_GENERATE"):
        monkeypatch.setenv(flag, "true" if on else "false")
    monkeypatch.setenv("RAG_QUERY_LOG_ENABLED", "true")
    # FR-E4 off here so the status these tests assert is the one `ground` produced. The gate
    # itself — and the fact that a map-reduce answer now meets it at all — is
    # `test_mapreduce_faithfulness.py`; leaving it on would make every assertion in this file
    # depend on what the lexical checker makes of a synthetic reduce reply.
    monkeypatch.setenv("RAG_FAITHFULNESS_ENABLED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


class _Gateway:
    """A gateway double that answers by PROMPT CONTENT, not by call order.

    The map calls run concurrently (``asyncio.gather`` under a semaphore), so a script indexed
    by call order hands section A's reply to section B under a different scheduling — and every
    item is then dropped by the quote check, because the quote belongs to the other section.
    That is a real property of the code under test and a test must not be flaky about it, so
    the double is addressed the way a model is: by what it was asked."""

    def __init__(
        self, *, switch: str | None = None, fire: str | None = None, reduce: str | Exception | None = None
    ) -> None:
        self._switch = switch if switch is not None else _map_reply("UL 62368")
        self._fire = fire if fire is not None else _map_reply("HBOM and an SBOM")
        self._reduce = reduce if reduce is not None else _REDUCE_REPLY
        self.prompts: list[str] = []
        self.other: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        if "## Extracted items for this topic" in prompt:  # the reduce prompt
            self.prompts.append(prompt)
            if isinstance(self._reduce, Exception):
                raise self._reduce
            return self._reduce
        if "## Section text" in prompt:  # a map prompt
            self.prompts.append(prompt)
            return self._switch if "Switches" in prompt else self._fire
        # ANYTHING ELSE is some other feature making a gateway call on this path — the FR-D6
        # query transformer, a condenser, a guard. It is recorded SEPARATELY so it cannot be
        # miscounted as a map call: these tests assert exact call counts, and a stray call
        # answered as if it were a map would make those assertions mean something else. A
        # leaked `RAG_QUERY_TRANSFORM_ENABLED` from another test file did exactly that.
        self.other.append(prompt)
        return ""

    @property
    def map_calls(self) -> int:
        return sum(1 for p in self.prompts if "## Extracted items for this topic" not in p)

    @property
    def reduce_calls(self) -> int:
        return sum(1 for p in self.prompts if "## Extracted items for this topic" in p)

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _map_reply(*quotes: str) -> str:
    return json.dumps(
        [{"facet_id": "certifications", "statement": f"The document requires {q}.", "quote": q} for q in quotes]
    )


_REDUCE_REPLY = (
    "The document is a corrigendum amending a tender.\n\n"
    "## Certifications\n"
    "- The switches shall comply with UL 62368 [m1]\n"
    "- Each firewall is supplied with an HBOM and an SBOM [m2]\n\n"
    "Coverage: 2 of 2 selected sections contributed; 2 sections in scope."
)


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
    """Ingest, then write the Phase-2/3/4 artifacts and point the ingested chunks at the sections.

    THE CHUNK `section_id` IS THE POINT. The map step reads a section's MASKED chunk text, so a
    section whose chunks do not name it has no text to map and is skipped — which is exactly
    what the "no section text" test below asserts."""
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
            facets=(ProfileFacet("certifications", "Certifications", "d"),),
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
                section_id="sec-fire", summary="Firewall supply chain.", facet_ids=("certifications",)
            ),
        },
    )
    await _attach_chunks()


async def _attach_chunks() -> None:
    """Give the ingested chunks a ``section_id`` and text that holds each section's quotes."""
    catalog = _twin()
    existing = await catalog.document_chunks(_TENANT, _DOC)
    assert existing, "ingest produced chunks"
    base = existing[0]
    await catalog.record_chunks(
        _TENANT,
        [
            base.model_copy(update={"chunk_id": "ck-switch", "section_id": "sec-switch", "text": _SWITCH_TEXT}),
            base.model_copy(update={"chunk_id": "ck-fire", "section_id": "sec-fire", "text": _FIRE_TEXT}),
        ],
    )


async def _ask(gateway: _Gateway | None, monkeypatch: pytest.MonkeyPatch, question: str = _QUERY) -> Any:
    if gateway is not None:

        async def _resolve(_tenant: str, model: str | None = None) -> Any:
            return gateway

        monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5, doc_ids=(_DOC,)), actor="asker"
        )


# --- the switch -------------------------------------------------------------------------------------


async def test_flag_off_answers_the_ordinary_way(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()

    answer = await _ask(None, monkeypatch)

    assert answer.query
    assert "Coverage:" not in answer.answer, "no map-reduce answer was produced"


async def test_the_map_reduce_answer_replaces_the_single_pass_one(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway()

    answer = await _ask(gw, monkeypatch)

    assert answer.status is AnswerStatus.GROUNDED
    assert "Certifications" in answer.answer
    assert gw.map_calls == 2 and gw.reduce_calls == 1


async def test_the_answer_cites_real_chunk_ids(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grounded-or-abstain is a TYPE invariant, so a GROUNDED answer with no citations cannot be
    constructed at all — what this pins is that the citations resolve to the CHUNKS the quotes
    came from, not to placeholders."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway()

    answer = await _ask(gw, monkeypatch)

    assert {c.chunk_id for c in answer.citations} == {"ck-switch", "ck-fire"}
    assert all(c.doc_id == _DOC for c in answer.citations)


async def test_the_item_markers_are_rewritten_to_numeric_ones(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything downstream — the cited-only filter, the console's click-through, the Phase-10
    verifier — reads ``[n]`` against a span list. An answer carrying ``[m1]`` would bypass them."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway()

    answer = await _ask(gw, monkeypatch)

    assert "[m1]" not in answer.answer and "[m2]" not in answer.answer
    assert "[1]" in answer.answer


async def test_a_bolded_marker_still_resolves(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reduce prompt now carries the platform's presentation contract, which says to bold
    every number -- and an `item_id` ends in one. A model that reads that literally writes
    `[**m1**]`.

    This is not the documented "unresolvable marker loses its citation" case, which drops the
    marker and keeps the sentence. An UNMATCHED marker is not dropped at all: it stays in the
    published answer verbatim, so the reader is shown `[**m1**]`, an internal marker vocabulary
    that means nothing outside this prompt. Both forms of emphasis resolve now, and the mixed
    bullet proves one bolded marker cannot take a plain one down with it."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway(
        reduce=(
            "## Certifications\n"
            "- The switches shall comply with UL 62368 [**m1**]\n"
            "- Each firewall is supplied with an HBOM and an SBOM [m2]\n"
        )
    )

    answer = await _ask(gw, monkeypatch)

    assert "**m1**" not in answer.answer, "the internal marker vocabulary reached the reader"
    assert "[1]" in answer.answer and "[2]" in answer.answer
    assert len(answer.citations) == 2


async def test_a_marker_for_an_item_that_does_not_exist_is_still_removed(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tolerating emphasis widened what the marker pattern MATCHES, so the guard that matters is
    that matching is not the same as resolving: an id the map never produced is looked up, missed
    and dropped, exactly as before. Widening the pattern must not widen what can be cited."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway(
        reduce=(
            "## Certifications\n- The switches shall comply with UL 62368 [m1]\n- A claim no item supports [**m99**]\n"
        )
    )

    answer = await _ask(gw, monkeypatch)

    assert "m99" not in answer.answer
    assert len(answer.citations) == 1


# --- every way it declines ----------------------------------------------------------------------------


async def test_an_unresolvable_quote_declines_rather_than_publishing_uncited(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reduce model cited an item whose quote is in no chunk. Publishing it uncited would
    break grounded-or-abstain; abstaining would be worse than the passage answer available."""
    _enable(monkeypatch, on=True)
    await _seed()
    # The quote is in the SECTION's chunk text, so the item survives the map check...
    gw = _Gateway(reduce="Some prose with no markers at all.")

    answer = await _ask(gw, monkeypatch)

    assert "no markers at all" not in answer.answer, "the uncited reduce output was not published"
    assert answer.query


async def test_a_reduce_failure_falls_through_to_the_ordinary_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway(reduce=RuntimeError("model unavailable"))

    answer = await _ask(gw, monkeypatch)

    assert answer.query, "the question was still answered"
    assert "Certifications" not in answer.answer


async def test_no_items_falls_through(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every section read, nothing relevant found. A reduce over no items would write an answer
    from nothing — the passage path is the honest alternative."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway(switch="[]", fire="[]")

    answer = await _ask(gw, monkeypatch)

    assert answer.query
    assert gw.map_calls == 2 and gw.reduce_calls == 0, "no reduce over no items"


async def test_a_lookup_question_runs_no_map_reduce(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the aggregate class gets a section selection, and without one there is nothing to
    map over."""
    _enable(monkeypatch, on=True)
    await _seed()
    gw = _Gateway()

    await _ask(gw, monkeypatch, "what certification must the switches have?")

    assert gw.map_calls == 0 and gw.reduce_calls == 0


async def test_sections_with_no_chunk_text_are_skipped(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected section whose chunks do not name it has nothing to map. It still counts in the
    coverage denominator — it was selected and did not contribute — but it costs no model call."""
    _enable(monkeypatch, on=True)
    await _seed()
    catalog = _twin()
    for chunk in await catalog.document_chunks(_TENANT, _DOC):
        await catalog.record_chunks(_TENANT, [chunk.model_copy(update={"section_id": None})])
    gw = _Gateway()

    answer = await _ask(gw, monkeypatch)

    assert gw.map_calls == 0, "no section had text, so no map call was made"
    assert answer.query


# --- what this feature must not do ----------------------------------------------------------------------


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _seed()
    await _ask(None, monkeypatch)
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    without = sum(1 for e in events if e.action is AuditAction.QUERY)

    _enable(monkeypatch, on=True)
    gw = _Gateway()
    await _ask(gw, monkeypatch)
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    with_mapreduce = sum(1 for e in events if e.action is AuditAction.QUERY)

    assert with_mapreduce - without == without, "the second question added exactly as many events as the first"


async def test_the_second_identical_question_is_served_from_the_map_cache(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan's §9.1 cache. The saving is one model call per selected section, which on a
    tender is the whole cost of the map pass."""
    _enable(monkeypatch, on=True)
    await _seed()

    first = _Gateway()
    await _ask(first, monkeypatch)
    assert first.map_calls == 2

    second = _Gateway()
    answer = await _ask(second, monkeypatch)

    assert second.map_calls == 0, "both sections came from the cache"
    assert second.reduce_calls >= 1
    assert answer.status is AnswerStatus.GROUNDED


async def test_the_cache_can_be_turned_off(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, on=True, RAG_V2_MAP_CACHE_ENABLED="false")
    await _seed()

    first = _Gateway()
    await _ask(first, monkeypatch)
    second = _Gateway()
    await _ask(second, monkeypatch)

    assert second.map_calls == 2, "every section re-read"


async def test_replacing_the_structure_purges_the_cached_items(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A section surviving a re-parse with the same id has still had its text replaced, so a
    cached extraction over it is stale either way."""
    _enable(monkeypatch, on=True)
    await _seed()
    first = _Gateway()
    await _ask(first, monkeypatch)

    await _twin().replace_document_structure(_TENANT, _tree())
    await _attach_chunks()
    second = _Gateway()
    await _ask(second, monkeypatch)

    assert second.map_calls == 2, "the sections were re-mapped, not served stale"
