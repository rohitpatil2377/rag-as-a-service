"""Section enrichment through the real ingest pipeline (RAG v2 Phase 4).

`libs/rag-core/tests/test_section_enrich.py` pins the parser and states plainly what cannot be
verified without the AEGCL PDF. This file pins the WIRING — the claims that have to hold whatever
any model answers:

1. **Flag off ⇒ no enrichment call is made at all**, and the index is byte-identical to Phase 3.
   The same holds for `section_enrich_enabled=false`, which exists precisely so an operator can
   decline one-call-per-section without giving up v2 chunking and the profile card.
2. **The section text and breadcrumb in the prompt are MASKED.** They come off the structure
   tree, which is captured PRE-masking, and what comes back is written to `sections.summary` and
   indexed as `Chunk.text` at two levels. This runs the REAL `RegexPIIMasker`.
3. **A model failure costs that section and nothing else** — at one call per section, a document
   whose ingest depended on every call succeeding would fail far more often than Phase 3's.
4. **The summaries and propositions reach the index as chunks**, and the enrichment reaches
   `sections` — written AFTER the structure tree, which replaces the very rows it updates.
5. **A section-level `amends` becomes a declared relation**, which is the second half of Phase
   6's graph seeding and the half that had no producer at all until now.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline, wiring
from svc_doc_intelligence.ocr import CAP_TABLES, CAP_TEXT, Extraction, PageRecord
from svc_doc_intelligence.service import DocIntelligenceService

from rag_clients.factory import get_catalog
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkLevel, Modality, SourceKind

_TENANT = "enrich4kb"
_PNG = b"\x89PNG\r\n fake image bytes of a scanned corrigendum page"

# A real identifier shape INSIDE a section's body, so "the section text was masked before it
# reached the prompt" is a fact about this document rather than a claim about the code path.
_PHONE = "98200-10005"
_MAAT = (
    "The bidder shall have a minimum average annual turnover of Rs. 19.44 Crores over the last "
    "three financial years. This supersedes the figure of Rs. 35 Crores stated in Corrigendum-II. "
    f"Queries on this clause may be directed to {_PHONE} before the bid due date."
)
_SCOPE = (
    "This corrigendum amends Corrigendum-II dated 31.08.2026 and must be read together with the "
    "original bidding document and every corrigendum issued before it. Bidders are advised to "
    "submit all undertakings in the prescribed format."
)
_MARKUP = (
    "<h1>Corrigendum-V</h1>"
    f"<h2>1. Scope</h2><p>{_SCOPE}</p>"
    f"<h2>4.2 Financial qualification</h2><p>{_MAAT}</p>"
    "<h2>9. Annexures</h2><p>See attached.</p>"
)
_TEXT = f"Corrigendum-V\n1. Scope\n{_SCOPE}\n4.2 Financial qualification\n{_MAAT}\n9. Annexures\nSee attached.\n"

_CARD = {
    "doc_type": "corrigendum",
    "purpose": "Amends the bidding document.",
    "issuer": "AEGCL",
    "facet_taxonomy": [{"facet_id": "financial_qualification", "name": "Financial", "description": "turnover"}],
}
_SUMMARY = "Sets the minimum average annual turnover and supersedes the earlier figure."
_ENRICHMENT = {
    "summary": _SUMMARY,
    "facet_ids": ["financial_qualification"],
    "key_values": [
        {
            "name": "average annual turnover",
            "value": "19.44",
            "unit": "Rs. Crores",
            "quote": "minimum average annual turnover of Rs. 19.44 Crores",
        }
    ],
    "amends": {
        "target_ref": "Corrigendum-II",
        "old": "Rs. 35 Crores",
        "new": "Rs. 19.44 Crores",
        "quote": "supersedes the figure of Rs. 35 Crores stated in Corrigendum-II",
    },
    "propositions": [
        {
            "text": "The bidder must have an average annual turnover of Rs. 19.44 Crores.",
            "quote": "The bidder shall have a minimum average annual turnover",
        }
    ],
}


class _MarkupOCR:
    """Stands in for Chandra: an engine that answers in markup (no network, no GPU)."""

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[
                PageRecord(
                    page_no=1,
                    text=_TEXT,
                    extractor="chandra",
                    capabilities=(CAP_TEXT, CAP_TABLES),
                    markup=_MARKUP,
                    markup_format="html",
                )
            ],
            modality=Modality.OCR,
        )


class _RoutingGateway:
    """Answers the profile prompt with a card and every section prompt with an enrichment.

    Routed on the prompt's own opening line rather than on call order, because the section calls
    are issued concurrently and their order is not a contract.
    """

    def __init__(self, *, enrichment: object = _ENRICHMENT, fail_sections: bool = False) -> None:
        self._enrichment = enrichment
        self._fail = fail_sections
        self.profile_prompts: list[str] = []
        self.section_prompts: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        if prompt.startswith("You are enriching ONE section"):
            self.section_prompts.append(prompt)
            if self._fail:
                raise RuntimeError("model unavailable")
            return json.dumps(self._enrichment)
        self.profile_prompts.append(prompt)
        return json.dumps(_CARD)

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused here
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _use_gateway(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _RoutingGateway:
    gateway = _RoutingGateway(**kwargs)

    async def _resolve(tenant_id: str, model: str | None = None) -> _RoutingGateway:
        return gateway

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)
    return gateway


def _use_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _for(tenant_id: str) -> DocIntelligenceService:
        # The default `RegexPIIMasker`, which is what makes the masking assertion real.
        return DocIntelligenceService(ocr_engine=_MarkupOCR())

    monkeypatch.setattr(wiring, "doc_intel_service_for", _for)


def _flag(monkeypatch: pytest.MonkeyPatch, *, on: bool = True, enrich: bool = True, **env: str) -> None:
    monkeypatch.setenv("RAG_V2_INGEST", "true" if on else "false")
    monkeypatch.setenv("RAG_SECTION_ENRICH_ENABLED", "true" if enrich else "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _doc(doc_id: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=_TENANT,
        doc_id=doc_id,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="image/png",
        content_b64=base64.b64encode(_PNG).decode(),
        metadata={"filename": "Corrigendum-V.pdf"},
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


async def _chunks(doc_id: str) -> list[Chunk]:
    return await get_catalog().document_chunks(_TENANT, doc_id)


# --- the switches ---------------------------------------------------------------------------------


async def test_flag_off_makes_no_enrichment_call(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("off"), actor="uploader")

    assert gateway.section_prompts == [], "no enrichment call with the v2 flag off"
    assert await _twin().section_enrichment(_TENANT, "off") == {}
    assert [c for c in await _chunks("off") if c.level in (ChunkLevel.SECTION, ChunkLevel.PROPOSITION)] == []


async def test_enrichment_can_be_declined_without_giving_up_the_rest_of_v2(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call per SECTION is a different order of spend from Phase 3's one per document, which
    is the whole reason this has its own switch."""
    _flag(monkeypatch, on=True, enrich=False)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("noenrich"), actor="uploader")

    assert gateway.section_prompts == []
    assert gateway.profile_prompts, "...but the Phase-3 card is still built"
    assert await _twin().document_profile(_TENANT, "noenrich") is not None
    assert [c for c in await _chunks("noenrich") if c.level is ChunkLevel.SECTION] == []


# --- what reaches the prompt -------------------------------------------------------------------------


async def test_the_section_text_and_breadcrumb_reaching_the_prompt_are_masked(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structure tree is captured PRE-masking, so a section body carries PII exactly as the
    extractor read it. What comes back is stored in `sections.summary` and indexed as chunk text
    at two levels, so an identifier that reaches this prompt reaches all three."""
    _flag(monkeypatch, RAG_SECTION_ENRICH_MIN_CHARS="20")
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("masked"), actor="uploader")

    assert gateway.section_prompts, "sections were offered for enrichment"
    assert _PHONE in _TEXT, "the fixture really does carry an identifier"
    for prompt in gateway.section_prompts:
        assert _PHONE not in prompt


async def test_the_prompt_carries_the_document_card_and_its_facet_vocabulary(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, RAG_SECTION_ENRICH_MIN_CHARS="20")
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("carded"), actor="uploader")

    prompt = gateway.section_prompts[0]
    assert "corrigendum" in prompt and "- financial_qualification" in prompt


async def test_a_short_section_is_not_worth_a_call(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the default floor only the two substantial clauses are offered; "9. Annexures / See
    attached." carries nothing for a summary to summarise that its own chunk does not already
    hold verbatim."""
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("short"), actor="uploader")

    assert len(gateway.section_prompts) == 2
    assert all("See attached." not in p.split("## Section text")[-1] for p in gateway.section_prompts)


async def test_the_section_cap_bounds_the_calls(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _flag(monkeypatch, RAG_SECTION_ENRICH_MIN_CHARS="20", RAG_SECTION_ENRICH_MAX_SECTIONS="1")
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)

    await pipeline.ingest_document(_doc("capped"), actor="uploader")

    assert len(gateway.section_prompts) == 1


# --- the fail-safe -----------------------------------------------------------------------------------


async def test_a_model_failure_costs_the_enrichment_and_not_the_document(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, fail_sections=True)

    outcome = await pipeline.ingest_document(_doc("broken"), actor="uploader")

    assert outcome.outcome in ("created", "updated")
    assert await _twin().section_enrichment(_TENANT, "broken") == {}
    chunks = await _chunks("broken")
    assert chunks, "the document is indexed"
    assert [c for c in chunks if c.level in (ChunkLevel.SECTION, ChunkLevel.PROPOSITION)] == []
    assert await _twin().document_profile(_TENANT, "broken") is not None, "...and keeps its card"


async def test_an_unusable_reply_leaves_the_document_indexed(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, enrichment={"not": "an enrichment"})

    await pipeline.ingest_document(_doc("unusable"), actor="uploader")

    assert await _twin().section_enrichment(_TENANT, "unusable") == {}
    assert await _chunks("unusable")


# --- what is stored and what is indexed ----------------------------------------------------------------


async def test_the_enrichment_is_written_onto_the_sections_rows(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AFTER `_persist_layout_structure`, which DELETEs and re-INSERTs the very rows this
    updates — written before it, every summary would be erased with nothing saying so."""
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch)
    await pipeline.ingest_document(_doc("stored"), actor="uploader")

    stored = await _twin().section_enrichment(_TENANT, "stored")
    assert len(stored) == 2, "both substantial clauses survived the structure replace"
    assert all(e.summary == _SUMMARY for e in stored.values())
    assert all(e.facet_ids == ("financial_qualification",) for e in stored.values())
    # The gateway answered BOTH sections with the MAAT clause's quotes, and only the section whose
    # own text contains them keeps the quoted items. That is the plan's validation rule holding
    # end to end rather than only in the parser's own unit tests.
    quoted = [e for e in stored.values() if e.key_values]
    assert len(quoted) == 1
    assert [kv.value for kv in quoted[0].key_values] == ["19.44"]
    assert quoted[0].amends is not None and quoted[0].amends.old == "Rs. 35 Crores"
    assert [e.amends for e in stored.values() if e.amends is not None] == [quoted[0].amends]


async def test_the_summary_and_the_propositions_reach_the_index(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch)
    await pipeline.ingest_document(_doc("indexed"), actor="uploader")

    chunks = await _chunks("indexed")
    summaries = [c for c in chunks if c.level is ChunkLevel.SECTION]
    propositions = [c for c in chunks if c.level is ChunkLevel.PROPOSITION]
    assert summaries and all(c.text == _SUMMARY for c in summaries)
    assert propositions and all("19.44 Crores" in c.text for c in propositions)
    # The Phase-3 card still prefixes what is EMBEDDED, and `text` is still what a citation shows.
    assert all((c.embed_text or "").startswith("[corrigendum:") for c in summaries)


async def test_a_section_level_amends_becomes_a_declared_relation(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second half of Phase 6's graph seeding. `doc_relations` got a producer in Phase 3;
    `sections.amends` had none at all until this phase, and `document_relations` said so."""
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch)
    await pipeline.ingest_document(_doc("amends"), actor="uploader")

    relations = await _twin().document_relations(_TENANT, "amends")
    section_level = [r for r in relations if r.source_ref not in ("Corrigendum-V.pdf", "amends")]
    assert section_level, "a section-level amendment reached the seeding input"
    assert all(r.rel_type == "amends" and r.target_ref == "Corrigendum-II" for r in section_level)
    assert all("supersedes" in r.evidence_quote for r in section_level)


async def test_a_reindex_re_enriches_rather_than_reusing_what_is_stored(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay re-runs extraction, so section ids and their text can differ from the run that
    produced the stored summaries. It is also the only way a corpus ingested before the flag went
    on acquires summaries at all."""
    _flag(monkeypatch)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch)
    await pipeline.ingest_document(_doc("reindexed"), actor="uploader")
    first = len(gateway.section_prompts)

    await pipeline.reindex_document(_TENANT, "reindexed", actor="ops")

    assert len(gateway.section_prompts) == first * 2
    assert await _twin().section_enrichment(_TENANT, "reindexed")
    assert [c for c in await _chunks("reindexed") if c.level is ChunkLevel.SECTION]
