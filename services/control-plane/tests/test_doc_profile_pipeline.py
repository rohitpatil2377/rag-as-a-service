"""The document profile card through the real ingest pipeline (RAG v2 Phase 3).

`libs/rag-core/tests/test_doc_profile.py` pins the parser and states plainly what cannot be
verified without the AEGCL PDF. This file pins the WIRING, which is where the claims that have to
hold regardless of what any model answers live:

1. **Flag off ⇒ no profile call is made at all**, and chunking is byte-identical to Phase 5 —
   asserted by a gateway that records every call it receives, so "no card was stored" cannot be
   confused with "a card was built and discarded".
2. **The breadcrumbs in the prompt are MASKED.** They come off the structure tree, which is
   captured PRE-masking (`_persist_layout_structure` states the position), so unlike
   `extracted.text` they carry PII as the extractor read it — and this prompt's answer is stored
   in `documents.purpose` and embedded into every v2 vector in the document. This runs the REAL
   `RegexPIIMasker` through the real `DocIntelligenceService`, not a stand-in.
3. **A model failure costs the document its card and nothing else** — the fail-safe is the reason
   a profile call is allowed on the ingest path at all.
4. **The card reaches `embed_text` and not `text`**, which is the ADR-0007 seam this phase rides.
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
from rag_schemas.common import Modality, SourceKind

_TENANT = "profile3kb"
_PNG = b"\x89PNG\r\n fake image bytes of a scanned corrigendum page"

# A heading carrying a real identifier shape, so "the breadcrumb was masked before it reached the
# prompt" is a fact about this document rather than an assertion about the code path.
_PHONE = "98200-10005"
# Long enough that the head window's 500-character floor truncates BEFORE the financial clause,
# which is what makes the bounding test a measurement rather than a tautology.
_FILLER = (
    "Bidders are advised to read this corrigendum together with the original bidding document "
    "and all previously issued corrigenda, and to ensure that every undertaking required under "
    "the instructions to bidders is submitted in the prescribed format and signed by an "
    "authorised signatory whose authority is evidenced by a board resolution or a power of "
    "attorney enclosed with the technical bid, failing which the bid may be treated as "
    "non-responsive at the sole discretion of the evaluating authority."
)
_MARKUP = (
    "<h1>Corrigendum-V</h1>"
    "<p>This corrigendum amends Corrigendum-II dated 31.08.2026.</p>"
    f"<h2>Clarifications desk {_PHONE}</h2><p>Queries are received until 30.09.2026.</p>"
    f"<h2>General instructions</h2><p>{_FILLER}</p>"
    "<h2>Financial qualification</h2><p>The minimum average annual turnover is Rs. 19.44 Crores.</p>"
)
_TEXT = (
    "Corrigendum-V\n"
    "This corrigendum amends Corrigendum-II dated 31.08.2026.\n"
    f"Clarifications desk {_PHONE}\n"
    "General instructions\n"
    f"{_FILLER}\n"
    "Financial qualification\n"
    "The minimum average annual turnover is Rs. 19.44 Crores.\n"
)

_CARD = {
    "doc_type": "Corrigendum",
    "purpose": "Amends the bidding document.",
    "issuer": "AEGCL",
    "issue_date": "2026-09-24",
    "lang": "en",
    "relations": [{"type": "amends", "target_ref": "Corrigendum-II", "evidence_quote": "amends Corrigendum-II"}],
    "facet_taxonomy": [{"facet_id": "financial_qualification", "name": "Financial", "description": "turnover"}],
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


class _RecordingGateway:
    """A scripted gateway that keeps every prompt it was given — the only way to assert that the
    flag-off path makes NO call, and the only way to see what text actually reached the model."""

    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused by the profiler
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _use_gateway(monkeypatch: pytest.MonkeyPatch, reply: str | Exception) -> _RecordingGateway:
    gateway = _RecordingGateway(reply)

    async def _resolve(tenant_id: str, model: str | None = None) -> _RecordingGateway:
        return gateway

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)
    return gateway


def _use_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _for(tenant_id: str) -> DocIntelligenceService:
        # The default `RegexPIIMasker`, which is what makes the masking assertion real.
        return DocIntelligenceService(ocr_engine=_MarkupOCR())

    monkeypatch.setattr(wiring, "doc_intel_service_for", _for)


def _flag(monkeypatch: pytest.MonkeyPatch, *, on: bool) -> None:
    monkeypatch.setenv("RAG_V2_INGEST", "true" if on else "false")
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


# --- flag OFF: the call is not made ---------------------------------------------------------------


async def test_flag_off_makes_no_profile_call_and_stores_no_card(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("off"), actor="uploader")

    assert gateway.prompts == [], "the profile call must not be made with the flag off"
    assert await _twin().document_profile(_TENANT, "off") is None
    assert await _twin().document_relations(_TENANT, "off") == []


async def test_flag_off_leaves_embed_text_exactly_as_phase_five_left_it(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("offembed"), actor="uploader")

    assert all(c.embed_text is None for c in await _chunks("offembed"))


# --- flag ON: the card is built, stored and prefixed ----------------------------------------------


async def test_the_card_is_stored_with_its_facets_and_declared_relations(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("on"), actor="uploader")

    profile = await _twin().document_profile(_TENANT, "on")
    assert profile is not None
    assert profile.doc_type == "corrigendum"  # lowercased: it is a filter value
    assert profile.issuer == "AEGCL"
    assert [f.facet_id for f in profile.facets] == ["financial_qualification"]
    assert [r.target_ref for r in profile.relations] == ["Corrigendum-II"]


async def test_the_declared_relation_reaches_the_phase_six_graph_seeding_input(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 6 shipped this consumer COMPLETE but INERT because nothing wrote `doc_relations`
    (discovery §8.12). This ingest is its producer."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("seed"), actor="uploader")

    relations = await _twin().document_relations(_TENANT, "seed")
    assert [(r.rel_type, r.source_ref, r.target_ref) for r in relations] == [
        ("amends", "Corrigendum-V.pdf", "Corrigendum-II")
    ]


async def test_the_card_prefixes_embed_text_and_never_the_stored_text(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ADR-0007 seam: the catalog keeps what the document said, the index gets the context."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("prefix"), actor="uploader")

    chunks = await _chunks("prefix")
    assert chunks, "the document produced chunks"
    carded = [c for c in chunks if c.embed_text and c.embed_text.startswith("[corrigendum: ")]
    assert carded, "at least one unit carries the profile card"
    for chunk in carded:
        assert "Amends the bidding document." in (chunk.embed_text or "")
        assert not chunk.text.startswith("[corrigendum: "), "the citable text is untouched"


# --- the masking invariant ------------------------------------------------------------------------


async def test_a_heading_carrying_an_identifier_is_masked_before_it_reaches_the_prompt(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The breadcrumbs come off the PRE-masking structure tree. The prompt's answer is stored in
    `documents.purpose` and embedded into every v2 vector, so an identifier reaching it reaches
    all three."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("masked"), actor="uploader")

    assert gateway.prompts, "the profile call was made"
    prompt = gateway.prompts[0]
    assert _PHONE not in prompt, "the raw identifier must not reach the model"
    assert "Clarifications desk" in prompt, "the rest of the heading still did"


async def test_the_prompt_carries_the_document_outline_and_its_opening_text(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("inputs"), actor="uploader")

    prompt = gateway.prompts[0]
    assert "Financial qualification" in prompt  # a breadcrumb
    assert "Rs. 19.44 Crores" in prompt  # the opening text
    assert "Corrigendum-V.pdf" in prompt  # the platform's own title
    # The v1 heuristic doc_type is NOT handed to the model — this phase replaces it, and showing
    # it would anchor a better classifier to a worse one.
    assert "auto_doc_type" not in prompt


async def test_the_head_text_is_bounded_by_the_configured_window(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    monkeypatch.setenv("RAG_DOC_PROFILE_HEAD_CHARS", "500")
    get_settings.cache_clear()
    _use_engine(monkeypatch)
    gateway = _use_gateway(monkeypatch, json.dumps(_CARD))

    await pipeline.ingest_document(_doc("bounded"), actor="uploader")

    body = gateway.prompts[0].split("## Document text (opening pages)")[1].strip()
    assert 0 < len(body) <= 500
    # The bound BIT: the later part of the document did not reach the model, which is also what
    # narrows the window an evidence quote may legitimately come from.
    assert "Rs. 19.44 Crores" not in body
    # ...and what WAS sent is the document's opening, verbatim from the masked text layer — a
    # truncation, not a summary. (It differs from `_TEXT` by exactly the masking, which the test
    # above pins.)
    assert body.startswith("Corrigendum-V\nThis corrigendum amends Corrigendum-II dated 31.08.2026.")
    assert _FILLER[:40] in body, "the window reaches into the body before it stops"


# --- the fail-safe ---------------------------------------------------------------------------------


async def test_a_model_outage_costs_the_document_its_card_and_nothing_else(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, TimeoutError("gateway unreachable"))

    outcome = await pipeline.ingest_document(_doc("outage"), actor="uploader")

    assert outcome.chunk_count > 0, "the document still indexed"
    assert await _twin().document_profile(_TENANT, "outage") is None
    assert all(not (c.embed_text or "").startswith("[corrigendum") for c in await _chunks("outage"))


async def test_an_unusable_reply_leaves_the_document_indexed_without_a_card(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, "I am not able to analyse this document.")

    outcome = await pipeline.ingest_document(_doc("unusable"), actor="uploader")

    assert outcome.chunk_count > 0
    assert await _twin().document_profile(_TENANT, "unusable") is None


# --- reindex ----------------------------------------------------------------------------------------


async def test_a_reindex_re_profiles_rather_than_reusing_the_stored_card(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay re-runs extraction, so a card built from the previous outline may describe headings
    this text no longer has. It is also the only way a corpus ingested before the flag went on
    acquires a card at all."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))
    await pipeline.ingest_document(_doc("replay"), actor="uploader")

    revised = dict(_CARD, doc_type="circular", purpose="A revised reading.")
    gateway = _use_gateway(monkeypatch, json.dumps(revised))
    await pipeline.reindex_document(_TENANT, "replay", actor="operator")

    assert gateway.prompts, "the reindex made its own profile call"
    profile = await _twin().document_profile(_TENANT, "replay")
    assert profile is not None and profile.doc_type == "circular"


async def test_a_reindex_whose_profile_call_fails_keeps_the_card_it_already_had(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None is an ORDINARY outcome here, so purging a good card because this run's model call timed
    out would turn a transient model fault into data loss."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_gateway(monkeypatch, json.dumps(_CARD))
    await pipeline.ingest_document(_doc("keep"), actor="uploader")

    _use_gateway(monkeypatch, TimeoutError("gateway unreachable"))
    await pipeline.reindex_document(_TENANT, "keep", actor="operator")

    profile = await _twin().document_profile(_TENANT, "keep")
    assert profile is not None and profile.doc_type == "corrigendum"


# --- the facet vocabulary is pinned to the document revision ------------------------------------


def _card(*facets: str) -> str:
    """The same card with a different facet taxonomy — which is what a second Phase-3 call really
    returns. Measured on `corrigendum-5.pdf`: three ingests of identical bytes produced three
    10-facet vocabularies sharing ONE id."""
    return json.dumps({**_CARD, "facet_taxonomy": [{"facet_id": f, "name": f, "description": "d"} for f in facets]})


class _ScriptedGateway(_RecordingGateway):
    """One reply per call, so the first and second profile calls can disagree."""

    def __init__(self, *replies: str) -> None:
        super().__init__(replies[0])
        self._queue = list(replies)

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]


def _use_scripted(monkeypatch: pytest.MonkeyPatch, *replies: str) -> _ScriptedGateway:
    gateway = _ScriptedGateway(*replies)

    async def _resolve(tenant_id: str, model: str | None = None) -> _ScriptedGateway:
        return gateway

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)
    return gateway


async def test_reindexing_reuses_the_stored_vocabulary_instead_of_the_one_just_proposed(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ACCEPTANCE TEST: the same document revision produces the same facet ids.

    A facet id is Phase 8's facet-union join key AND, de-underscored, its cross-encoder probe
    text, so a re-profile that renames the vocabulary re-rolls that document's section selection.
    Here the second call proposes an entirely different taxonomy and the stored one wins."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    gateway = _use_scripted(
        monkeypatch,
        _card("financial_qualification", "eligibility"),
        _card("financial_capacity", "bidder_experience_requirements"),
    )

    await pipeline.ingest_document(_doc("pinned"), actor="uploader")
    first = await _twin().document_profile(_TENANT, "pinned")
    assert first is not None and [f.facet_id for f in first.facets] == ["financial_qualification", "eligibility"]

    await pipeline.reindex_document(_TENANT, "pinned", actor="operator")

    again = await _twin().document_profile(_TENANT, "pinned")
    assert again is not None
    assert [f.facet_id for f in again.facets] == ["financial_qualification", "eligibility"], (
        "the reindex adopted the newly proposed taxonomy and re-rolled the document's selection"
    )
    # Counted by SHAPE, because this gateway also serves Phase 4's enrichment calls and their
    # retries: 2 of the 6 prompts it saw are profile prompts.
    profile_calls = [p for p in gateway.prompts if "build its profile" in p]
    assert len(profile_calls) == 2, "the profile call still ran — this is reuse, not a skipped call"


async def test_changed_bytes_take_the_new_taxonomy_rather_than_inheriting_the_old(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An amended document must not keep facets that may no longer describe it. The revision is
    the content hash, so new bytes are a new revision and the fresh proposal is adopted."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch)
    _use_scripted(monkeypatch, _card("financial_qualification"), _card("new_requirement_class"))

    await pipeline.ingest_document(_doc("amended"), actor="uploader")
    changed = _doc("amended")
    object.__setattr__(changed, "content_b64", base64.b64encode(_PNG + b" amended").decode())
    await pipeline.ingest_document(changed, actor="uploader")

    profile = await _twin().document_profile(_TENANT, "amended")
    assert profile is not None
    assert [f.facet_id for f in profile.facets] == ["new_requirement_class"]
