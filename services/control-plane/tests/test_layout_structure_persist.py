"""RAG v2 Phase 2, slice B2: the parsed structure tree is PERSISTED into ``sections``/``blocks``.

Slice A stores the overlay; this is what reads it. These tests pin the pipeline wiring: the flag's
two states, both write paths (ingest and reindex), the replace-not-append contract a reindex
depends on, the plain-text document that still gets a tree, an overlay this build cannot parse not
taking down the ingest, and the erasure path emptying both tables.

The row shape itself is ``libs/clients/tests/test_layout_structure_rows.py``; the live Postgres
round trip is ``tests/integration/test_v2_structure_roundtrip.py``.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest
from svc_control_plane import pipeline, wiring
from svc_doc_intelligence.ocr import CAP_TABLES, CAP_TEXT, LAYOUT_OVERLAY_VERSION, Extraction, PageRecord
from svc_doc_intelligence.service import DocIntelligenceService

from rag_clients.factory import get_catalog, get_sparse_index, get_vector_store
from rag_clients.layout_rows import BlockRow, SectionRow
from rag_clients.memory import InMemoryCatalog, InMemorySparseIndex, InMemoryVectorStore
from rag_common import get_settings
from rag_common.tenancy import sparse_index, vector_collection
from rag_core import layout
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_TENANT = "structpipe"
_MARKUP = (
    "<h1>Night shift policy</h1><p>The night shift allowance is Rs.200 per day.</p>"
    "<table><tr><th>Grade</th><th>Allowance</th></tr><tr><td>M1</td><td>Rs.200</td></tr></table>"
)
_TEXT = "Night shift policy. The night shift allowance is Rs.200 per day, paid monthly."
_PNG = b"\x89PNG\r\n fake image bytes of a scanned policy page"


class _MarkupOCR:
    """Stands in for Chandra: an engine that answers in markup (no network, no GPU)."""

    def __init__(self, markup: str = _MARKUP, text: str = _TEXT) -> None:
        self._markup = markup
        self._text = text

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[
                PageRecord(
                    page_no=1,
                    text=self._text,
                    extractor="chandra",
                    capabilities=(CAP_TEXT, CAP_TABLES),
                    markup=self._markup,
                    markup_format="html",
                )
            ],
            modality=Modality.OCR,
        )


class _PlainOCR:
    """Every non-Chandra engine: plain text, no markup at all, so the overlay has no pages."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[PageRecord(page_no=1, text=self._text, extractor="tesseract", capabilities=(CAP_TEXT,))],
            modality=Modality.OCR,
        )


def _use_engine(monkeypatch: pytest.MonkeyPatch, engine: object) -> None:
    async def _service(tenant_id: str) -> DocIntelligenceService:
        return DocIntelligenceService(ocr_engine=engine)  # type: ignore[arg-type]

    monkeypatch.setattr(wiring, "doc_intel_service_for", _service)


def _flag(monkeypatch: pytest.MonkeyPatch, *, on: bool) -> None:
    monkeypatch.setenv("RAG_V2_INGEST", "true" if on else "false")
    get_settings.cache_clear()


def _doc(doc_id: str, raw: bytes = _PNG) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=_TENANT,
        doc_id=doc_id,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="image/png",
        content_b64=base64.b64encode(raw).decode(),
        ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _structure(doc_id: str) -> tuple[list[SectionRow], list[BlockRow]]:
    return _twin().document_structure(_TENANT, doc_id)


# --- the contract between the overlay writer and the parser --------------------------------


def test_the_writers_overlay_version_is_one_the_parser_accepts() -> None:
    """The one assertion that makes the structure tables reachable in production at all.

    An overlay the parser refuses is handled deliberately — the ingest survives and the document
    simply has no structure (see below) — so if the writer's own version were outside the parser's
    accepted set, that deliberate handling would be the ONLY path there is and the tables would
    never be written by anything. The two constants live in different packages and are bumped by
    different changes, which is exactly why their agreement is asserted rather than assumed."""
    assert LAYOUT_OVERLAY_VERSION in layout.SUPPORTED_SCHEMA_VERSIONS


# --- flag OFF ------------------------------------------------------------------------------


async def test_flag_off_writes_no_sections_and_no_blocks(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch, _MarkupOCR())

    outcome = await pipeline.ingest_document(_doc("off"), actor="uploader")

    assert outcome.outcome == "created"
    assert _structure("off") == ([], [])
    # The rest of the path is untouched — the structure tables are additive.
    assert await get_catalog().document_chunks(_TENANT, "off")


async def test_turning_the_flag_off_leaves_an_already_stored_tree_alone(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Purging on flag-off would destroy structure a re-index would have to rebuild, and the only
    consumers are gated on the same flag, so a stored tree is simply unread while it is off."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("kept"), actor="uploader")
    stored = _structure("kept")
    assert stored[0] and stored[1]

    _flag(monkeypatch, on=False)
    await pipeline.reindex_document(_TENANT, "kept", actor="operator")

    assert _structure("kept") == stored


# --- flag ON: ingest --------------------------------------------------------------------------


async def test_ingest_persists_the_markup_documents_tree(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())

    await pipeline.ingest_document(_doc("on"), actor="uploader")

    sections, blocks = _structure("on")
    assert [s.heading for s in sections] == ["Night shift policy"]
    assert sections[0].doc_id == "on" and sections[0].breadcrumb == "Night shift policy"
    assert [b.block_type for b in blocks] == ["heading", "paragraph", "table", "table_row"]
    assert {b.doc_id for b in blocks} == {"on"}
    assert {b.section_id for b in blocks} == {sections[0].section_id}
    table = next(b for b in blocks if b.block_type == "table")
    assert next(b for b in blocks if b.block_type == "table_row").parent_block_id == table.block_id
    assert table.html, "a table keeps the engine's own markup"


async def test_a_plain_text_document_gets_a_text_derived_tree_with_no_blocks(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the Chandra engines emit markup, so markup-only structure would leave most of a real
    corpus out of the tables. The text tree is persisted, and its total absence of blocks is what
    tells a reader which kind it is."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _PlainOCR("1.1 Night shift\nThe allowance is Rs.200.\n1.1.1 Eligibility\nAll staff.\n"))

    await pipeline.ingest_document(_doc("plain"), actor="uploader")

    sections, blocks = _structure("plain")
    assert blocks == [], "the text layer has no block stream"
    assert [s.heading for s in sections] == ["1.1 Night shift", "1.1.1 Eligibility"]
    assert sections[1].parent_section_id == sections[0].section_id
    assert all(s.page_start is None for s in sections)
    # Slice A writes no overlay object for this document, so the structure path cannot have been
    # reading one back from storage.
    assert _twin().layout_key(_TENANT, "plain") is None


async def test_a_failed_ingest_leaves_no_structure_behind(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tree is written AFTER indexing, so a failure never reaches it; the rollback's
    `delete_document` would purge it even if it had."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())

    class _BrokenIndexing:
        async def index_document(self, *a: object, **kw: object) -> object:
            raise RuntimeError("vector store unreachable")

    async def _indexing(tenant_id: str) -> _BrokenIndexing:
        return _BrokenIndexing()

    monkeypatch.setattr(wiring, "indexing_service_for", _indexing)

    with pytest.raises(RuntimeError):
        await pipeline.ingest_document(_doc("rollback"), actor="uploader")

    assert _structure("rollback") == ([], [])


# --- flag ON: reindex, and the replace-not-append contract -----------------------------------


async def test_a_reindex_does_not_double_the_tree(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the write path replaces instead of appending. Re-running the same extraction must
    leave the same rows, not two copies of them."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("replay"), actor="uploader")
    first = _structure("replay")

    await pipeline.reindex_document(_TENANT, "replay", actor="operator")
    await pipeline.reindex_document(_TENANT, "replay", actor="operator")

    assert _structure("replay") == first
    assert len(first[0]) == 1 and len(first[1]) == 4


async def test_a_reindex_replaces_a_tree_the_new_extraction_no_longer_produces(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chandra is a stochastic VLM, so a replay's headings and tables genuinely differ. Appending
    would leave the old sections describing text that is no longer served."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("revised"), actor="uploader")
    assert any(b.block_type == "table" for b in _structure("revised")[1])

    revised = "<h1>Night shift policy (revised)</h1><p>The allowance is now Rs.250 per day.</p>"
    _use_engine(monkeypatch, _MarkupOCR(markup=revised, text="The allowance is now Rs.250 per day."))
    await pipeline.reindex_document(_TENANT, "revised", actor="operator")

    sections, blocks = _structure("revised")
    assert [s.heading for s in sections] == ["Night shift policy (revised)"]
    assert [b.block_type for b in blocks] == ["heading", "paragraph"], "the old table is gone, not kept"


async def test_reindex_backfills_a_document_ingested_before_the_flag(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ingest write only ever covers new uploads, so reindex is the only way an
    already-ingested corpus acquires structure at all."""
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("backfill"), actor="uploader")
    assert _structure("backfill") == ([], [])

    _flag(monkeypatch, on=True)
    await pipeline.reindex_document(_TENANT, "backfill", actor="operator")

    sections, blocks = _structure("backfill")
    assert [s.heading for s in sections] == ["Night shift policy"]
    assert blocks


# --- an overlay this build cannot parse -------------------------------------------------------


async def test_an_unsupported_overlay_version_does_not_take_down_the_ingest(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored artifact from a future build is the parser's refusal to guess, not a bad document.
    The ingest completes, the chunks are indexed and served, and the document simply has no
    structure until a build that understands the overlay re-indexes it."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    monkeypatch.setattr(layout, "SUPPORTED_SCHEMA_VERSIONS", frozenset())  # nothing is parseable

    outcome = await pipeline.ingest_document(_doc("future"), actor="uploader")

    assert outcome.outcome == "created"
    assert await get_catalog().document_chunks(_TENANT, "future"), "the document is indexed and serving"
    assert _structure("future") == ([], [])


async def test_an_unsupported_overlay_version_leaves_the_previous_tree_intact(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping the replace is what makes this safe: the stored tree still matches the chunks
    write-then-swap left serving."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("survivor"), actor="uploader")
    stored = _structure("survivor")

    monkeypatch.setattr(layout, "SUPPORTED_SCHEMA_VERSIONS", frozenset())
    await pipeline.reindex_document(_TENANT, "survivor", actor="operator")

    assert _structure("survivor") == stored


# --- erasure ---------------------------------------------------------------------------------


async def test_delete_document_empties_both_tables_for_that_document(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`blocks.text` is the first place the structure tables hold real document text, so the DPDP
    erasure path (`compliance.erase` cascades through `delete_document`) has to reach it."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("doomed"), actor="uploader")
    await pipeline.ingest_document(_doc("spared", raw=_PNG + b" other"), actor="uploader")
    assert any("Rs.200" in b.text for b in _structure("doomed")[1]), "document text really is in blocks.text"

    await pipeline.delete_document(_TENANT, "doomed", actor="dpo")

    assert _structure("doomed") == ([], [])
    assert _structure("spared")[0], "another document's structure is untouched"


async def test_delete_document_empties_both_tables_even_after_the_flag_is_turned_off(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document ingested while the flag was ON must stay erasable after it is turned OFF."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("doomed"), actor="uploader")

    _flag(monkeypatch, on=False)
    await pipeline.delete_document(_TENANT, "doomed", actor="dpo")

    assert _structure("doomed") == ([], [])


# --- the catalog seam -------------------------------------------------------------------------


async def test_an_unknown_catalog_adapter_fails_closed(inmemory: None) -> None:
    """A backend that silently stored nothing is indistinguishable from a document with no
    structure, which is also what a legitimate plain-text document looks like."""
    from rag_clients import factory
    from rag_common.errors import ConfigError

    class _UnknownCatalog:
        """Neither Postgres nor the in-memory twin — a future backend."""

    def _unknown() -> _UnknownCatalog:
        return _UnknownCatalog()

    factory.get_catalog.cache_clear()
    original = factory.get_catalog
    factory.get_catalog = _unknown  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError) as ei:
            await factory.persist_document_structure(_TENANT, layout.LayoutTree("d", (), "markup"))
        assert "_UnknownCatalog" in str(ei.value), "the error must name the backend that was not handled"
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


# --- the PII asymmetry Phase 5 must not inherit blindly ----------------------------------------


async def test_block_text_is_unmasked_while_the_indexed_text_is_masked(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`blocks.text` holds PII as the extractor read it. The INDEX does not. Pinned, not assumed.

    `DocIntelligenceService.extract` captures `layout` from the UNMASKED engine output (it says so
    at the site) and masks only the rendered text, so a markup document's structure rows carry the
    original identifiers while everything reaching the chunker is de-identified. That is acceptable
    for these rows on the same grounds as the stored raw bytes — tenant-scoped, never embedded,
    nothing retrieves them, purged by the same `delete_document`.

    It is pinned here because PHASE 5 IS THE TRAP: emitting section/row/proposition chunks from a
    block's text is the obvious implementation and would put unmasked PII straight into the index,
    breaking the platform's "only de-identified text is chunked/embedded/indexed" constraint. Phase
    5 must chunk the MASKED text and use these rows only for structure. If that invariant is ever
    deliberately changed, this test is where the decision gets recorded."""
    _flag(monkeypatch, on=True)
    email = "blake@acme.example"
    markup = f"<h1>Contact</h1><p>Escalate to {email} within 24 hours.</p>"
    _use_engine(monkeypatch, _MarkupOCR(markup=markup, text=f"Escalate to {email} within 24 hours."))

    await pipeline.ingest_document(_doc("piidoc"), actor="uploader")

    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog)
    _sections, blocks = catalog.document_structure(_TENANT, "piidoc")
    block_text = " ".join(b.text for b in blocks)
    assert email in block_text, "blocks.text is captured pre-masking — if this changes, the ids move too"

    chunks = await catalog.document_chunks(_TENANT, "piidoc")
    indexed = " ".join(c.text for c in chunks)
    assert email not in indexed, "the INDEX must never carry the raw identifier"
    assert "REDACTED" in indexed or "@" not in indexed, "the indexed text must be de-identified"


# --- the rollback purge gap ----------------------------------------------------------------------


async def test_a_failure_after_indexing_purges_the_chunks_it_already_wrote(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window the other rollback test cannot see: it fails INSIDE `index_document`, so no chunks
    were ever written. Here indexing SUCCEEDS and a later step raises — `register_document`,
    `_persist_layout_structure` and `_record_duplicate_chunks` all run after the index write, and the
    structure write is a Postgres delete+insert under an advisory lock that can time out.

    The rollback deleted the catalog row and discarded the chunk ids it returned. The catalog row is
    the ONLY thing that can enumerate a document's chunks, so the vectors and BM25 documents stayed
    behind with nothing left to find them by — unreachable by `delete_document` and, which is why
    this matters, by DPDP right-to-erasure.
    """
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())

    async def _boom(*a: object, **kw: object) -> None:
        raise RuntimeError("advisory lock timeout")

    monkeypatch.setattr(pipeline, "_persist_layout_structure", _boom)

    with pytest.raises(RuntimeError):
        await pipeline.ingest_document(_doc("orphan"), actor="uploader")

    assert await get_catalog().list_documents(_TENANT) == [], "the catalog row is rolled back"
    # ...and nothing derived from it survives the row that could have enumerated it.
    vectors = get_vector_store()
    sparse = get_sparse_index()
    assert isinstance(vectors, InMemoryVectorStore)
    assert isinstance(sparse, InMemorySparseIndex)
    assert vectors._c[vector_collection(_TENANT)] == {}, "orphaned vectors: unreachable by erasure"
    assert sparse._i[sparse_index(_TENANT)] == {}, "orphaned BM25 docs: unreachable by erasure"


async def test_a_derived_write_failure_on_reindex_still_reaches_the_swap(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling gap. `reindex_document` had no try/except, and the derived-artifact writes sit
    between `register_document` and the swap — so a structure-write timeout skipped the swap, the
    INDEX audit event, the semantic-cache invalidation and the community rebuild. The document then
    answered from BOTH chunk generations at once (the old ids were never purged), with nothing
    recording that a reindex had happened. A stale derived artifact is detectable (the overlay
    carries the raw object's content_hash); two live generations are not.
    """
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("swapme"), actor="uploader")
    before = {c.chunk_id for c in await get_catalog().document_chunks(_TENANT, "swapme")}
    assert before

    async def _boom(*a: object, **kw: object) -> None:
        raise RuntimeError("advisory lock timeout")

    monkeypatch.setattr(pipeline, "_persist_layout_structure", _boom)
    # Re-extract to DIFFERENT text, so the new chunk ids cannot coincide with the old ones and the
    # swap has something real to purge.
    _use_engine(monkeypatch, _MarkupOCR(markup="<h1>Revised</h1><p>The allowance is now Rs.400.</p>"))

    await pipeline.reindex_document(_TENANT, "swapme", actor="admin")

    after = {c.chunk_id for c in await get_catalog().document_chunks(_TENANT, "swapme")}
    assert after, "the new version is serving"
    assert not (before & after), "the OLD generation must not still be in the catalog"
