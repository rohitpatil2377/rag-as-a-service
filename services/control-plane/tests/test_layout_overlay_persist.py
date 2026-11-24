"""RAG v2 Phase 2, slice A: the layout overlay is PERSISTED, and purged wherever the raw bytes are.

The overlay is an extraction artifact in the same class as the raw object (see
`pipeline._persist_layout_overlay` for the compliance position): captured before PII masking,
retained for re-extraction and provenance, never chunked/embedded/indexed, and purged by the same
paths. These tests pin all four halves of that: the flag's two states, the key, the catalog
column, and both purge paths.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

import pytest
from svc_control_plane import pipeline, wiring
from svc_doc_intelligence.ocr import CAP_TABLES, CAP_TEXT, Extraction, PageRecord
from svc_doc_intelligence.service import DocIntelligenceService

from rag_clients.factory import get_catalog, get_object_store
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, layout_object_key, raw_object_key
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_TENANT = "layoutkb"
_MARKUP = "<h1>Night shift policy</h1><table><tr><td>Allowance</td><td>Rs.200</td></tr></table>"
_OCR_TEXT = "Night shift policy. The night shift allowance is Rs.200 per day, paid monthly."
_PNG = b"\x89PNG\r\n fake image bytes of a scanned policy page"


class _MarkupOCR:
    """Stands in for Chandra: an engine that answers in markup (no network, no GPU).

    ``markup``/``text`` are overridable so a re-extraction can legitimately return something
    different from the first one — which is what makes a stale overlay observable."""

    def __init__(self, markup: str = _MARKUP, text: str = _OCR_TEXT) -> None:
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
    """Every non-Chandra engine: plain text, no markup at all."""

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[PageRecord(page_no=1, text=_OCR_TEXT, extractor="tesseract", capabilities=(CAP_TEXT,))],
            modality=Modality.OCR,
        )


def _use_engine(monkeypatch: pytest.MonkeyPatch, engine: object) -> None:
    """Route this tenant's extraction through `engine`. A bare DocIntelligenceService (no image
    describer) so the test exercises OCR only, with no vision caption in the way."""

    async def _service(tenant_id: str) -> DocIntelligenceService:
        return DocIntelligenceService(ocr_engine=engine)  # type: ignore[arg-type]

    monkeypatch.setattr(wiring, "doc_intel_service_for", _service)


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
        ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


async def _chunk_texts(doc_id: str) -> list[str]:
    return [c.text for c in await get_catalog().document_chunks(_TENANT, doc_id)]


# --- flag OFF: nothing written, nothing recorded, nothing changed ----------------------------


async def test_flag_off_writes_no_object_and_leaves_the_column_null(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch, _MarkupOCR())

    outcome = await pipeline.ingest_document(_doc("off"), actor="uploader")
    assert outcome.outcome == "created"

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("off"))
    assert _twin().layout_key(_TENANT, "off") is None
    # The rest of the path is untouched: the raw object and the chunks are exactly as before.
    assert await get_object_store().get(_TENANT, raw_object_key("off")) == _PNG
    assert await _chunk_texts("off")


async def test_the_overlay_write_itself_still_changes_nothing_about_the_indexed_text(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SUPERSEDED BY PHASE 5, and narrowed rather than deleted.

    This test used to assert that the same document ingested with the flag off and on produced
    byte-identical chunks, because slice A only STORED the overlay and nothing re-derived anything
    from it. Phase 5 deliberately ends that: with the flag on, a document that has a block stream
    is chunked FROM its structure tree, so the chunks genuinely differ — that is the phase's whole
    point, and the byte-identity that still matters (flag OFF versus today) is pinned in
    ``test_layout_chunking_pipeline.py``.

    What survives here is the narrower claim this file is actually about: the overlay WRITE is
    additive. A document whose extractor emits no markup gets no overlay and no chunkable tree, so
    for that document the flag still changes nothing about the indexed text."""
    _use_engine(monkeypatch, _PlainOCR())

    _flag(monkeypatch, on=False)
    await pipeline.ingest_document(_doc("off"), actor="uploader")
    off = await _chunk_texts("off")

    _flag(monkeypatch, on=True)
    await pipeline.ingest_document(_doc("on"), actor="uploader")
    on = await _chunk_texts("on")

    assert off == on != []


# --- flag ON: the artifact, its key, and the catalog pointer --------------------------------


async def test_flag_on_stores_the_overlay_at_the_expected_key_and_records_it(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())

    await pipeline.ingest_document(_doc("on"), actor="uploader")

    key = layout_object_key("on")
    payload = json.loads(await get_object_store().get(_TENANT, key))
    assert payload["schema_version"] == 2, "a stored artifact must say which shape it is"
    assert payload["extractor"] == "chandra"
    assert payload["pages"] == [{"page_no": 1, "markup": _MARKUP, "markup_format": "html", "degradations": []}]
    # `documents.minio_layout_key` — the Phase 1 column — now points at it.
    assert _twin().layout_key(_TENANT, "on") == key


async def test_flag_on_with_a_plain_text_engine_writes_nothing(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """No markup means no artifact. An empty object would assert "captured, and it was empty"
    about a document that was never captured with markup at all."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _PlainOCR())

    await pipeline.ingest_document(_doc("plain"), actor="uploader")

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("plain"))
    assert _twin().layout_key(_TENANT, "plain") is None


async def test_a_crawl_style_doc_id_stores_the_overlay_under_a_legal_key(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crawl doc id is a URL. The overlay must land under the SAME safe-key encoding the raw
    bytes use, or a crawl document's artifacts diverge and one of them is unreachable."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    doc_id = "http://host/path"

    await pipeline.ingest_document(_doc(doc_id), actor="crawler")

    key = layout_object_key(doc_id)
    assert key == "layout/" + raw_object_key(doc_id).removeprefix("raw/")
    assert ":" not in key.removeprefix("layout/")
    assert await get_object_store().get(_TENANT, key)
    assert _twin().layout_key(_TENANT, doc_id) == key


# --- purge: every path that removes the raw object removes this one ------------------------


async def test_delete_document_purges_the_overlay(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """This is the function DPDP right-to-erasure cascades through (`compliance.erase`) and the
    one `delete_knowledge_base` calls per document, so the overlay is covered by both."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("doomed"), actor="uploader")
    assert await get_object_store().get(_TENANT, layout_object_key("doomed"))

    await pipeline.delete_document(_TENANT, "doomed", actor="dpo")

    for key in (raw_object_key("doomed"), layout_object_key("doomed")):
        with pytest.raises(KeyError):
            await get_object_store().get(_TENANT, key)
    assert _twin().layout_key(_TENANT, "doomed") is None


async def test_delete_document_purges_the_overlay_even_after_the_flag_is_turned_off(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The purge must not be gated on the flag: a document ingested while it was ON has to stay
    erasable after it is turned OFF, or the platform cannot honour an erasure request."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("doomed"), actor="uploader")

    _flag(monkeypatch, on=False)
    await pipeline.delete_document(_TENANT, "doomed", actor="dpo")

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("doomed"))


async def test_a_failed_ingest_does_not_orphan_the_overlay(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rollback case. The overlay is written before indexing, so an indexing failure on a NEW
    document would leave an artifact whose catalog row has just been deleted — an orphan no purge
    path enumerates, because every purge path works from the catalog."""
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

    for key in (raw_object_key("rollback"), layout_object_key("rollback")):
        with pytest.raises(KeyError):
            await get_object_store().get(_TENANT, key)
    assert await get_catalog().list_documents(_TENANT) == []


# --- the catalog seam ----------------------------------------------------------------------


async def test_recording_the_key_for_an_unknown_document_reports_false(inmemory: None) -> None:
    """So the caller can tell "recorded" from "silently did nothing" — a stored object with no
    catalog row pointing at it is the orphan case above."""
    from rag_clients import factory

    assert await factory.record_layout_key(_TENANT, "never-ingested", "layout/x") is False


async def test_an_unknown_catalog_adapter_fails_closed(inmemory: None) -> None:
    """An object has already been WRITTEN by the time this runs. A backend this seam does not
    know how to record into must stop the ingest, not leave the artifact unreferenced."""
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
            await factory.record_layout_key(_TENANT, "d", "layout/d")
        assert "_UnknownCatalog" in str(ei.value), "the error must name the backend that was not handled"
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


# --- reindex is the backfill path -------------------------------------------------------------


async def test_reindex_backfills_the_overlay_for_a_document_ingested_before_the_flag(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is on the ingest path, so without this a document ingested while the flag was off
    could NEVER acquire an overlay — and an already-ingested corpus is exactly what the structure
    tree has to run against. Reindex is the only backfill mechanism there is."""
    _flag(monkeypatch, on=False)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("backfill"), actor="uploader")
    assert _twin().layout_key(_TENANT, "backfill") is None, "nothing stored while the flag was off"

    _flag(monkeypatch, on=True)
    await pipeline.reindex_document(_TENANT, "backfill", actor="operator")

    key = layout_object_key("backfill")
    assert _twin().layout_key(_TENANT, "backfill") == key
    payload = json.loads(await get_object_store().get(_TENANT, key))
    assert payload["pages"] == [{"page_no": 1, "markup": _MARKUP, "markup_format": "html", "degradations": []}]
    # A backfilled overlay is bound to the bytes it was re-extracted from, like any other.
    assert payload["content_hash"] == hashlib.sha256(_PNG).hexdigest()


async def test_reindex_refreshes_a_stale_overlay(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay re-runs extraction, so an overlay from the previous run describes structure that no
    longer matches this text — the same argument the provenance stamp beside it already makes."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("stale"), actor="uploader")

    revised = "<h1>Night shift policy (revised)</h1><p>The allowance is now Rs.250 per day.</p>"
    _use_engine(monkeypatch, _MarkupOCR(markup=revised))
    await pipeline.reindex_document(_TENANT, "stale", actor="operator")

    payload = json.loads(await get_object_store().get(_TENANT, layout_object_key("stale")))
    assert payload["pages"][0]["markup"] == revised, "the overlay still describes the old extraction"


# --- the overlay names the bytes it describes ------------------------------------------------


async def test_the_stored_overlay_is_bound_to_the_stored_raw_object(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness check itself, run exactly as a consumer would: re-hash the raw object and
    compare. Without this, overlay staleness is only an argument about write paths; with it, it
    is a fact a reader can establish — and a reader can skip re-parsing when it matches."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("bound"), actor="uploader")

    payload = json.loads(await get_object_store().get(_TENANT, layout_object_key("bound")))
    raw = await get_object_store().get(_TENANT, raw_object_key("bound"))
    assert payload["content_hash"] == hashlib.sha256(raw).hexdigest()


async def test_replacing_the_bytes_rebinds_the_overlay(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """An update re-extracts, so the overlay must name the NEW bytes. If it kept the old hash the
    staleness check above would report a stale overlay that is in fact current — a false alarm is
    as useless as no alarm."""
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("rebound"), actor="uploader")
    first = json.loads(await get_object_store().get(_TENANT, layout_object_key("rebound")))["content_hash"]

    revised_bytes = _PNG + b" revised"
    doc = _doc("rebound").model_copy(update={"content_b64": base64.b64encode(revised_bytes).decode()})
    await pipeline.ingest_document(doc, actor="uploader")

    payload = json.loads(await get_object_store().get(_TENANT, layout_object_key("rebound")))
    assert payload["content_hash"] == hashlib.sha256(revised_bytes).hexdigest() != first


# --- reindex writes the overlay only once the new version is durable ------------------------


async def test_a_failed_reindex_leaves_the_overlay_describing_the_text_still_being_served(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reindex_document` has NO try/except, and write-then-swap leaves the OLD chunks serving if
    indexing raises. So the overlay must be written AFTER the new version is durable: written
    before, it would describe a re-extraction that never reached the index, while the structure
    tree joins it to the old chunk text. Chandra is a stochastic VLM, so the two genuinely differ.
    """
    _flag(monkeypatch, on=True)
    _use_engine(monkeypatch, _MarkupOCR())
    await pipeline.ingest_document(_doc("survivor"), actor="uploader")
    served = await _chunk_texts("survivor")

    class _BrokenIndexing:
        async def index_document(self, *a: object, **kw: object) -> object:
            raise RuntimeError("vector store unreachable")

    async def _indexing(tenant_id: str) -> _BrokenIndexing:
        return _BrokenIndexing()

    revised = "<h1>Night shift policy (revised)</h1><p>The allowance is now Rs.250 per day.</p>"
    _use_engine(monkeypatch, _MarkupOCR(markup=revised, text="The allowance is now Rs.250 per day."))
    monkeypatch.setattr(wiring, "indexing_service_for", _indexing)

    with pytest.raises(RuntimeError):
        await pipeline.reindex_document(_TENANT, "survivor", actor="operator")

    payload = json.loads(await get_object_store().get(_TENANT, layout_object_key("survivor")))
    assert payload["pages"][0]["markup"] == _MARKUP, "the overlay ran ahead of the index"
    assert await _chunk_texts("survivor") == served, "write-then-swap kept the old version serving"


# --- the BORN-DIGITAL path reaches the same persist with no change to it (Phase 2c-1) -------


def _pdf_fixture() -> bytes:
    """A two-page born-digital PDF with a ruled table — the shape this plan was written about.

    Built from the doc-intelligence test fixture rather than a second PDF writer; same
    sys.path insert as `tests/e2e/test_doc_pipeline.py` uses for the OOXML fixtures."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "doc-intelligence" / "tests"))
    from pdf_fixtures import make_pdf

    return make_pdf(
        [["SECTION 1 SCOPE OF WORK", "The tenderer shall supply the items below."], ["Delivery within sixty days."]],
        tables={1: [["Item", "Qty"], ["LED panel", "12"]]},
        table_after_line={1: 2},
        styles={1: {0: ("F2", 18.0)}},
    )


def _real_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extraction built by `wiring.build_doc_intel_service` — i.e. with the layout capture gated
    by the REAL flag, not by a value the test passed in. This is what proves the gate is wired."""

    async def _service(tenant_id: str) -> DocIntelligenceService:
        return wiring.build_doc_intel_service(_PlainOCR())

    monkeypatch.setattr(wiring, "doc_intel_service_for", _service)


def _pdf_doc(doc_id: str) -> DocumentEnvelope:
    return _doc(doc_id).model_copy(
        update={"mime": "application/pdf", "content_b64": base64.b64encode(_pdf_fixture()).decode()}
    )


async def test_a_born_digital_pdf_now_persists_a_layout_overlay(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement this slice exists for: `application/pdf` is `DocKind.NATIVE_PDF` and never
    reaches Chandra, so before this it produced no markup and no overlay was ever written for the
    tender/policy PDFs the plan is about. `_persist_layout_overlay` needed NO change — it stores
    whatever pages the overlay has, under the same key, bound to the same content hash."""
    _flag(monkeypatch, on=True)
    _real_wiring(monkeypatch)

    await pipeline.ingest_document(_pdf_doc("tender"), actor="uploader")

    payload = json.loads(await get_object_store().get(_TENANT, layout_object_key("tender")))
    assert payload["extractor"] == "pdfplumber"
    assert [p["page_no"] for p in payload["pages"]] == [1, 2]
    assert {p["markup_format"] for p in payload["pages"]} == {"layout-json"}
    raw = await get_object_store().get(_TENANT, raw_object_key("tender"))
    assert payload["content_hash"] == hashlib.sha256(raw).hexdigest()
    assert _twin().layout_key(_TENANT, "tender") == layout_object_key("tender")
    # The blocks carry what only this path can: coordinates and per-line typography.
    heading = json.loads(payload["pages"][0]["markup"])["blocks"][0]
    assert heading["bbox"] and heading["font_size"] == 18.0 and heading["bold"] is True


async def test_the_flag_off_writes_no_overlay_for_a_born_digital_pdf_either(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the gate is on the CAPTURE, not just the persist: with the flag off the layout pass
    does not run at all (pinned at the extractor in
    `doc-intelligence/tests/test_layout_overlay_native_pdf.py`)."""
    _flag(monkeypatch, on=False)
    _real_wiring(monkeypatch)

    await pipeline.ingest_document(_pdf_doc("tender-off"), actor="uploader")

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("tender-off"))
    assert _twin().layout_key(_TENANT, "tender-off") is None
    assert await _chunk_texts("tender-off"), "the document still ingested normally"


async def test_deleting_the_document_purges_the_born_digital_overlay_too(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified rather than assumed: the native overlay is written by the same helper at the same
    `layout_object_key`, so `delete_document` — which DPDP erasure and KB delete both cascade
    through — purges it unconditionally, exactly as it does the Chandra one and the raw bytes."""
    _flag(monkeypatch, on=True)
    _real_wiring(monkeypatch)
    await pipeline.ingest_document(_pdf_doc("purge-me"), actor="uploader")
    assert await get_object_store().get(_TENANT, layout_object_key("purge-me"))

    await pipeline.delete_document(_TENANT, "purge-me", actor="dpo")

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("purge-me"))
    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, raw_object_key("purge-me"))


# --- the size ceiling ----------------------------------------------------------------------------


async def test_an_overlay_over_the_byte_ceiling_is_refused_not_truncated(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ocr._layout_capture_allowed` bounds the normal case from the page count; this catches what a
    page count cannot see — a pathological page of thousands of tiny text runs.

    Refused rather than truncated: a truncated overlay is not a smaller overlay, it is one that
    lies about where the document ends, and the parser, the structure rows and any later re-parse
    would all believe it. The absence is detectable instead of silent, because
    `documents.minio_layout_key` stays NULL — which is the same "never captured" signal this
    function already uses for a plain-text document.
    """
    _flag(monkeypatch, on=True)
    monkeypatch.setenv("RAG_LAYOUT_OVERLAY_MAX_BYTES", "65536")  # the floor the field allows
    get_settings.cache_clear()
    # 64 KiB is the smallest ceiling the field allows, so the overlay has to really exceed it.
    big = "<h1>Night shift policy</h1>" + "".join(f"<p>Clause {i} applies in full.</p>" for i in range(2000))
    _use_engine(monkeypatch, _MarkupOCR(markup=big))

    await pipeline.ingest_document(_doc("toobig"), actor="uploader")

    with pytest.raises(KeyError):
        await get_object_store().get(_TENANT, layout_object_key("toobig"))
    assert _twin().layout_key(_TENANT, "toobig") is None, "no key may point at an object never written"
    # The ingest itself must succeed — the overlay is additive, and refusing it is not a failure.
    assert [d.doc_id for d in await get_catalog().list_documents(_TENANT)] == ["toobig"]


async def test_an_overlay_under_the_ceiling_is_stored(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is a guard on the pathological case, not a policy that bites a normal document."""
    _flag(monkeypatch, on=True)
    monkeypatch.setenv("RAG_LAYOUT_OVERLAY_MAX_BYTES", str(32 * 1024 * 1024))
    get_settings.cache_clear()
    _use_engine(monkeypatch, _MarkupOCR())

    await pipeline.ingest_document(_doc("fits"), actor="uploader")

    assert await get_object_store().get(_TENANT, layout_object_key("fits"))
    assert _twin().layout_key(_TENANT, "fits") == layout_object_key("fits")
