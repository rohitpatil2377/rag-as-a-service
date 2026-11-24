"""The layout overlay: the engine markup that already arrives is KEPT instead of discarded.

Chandra is asked for Markdown and answers in HTML (`ocr_check.normalize_ocr_markup`'s docstring,
measured on the live corpus). That HTML was flattened to text on arrival and then dropped, so the
only structural signal the platform receives was thrown away. These tests pin what is kept, where
it is kept, and — the part that matters most — that keeping it changes `text` not at all.

`PageRecord.markup` is an OVERLAY: `text` stays authoritative and is never rebuilt from it, per
the rule in `PageRecord`'s own docstring ("structural information (table/column regions) belongs
in a separate overlay alongside `text`, never in a shape `text` is rebuilt from").
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime

import httpx
import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import (
    CAP_TABLES,
    CAP_TEXT,
    LAYOUT_OVERLAY_VERSION,
    OCR_EXTRACTOR_CHANDRA,
    REASON_OCR_DEGENERATE,
    ChandraOCREngine,
    DatalabOCREngine,
    Extraction,
    PageRecord,
)
from svc_doc_intelligence.service import DocIntelligenceService

from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_HTML = (
    "<h1>Tender Corrigendum</h1><table><tr><th>Item</th><th>Qty</th></tr><tr><td>LED panel</td><td>12</td></tr></table>"
)
_IMG = b"\x89PNG\r\n fake image bytes of a table"
_TEXT_LINE = "This page carries a genuine text layer with ample characters on it."


def _chandra_returning(answer: str) -> ChandraOCREngine:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    # localhost is egress-allowlisted, so the engine's default client construction passes the
    # sovereignty guard; we then swap in a mock transport (no real Chandra).
    eng = ChandraOCREngine(url="http://localhost/v1", model="chandra-test")
    eng._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost/v1")
    return eng


# --- what the Chandra path records ---------------------------------------------------------


async def test_chandra_keeps_the_markup_it_already_answered_with(inmemory: None) -> None:
    extraction = await _chandra_returning(_HTML).ocr(_IMG, "image/png")
    page = extraction.pages[0]

    assert page.markup == _HTML, "stored exactly as returned — pre-normalisation, not re-derived"
    assert page.markup_format == "html"
    # `text` is the flattened reading-order rendering, unchanged by this slice.
    assert page.text == "# Tender Corrigendum\n\n| Item | Qty |\n| --- | --- |\n| LED panel | 12 |"
    assert "<table>" not in page.text
    assert page.extractor == OCR_EXTRACTOR_CHANDRA
    assert CAP_TEXT in page.capabilities and CAP_TABLES in page.capabilities


async def test_keeping_the_markup_does_not_change_the_text_or_the_page_map(inmemory: None) -> None:
    """The whole safety claim of this slice in one assertion: `text` and its page map are
    byte-identical to what a build with no `markup` field produced. The overlay is additive."""
    extraction = await _chandra_returning(_HTML).ocr(_IMG, "image/png")
    rendered = extraction.render()

    stripped = Extraction(
        pages=[replace(p, markup=None, markup_format="") for p in extraction.pages],
        modality=extraction.modality,
    )
    assert rendered.text == stripped.render().text
    assert rendered.page_map == stripped.render().page_map


async def test_a_page_the_model_looped_on_still_records_its_markup(inmemory: None) -> None:
    """A degenerate page is the case most worth capturing: the markup is the evidence of what
    the model actually emitted. The degradation already rides on the record, so nothing is
    hidden by keeping it."""
    looping = f"<p>{_TEXT_LINE}</p>" + "<p>ce</p>" * 400
    extraction = await _chandra_returning(looping).ocr(_IMG, "image/png")
    page = extraction.pages[0]

    assert REASON_OCR_DEGENERATE in page.degradations
    assert page.markup == looping
    assert page.text.count("ce") < 10, "the loop is still trimmed out of the indexed text"
    # ...and the overlay carries the degradation, so a consumer reading ONLY the stored artifact
    # knows this page was not read properly.
    overlay = extraction.layout_overlay()
    assert overlay.pages[0].degradations == (REASON_OCR_DEGENERATE,)


# --- what every other extractor records: nothing --------------------------------------------


def test_a_native_pdf_page_records_no_markup() -> None:
    """pdfplumber produces plain text. Fabricating markup for it would invent structure the
    extractor never reported."""
    service = DocIntelligenceService()
    pdf = make_pdf([[_TEXT_LINE], [_TEXT_LINE]])
    extraction = _route(service, pdf)

    assert [p.markup for p in extraction.pages] == [None, None]
    overlay = extraction.layout_overlay()
    assert overlay.pages == ()
    assert overlay.extractor == ""


def _doc(mime: str = "application/pdf") -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id="mixed.pdf",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime=mime,
        ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


def _route(service: DocIntelligenceService, raw: bytes) -> Extraction:
    import asyncio

    return asyncio.run(service._route(_doc(), raw))


# --- the overlay's shape --------------------------------------------------------------------


class _MarkupOCR:
    """An engine that answers in markup, standing in for Chandra on the pages it is given."""

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[
                PageRecord(
                    page_no=1,
                    text="OCR TEXT",
                    extractor="markup-ocr",
                    capabilities=(CAP_TEXT,),
                    markup="<h2>OCR TEXT</h2>",
                    markup_format="html",
                )
            ],
            modality=Modality.OCR,
        )


def test_the_overlay_lists_only_the_pages_that_have_markup_by_their_real_page_number() -> None:
    """A mixed document: image pages go to the markup engine, text pages are read directly. The
    overlay must carry the DOCUMENT's page numbers, not its own indices, or a parser built on it
    would attribute structure to the wrong pages."""
    pages: list[list[str]] = [[], [_TEXT_LINE], [], [_TEXT_LINE]]
    pdf = make_pdf(pages, image_pages=(1, 3))
    extraction = _route(DocIntelligenceService(ocr_engine=_MarkupOCR()), pdf)

    assert [p.page_no for p in extraction.pages] == [1, 2, 3, 4]
    overlay = extraction.layout_overlay()
    assert [p.page_no for p in overlay.pages] == [1, 3]
    assert all(p.markup == "<h2>OCR TEXT</h2>" for p in overlay.pages)
    assert all(p.markup_format == "html" for p in overlay.pages)
    # Only the markup pages' extractor is named — the native-text pages must not be credited
    # with markup they never produced (the document-level `extractor` says "markup-ocr,pdfplumber").
    assert overlay.extractor == "markup-ocr"
    assert extraction.extractor == "markup-ocr,pdfplumber"


def test_empty_markup_is_distinguishable_from_no_markup() -> None:
    """A markup engine that answered a page with nothing is NOT the same fact as an engine that
    emits no markup at all, and a consumer has to be able to tell them apart."""
    extraction = Extraction(
        pages=[
            PageRecord(
                page_no=1, text="a", extractor="markup-ocr", capabilities=(CAP_TEXT,), markup="", markup_format="html"
            ),
            PageRecord(page_no=2, text="b", extractor="pdfplumber", capabilities=(CAP_TEXT,)),
        ],
        modality=Modality.TEXT,
    )
    overlay = extraction.layout_overlay()
    assert [(p.page_no, p.markup) for p in overlay.pages] == [(1, "")]


def test_a_document_with_no_markup_at_all_yields_an_empty_overlay() -> None:
    extraction = Extraction(
        pages=[PageRecord(page_no=1, text="a", extractor="pypdf", capabilities=(CAP_TEXT,))],
        modality=Modality.TEXT,
    )
    assert extraction.layout_overlay().pages == ()


def _one_markup_page() -> Extraction:
    return Extraction(
        pages=[
            PageRecord(
                page_no=7,
                text="a",
                extractor="markup-ocr",
                capabilities=(CAP_TEXT,),
                degradations=(REASON_OCR_DEGENERATE,),
                markup="<p>a</p>",
                markup_format="html",
            )
        ],
        modality=Modality.OCR,
    )


def test_the_overlay_is_json_serialisable_and_carries_a_schema_version() -> None:
    """It is PERSISTED, so a future shape change has to be detectable in a stored artifact
    rather than guessed at. The whole shape is pinned, not sampled: a field added or renamed
    without a version bump fails here."""
    payload = json.loads(json.dumps(_one_markup_page().layout_overlay().as_dict()))
    assert payload == {
        "schema_version": LAYOUT_OVERLAY_VERSION,
        "extractor": "markup-ocr",
        "content_hash": "",
        "pages": [
            {
                "page_no": 7,
                "markup": "<p>a</p>",
                "markup_format": "html",
                "degradations": [REASON_OCR_DEGENERATE],
            }
        ],
    }
    assert LAYOUT_OVERLAY_VERSION == 2, "bumped when content_hash + markup_format were added"


def test_an_unbound_overlay_claims_no_content_hash() -> None:
    """Extraction does not know how its bytes are addressed, so the hash is empty until the
    ingest path binds it. Empty must read as "not bound", never as a hash of nothing."""
    assert _one_markup_page().layout_overlay().content_hash == ""


def test_binding_the_overlay_to_its_bytes_changes_nothing_else() -> None:
    """What makes staleness checkable: the stored artifact names the bytes it describes, so a
    consumer re-hashes the raw object and compares instead of trusting the write paths."""
    overlay = _one_markup_page().layout_overlay()
    bound = overlay.for_content("deadbeef")

    assert bound.content_hash == "deadbeef"
    assert bound.as_dict()["content_hash"] == "deadbeef"
    assert overlay.content_hash == "", "binding returns a new value; the original is untouched"
    assert bound.pages == overlay.pages and bound.extractor == overlay.extractor


def test_render_does_not_grow_a_third_concern() -> None:
    """`render()` returns text + page map because neither is usable without the other. The
    overlay is a separate call on purpose: a text consumer has no use for it."""
    rendered = Extraction(pages=[], modality=Modality.TEXT).render()
    assert {f.name for f in fields(rendered)} == {"text", "page_map"}


# --- and it reaches the pipeline on the ExtractedDoc ----------------------------------------


async def test_extract_carries_the_overlay_onto_the_extracted_doc(inmemory: None) -> None:
    pdf = make_pdf([[], [_TEXT_LINE]], image_pages=(1,))
    service = DocIntelligenceService(ocr_engine=_MarkupOCR())
    extracted = await service.extract(_doc(), pdf)

    assert extracted.layout is not None
    assert [p.page_no for p in extracted.layout.pages] == [1]
    assert extracted.layout.pages[0].markup == "<h2>OCR TEXT</h2>"
    # The overlay rode along; the indexed text did not change shape because of it.
    assert sum(extracted.page_map) == len(extracted.text)


async def test_extract_reports_an_empty_overlay_for_a_plain_text_document(inmemory: None) -> None:
    """Not None: `extract` always reports, so "no page had markup" is a stated fact rather than
    an absence the caller has to interpret."""
    extracted = await DocIntelligenceService().extract(_doc(), make_pdf([[_TEXT_LINE]]))
    assert extracted.layout is not None
    assert extracted.layout.pages == ()


@pytest.mark.parametrize("page_no", [1, 12])
def test_the_overlay_page_number_is_the_page_s_own_not_its_index(page_no: int) -> None:
    extraction = Extraction(
        pages=[
            PageRecord(
                page_no=page_no, text="a", extractor="m", capabilities=(CAP_TEXT,), markup="<p/>", markup_format="html"
            )
        ],
        modality=Modality.OCR,
    )
    assert extraction.layout_overlay().pages[0].page_no == page_no


# --- the hosted Datalab engine: same model, same extractor id, DIFFERENT markup language ----


class _StubDatalabClient:
    """Stands in for DatalabClient — records what output_format was asked for."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.asked_for = ""

    async def convert(self, data: bytes, mime: str, *, output_format: str = "markdown") -> str:
        self.asked_for = output_format
        return self._answer


def _datalab_returning(answer: str) -> tuple[DatalabOCREngine, _StubDatalabClient]:
    eng = DatalabOCREngine(url="http://localhost")  # allowlisted; the client is then replaced
    stub = _StubDatalabClient(answer)
    eng._client = stub  # type: ignore[assignment]
    return eng, stub


_MARKDOWN = "# Tender Corrigendum\n\n| Item | Qty |\n| --- | --- |\n| LED panel | 12 |"


async def test_the_hosted_engine_keeps_its_markup_too(inmemory: None) -> None:
    """`text` here is this exact string flattened by `normalize_ocr_markup` — the function whose
    existence is the evidence this model emits markup — so discarding it would have thrown away
    the only structural signal a `chandra_api` deployment ever receives."""
    eng, stub = _datalab_returning(_MARKDOWN)
    extraction = await eng.ocr(_IMG, "image/png")
    page = extraction.pages[0]

    assert page.markup == _MARKDOWN
    assert page.markup_format == "markdown"
    assert stub.asked_for == "markdown", "the declared format is the one actually requested"
    assert extraction.layout_overlay().pages[0].markup == _MARKDOWN


async def test_both_chandra_engines_share_an_extractor_id_but_declare_different_formats(
    inmemory: None,
) -> None:
    """The regression this field exists for: a slice-B consumer keying off `extractor ==
    "chandra"` and assuming HTML would mis-parse every `chandra_api` deployment. The extractor id
    does not determine the markup language; `markup_format` does."""
    hosted, _ = _datalab_returning(_MARKDOWN)
    self_hosted_page = (await _chandra_returning(_HTML).ocr(_IMG, "image/png")).pages[0]
    hosted_page = (await hosted.ocr(_IMG, "image/png")).pages[0]

    assert self_hosted_page.extractor == hosted_page.extractor == OCR_EXTRACTOR_CHANDRA
    assert self_hosted_page.markup_format != hosted_page.markup_format
    assert {self_hosted_page.markup_format, hosted_page.markup_format} == {"html", "markdown"}


def test_a_page_with_no_markup_declares_no_format() -> None:
    """ "" and None travel together: a format without markup would describe nothing."""
    page = PageRecord(page_no=1, text="a", extractor="pypdf", capabilities=(CAP_TEXT,))
    assert page.markup is None and page.markup_format == ""


# --- the markup/format pairing is an invariant, not a convention --------------------------------


def test_markup_without_a_declared_format_is_refused() -> None:
    """The parser branches on `markup_format`, so undeclared markup is markup it cannot read. A
    new engine is where this would be forgotten, and the only symptom would be an empty section
    tree much later — so it fails at construction instead."""
    import pytest

    with pytest.raises(ValueError, match="no markup_format"):
        PageRecord(page_no=1, text="t", extractor="newengine", capabilities=(CAP_TEXT,), markup="<h1>x</h1>")


def test_a_declared_format_without_markup_is_refused() -> None:
    """The mirror case: a format is a claim about markup that isn't there."""
    import pytest

    with pytest.raises(ValueError, match="declared with no markup"):
        PageRecord(page_no=1, text="t", extractor="newengine", capabilities=(CAP_TEXT,), markup_format="html")


def test_a_plain_text_page_declares_neither_and_is_valid() -> None:
    """Every non-markup engine must keep constructing exactly as before."""
    rec = PageRecord(page_no=1, text="t", extractor="pdfplumber", capabilities=(CAP_TEXT,))
    assert rec.markup is None and rec.markup_format == ""
