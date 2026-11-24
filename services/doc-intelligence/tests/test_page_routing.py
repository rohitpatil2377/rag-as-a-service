"""Per-page scanned/OCR routing (FR-B4).

The scanned decision used to be a whole-document average: a PDF whose diagram pages
outnumbered its text pages was pushed wholesale through OCR, losing the perfectly good text
layer on the pages that had one. It is now decided per page, and the OCR'd pages are merged
back into the same record list.

OCR here is a fake engine: pdf2image/pytesseract live only in the prod OCR image, and what
needs testing is the ROUTING, not Tesseract's accuracy.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import (
    CAP_TABLES,
    CAP_TEXT,
    REASON_ISOLATED_NORMALIZATION,
    REASON_OCR_UNAVAILABLE,
    Extraction,
    PageRecord,
)
from svc_doc_intelligence.service import DocIntelligenceService

from rag_common import ValidationError
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_OCR_EXTRACTOR = "fake-ocr"
_TEXT_LINE = "This page carries a genuine text layer with ample characters on it."


class FakeOCR:
    """Records every call so we can assert WHAT was sent to OCR, not just that it ran."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        self.calls.append(data)
        import io

        import pdfplumber  # the fake reads the real page count so merging is exercised honestly

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            count = len(pdf.pages)
        return Extraction(
            pages=[
                PageRecord(
                    page_no=i,
                    text=f"OCR TEXT FOR PAGE {i}",
                    extractor=_OCR_EXTRACTOR,
                    capabilities=(CAP_TEXT,),
                    confidence=0.9,
                )
                for i in range(1, count + 1)
            ],
            modality=Modality.TEXT,
        )


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id="mixed.pdf",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime.now(UTC),
    )


def _service(ocr_engine: object | None = None, **kw: object) -> DocIntelligenceService:
    return DocIntelligenceService(ocr_engine=ocr_engine, **kw)  # type: ignore[arg-type]


def _route(service: DocIntelligenceService, raw: bytes) -> Extraction:
    return asyncio.run(service._route(_doc(), raw))


# --- the misjudged diagram-heavy document --------------------------------------------------


def _diagram_heavy() -> bytes:
    """8 image-only pages and 2 real text pages. The whole-document average puts this under
    the threshold, so today the ENTIRE document — text pages included — goes to OCR."""
    pages: list[list[str]] = [[] for _ in range(8)]
    pages.insert(3, [_TEXT_LINE])
    pages.append([_TEXT_LINE])
    image_pages = tuple(i for i, lines in enumerate(pages, start=1) if not lines)
    return make_pdf(pages, image_pages=image_pages)


def test_text_pages_of_a_diagram_heavy_pdf_are_not_routed_to_ocr() -> None:
    ocr = FakeOCR()
    extraction = _route(_service(ocr), _diagram_heavy())

    text_pages = [p for p in extraction.pages if p.extractor != _OCR_EXTRACTOR]
    assert [p.page_no for p in text_pages] == [4, 10]
    assert all(_TEXT_LINE in p.text for p in text_pages)
    assert all(p.extractor == "pdfplumber" for p in text_pages)
    assert all(CAP_TABLES in p.capabilities for p in text_pages)  # text path keeps table capture

    ocr_pages = [p for p in extraction.pages if p.extractor == _OCR_EXTRACTOR]
    assert [p.page_no for p in ocr_pages] == [1, 2, 3, 5, 6, 7, 8, 9]
    # Only the image pages were ever sent to an OCR engine — one single-page PDF each.
    assert len(ocr.calls) == len(ocr_pages)


def test_ocr_pages_merge_into_one_ordered_record_list() -> None:
    extraction = _route(_service(FakeOCR()), _diagram_heavy())
    assert [p.page_no for p in extraction.pages] == list(range(1, 11))
    # The engine only ever sees a ONE-page PDF, so it can only report page 1. The document's
    # page number has to be restored on merge, or every OCR'd page would collapse onto page 1.
    ocr_pages = [p for p in extraction.pages if p.extractor == _OCR_EXTRACTOR]
    assert all(p.text == "OCR TEXT FOR PAGE 1" for p in ocr_pages)  # what the engine saw
    assert [p.page_no for p in ocr_pages] == [1, 2, 3, 5, 6, 7, 8, 9]  # what the document says
    # Document view stays derived: OCR pages have no tables => the document has none.
    assert extraction.capabilities == (CAP_TEXT,)
    assert extraction.extractor == f"{_OCR_EXTRACTOR},pdfplumber"
    assert extraction.min_confidence == pytest.approx(0.9)  # weakest page, from the OCR records


# --- blank is not scanned -------------------------------------------------------------------


def test_a_blank_page_is_not_sent_to_ocr() -> None:
    """No text AND no image content means an empty page, not an unreadable one. Routing it to
    OCR would burn CPU and stamp a degradation on every duplex back and separator sheet."""
    ocr = FakeOCR()
    pdf = make_pdf([[_TEXT_LINE], [], [_TEXT_LINE]])  # page 2 blank: no text, no image
    extraction = _route(_service(ocr), pdf)

    assert ocr.calls == []
    assert all(p.extractor == "pdfplumber" for p in extraction.pages)
    assert extraction.degradations == ()


# --- no OCR engine configured ---------------------------------------------------------------


def test_a_page_needing_ocr_without_an_engine_keeps_the_other_pages() -> None:
    """Previously this document 422'd outright. The text pages are good — keep them, and stamp
    the pages we could not read so the loss is queryable."""
    extraction = _route(_service(None), _diagram_heavy())  # NoOpOCR

    text_pages = [p for p in extraction.pages if not p.degradations]
    assert [p.page_no for p in text_pages] == [4, 10]
    unreadable = [p for p in extraction.pages if REASON_OCR_UNAVAILABLE in p.degradations]
    assert len(unreadable) == 8
    assert extraction.degradations == (REASON_OCR_UNAVAILABLE,)


def test_a_fully_scanned_pdf_without_an_engine_still_fails_loudly() -> None:
    """Unchanged contract: when NO page has text there is nothing to keep, so refusing with an
    actionable 422 beats indexing an empty document."""
    pdf = make_pdf([[], [], []], image_pages=(1, 2, 3))
    with pytest.raises(ValidationError, match="OCR"):
        _route(_service(None), pdf)


def test_a_fully_scanned_pdf_goes_to_the_engine_in_one_call() -> None:
    """Every page taking the same path => one whole-document call, so cross-page normalization
    inside the engine still sees the full page set."""
    ocr = FakeOCR()
    pdf = make_pdf([[], [], []], image_pages=(1, 2, 3))
    extraction = _route(_service(ocr), pdf)

    assert len(ocr.calls) == 1
    assert ocr.calls[0] == pdf  # the whole document, not a split page
    assert [p.page_no for p in extraction.pages] == [1, 2, 3]
    assert all(p.extractor == _OCR_EXTRACTOR for p in extraction.pages)
    assert REASON_ISOLATED_NORMALIZATION not in extraction.degradations


def test_pages_ocrd_in_isolation_record_what_that_cost() -> None:
    """A page OCR'd on its own cannot participate in running header/footer detection, which
    needs the full page set. Visible degradation, not a silent quality drop."""
    extraction = _route(_service(FakeOCR()), _diagram_heavy())
    ocr_pages = [p for p in extraction.pages if p.extractor == _OCR_EXTRACTOR]
    assert all(REASON_ISOLATED_NORMALIZATION in p.degradations for p in ocr_pages)
    assert REASON_ISOLATED_NORMALIZATION in extraction.degradations


# --- the threshold is a tunable --------------------------------------------------------------


def test_the_scanned_threshold_is_configurable() -> None:
    """A page with a short-but-real text layer is scanned or not depending on the operator's
    chars-per-page setting — nothing about it is baked in."""
    # Both pages carry an image, so the CHARACTER threshold is the only thing deciding.
    pdf = make_pdf([["Figure 1."], [_TEXT_LINE]], image_pages=(1, 2))  # 9 chars vs 67 chars

    lenient = _route(_service(FakeOCR(), scanned_chars_per_page=5), pdf)
    assert [p.extractor for p in lenient.pages] == ["pdfplumber", "pdfplumber"]

    between = _route(_service(FakeOCR(), scanned_chars_per_page=20), pdf)
    assert [p.extractor for p in between.pages] == [_OCR_EXTRACTOR, "pdfplumber"]

    strict = _route(_service(FakeOCR(), scanned_chars_per_page=100), pdf)
    assert [p.extractor for p in strict.pages] == [_OCR_EXTRACTOR, _OCR_EXTRACTOR]
