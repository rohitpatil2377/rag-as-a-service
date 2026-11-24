"""The ingestion OCR check: every document is asked whether it holds text only OCR can read.

Measured live before this existed: a JPEG e-ticket was indexed as a 32-character caption while
the OCR engine — which read 4,544 characters from the same file — was never called. These tests
pin that every route that can carry pixels-as-text reaches OCR, that a text layer is never
replaced by OCR, and that anything OCR could not read is stamped rather than silently dropped.

OCR is a fake engine: what is under test is the ROUTING and the merge, not a model's accuracy.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from ooxml_fixtures import make_docx, make_pptx
from pdf_fixtures import make_pdf
from PIL import Image
from svc_doc_intelligence.ocr import (
    CAP_CAPTION,
    CAP_TEXT,
    REASON_OCR_FAILED,
    REASON_OCR_IMAGE_CAP,
    Extraction,
    PageRecord,
)
from svc_doc_intelligence.ocr_check import (
    PdfImageBox,
    image_worth_ocr,
    normalize_ocr_markup,
    pdf_images_worth_ocr,
)
from svc_doc_intelligence.service import DocIntelligenceService

from rag_common import ValidationError
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_TEXT_LINE = "This page carries a genuine text layer with ample characters on it."


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=200).save(buf, format="PNG")
    return buf.getvalue()


def _doc(doc_id: str, mime: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id=doc_id,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime=mime,
        ingested_at=datetime.now(UTC),
    )


class FakeOCR:
    """Answers every call with a fixed transcription and records what it was sent."""

    def __init__(self, text: str = "OCR READ THIS", *, fail: Exception | None = None) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[bytes, str]] = []

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        self.calls.append((data, mime))
        if self.fail is not None:
            raise self.fail
        return Extraction(
            pages=[PageRecord(page_no=1, text=self.text, extractor="fake-ocr", capabilities=(CAP_TEXT,))],
            modality=Modality.OCR,
        )


async def _caption(data: bytes, mime: str, tenant_id: str) -> str:
    return "An airline e-ticket with a barcode."


# --- markup -> text ---------------------------------------------------------------------------


def test_ocr_html_becomes_text_with_markdown_tables() -> None:
    """Chandra is asked for Markdown and answers in HTML; live, 18 chunks were indexed full of
    <td> tags. The label must stay beside its value, which a flat cell stream would lose."""
    html = (
        "<div><h2>e-Ticket Receipt &amp; Itinerary</h2><p>Economy passengers report 3 hours prior.</p>"
        '<img alt="Emirates logo"/><table><thead><tr><th>E-TICKET NUMBER</th><th>176 2139738899</th></tr>'
        "</thead><tbody><tr><td>BAGGAGE</td><td>30KGS</td></tr></tbody></table></div>"
    )
    text = normalize_ocr_markup(html)
    assert "<" not in text and ">" not in text
    assert "## e-Ticket Receipt & Itinerary" in text
    assert "| E-TICKET NUMBER | 176 2139738899 |" in text
    assert "| BAGGAGE | 30KGS |" in text
    assert "[Image: Emirates logo]" in text


def test_ocr_output_without_markup_is_left_untouched() -> None:
    """An engine that already returns Markdown must not be rewritten by the converter."""
    markdown = "# Title\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nAmount 5 < 10 and 7 > 3."
    assert normalize_ocr_markup(markdown) == markdown


# --- what is worth OCR ---------------------------------------------------------------------------


def test_small_images_are_decoration_and_large_ones_are_worth_reading() -> None:
    assert image_worth_ocr(_png(1200, 800)) is True
    assert image_worth_ocr(_png(320, 64)) is False  # a letterhead logo: wide but short
    assert image_worth_ocr(b"not an image at all") is False


def test_a_pdf_image_must_be_large_on_the_page_and_large_in_pixels() -> None:
    """Share alone would OCR a low-resolution banner; pixels alone a thumbnail-sized logo."""
    page_w, page_h = 612.0, 792.0
    scan = PdfImageBox(x0=72, top=200, x1=540, bottom=600, src_width=1400, src_height=1200)
    logo = PdfImageBox(x0=20, top=20, x1=120, bottom=60, src_width=2000, src_height=800)
    banner = PdfImageBox(x0=0, top=0, x1=612, bottom=120, src_width=600, src_height=40)
    kept = pdf_images_worth_ocr([scan, logo, banner], page_width=page_w, page_height=page_h)
    assert kept == [scan]


# --- image files -------------------------------------------------------------------------------------


async def test_an_image_file_is_ocrd_and_the_caption_is_added_not_substituted() -> None:
    """The live failure: with captioning on, the caption REPLACED OCR. Both must be indexed."""
    ocr = FakeOCR("E-TICKET NUMBER 176 2139738899")
    service = DocIntelligenceService(ocr_engine=ocr, image_describer=_caption)
    out = await service.extract(_doc("ticket.jpg", "image/jpeg"), _png(600, 780))

    assert len(ocr.calls) == 1
    assert "E-TICKET NUMBER 176 2139738899" in out.text
    assert "[Image description] An airline e-ticket with a barcode." in out.text
    assert out.provenance["ocr_applied"] == "true"
    assert out.provenance["ocr_reasons"] == "image_file"
    assert CAP_CAPTION in out.provenance["extraction_capabilities"]
    assert out.provenance["extraction_degraded"] == "false"


async def test_an_image_without_captioning_is_still_ocrd() -> None:
    ocr = FakeOCR("FORM 16 PART A")
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("form.png", "image/png"), _png(800, 1000))
    assert out.text == "FORM 16 PART A"
    assert out.provenance["ocr_applied"] == "true"


async def test_when_ocr_fails_the_caption_is_kept_and_the_gap_is_stamped() -> None:
    """Degrade visibly: indexing the caption as though it were a full reading would hide the loss."""
    ocr = FakeOCR(fail=RuntimeError("chandra 503"))
    service = DocIntelligenceService(ocr_engine=ocr, image_describer=_caption)
    out = await service.extract(_doc("ticket.jpg", "image/jpeg"), _png(600, 780))

    assert out.text == "[Image description] An airline e-ticket with a barcode."
    assert out.provenance["extraction_degraded"] == "true"
    assert REASON_OCR_FAILED in out.provenance["extraction_degradation"]
    assert out.provenance["ocr_applied"] == "false"


async def test_when_ocr_fails_and_there_is_no_caption_ingestion_fails_loudly() -> None:
    ocr = FakeOCR(fail=RuntimeError("chandra 503"))
    with pytest.raises(RuntimeError, match="chandra 503"):
        await DocIntelligenceService(ocr_engine=ocr).extract(_doc("ticket.jpg", "image/jpeg"), _png(600, 780))


async def test_an_image_with_no_ocr_engine_and_no_caption_is_refused_with_a_reason() -> None:
    with pytest.raises(ValidationError, match="needs OCR"):
        await DocIntelligenceService().extract(_doc("ticket.jpg", "image/jpeg"), _png(600, 780))


async def test_an_image_ocr_reads_nothing_from_is_refused_rather_than_indexed_empty() -> None:
    with pytest.raises(ValidationError, match="no readable text"):
        await DocIntelligenceService(ocr_engine=FakeOCR("   ")).extract(_doc("blank.png", "image/png"), _png(600, 600))


# --- PDF pages that keep their text layer --------------------------------------------------------


async def test_a_large_image_on_a_texty_pdf_page_is_read_and_the_text_layer_kept() -> None:
    """Only the image region is OCR'd: the exact text layer must survive untouched, with what the
    image says appended to the same page."""
    raw = make_pdf([[_TEXT_LINE], [_TEXT_LINE]], image_pages=(2,), image_px=400)
    ocr = FakeOCR("TABLE FROM A PASTED SCREENSHOT")
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("report.pdf", "application/pdf"), raw)

    assert len(ocr.calls) == 1
    assert ocr.calls[0][1] == "image/png"
    page_two = out.text[out.page_map[0] :]
    assert _TEXT_LINE in page_two
    assert "[Text read from an embedded image]\nTABLE FROM A PASTED SCREENSHOT" in page_two
    assert "TABLE FROM A PASTED SCREENSHOT" not in out.text[: out.page_map[0]]
    assert out.provenance["ocr_reasons"] == "embedded_images"
    assert out.provenance["ocr_images"] == "1"


async def test_a_decorative_pdf_image_does_not_trigger_ocr() -> None:
    raw = make_pdf([[_TEXT_LINE]], image_pages=(1,))  # the 2x2 swatch
    ocr = FakeOCR()
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("letter.pdf", "application/pdf"), raw)
    assert ocr.calls == []
    assert out.provenance["ocr_applied"] == "false"


async def test_embedded_image_ocr_can_be_switched_off() -> None:
    raw = make_pdf([[_TEXT_LINE]], image_pages=(1,), image_px=400)
    ocr = FakeOCR()
    await DocIntelligenceService(ocr_engine=ocr, embedded_image_ocr=False).extract(
        _doc("report.pdf", "application/pdf"), raw
    )
    assert ocr.calls == []


# --- Office files ------------------------------------------------------------------------------------


def _with_parts(archive: bytes, extra: Mapping[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(buf, "w") as out:
        for item in src.infolist():
            out.writestr(item, src.read(item.filename))
        for name, body in extra.items():
            out.writestr(name, body)
    return buf.getvalue()


def _image_rels(*targets: str) -> str:
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    body = "".join(f'<Relationship Id="rId{i}" Type="{rel}" Target="{t}"/>' for i, t in enumerate(targets, start=10))
    return (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{body}</Relationships>"
    )


async def test_a_screenshot_pasted_on_a_slide_is_read_onto_that_slide() -> None:
    """The fixture deck numbers its slide parts in REVERSE of presentation order, so attaching by
    part name would put the text on the wrong slide."""
    deck = make_pptx([["Agenda"], ["Quarterly numbers"]])  # slide 2 is part slide1.xml
    raw = _with_parts(
        deck,
        {
            "ppt/slides/_rels/slide1.xml.rels": _image_rels("../media/screenshot.png"),
            "ppt/slides/_rels/slide2.xml.rels": _image_rels("../media/icon.png"),
            "ppt/media/screenshot.png": _png(1280, 720),
            "ppt/media/icon.png": _png(48, 48),
        },
    )
    ocr = FakeOCR("REVENUE 42 CRORE")
    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("deck.pptx", mime), raw)

    assert len(ocr.calls) == 1, "the icon is decoration and must not be sent to OCR"
    slide_one, slide_two = out.text[: out.page_map[0]], out.text[out.page_map[0] :]
    assert "REVENUE 42 CRORE" in slide_two
    assert "REVENUE 42 CRORE" not in slide_one
    assert out.provenance["ocr_images"] == "1"


async def test_a_scan_pasted_into_a_word_document_is_read_after_the_body() -> None:
    raw = _with_parts(make_docx(["Please find the signed letter below."]), {"word/media/image1.jpeg": _png(900, 1200)})
    ocr = FakeOCR("SANCTION LETTER REF 2026/114")
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("letter.docx", mime), raw)

    assert out.text.index("Please find the signed letter below.") < out.text.index("SANCTION LETTER REF 2026/114")
    assert out.provenance["ocr_reasons"] == "embedded_images"


async def test_images_past_the_cap_are_not_read_and_the_document_says_so() -> None:
    media = {f"word/media/image{i}.png": _png(800, 800) for i in range(1, 4)}
    raw = _with_parts(make_docx(["Annexures follow."]), media)
    ocr = FakeOCR("ANNEXURE")
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    out = await DocIntelligenceService(ocr_engine=ocr, max_ocr_images=2).extract(_doc("annex.docx", mime), raw)

    assert len(ocr.calls) == 2
    assert out.provenance["ocr_images"] == "2"
    assert REASON_OCR_IMAGE_CAP in out.provenance["extraction_degradation"]


async def test_one_unreadable_embedded_image_does_not_sink_the_document() -> None:
    raw = _with_parts(make_docx(["Body text survives."]), {"word/media/image1.png": _png(800, 800)})
    ocr = FakeOCR(fail=RuntimeError("timeout"))
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    out = await DocIntelligenceService(ocr_engine=ocr).extract(_doc("memo.docx", mime), raw)

    assert "Body text survives." in out.text
    assert REASON_OCR_FAILED in out.provenance["extraction_degradation"]
    assert out.provenance["ocr_applied"] == "false"


async def test_a_plain_text_document_reports_no_ocr() -> None:
    out = await DocIntelligenceService(ocr_engine=FakeOCR()).extract(_doc("notes.txt", "text/plain"), b"just text")
    assert out.provenance["ocr_applied"] == "false"
    assert out.provenance["ocr_reasons"] == ""
