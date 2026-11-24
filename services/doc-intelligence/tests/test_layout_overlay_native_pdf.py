"""The BORN-DIGITAL layout overlay: the structure tree finally reaches the documents that matter.

Phase 2 built the tree from Chandra's HTML, but `application/pdf` classifies as
`DocKind.NATIVE_PDF` and `_route_pdf` only sends a page to OCR when `page_needs_ocr` fires — so a
word-processor tender or policy PDF never reaches Chandra, produced no markup, and got no blocks.
Measured on the live corpus: 1 document in 6 had any. These tests pin the overlay this path CAN
give (coordinates and per-glyph typography, which a chat-shaped VLM cannot), and — the part the
whole safety case rests on — that producing it changes `text` not at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pdfplumber.page
import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence import ocr
from svc_doc_intelligence.ocr import (
    CAP_TEXT,
    LAYOUT_JSON_VERSION,
    MARKUP_FORMAT_LAYOUT_JSON,
    REASON_EXTRACTOR_FALLBACK,
    Extraction,
    PageRecord,
    extract_pdf_text,
    provenance_tags,
)
from svc_doc_intelligence.service import DocIntelligenceService

from rag_common import get_settings
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality, SourceKind

_HEADING = "SECTION 1 SCOPE OF WORK"
_BODY = "The tenderer shall supply the items listed in the schedule below."
_TAIL = "All rates are inclusive of taxes and delivery."
_SECOND = "Delivery shall be completed within sixty days of the award."
_ROWS = [["Item", "Qty"], ["LED panel", "12"], ["Switchgear", "4"]]


def _tender_pdf() -> bytes:
    """A two-page born-digital PDF: a bigger, bolder heading, prose, a ruled table mid-page
    between two paragraphs, and a second page of prose."""
    return make_pdf(
        [[_HEADING, _BODY, _TAIL], [_SECOND]],
        tables={1: _ROWS},
        table_after_line={1: 2},
        styles={1: {0: ("F2", 18.0)}},  # the heading: Helvetica-Bold at 18pt over 12pt body
    )


def _payload(page: PageRecord) -> dict[str, Any]:
    assert page.markup is not None
    return dict(json.loads(page.markup))


def _blocks(page: PageRecord, kind: str | None = None) -> list[dict[str, Any]]:
    return [b for b in _payload(page)["blocks"] if kind is None or b["block_type"] == kind]


# --- the claim everything else rests on: `text` is untouched --------------------------------


def test_the_overlay_does_not_change_the_text_on_any_page() -> None:
    """Byte-identical, with the capture on and off, on the same bytes. Not "equivalent" — the
    overlay is derived from the pieces the splice already produced, so there is no second
    detection that could move a table or re-order a band."""
    pdf = _tender_pdf()
    on = extract_pdf_text(pdf, doc_id="tender.pdf", capture_layout=True)
    off = extract_pdf_text(pdf, doc_id="tender.pdf")

    assert [p.text for p in on.pages] == [p.text for p in off.pages]
    assert on.render().text == off.render().text
    assert on.render().page_map == off.render().page_map
    # And nothing derived from the text moved either: table shares, counts, boilerplate.
    assert provenance_tags(on) == provenance_tags(off)


def test_the_capture_off_does_no_pdfplumber_layout_work_at_all() -> None:
    """A pdfplumber layout pass is real work per page, which is why the CAPTURE is gated and not
    just the persist (Chandra's markup arrived free in a response that was already made). Flag
    off must mean the line extraction never runs, not that its result is thrown away."""
    calls: list[int] = []
    real = pdfplumber.page.Page.extract_text_lines

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(self, *args, **kwargs)

    pdf = _tender_pdf()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pdfplumber.page.Page, "extract_text_lines", counted)
        off = extract_pdf_text(pdf, doc_id="tender.pdf")
        assert calls == [], "the flag was off and the layout pass still ran"
        assert [p.markup for p in off.pages] == [None, None]
        assert off.layout_overlay().pages == ()

        extract_pdf_text(pdf, doc_id="tender.pdf", capture_layout=True)
        assert calls, "with the flag on it must actually be the line extraction doing the work"


# --- the format and the invariant -----------------------------------------------------------


def test_every_captured_page_declares_the_new_format_and_only_that_format() -> None:
    extraction = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True)

    assert [p.markup_format for p in extraction.pages] == [MARKUP_FORMAT_LAYOUT_JSON] * 2
    assert MARKUP_FORMAT_LAYOUT_JSON not in ("html", "markdown")
    overlay = extraction.layout_overlay()
    assert [p.page_no for p in overlay.pages] == [1, 2]
    assert {p.markup_format for p in overlay.pages} == {MARKUP_FORMAT_LAYOUT_JSON}
    assert overlay.extractor == "pdfplumber"


def test_the_markup_format_pairing_holds_for_every_page_this_path_emits() -> None:
    """`PageRecord.__post_init__` refuses markup with no format and a format with no markup. The
    native path now emits both shapes (captured / not captured), so pin that it satisfies the
    invariant in both — a page built here must never be a page the parser cannot read."""
    for capture in (True, False):
        for page in extract_pdf_text(_tender_pdf(), doc_id="t.pdf", capture_layout=capture).pages:
            assert (page.markup is not None) == bool(page.markup_format)
            # The construction is not merely consistent by luck — re-running it through the
            # dataclass re-runs __post_init__ on exactly these values.
            PageRecord(
                page_no=page.page_no,
                text=page.text,
                extractor=page.extractor,
                capabilities=page.capabilities,
                markup=page.markup,
                markup_format=page.markup_format,
            )


def test_the_payload_names_its_own_format_version_and_page_geometry() -> None:
    """A bbox is uninterpretable without the page it is in: "this block sits in the top 8% of the
    page" is what the plan's header/footer region rule asks, and coordinates alone cannot say."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[1]
    payload = _payload(page)

    assert payload["format"] == MARKUP_FORMAT_LAYOUT_JSON
    assert payload["version"] == LAYOUT_JSON_VERSION
    assert payload["page_no"] == 2
    assert (payload["page_width"], payload["page_height"]) == (612.0, 792.0)


def test_every_block_carries_the_full_key_set_whatever_its_type() -> None:
    """A consumer branches on `block_type`, never on which keys it happens to find."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    keys = {"block_type", "text", "page", "bbox", "font_size", "bold", "cells"}

    assert [b["block_type"] for b in _blocks(page)] == ["line", "line", "table", "line"]
    assert all(set(b) == keys for b in _blocks(page))
    assert all(b["page"] == 1 for b in _blocks(page))


# --- typography: the signal this path has and Chandra does not ------------------------------


def test_a_heading_line_is_measurably_bigger_and_bolder_than_the_body_around_it() -> None:
    """The plan's heading-detection cue ("font size / bold"), which Phase 2 had to document as
    permanently unavailable because a chat-shaped VLM returns no metrics. On this path it is
    measured per line from `page.chars` — the overlay carries the EVIDENCE; classifying a line
    as a heading is a judgement about the document's own size distribution and belongs to the
    parser (Phase 2c-2)."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    by_text = {b["text"]: b for b in _blocks(page, "line")}

    assert by_text[_HEADING]["font_size"] == 18.0
    assert by_text[_HEADING]["bold"] is True
    assert by_text[_BODY]["font_size"] == 12.0
    assert by_text[_BODY]["bold"] is False
    assert by_text[_HEADING]["font_size"] > by_text[_BODY]["font_size"]


def test_a_table_block_claims_no_typography_rather_than_an_arbitrary_line_s() -> None:
    """A band spans many lines in whatever faces its cells use; reporting one of them as "the"
    size of the band would be a claim about nothing. None reads as "not measured"."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    table = _blocks(page, "table")[0]

    assert table["font_size"] is None and table["bold"] is None
    assert all(b["cells"] is None for b in _blocks(page, "line")), "the mirror case"


def test_the_line_size_is_a_median_so_one_outlier_glyph_cannot_move_it() -> None:
    """Why median and not max or mean: a superscript marker, a trademark sign or a dropped
    capital would each drag a maximum (and, on a short line, a mean) off the size a reader would
    call the line's. It takes more than half the glyphs to move an order statistic."""
    chars = [{"size": 11.0, "fontname": "Helvetica"}] * 9 + [{"size": 48.0, "fontname": "Helvetica"}]
    assert ocr._line_typography(chars) == (11.0, False)
    assert ocr._line_typography([]) == (None, None), "not measured must not read as size 0"


def test_a_bolded_lead_in_does_not_make_the_whole_paragraph_bold() -> None:
    """`bold` is a majority of the line's glyphs, not "any bold glyph": "**Note:** the tenderer
    shall..." is a paragraph with emphasis, not a heading."""
    lead_in = [{"size": 11.0, "fontname": "Arial-BoldMT"}] * 5
    rest = [{"size": 11.0, "fontname": "ArialMT"}] * 40
    assert ocr._line_typography(lead_in + rest) == (11.0, False)
    assert ocr._line_typography(lead_in) == (11.0, True)
    # The marker list covers how real PostScript names spell weight.
    for name in ("Arial-BoldMT", "TimesNewRoman,Bold", "Calibri-Semibold", "Helvetica-Black"):
        assert ocr._line_typography([{"size": 10.0, "fontname": name}])[1] is True


# --- geometry -------------------------------------------------------------------------------


def test_every_bbox_is_plausible_and_inside_its_page() -> None:
    extraction = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True)
    for page in extraction.pages:
        payload = _payload(page)
        width, height = payload["page_width"], payload["page_height"]
        for block in payload["blocks"]:
            x0, top, x1, bottom = block["bbox"]
            assert 0 <= x0 < x1 <= width, block
            assert 0 <= top < bottom <= height, block


def test_the_blocks_run_down_the_page_in_the_order_the_text_does() -> None:
    """Reading order, not a bag: a parser that walks the blocks must see the heading before the
    paragraph and the table where the text has it."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    tops = [b["bbox"][1] for b in _blocks(page)]

    assert tops == sorted(tops)
    assert [b["text"] for b in _blocks(page, "line")] == [_HEADING, _BODY, _TAIL]


# --- the overlay and the text cannot disagree about tables ----------------------------------


def test_the_overlay_s_tables_are_the_text_s_tables_count_text_and_position() -> None:
    """ONE detection produces both. The splice emits exactly one serialized body per BAND, and
    the overlay emits one `table` block per band carrying that body VERBATIM — so the counts
    match, the strings match, and the block's position in the stream is the body's position in
    the text. If these could disagree, a Phase-5 chunk's `section_id` would point at structure
    the text never had."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    tables = _blocks(page, "table")

    assert len(tables) == 1
    body = tables[0]["text"]
    assert body in page.text, "the overlay's table text is the text's own table text"
    # Position: the narrative lines before the table block are before it in the text too.
    before = [b for b in _blocks(page) if b["bbox"][1] < tables[0]["bbox"][1]]
    for block in before:
        assert page.text.index(block["text"]) < page.text.index(body)
    assert page.text.index(_TAIL) > page.text.index(body), "and the line below it is below it"


def test_a_table_block_carries_its_real_box_and_pdfplumber_s_own_cells() -> None:
    """Cells come from `table.extract()` — the SAME grid `serialize_table` rendered into the
    text — so a cell is a cell rather than a guess from spacing. Note the block's `text` is the
    LABELLED serialization (the header row became labels); `cells` is the grid behind it."""
    page = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages[0]
    table = _blocks(page, "table")[0]

    assert table["cells"] == _ROWS
    x0, top, x1, bottom = table["bbox"]
    # The table is drawn at x 72..330 by the fixture, mid-page between the two paragraphs.
    assert (x0, x1) == (72.0, 330.0)
    lines = _blocks(page, "line")
    assert lines[1]["bbox"][3] <= top and bottom <= lines[2]["bbox"][1]


def test_a_page_with_no_table_still_reports_its_lines() -> None:
    extraction = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True)
    page = extraction.pages[1]

    assert _blocks(page, "table") == []
    assert [b["text"] for b in _blocks(page, "line")] == [_SECOND]
    assert page.tables_found is False


def test_a_block_s_text_appears_in_its_page_s_text() -> None:
    for page in extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages:
        assert _blocks(page), "a page with no blocks would pass this vacuously"
        for block in _blocks(page):
            assert block["text"] in page.text, block


def test_the_overlay_is_pre_normalization_like_chandra_s_and_says_so() -> None:
    """The HONEST LIMIT, pinned rather than left to be rediscovered. `_normalize_pages` runs over
    the JOINED page string — running header/footer removal, de-hyphenation — so a stripped
    running header is still a block in the overlay although it is gone from `text`. Exactly the
    Chandra path's position (`markup` is stored pre-`normalize_ocr_markup`), and the reason
    `parse_layout` is handed `text` separately instead of deriving it from blocks."""
    header = "ACME LTD CONFIDENTIAL"
    pdf = make_pdf([[header, _BODY], [header, _TAIL], [header, _SECOND]])
    pages = extract_pdf_text(pdf, doc_id="report.pdf", capture_layout=True).pages

    assert all(p.boilerplate_lines == 1 for p in pages), "the running header was stripped"
    assert all(header not in p.text for p in pages)
    assert all(header in [b["text"] for b in _blocks(p, "line")] for p in pages)


# --- failure containment --------------------------------------------------------------------


def test_a_pypdf_fallback_page_records_no_overlay() -> None:
    """pypdf exposes no per-glyph geometry, so there is nothing to measure. Fabricating a
    bbox-less block would claim this path can do what it cannot."""
    real = ocr._page_tables
    seen: list[int] = []

    def boom_on_second(page: Any) -> Any:
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("pdfplumber tripped on this page")
        return real(page)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ocr, "_page_tables", boom_on_second)
        pages = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True).pages

    assert REASON_EXTRACTOR_FALLBACK in pages[1].degradations
    assert (pages[1].markup, pages[1].markup_format) == (None, "")
    assert pages[0].markup is not None, "the page that read fine keeps its overlay"
    assert [p.page_no for p in extract_pdf_text(_tender_pdf(), doc_id="x", capture_layout=True).pages] == [1, 2]


def test_a_layout_capture_failure_costs_the_page_its_overlay_never_its_text() -> None:
    """The overlay is additive, so a bug in it must not fall through to the pypdf fallback — that
    would drop the page's TABLES to save a block list."""
    pdf = _tender_pdf()
    expected = [p.text for p in extract_pdf_text(pdf, doc_id="tender.pdf").pages]

    def boom(crop: Any, page_no: int) -> Any:
        raise RuntimeError("layout derivation is broken")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ocr, "_narrative_blocks", boom)
        pages = extract_pdf_text(pdf, doc_id="tender.pdf", capture_layout=True).pages

    assert [p.text for p in pages] == expected
    assert [p.markup for p in pages] == [None, None]
    assert [p.extractor for p in pages] == ["pdfplumber", "pdfplumber"], "no fallback was taken"
    assert all(p.degradations == () for p in pages)


# --- the realistic file: a tender with scanned annexures appended ---------------------------


class _MarkupOCR:
    """Stands in for Chandra on the pages routed to it: answers HTML, declares HTML."""

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        return Extraction(
            pages=[
                PageRecord(
                    page_no=1,
                    text="ANNEXURE A",
                    extractor="markup-ocr",
                    capabilities=(CAP_TEXT,),
                    markup="<h2>ANNEXURE A</h2>",
                    markup_format="html",
                )
            ],
            modality=Modality.OCR,
        )


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id="tender.pdf",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


async def test_a_mixed_pdf_declares_its_own_markup_format_per_page(inmemory: None) -> None:
    """The realistic case for a scanned-and-appended tender, and the reason `markup_format` is
    PER PAGE: page 1 is born-digital and answers `layout-json`, page 2 is a scan routed to OCR
    and answers `html`, in ONE document, each page saying which language it is in."""
    pdf = make_pdf([[_HEADING, _BODY], []], image_pages=(2,))
    service = DocIntelligenceService(ocr_engine=_MarkupOCR(), pdf_layout_overlay=True)

    extraction = await service._route(_doc(), pdf)

    assert [(p.page_no, p.markup_format) for p in extraction.pages] == [
        (1, MARKUP_FORMAT_LAYOUT_JSON),
        (2, "html"),
    ]
    overlay = extraction.layout_overlay()
    assert [(p.page_no, p.markup_format) for p in overlay.pages] == [(1, MARKUP_FORMAT_LAYOUT_JSON), (2, "html")]
    # Both extractors are named, neither credited with the other's markup.
    assert overlay.extractor == "markup-ocr,pdfplumber"
    assert json.loads(overlay.pages[0].markup)["blocks"][0]["text"] == _HEADING
    assert overlay.pages[1].markup == "<h2>ANNEXURE A</h2>"


async def test_the_service_gate_is_what_decides_whether_the_pass_runs(inmemory: None) -> None:
    """Wired from `v2_ingest` at `wiring.build_doc_intel_service`. Default off, and off means
    the born-digital pages carry no overlay at all — the Chandra pages still do, because that
    markup was already in a response that had to be made."""
    pdf = make_pdf([[_HEADING, _BODY], []], image_pages=(2,))
    extraction = await DocIntelligenceService(ocr_engine=_MarkupOCR())._route(_doc(), pdf)

    assert [(p.page_no, p.markup_format) for p in extraction.pages] == [(1, ""), (2, "html")]
    assert [p.page_no for p in extraction.layout_overlay().pages] == [2]


async def test_extract_carries_the_native_overlay_onto_the_extracted_doc(inmemory: None) -> None:
    """It has to reach `pipeline._persist_layout_overlay` on the existing seam — which needs no
    change, because it stores whatever pages the overlay has under the same key."""
    service = DocIntelligenceService(pdf_layout_overlay=True)
    extracted = await service.extract(_doc(), _tender_pdf())

    assert extracted.layout is not None
    assert [p.page_no for p in extracted.layout.pages] == [1, 2]
    assert {p.markup_format for p in extracted.layout.pages} == {MARKUP_FORMAT_LAYOUT_JSON}
    # The artifact is JSON-serialisable exactly as stored, bound to the bytes it describes.
    body = json.dumps(extracted.layout.for_content("sha256:abc").as_dict(), ensure_ascii=False)
    stored = json.loads(body)
    assert stored["extractor"] == "pdfplumber"
    assert stored["content_hash"] == "sha256:abc"
    assert json.loads(stored["pages"][0]["markup"])["page_no"] == 1
    # ...and the indexed text still matches its page map.
    assert sum(extracted.page_map) == len(extracted.text)


# --- the capture guards --------------------------------------------------------------------------


def test_a_document_over_the_page_cap_captures_no_layout_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture is a SECOND pdfplumber pass per page plus a JSON object holding every line's
    bbox and font metrics, both linear in page count with no bound. Measured at 29 blocks/page on a
    tender-shaped fixture: +0-20% extraction time and 6.7 KiB/page, so 2000 pages is ~13 MiB and
    ~58k blocks serialized, stored, re-fetched and re-parsed inside the API event loop.

    Skipped for the WHOLE document rather than the first N pages: a partial overlay would make
    `pipeline._chunkable_layout` decline the document anyway, since its block stream has to cover
    every page that produced text — so a partial capture is pure cost for the same outcome.
    """
    monkeypatch.setenv("RAG_LAYOUT_CAPTURE_MAX_PAGES", "2")
    get_settings.cache_clear()
    pdf = make_pdf([["Clause 1. The contractor shall comply."]] * 3)

    extraction = extract_pdf_text(pdf, doc_id="over-cap", capture_layout=True)

    assert all(p.markup is None for p in extraction.pages), "no page may carry an overlay"
    assert extraction.layout_overlay().pages == ()
    # NOT a `degradations` reason: that channel means the extracted TEXT lost fidelity and it
    # drives `extraction_degraded` in provenance. Here the text is byte-identical and only an
    # optional overlay is missing, so a perfectly-extracted document must not report itself
    # degraded. The absence of markup is itself the signal every consumer already reads.
    plain = extract_pdf_text(pdf, doc_id="over-cap", capture_layout=False)
    assert [p.text for p in extraction.pages] == [p.text for p in plain.pages]
    assert provenance_tags(extraction) == provenance_tags(plain), "provenance must not claim a degradation"


def test_a_document_at_the_page_cap_still_captures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is inclusive, and it is a guard rather than a size policy — the default sits far
    above any real tender on purpose."""
    monkeypatch.setenv("RAG_LAYOUT_CAPTURE_MAX_PAGES", "3")
    get_settings.cache_clear()
    pdf = make_pdf([["Clause 1. The contractor shall comply."]] * 3)

    extraction = extract_pdf_text(pdf, doc_id="at-cap", capture_layout=True)

    assert all(p.markup is not None for p in extraction.pages)
