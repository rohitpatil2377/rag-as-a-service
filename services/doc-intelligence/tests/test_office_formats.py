"""Office archives are routed by what is IN them, not by their first four bytes (P8).

`.docx`, `.xlsx`, `.pptx`, `.odt` and `.epub` are all zip containers, so they all start
`PK\\x03\\x04`. The magic-byte table mapped that one signature to a single OFFICE kind, which
the service handed to the Word extractor — so every spreadsheet and deck died on
`KeyError: no item named 'word/document.xml'`, surfacing as a 500.

A zip's identity is in its namelist, and the bytes are already in hand at classification time.
"""

from __future__ import annotations

import pytest
from ooxml_fixtures import make_docx, make_epub, make_pptx, make_xlsx
from svc_doc_intelligence.classify import DocKind, classify
from svc_doc_intelligence.ocr import CAP_TABLES, CAP_TEXT, extract_office_text

from rag_common import ValidationError

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_ROWS = {"Leave": [["Grade", "Days"], ["Officer", "24"], ["Manager", "30"]]}


# --- classification -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (make_docx(["hello"]), DocKind.OFFICE_DOC),
        (make_xlsx(_ROWS), DocKind.OFFICE_SHEET),
        (make_pptx([["Slide one"]]), DocKind.OFFICE_SLIDES),
    ],
)
def test_a_zip_is_identified_from_its_contents_not_its_signature(data: bytes, expected: DocKind) -> None:
    """Declared MIME absent (a generic upload) — only the bytes can answer."""
    assert classify("application/octet-stream", data) is expected


def test_the_declared_mime_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    assert classify(_XLSX_MIME, make_xlsx(_ROWS)) is DocKind.OFFICE_SHEET
    assert classify(_PPTX_MIME, make_pptx([["a"]])) is DocKind.OFFICE_SLIDES
    assert classify(_DOCX_MIME, make_docx(["a"])) is DocKind.OFFICE_DOC


def test_a_declared_mime_that_contradicts_the_bytes_loses() -> None:
    """A spreadsheet mislabelled as a Word document is the exact upload that used to 500.
    The contents are the ground truth — a wrong label must not pick the wrong extractor."""
    assert classify(_DOCX_MIME, make_xlsx(_ROWS)) is DocKind.OFFICE_SHEET


# --- extraction -----------------------------------------------------------------------------


def test_a_workbook_extracts_as_self_describing_rows() -> None:
    ex = extract_office_text(make_xlsx(_ROWS), doc_id="leave.xlsx")
    text = ex.render().text
    # Through the SHARED serializer, so a spreadsheet row and a PDF table row read alike.
    assert "Grade = Officer | Days = 24" in text
    assert "Grade = Manager | Days = 30" in text
    assert CAP_TABLES in ex.capabilities  # a spreadsheet is row/column structure by definition


def test_shared_strings_are_resolved() -> None:
    """Excel stores cell text in a shared-string table and puts an INDEX in the cell. A reader
    that takes the cell value literally indexes the corpus with integers."""
    assert "0" not in extract_office_text(make_xlsx({"S": [["Officer"]]}), doc_id="s.xlsx").render().text


def test_sheets_are_read_in_workbook_order_not_filename_order() -> None:
    """Sheet order lives in the relationship table, not in the part names. The fixture numbers
    the parts backwards on purpose, so a reader that sorts filenames reads the workbook
    inside out — and every citation would name the wrong sheet."""
    ex = extract_office_text(
        make_xlsx({"First": [["alpha"]], "Second": [["beta"]], "Third": [["gamma"]]}), doc_id="w.xlsx"
    )
    assert [p.page_no for p in ex.pages] == [1, 2, 3]
    assert "alpha" in ex.pages[0].text and "gamma" in ex.pages[2].text


def test_each_slide_is_its_own_page() -> None:
    """A deck genuinely paginates, so slides map to pages and a citation can name one."""
    ex = extract_office_text(make_pptx([["Agenda"], ["Budget", "FY26"]]), doc_id="deck.pptx")
    assert len(ex.pages) == 2
    assert "Agenda" in ex.pages[0].text
    assert "Budget" in ex.pages[1].text and "FY26" in ex.pages[1].text
    assert ex.capabilities == (CAP_TEXT,)  # slide text only; no table structure claimed


def test_slides_are_read_in_presentation_order() -> None:
    ex = extract_office_text(make_pptx([["one"], ["two"], ["three"]]), doc_id="d.pptx")
    assert [p.text.strip() for p in ex.pages] == ["one", "two", "three"]


def test_a_word_document_still_works() -> None:
    ex = extract_office_text(make_docx(["Annual leave is 24 days."]), doc_id="memo.docx")
    assert "Annual leave is 24 days." in ex.render().text


# --- the archives we do not extract ----------------------------------------------------------


def test_an_unsupported_archive_is_named_rather_than_misparsed() -> None:
    """An .epub or .odt is a zip too. It must not be pushed through the Word extractor to die
    on a missing part — the uploader gets told what we think it is and what we support."""
    with pytest.raises(ValidationError) as raised:
        extract_office_text(make_epub(), doc_id="book.epub")
    message = str(raised.value)
    assert "book.epub" in message
    assert raised.value.http_status == 422
