"""An unreadable upload is a 422 about the file, not a 500 about us (P9).

Every extractor ends in a third-party library, and each one raises its own exception type on
a file it cannot parse: pypdf a PdfStreamError, zipfile a BadZipFile, a missing archive part a
bare KeyError. None of those subclass PlatformError, so the middleware had nothing to map and the
caller got an opaque 500 — a server fault for what is actually a broken input.

The translation is deliberately NARROW. Only the "we have exhausted every extractor for this
format" path is converted; anything else still propagates, because turning our own bugs into
422s would blame the user for them.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import extract_docx_text, extract_pdf_text

from rag_common import PlatformError, ValidationError

_CORRUPT_PDF = b"%PDF-1.4\n" + b"\x00garbage" * 50
_CORRUPT_ZIP = b"PK\x03\x04" + b"\x00garbage" * 50


def _zip_of(names: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in names.items():
            z.writestr(name, body)
    return buf.getvalue()


def test_a_pdf_neither_extractor_can_open_is_a_validation_error() -> None:
    """pdfplumber failing to open already falls back to pypdf. When pypdf cannot read it
    either there is no extractor left, and the raw PdfStreamError reached the API as a 500."""
    with pytest.raises(ValidationError) as raised:
        extract_pdf_text(_CORRUPT_PDF, doc_id="broken.pdf")
    assert "broken.pdf" in str(raised.value)
    assert raised.value.http_status == 422


def test_a_corrupt_office_archive_is_a_validation_error() -> None:
    with pytest.raises(ValidationError) as raised:
        extract_docx_text(_CORRUPT_ZIP, doc_id="broken.docx")
    assert raised.value.http_status == 422


def test_an_archive_without_the_expected_part_is_a_validation_error() -> None:
    """A valid zip that is not a .docx — the KeyError this produced was the most opaque of
    the three, naming an internal path the uploader has never heard of."""
    with pytest.raises(ValidationError):
        extract_docx_text(_zip_of({"xl/workbook.xml": "<x/>"}), doc_id="sheet.xlsx")


def test_a_readable_pdf_is_untouched() -> None:
    """The translation must not swallow working files, or a real regression looks like a
    user error."""
    ex = extract_pdf_text(make_pdf([["Annual Leave Policy"]]), doc_id="fine.pdf")
    assert "Annual Leave" in ex.render().text


def test_every_raised_error_is_part_of_the_taxonomy() -> None:
    """The property the API middleware depends on: it maps PlatformError to a status and lets
    everything else become a 500."""
    for data, doc_id in ((_CORRUPT_PDF, "a.pdf"), (_CORRUPT_ZIP, "b.docx")):
        extract = extract_pdf_text if doc_id.endswith(".pdf") else extract_docx_text
        with pytest.raises(PlatformError):
            extract(data, doc_id=doc_id)
