"""Document classifier (FR-B4) + OCR preprocessing/confidence (FR-B1).

The classifier tests are pure (no OCR deps) and always run. The real-Tesseract tests skip
automatically when the binary/bindings aren't in the environment (they run in the prod/OCR
image), so `make test` stays infra-free.
"""

from __future__ import annotations

import shutil

import pytest
from svc_doc_intelligence.classify import (
    DEFAULT_SCANNED_CHARS_PER_PAGE,
    DocKind,
    classify,
    page_needs_ocr,
    sniff_magic,
)
from svc_doc_intelligence.ocr import reconstruct_paddle_page

_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c63f8cfc0f01f0005000101ffa0a25"
    "e0000000049454e44ae426082"
)
_PDF_HEAD = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
_JPEG_HEAD = b"\xff\xd8\xff\xe0\x00\x10JFIF"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_classify_by_mime() -> None:
    from ooxml_fixtures import make_docx

    assert classify("application/pdf", _PDF_HEAD) is DocKind.NATIVE_PDF
    # A real archive, because the kind now comes from the namelist: a bare PK signature is a
    # zip nobody can open, and calling that a Word document is how spreadsheets reached the
    # Word extractor in the first place.
    assert classify(_DOCX_MIME, make_docx(["hi"])) is DocKind.OFFICE_DOC
    assert classify(_DOCX_MIME, b"PK\x03\x04") is DocKind.ARCHIVE
    assert classify("text/html", b"<html>") is DocKind.HTML
    assert classify("image/png", _PNG_1x1) is DocKind.IMAGE
    assert classify("image/tiff", b"II*\x00") is DocKind.IMAGE
    assert classify("text/plain", b"hello") is DocKind.TEXT


def test_classify_falls_back_to_magic_bytes_when_mime_is_generic() -> None:
    # A file uploaded as octet-stream must still be identified from its bytes.
    assert classify("application/octet-stream", _PNG_1x1) is DocKind.IMAGE
    assert classify("application/octet-stream", _JPEG_HEAD) is DocKind.IMAGE
    assert classify("application/octet-stream", _PDF_HEAD) is DocKind.NATIVE_PDF
    assert classify("", b"\x00\x01\x02random") is DocKind.UNKNOWN


def test_sniff_magic_webp() -> None:
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    assert sniff_magic(webp) is DocKind.IMAGE


def test_page_needs_ocr_is_decided_per_page() -> None:
    threshold = DEFAULT_SCANNED_CHARS_PER_PAGE
    # An image page with no text layer needs OCR; a page with real text does not.
    assert page_needs_ocr(text_layer_chars=2, has_image_content=True, min_chars_per_page=threshold) is True
    assert page_needs_ocr(text_layer_chars=1600, has_image_content=True, min_chars_per_page=threshold) is False
    # No text and no image is a BLANK page, not a scan — OCR would find nothing.
    assert page_needs_ocr(text_layer_chars=0, has_image_content=False, min_chars_per_page=threshold) is False
    # Undetermined image content is investigated, not silently written off.
    assert page_needs_ocr(text_layer_chars=0, has_image_content=None, min_chars_per_page=threshold) is True


def test_scanned_threshold_default_matches_the_config_layer() -> None:
    """One documented default, two places that must state it. Drift here would mean the shipped
    behaviour silently differs from the value operators read in config.py/.env.example."""
    from rag_common.config import Settings

    assert Settings().ocr_scanned_chars_per_page == DEFAULT_SCANNED_CHARS_PER_PAGE


def _paddle_box(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_reconstruct_paddle_page_orders_and_scores() -> None:
    # PaddleOCR shape: [[ [box, (text, score)], ... ]] — deliberately OUT of reading order.
    raw = [
        [
            [_paddle_box(20, 200, 180, 230), ("TOTAL 50.16", 0.98)],
            [_paddle_box(20, 20, 300, 50), ("THE CITY OF NEW YORK", 0.95)],
            [_paddle_box(20, 110, 200, 140), ("ORANGE SLICE 1.40", 0.90)],
        ]
    ]
    text, conf = reconstruct_paddle_page(raw)
    # Restored top-to-bottom reading order.
    assert text.splitlines() == ["THE CITY OF NEW YORK", "ORANGE SLICE 1.40", "TOTAL 50.16"]
    # Mean of the per-line confidences.
    assert conf == pytest.approx((0.98 + 0.95 + 0.90) / 3)


def test_reconstruct_paddle_page_empty() -> None:
    assert reconstruct_paddle_page([[]]) == ("", 0.0)
    assert reconstruct_paddle_page(None) == ("", 0.0)


# --- Real OCR (skips unless tesseract + Pillow + pytesseract are installed) ---------------

_HAS_OCR = shutil.which("tesseract") is not None


@pytest.mark.skipif(not _HAS_OCR, reason="tesseract binary not installed in this environment")
def test_preprocess_and_ocr_reads_low_contrast_text() -> None:
    pytest.importorskip("pytesseract")
    from PIL import Image, ImageDraw
    from svc_doc_intelligence.ocr import TesseractOCR, preprocess_for_ocr

    # Render faint grey text on an off-white background — mimics a thermal receipt / halftone.
    img = Image.new("RGB", (900, 140), (238, 238, 238))
    ImageDraw.Draw(img).text((20, 45), "ORANGE SLICE 1.40 TOTAL 50.16", fill=(150, 150, 150))
    # Preprocessing must return a usable image and must not raise.
    assert preprocess_for_ocr(img) is not None

    import asyncio
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = asyncio.run(TesseractOCR(preprocess=True).ocr(buf.getvalue(), "image/png"))
    assert "TOTAL" in result.render().text.upper()
    assert 0.0 <= result.min_confidence <= 1.0  # real confidence, not a hardcoded 0.7
