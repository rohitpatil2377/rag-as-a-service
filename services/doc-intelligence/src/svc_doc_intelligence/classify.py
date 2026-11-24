"""Document-type classifier (FR-B4) — decide the BEST extraction path per file.

One extractor cannot handle every input: a born-digital PDF has a real text layer (read it
directly, perfectly), a scanned PDF or a photo/PNG has none (must be OCR'd), an Office doc is
zipped XML, HTML is tags. This module identifies the kind from the declared MIME *and* the
raw magic bytes (so a mislabeled upload — e.g. a PNG sent as octet-stream — still routes
right), and the service maps each kind to its extractor. It is the seam where a learned
classifier (visual layout / ColPali) can later replace the heuristic with no call-site change.
"""

from __future__ import annotations

import io
from enum import StrEnum

# Below this many real characters, a PDF page has no usable text layer. Operator-tunable via
# RAG_OCR_SCANNED_CHARS_PER_PAGE (see rag_common.config); this is the documented default and
# the single source of it — the Settings field must not drift from it (pinned by a test).
DEFAULT_SCANNED_CHARS_PER_PAGE = 20


class DocKind(StrEnum):
    """The extraction path a document should take."""

    NATIVE_PDF = "native_pdf"  # born-digital PDF with a text layer -> direct extract
    SCANNED_PDF = "scanned_pdf"  # image-only PDF -> rasterise + OCR
    IMAGE = "image"  # png/jpg/tiff/... -> OCR
    # The three OOXML families. They are separate kinds because they need separate extractors:
    # a single "office" kind is what sent every workbook and deck to the Word extractor.
    OFFICE_DOC = "office_doc"  # .docx -> paragraphs + tables
    OFFICE_SHEET = "office_sheet"  # .xlsx -> sheets of rows
    OFFICE_SLIDES = "office_slides"  # .pptx -> slides of text
    ARCHIVE = "archive"  # a zip container we have no extractor for (.odt/.epub/...)
    HTML = "html"  # web page -> visible-text extract
    TEXT = "text"  # plain text
    UNKNOWN = "unknown"  # fall back to OCR (safest for an unknown binary)


_IMAGE_MIME_PREFIX = "image/"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")  # normal, empty, spanned

# Which archive MEMBER identifies each OOXML family. A zip's identity is in its namelist, not
# in its first four bytes — .docx, .xlsx, .pptx, .odt and .epub share the same signature, so
# the signature can only ever say "this is a zip". Each of these is the part the format's own
# spec requires, so the test is the format's definition rather than a guess.
_OOXML_MARKERS: tuple[tuple[str, DocKind], ...] = (
    ("word/document.xml", DocKind.OFFICE_DOC),
    ("xl/workbook.xml", DocKind.OFFICE_SHEET),
    ("ppt/presentation.xml", DocKind.OFFICE_SLIDES),
)

_EXACT_MIME: dict[str, DocKind] = {"application/pdf": DocKind.NATIVE_PDF, "text/html": DocKind.HTML}

# MIME types that CLAIM to be Office. Which family they are is not taken from the label — the
# namelist decides that — so only membership matters here. Their purpose is to stop the
# `text/` and `image/` prefix rules from firing on a mislabelled or legacy binary Office file.
_OFFICE_MIME = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)

# Magic-byte signatures — used when the MIME is missing/generic (application/octet-stream).
_MAGIC: tuple[tuple[bytes, DocKind], ...] = (
    (b"%PDF", DocKind.NATIVE_PDF),  # PDF native-vs-scanned is refined later once text is known
    (b"\x89PNG\r\n\x1a\n", DocKind.IMAGE),
    (b"\xff\xd8\xff", DocKind.IMAGE),  # JPEG
    (b"GIF87a", DocKind.IMAGE),
    (b"GIF89a", DocKind.IMAGE),
    (b"II*\x00", DocKind.IMAGE),  # little-endian TIFF
    (b"MM\x00*", DocKind.IMAGE),  # big-endian TIFF
    (b"BM", DocKind.IMAGE),  # BMP
)


def looks_like_zip(raw: bytes) -> bool:
    return raw[:4] in _ZIP_SIGNATURES


def sniff_archive(raw: bytes) -> DocKind:
    """Which OOXML family is this zip, read from its namelist.

    Returns ``ARCHIVE`` for a zip that is a real archive but not one we extract (.odt, .epub,
    a plain .zip), so the caller can say what it is instead of feeding it to the wrong parser.
    A zip we cannot even open is also ``ARCHIVE``: the extractor raises the actionable error,
    because that is where the document id is."""
    import zipfile  # noqa: PLC0415 - stdlib but only needed on the archive path

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = set(z.namelist())
    except Exception:
        return DocKind.ARCHIVE
    for marker, kind in _OOXML_MARKERS:
        if marker in names:
            return kind
    return DocKind.ARCHIVE


def sniff_magic(raw: bytes) -> DocKind | None:
    """Identify a file from its leading bytes (RIFF/WEBP handled specially), or None.

    A zip is resolved further by :func:`sniff_archive` — its signature alone cannot tell a
    spreadsheet from a Word document, and pretending otherwise is what routed both to the
    same extractor."""
    head = raw[:16]
    if head[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return DocKind.IMAGE
    for signature, kind in _MAGIC:
        if head.startswith(signature):
            return kind
    if looks_like_zip(raw):
        return sniff_archive(raw)
    return None


def classify(mime: str, raw: bytes) -> DocKind:
    """Coarse kind from the declared MIME, then magic bytes as a fallback/override.

    PDFs come back as ``NATIVE_PDF`` here; whether a PDF is actually scanned is only knowable
    after trying to read its text layer — call :func:`is_scanned_pdf` for that refinement.

    For a zip the CONTENTS win over the declared MIME. A mislabelled upload is common (a
    spreadsheet sent as ``application/msword``, or anything sent as octet-stream) and the
    namelist is ground truth, so trusting the label would keep choosing the wrong extractor
    for exactly the files that used to fail.
    """
    if looks_like_zip(raw):
        return sniff_archive(raw)
    mime = (mime or "").lower().strip()
    if mime in _EXACT_MIME:
        return _EXACT_MIME[mime]
    # An Office MIME on bytes that are not a zip is a legacy binary .doc/.xls or a mislabel.
    # Either way the prefix rules below would be wrong about it, so skip straight to the bytes.
    if mime not in _OFFICE_MIME:
        if mime.startswith(_IMAGE_MIME_PREFIX):
            return DocKind.IMAGE
        if mime.startswith("text/"):
            return DocKind.TEXT
    # MIME was generic, unknown or contradicted — trust the bytes.
    return sniff_magic(raw) or DocKind.UNKNOWN


def page_needs_ocr(*, text_layer_chars: int, has_image_content: bool | None, min_chars_per_page: int) -> bool:
    """Does THIS page need OCR? Decided per page, never as a document average.

    Averaging over the document is what let a diagram-heavy but born-digital PDF be pushed
    wholesale through OCR: one rich page hid ten empty ones, or ten empty ones condemned two
    good ones. Two independent conditions, both necessary:

    * the page's own text layer is below ``min_chars_per_page`` (the operator's threshold), and
    * there is something for OCR to read. A page with no text AND no image is BLANK — a duplex
      back or a separator sheet — and OCR would burn CPU to produce nothing while stamping a
      degradation on it. ``None`` (undetermined) counts as "might have ink", so an unknown page
      is investigated rather than silently written off.
    """
    return text_layer_chars < min_chars_per_page and has_image_content is not False
