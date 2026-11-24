"""The ingestion OCR check — does this document hold text that only OCR can read?

Every document is asked this at ingest, whatever its type, because a text layer and the text a
reader sees are not the same thing:

* an **image file** has no text layer at all — a photographed or scanned page, a ticket, a form;
* a **scanned PDF page** has none either (routed per page in ``DocIntelligenceService``);
* a **PDF page WITH a text layer** can still carry a pasted scan, a screenshot of a table or a
  chart whose words exist only as pixels;
* a **Word / PowerPoint / Excel file** stores pasted scans and screenshots as embedded media that
  the XML extractors never open.

Measured live before this module existed: an Emirates e-ticket uploaded as a JPEG was indexed as
32 characters of caption ("Flight Number: EK 602 and EK 605"); OCR, which read 4,544 characters
from the same image, was never called. Every image across two OCR knowledge bases had been
indexed the same way.

This module holds the pure decisions and conversions — which embedded images are worth OCR, and
turning an OCR engine's markup into indexable text. Running the engine stays in the service.
"""

from __future__ import annotations

import io
import posixpath
import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Why OCR was applied to a page. Recorded per page and rolled up onto the document's metadata,
# so "was this document OCR'd, and why?" is answerable from the catalog without re-reading it.
OCR_REASON_IMAGE_FILE = "image_file"  # the upload itself is an image
OCR_REASON_SCANNED_PAGE = "scanned_page"  # a PDF page with no usable text layer
OCR_REASON_PAGE_IMAGES = "page_images"  # "ocr" mode: every PDF page carrying an image
OCR_REASON_EMBEDDED_IMAGES = "embedded_images"  # large images inside a texty page or an Office file

# Below these, an embedded image is decoration — a logo, an icon, a bullet, a rule — and OCR
# would spend a model call reading "Acme" off a letterhead on every page. The operator's values
# come from config; these are the documented defaults.
DEFAULT_MIN_IMAGE_EDGE_PX = 200  # the image's SHORTER side, in its own pixels
DEFAULT_MIN_PAGE_SHARE = 0.10  # share of the PDF page's area the image is drawn over
DEFAULT_MAX_IMAGES_PER_DOC = 20  # a hard bound on OCR calls one document can trigger

_MARKUP_TAG = re.compile(
    r"<\s*/?\s*(?:div|p|span|table|thead|tbody|tr|td|th|h[1-6]|ul|ol|li|br|img|section|header|footer|a|b|strong|em|i)\b",
    re.IGNORECASE,
)
_BLANK_RUNS = re.compile(r"\n{3,}")
_BLOCK_TAGS = frozenset({"div", "p", "section", "header", "footer", "article", "ul", "ol", "table", "br"})


def image_worth_ocr(data: bytes, *, min_edge_px: int = DEFAULT_MIN_IMAGE_EDGE_PX) -> bool:
    """Whether an embedded image is big enough, in its own pixels, to hold readable text.

    Undecodable bytes (EMF/WMF vector art, a corrupt part) are NOT worth OCR: no engine here
    reads them, and a call that is certain to fail only adds a degradation stamp to a document
    that lost nothing."""
    try:
        from PIL import Image  # noqa: PLC0415 - heavy dep, imported on use

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        return False
    return min(width, height) >= min_edge_px


@dataclass(frozen=True, slots=True)
class PdfImageBox:
    """One image drawn on a PDF page, in the page's own coordinates (points, top-left origin)."""

    x0: float
    top: float
    x1: float
    bottom: float
    src_width: int  # the image's intrinsic pixel size — a 2x2 swatch scaled up is not a scan
    src_height: int

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.bottom - self.top)


def pdf_images_worth_ocr(
    boxes: list[PdfImageBox],
    *,
    page_width: float,
    page_height: float,
    min_page_share: float = DEFAULT_MIN_PAGE_SHARE,
    min_edge_px: int = DEFAULT_MIN_IMAGE_EDGE_PX,
) -> list[PdfImageBox]:
    """The images on a TEXTY PDF page that may carry words the text layer does not.

    Both conditions are required. Page share alone would OCR a low-resolution banner stretched
    across the header; pixel size alone would OCR a high-resolution logo printed thumbnail-small.
    A pasted scan or screenshot is both large on the page and large in its own pixels."""
    page_area = page_width * page_height
    if page_area <= 0:
        return []
    return [
        box
        for box in boxes
        if box.area / page_area >= min_page_share and min(box.src_width, box.src_height) >= min_edge_px
    ]


# Resolution a cropped image region is rendered at before OCR. Chandra reads 200 DPI page
# renders elsewhere in this service (ChandraOCREngine._to_images), so a crop matches it.
_CROP_DPI = 200


def pdf_page_image_crops(
    raw: bytes,
    page_nos: list[int],
    *,
    min_page_share: float = DEFAULT_MIN_PAGE_SHARE,
    min_edge_px: int = DEFAULT_MIN_IMAGE_EDGE_PX,
) -> dict[int, list[bytes]]:
    """PNG renders of the images worth OCR on the given 1-based PDF pages.

    Only the IMAGE REGION is rendered, never the whole page: the page's text layer is exact and
    already extracted, and OCR over it would add a second, fallible copy of every sentence (and
    live, Chandra misread "EK 602" as "EX 602" — a text layer never does). Synchronous and
    CPU-bound; the service runs it off the event loop."""
    import pdfplumber  # noqa: PLC0415 - heavy dep, imported on use

    wanted = set(page_nos)
    crops: dict[int, list[bytes]] = {}
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if page_no not in wanted:
                continue
            boxes = []
            for img in page.images:
                src = img.get("srcsize") or (0, 0)
                boxes.append(
                    PdfImageBox(
                        x0=max(0.0, float(img["x0"])),
                        top=max(0.0, float(img["top"])),
                        x1=min(float(page.width), float(img["x1"])),
                        bottom=min(float(page.height), float(img["bottom"])),
                        src_width=int(src[0] or 0),
                        src_height=int(src[1] or 0),
                    )
                )
            worth = pdf_images_worth_ocr(
                boxes,
                page_width=float(page.width),
                page_height=float(page.height),
                min_page_share=min_page_share,
                min_edge_px=min_edge_px,
            )
            for box in worth:
                rendered = page.crop((box.x0, box.top, box.x1, box.bottom)).to_image(resolution=_CROP_DPI).original
                buf = io.BytesIO()
                rendered.save(buf, format="PNG")
                crops.setdefault(page_no, []).append(buf.getvalue())
    return crops


def image_mime(data: bytes) -> str:
    """The MIME an OCR engine should be told for these image bytes (the vision API needs it)."""
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "PNG").lower()
    except Exception:
        return "image/png"
    return "image/jpeg" if fmt in ("jpeg", "mpo") else f"image/{fmt}"


# --- embedded media in Office files --------------------------------------------------------

_RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_MEDIA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp")


def _media_targets(z: object, part: str) -> list[str]:
    """The media files a part references through its relationship table, in table order."""
    import zipfile  # noqa: PLC0415

    from defusedxml import ElementTree as SafeET  # noqa: PLC0415 - untrusted archive XML

    assert isinstance(z, zipfile.ZipFile)
    folder, name = posixpath.split(part)
    rels_part = posixpath.join(folder, "_rels", f"{name}.rels")
    try:
        root = SafeET.fromstring(z.read(rels_part))
    except (KeyError, SafeET.ParseError):
        return []
    targets: list[str] = []
    for rel in root.iter(f"{_RELS_NS}Relationship"):
        target = rel.get("Target") or ""
        if rel.get("TargetMode") == "External" or not target.lower().endswith(_MEDIA_EXTENSIONS):
            continue
        resolved = posixpath.normpath(posixpath.join(folder, target))
        if resolved not in targets:
            targets.append(resolved)
    return targets


def office_embedded_images(raw: bytes, *, slide_parts: list[str] | None = None) -> dict[int, list[bytes]]:
    """Embedded images in an OOXML file, keyed by the 0-based PAGE index they belong to.

    A slide's images land on that slide, so a citation to text read from a pasted screenshot
    names the slide it was pasted on. Word and Excel have no page the extractor could honestly
    attribute an image to, so their images are keyed to page 0 of their own record list and the
    caller appends them after the body. ``slide_parts`` is the deck's slide order as the pptx
    extractor resolved it; when it is None the whole archive's media is collected."""
    import zipfile  # noqa: PLC0415

    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return {}
    with z:
        names = set(z.namelist())
        if slide_parts is not None:
            keyed: dict[int, list[bytes]] = {}
            for index, part in enumerate(slide_parts):
                images = [z.read(t) for t in _media_targets(z, part) if t in names]
                if images:
                    keyed[index] = images
            return keyed
        media = sorted(n for n in names if "/media/" in n and n.lower().endswith(_MEDIA_EXTENSIONS))
        return {0: [z.read(n) for n in media]} if media else {}


# --- OCR engine markup -> indexable text -------------------------------------------------


class _MarkupToText(HTMLParser):
    """HTML as an OCR engine emits it, rendered as the text a reader would see.

    Tables become Markdown tables rather than a flat run of cell text: the platform chunks,
    embeds and cites table rows, and "E-TICKET NUMBER | 176 2139738899" keeps the label beside
    its value where a flattened cell stream would not."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._prefix = ""

    def _emit(self, text: str) -> None:
        if self._cell is not None:
            self._cell.append(text)
        else:
            self._out.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._out.append("\n\n")
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag in _BLOCK_TAGS:
            self._emit("\n")
        elif re.fullmatch(r"h[1-6]", tag):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._emit(f" [Image: {alt}] ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()).replace("|", "/"))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._rows is not None:
            if any(c for c in self._row):
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            self._out.append(_markdown_table(self._rows))
            self._out.append("\n\n")
            self._rows = None
        elif tag in _BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        self._emit(data)

    def text(self) -> str:
        # Table rows keep their cell padding; every other line has runs of whitespace collapsed.
        lines = [
            line.strip() if line.lstrip().startswith("|") else " ".join(line.split())
            for line in "".join(self._out).splitlines()
        ]
        return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join(lines)


def normalize_ocr_markup(text: str) -> str:
    """OCR output as indexable text: HTML markup rendered out, anything else left untouched.

    Chandra is asked for Markdown and answers in HTML. Measured on the live corpus, 18 chunks
    across two knowledge bases were indexed carrying raw ``<div>``/``<td>`` tags — markup
    that spends a 700-character chunk budget on angle brackets and puts "td" and "tr" into
    keyword search. Output that carries no recognisable markup (Markdown, plain transcription)
    is returned unchanged, so an engine that already obeys the prompt is not rewritten."""
    if not _MARKUP_TAG.search(text):
        return text
    parser = _MarkupToText()
    parser.feed(text)
    parser.close()
    return parser.text()


# --- degenerate-generation guard ---------------------------------------------------------------
# A vision model given a page it cannot read does not raise: it transcribes what it can and then
# repeats one short cycle until its token budget is gone. Measured live on a handwritten chemistry
# worksheet: ~350 characters of real formulas followed by "ce " 1,500 times. Indexing that makes
# the page look transcribed, poisons retrieval with a meaningless token, and (at the platform's
# full token budget) takes long enough that the model gateway times the ingest out entirely.

# The cycle a degenerate loop repeats is short; a longer "repeat" is usually a real duplicated
# paragraph, which is the document's content and not ours to delete.
_MAX_CYCLE_WORDS = 8
# How many back-to-back repetitions stop being a document and start being a loop. Real pages do
# repeat: a table column of "N/A", a form of empty ticks. Twelve consecutive identical cycles with
# nothing after them is not that.
_MIN_CYCLE_REPEATS = 12
# Below this there is not enough text to tell a loop from a caption.
_MIN_WORDS_TO_JUDGE = 24


def trim_degenerate_tail(text: str) -> tuple[str, bool]:
    """Drop a repeated-cycle tail from OCR output. Returns ``(text, trimmed)``.

    Only a tail is trimmed, and only when the repetition runs to the END of the text: a cycle that
    repeats and is then followed by more document is how tables and forms legitimately read, and
    cutting there would delete what the page actually says. Everything before the loop is kept
    verbatim — that part is the model's real reading of the page."""
    words = text.split()
    if len(words) < _MIN_WORDS_TO_JUDGE:
        return text, False
    for size in range(1, _MAX_CYCLE_WORDS + 1):
        cycle = words[-size:]
        repeats = 0
        index = len(words)
        while index >= size and words[index - size : index] == cycle:
            repeats += 1
            index -= size
        if repeats >= _MIN_CYCLE_REPEATS:
            kept = " ".join(words[:index]).rstrip()
            return kept, True
    return text, False
