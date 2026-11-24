"""Text extraction + OCR seam (FR-B1/B2/B3).

Phase-0 extracts born-digital PDF/text directly (pypdf). Scanned documents route to
an :class:`OCREngine`; the production engine is PaddleOCR/docTR + Indic models on the
GPUaaS pool (TDD §4.2), provided as an optional adapter. The zero-GPU dev image uses
:class:`NoOpOCR`, so the walking skeleton runs on digital text without a GPU.

Per-field OCR confidence (FR-B1) is surfaced via ``min_confidence`` on the result;
digital extraction reports full confidence.
"""

from __future__ import annotations

import io
import json
import re
import statistics
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any, ClassVar, Protocol, runtime_checkable

from rag_common import ConfigError, ValidationError, get_logger, get_settings
from rag_schemas.common import Modality
from svc_doc_intelligence.classify import DocKind, sniff_archive
from svc_doc_intelligence.ocr_check import normalize_ocr_markup, trim_degenerate_tail
from svc_doc_intelligence.tables import LABEL_SEP, serialize_table

_log = get_logger(__name__)

_MIN_PAGES_FOR_EDGE_STRIP = 3  # header/footer detection needs a few pages to be reliable
# What share of pages a line must appear on, at the same edge, before it counts as furniture
# rather than content. Deliberately not a deployment knob, for the same reason PAGE_SEPARATOR
# is not: it decides what text is STORED, so changing it on a running install would silently
# rewrite every chunk indexed afterwards while leaving the earlier ones alone.
_FURNITURE_PAGE_SHARE = 0.6
_MAX_EDGE_BLOCK_LINES = 5  # a masthead is a few lines; beyond this we would be eating the page

# Tesseract likes ~300 DPI; images smaller than this on the long edge OCR poorly, so we
# upscale them first. Big enough to help small scans without blowing up memory on large ones.
_OCR_UPSCALE_TO_PX = 1600
_OCR_AUTOCONTRAST_CUTOFF = 2  # % of darkest/lightest pixels to clip — fights faint thermal print
_INK_LEVEL = 128  # grayscale value below which a pixel counts as "ink" (for deskew/threshold)
_MIN_INK_PIXELS_FOR_DESKEW = 50  # too few dark pixels -> nothing to align, skip deskew
_DESKEW_ANGLE_FLOOR = -45.0  # minAreaRect angle wrap-around boundary
_MIN_DESKEW_DEGREES = 0.5  # don't resample the page for a negligible tilt


# --- Extraction provenance -----------------------------------------------------------------
# Extractors are not interchangeable: some capture tables, some only a text layer. When we fall
# back from one to another the text still arrives, so the loss is invisible unless we record it.
# These names are a STABLE QUERY VOCABULARY — operators count the corpus by them — so they are
# deliberately NOT env-tunable: renaming one would silently break every saved query.
#
# Both vocabularies are OPEN. An extractor declares its own id and the capabilities it actually
# delivered; a failure path declares its own reason code. Nothing below enumerates a closed set,
# so adding an extractor or a degradation reason is a declaration, not a change to this code.
META_EXTRACTOR = "extraction_extractor"  # id of the extractor that produced the STORED text
META_CAPABILITIES = "extraction_capabilities"  # capabilities it actually delivered (sorted, comma-sep)
META_DEGRADED = "extraction_degraded"  # "true"/"false" — did anything downgrade on the way here
META_DEGRADATION = "extraction_degradation"  # reason codes (sorted, comma-sep); "" when clean

# Observed table content. Distinct from CAP_TABLES, which is what the extractor COULD recover:
# these say what it DID. Same boolean-for-filtering + detail-alongside shape as degraded/
# degradation above. Together they answer, without re-extracting anything, (a) how much of the
# corpus has tables and (b) whether tables sit on consecutive pages — the observable proxy for
# a table crossing a page break, since a table's position within its page is a coordinate and
# so belongs in an overlay rather than on the record.
META_TABLES_FOUND = "extraction_tables_found"  # "true"/"false" — any table on any page
META_TABLE_PAGES = "extraction_table_pages"  # how many pages had a table
META_TABLE_RUNS = "extraction_table_runs"  # maximal runs of CONSECUTIVE table pages
# Share of the document's source characters that table detection CLAIMED. A detected table's
# characters are removed from the narrative and replaced by rendered rows, so a false positive
# does not add noise beside the prose — it restructures it, invisibly. This is the number that
# makes that findable: a page that is 4% table is a table on a page; a page reported as 95%
# table that reads like prose is a detection failure. Deliberately an OBSERVATION, not a
# classifier — the extractor already decides ownership per character in order to do the splice.
META_TABLE_CHAR_SHARE = "extraction_table_char_share"
_TABLE_SHARE_DP = 4  # stored precision — a screening signal, not a measurement
# Running header/footer lines removed across the document. An observation like the table tags:
# it says what normalization DID, so an over-eager strip is findable without re-extracting.
META_BOILERPLATE_LINES = "extraction_boilerplate_lines"
# The ingestion OCR check, rolled up (see svc_doc_intelligence.ocr_check). Always emitted, so a
# re-ingest can never leave a stale "OCR'd" stamp on a document the check no longer OCRs.
META_OCR_APPLIED = "ocr_applied"  # "true"/"false" — did OCR contribute ANY of the stored text
META_OCR_REASONS = "ocr_reasons"  # why (sorted, comma-sep): image_file, scanned_page, embedded_images...
META_OCR_PAGES = "ocr_pages"  # how many pages OCR contributed text to
META_OCR_IMAGES = "ocr_images"  # how many embedded images were read by OCR
# pages/runs is the mean run length: 1.0 => tables never sit on adjacent pages; >1.0 => some do.

_META_SEP = ","  # separator for the multi-valued tags above (metadata values are strings)

# Capability tokens. Ask "what could this extractor recover from the source", not "which library".
CAP_TEXT = "text"  # a linear text layer
# NOTE: an ABILITY, not an observation — "this path preserves row/column structure where the
# page has any". A page with no table still has it; whether one was actually found is
# PageRecord.tables_found. Conflating the two would make the document-level intersection
# collapse to "has a table on every page", which is not what any caller is asking.
CAP_CAPTION = "caption"  # a vision model's description of what the image shows
CAP_TABLES = "tables"  # row/column structure preserved as retrievable rows

# Degradation reason codes.
REASON_EXTRACTOR_FALLBACK = "extractor_fallback"  # the preferred extractor failed; a lesser one took over
REASON_OCR_UNAVAILABLE = "ocr_unavailable"  # page has no text layer and no OCR engine is configured
REASON_OCR_FAILED = "ocr_failed"  # page needed OCR and the engine raised on it
# More embedded images qualified for OCR than one document may spend; the rest were not read.
REASON_OCR_IMAGE_CAP = "ocr_image_cap_reached"
# A page OCR'd on its own cannot take part in cross-page work (running header/footer detection
# needs the whole page set), so its text keeps boilerplate its neighbours had removed.
REASON_ISOLATED_NORMALIZATION = "isolated_normalization"
# Image cleaning is best-effort — it must never stop OCR — but a page read from an image that
# was NOT cleaned is a real accuracy loss on faint or skewed scans, and nothing downstream can
# tell. These say so. Note there is deliberately no code for "OpenCV is not installed": that is
# a deployment choice visible in the image manifest, and stamping every page of every document
# would bury the pages that actually lost something.
REASON_OCR_PREPROCESS_FAILED = "ocr_preprocess_failed"  # OCR read the raw, uncleaned image
REASON_OCR_DESKEW_FAILED = "ocr_deskew_failed"  # cv2 was present and raised on this page
# The model stopped reading and started repeating; the looping tail was dropped (see
# ocr_check.trim_degenerate_tail). What precedes it is kept — this says part of the page is gone.
REASON_OCR_DEGENERATE = "ocr_degenerate_output"
# The OCR engine rejected accuracy-tuning options (e.g. angle classification) — it still read
# the page, but with fewer of them applied than the deployment asked for.
REASON_OCR_TUNING_DROPPED = "ocr_tuning_dropped"


# --- Page joining -------------------------------------------------------------------------
# Inserted between pages when the document is rendered to one string. NOT env-tunable, and not
# a free choice: it is load-bearing for chunking. `rag_core.chunking._CLAUSE_RE` is
# ``(?<=[.;:\n])\s+`` — it has no blank-line rule; it splits on a whitespace RUN whose first
# character satisfies the lookbehind. So "\n\n" splits (the first \n is the lookbehind for the
# second) while a single "\n" after a word does NOT, and page breaks would stop being chunk
# boundaries. Anything shorter than two whitespace chars starting with a newline silently
# breaks that. Changing it is a code change with a test, not a deployment knob.
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class PageRecord:
    """One extracted page: its text, its page number, and how that text was produced.

    Provenance lives HERE, not only on the document, so a future per-page fallback or per-page
    OCR route records itself at the granularity it happens at; the document-level view is
    derived from these (see :class:`Extraction`) rather than set independently alongside them."""

    page_no: int  # 1-based, as printed and as cited
    text: str  # AUTHORITATIVE linear reading-order rendering of the page — never derived
    extractor: str  # open id, e.g. "pdfplumber" / "pypdf" / "tesseract"
    # What this path was ABLE to recover from this page (CAP_* tokens) — an ability, not a
    # census of what was there. `tables_found` below is the observation.
    capabilities: tuple[str, ...]
    degradations: tuple[str, ...] = ()  # open reason codes; empty => nothing was lost
    confidence: float = 1.0  # per-page confidence in [0,1] (FR-B1); digital extraction is exact
    # Does the page draw any image? A ROUTING SIGNAL, not layout data: a page below the text
    # threshold is only worth OCR if it has ink to read. None = undetermined (treated as "may
    # need OCR"). Deliberately a flat boolean — structural information (table/column regions)
    # belongs in a separate overlay alongside `text`, never in a shape `text` is rebuilt from.
    has_image_content: bool | None = None
    # Separate from has_image_content ON PURPOSE. has_image_content means "a raster picture is
    # carrying this page's content", and the pdf_image_pages policy uses it to decide whether to
    # OCR a page that already has text. This one means "there is SOMETHING here for OCR to read
    # at all", including glyphs traced into vector outlines, and only the blank-page test
    # consults it. Conflating them made every ruled table look like a page that needed OCR.
    has_vector_ink: bool = False
    # Was a table actually recovered from THIS page? Observed from the rows the serializer
    # actually emitted, not declared. A scalar fact about the page; WHERE the table sat is
    # deliberately absent — coordinates belong in an overlay, not here.
    tables_found: bool = False
    # How many of the page's source characters table detection claimed, and how many there
    # were. Two COUNTS rather than one ratio, because the document-level share has to be
    # derived exactly: averaging per-page ratios would weight a sparse page like a dense one.
    # Still scalars, so the record stays flat (no coordinates — see the note above).
    table_chars: int = 0
    source_chars: int = 0
    # Running header/footer lines removed from this page. An OBSERVATION, not a degradation:
    # the strip succeeded, and marking every page of every long document "degraded" would bury
    # the pages that genuinely lost something (see the note on the reason codes above).
    boilerplate_lines: int = 0
    # Why OCR contributed to this page ("" when it did not) — one of ocr_check.OCR_REASON_*. A
    # scalar on the page for the same reason as the fields above: the document-level view is
    # derived from pages, so it cannot claim OCR ran where no page says it did.
    ocr_reason: str = ""
    # How many EMBEDDED images on this page OCR read (0 for a page OCR'd as a whole).
    ocr_images: int = 0
    # The engine's RAW markup for this page, or None when the engine emits none. An OVERLAY
    # alongside `text`, never an input to it — exactly what the `has_image_content` note above
    # prescribes: "structural information (table/column regions) belongs in a separate overlay
    # alongside `text`, never in a shape `text` is rebuilt from". `text` stays AUTHORITATIVE and
    # byte-identical; nothing downstream rebuilds it from here.
    #
    # BOTH Chandra engines set it — `ChandraOCREngine` (self-hosted chat) and
    # `DatalabOCREngine` (hosted /convert) — and each declares its own `markup_format` below,
    # because the two emit DIFFERENT languages under the same extractor id. Neither was asked for
    # anything new: the markup already arrived and was flattened and discarded on the line that
    # built `text`. The BORN-DIGITAL PDF path sets it too, in a third language
    # (:data:`MARKUP_FORMAT_LAYOUT_JSON`) and only when asked to — see
    # :func:`_read_primary_page`. Every remaining path — pypdf, Tesseract, Paddle, Office —
    # produces plain text and leaves this None. None therefore means "this engine emits no
    # markup", NOT "the page was blank": a markup engine that returned nothing for a page records
    # "" (or, for layout-json, a payload with no blocks) and stays distinguishable from both.
    #
    # Stored EXACTLY as the engine returned it: post the client's `_sanitize` (so NUL/C0 are
    # already stripped) and PRE `normalize_ocr_markup`. `text` IS derived from this same engine
    # answer — that is where it has always come from, and the VERIFIED claim is narrower: adding
    # this field changed nothing about that derivation (`normalize_ocr_markup` +
    # `trim_degenerate_tail`, asserted byte-identical in the tests). The two then diverge further
    # downstream — `text` is normalised per page and may have a vision caption appended
    # (`service._route_image`) — so they are not interchangeable, and `text` must never be
    # REBUILT from here.
    #
    # HONEST LIMIT, and it is PER PATH, not per field. A chat-shaped VLM returns no coordinates,
    # so CHANDRA's markup carries no bounding boxes: `blocks.bbox` (Phase 1) stays NULL on that
    # path and a parser has only the `<h1>`-`<h6>`/`<b>` tags as typography cues, never font size
    # or weight. The born-digital path is the opposite case — `page.chars` carry `fontname`,
    # `size` and coordinates per glyph — which is why its markup is a different language rather
    # than more HTML: it has strictly more to say. See :data:`MARKUP_FORMAT_LAYOUT_JSON`.
    markup: str | None = None
    # Which markup language `markup` is written in — "" when there is no markup. An engine that
    # sets `markup` MUST set this, because the extractor id does not determine it: BOTH Chandra
    # engines stamp `OCR_EXTRACTOR_CHANDRA`, and the self-hosted chat path answers HTML while the
    # hosted Datalab /convert path is asked for (and answers) MARKDOWN. A consumer keying off the
    # extractor would be wrong on a `chandra_api` deployment, so it must branch on this instead.
    # It is what the engine was asked for or observably answers, not a verified parse of the
    # bytes — a consumer should tolerate the other, as `normalize_ocr_markup` already does.
    #
    # It is PER PAGE, and that is load-bearing rather than incidental: a tender with scanned
    # annexures appended has born-digital pages declaring `layout-json` and OCR'd pages declaring
    # `html`, in ONE document, each page saying which language it is in. A document-level format
    # could not describe that file at all.
    markup_format: str = ""  # "html" | "markdown" | "layout-json" | "" (no markup)

    def __post_init__(self) -> None:
        """Enforce the markup/format pairing rather than only documenting it.

        The structure parser branches on ``markup_format``, never on the extractor id, so markup
        that arrives without a declared language is markup the parser cannot read — and a NEW
        engine is exactly where that would be forgotten, silently, with the only symptom being an
        empty section tree much later. Both directions are wrong and both are cheap to catch here:
        markup with no format is unreadable, a format with no markup is a claim about nothing."""
        if self.markup is not None and not self.markup_format:
            raise ValueError(f"page {self.page_no}: markup was captured with no markup_format declared")
        if self.markup_format and self.markup is None:
            raise ValueError(f"page {self.page_no}: markup_format={self.markup_format!r} declared with no markup")

    @property
    def table_char_share(self) -> float:
        """0.0 when nothing was claimed, 1.0 when the whole page was read as table."""
        return self.table_chars / self.source_chars if self.source_chars else 0.0


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """An image handed to OCR, together with anything that went wrong producing it.

    The image and the reasons travel as ONE value for the same reason a page's text and its
    confidence do: a caller that received only the image could not know whether it had been
    cleaned, and would have to assume — which is how a silently-uncleaned scan became
    indistinguishable from a properly preprocessed one."""

    image: Any
    degradations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderedText:
    """The document as one string, together with THE page map for that exact string.

    The two are only ever produced together, by :meth:`Extraction.render`. ``page_map[i]`` is
    the number of characters page ``i+1`` occupies in ``text`` INCLUDING the separator emitted
    after it — which is what makes a consumer's running total (``_page_for_offset``) land on
    real page boundaries with no knowledge of the separator at all."""

    text: str
    page_map: list[int]

    def __post_init__(self) -> None:
        # Every character of `text` belongs to exactly one page. If a separator is ever emitted
        # without being counted (the drift this type exists to prevent), this fires immediately.
        if sum(self.page_map) != len(self.text):
            raise ValueError(
                f"page_map covers {sum(self.page_map)} chars but text is {len(self.text)} — "
                "the map must be derived in the same pass that joins the pages"
            )

    def apply_replacements(self, spans: list[tuple[int, int, str]]) -> RenderedText:
        """Rewrite regions of the text, moving the page boundaries with them.

        ``spans`` are ``(start, end, replacement)`` over THIS text, ascending and
        non-overlapping. A replacement is almost never the same length as what it replaces —
        "officer@acme.example" is 18 characters, "[REDACTED:EMAIL]" is 16 — so every offset after
        it shifts. Rebuilding the text while leaving the map alone is what made citations drift
        further and further down a document with no error anywhere.

        Text and map are rebuilt in ONE pass from ONE source, exactly as :meth:`Extraction.render`
        does. Do NOT "fix" a drifted map by rescaling it afterwards: that re-derives offsets from
        assumed lengths, which is the bug this type exists to make impossible."""
        if not spans:
            return self
        for i, (start, end, _) in enumerate(spans):
            if start > end:
                raise ValueError(f"replacement span {start}..{end} ends before it starts")
            if i and start < spans[i - 1][1]:
                raise ValueError(
                    f"replacement spans must be ascending and non-overlapping; {start} follows "
                    f"{spans[i - 1][1]}. Applying them out of order corrupts both text and map."
                )

        # Walk the pages and the spans together: a replacement belongs to the page whose range
        # it starts in, and only that page's entry changes length.
        parts: list[str] = []
        page_map: list[int] = []
        cursor = 0  # offset into the ORIGINAL text
        remaining = list(spans)
        for length in self.page_map:
            page_end = cursor + length
            emitted = 0
            at = cursor
            while remaining and remaining[0][0] < page_end:
                start, end, replacement = remaining.pop(0)
                parts.append(self.text[at:start])
                parts.append(replacement)
                emitted += (start - at) + len(replacement)
                at = end
            parts.append(self.text[at:page_end])
            emitted += page_end - at
            page_map.append(emitted)
            cursor = page_end
        return RenderedText(text="".join(parts), page_map=page_map)


# Bumped when the overlay's SHAPE changes, so a stored artifact written by an older build is
# detectable rather than silently misparsed. It is not the extractor's version and not the
# platform's: it versions this structure alone.
#
# 2 — adds `content_hash` (the overlay is now bound to the bytes it describes) and
#     `markup_format` per page. Both are new top-level keys a v1 reader would not find, which is
#     exactly the ambiguity the field exists to prevent, so this is a bump rather than an
#     "additive, compatible" non-change. Nothing anywhere has been persisted under v1: the write
#     is gated on `v2_ingest`, which defaults false and is set in no deployment.
# 1 — initial shape (schema_version, extractor, pages[page_no, markup, degradations]).
LAYOUT_OVERLAY_VERSION = 2


@dataclass(frozen=True, slots=True)
class LayoutPage:
    """One page's raw engine markup, carrying the page number it belongs to.

    `page_no` is the page's own 1-based number (as printed and as cited), NOT this page's index
    in the overlay: pages with no markup are absent, so the two do not line up."""

    page_no: int
    markup: str
    # Which markup language this page is in — see `PageRecord.markup_format`. Carried into the
    # stored artifact because the extractor id does not determine it.
    markup_format: str = ""
    # Copied from the page record so a consumer reading ONLY the stored artifact can see that a
    # page was degraded (e.g. `ocr_degenerate` — the model looped) without the record in hand.
    degradations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "markup": self.markup,
            "markup_format": self.markup_format,
            "degradations": list(self.degradations),
        }


@dataclass(frozen=True, slots=True)
class LayoutOverlay:
    """A document's engine markup: the pages that have any, in order, plus their provenance.

    An OVERLAY on the extracted text — it is persisted beside the raw bytes and read by a later
    parsing stage; it is never chunked, embedded or indexed, and `text` is never rebuilt from it
    (see `PageRecord.markup`). `rag_common.tenancy.layout_object_key` is where it is stored and
    `svc_control_plane.pipeline._persist_layout_overlay` states the compliance position.

    An overlay with NO pages means no page carried markup — distinct from a page present with
    `markup == ""`, which means the markup engine answered this page with nothing."""

    extractor: str  # the extractor(s) of the pages BELOW, comma-joined when they disagree
    pages: tuple[LayoutPage, ...]
    version: int = LAYOUT_OVERLAY_VERSION
    # SHA-256 of the raw bytes this overlay was extracted from — "" until it is bound (see
    # :meth:`for_content`; only the ingest path knows the hash). It is what makes overlay
    # STALENESS a fact rather than an argument: a consumer re-hashes the stored raw object and
    # compares, exactly as the platform already detects stale raw bytes, instead of trusting that
    # every write path refreshed the overlay. It also lets a consumer SKIP re-parsing when the
    # bytes have not changed.
    content_hash: str = ""

    def for_content(self, content_hash: str) -> LayoutOverlay:
        """This overlay, bound to the content hash of the bytes it was extracted from.

        A separate step because the hash belongs to the INGEST, not to extraction: the extractor
        is handed decoded bytes and has no opinion about how they are addressed, while the
        pipeline hashes exactly the object it stored."""
        return replace(self, content_hash=content_hash)

    def as_dict(self) -> dict[str, Any]:
        """The stable, JSON-serialisable form — this is what gets stored."""
        return {
            "schema_version": self.version,
            "extractor": self.extractor,
            "content_hash": self.content_hash,
            "pages": [page.as_dict() for page in self.pages],
        }


@dataclass(slots=True)
class Extraction:
    """An ordered list of extracted pages.

    There is deliberately NO ``text`` or ``page_map`` attribute: a caller cannot obtain the
    joined text without the page map that matches it — :meth:`render` returns both or neither.
    Document-level provenance is DERIVED from the pages, so it cannot disagree with them."""

    pages: list[PageRecord]
    modality: Modality

    def render(self) -> RenderedText:
        """Join the pages and derive the page map in ONE pass over ONE list.

        Each map entry is the length of the exact string appended for that page, so changing
        :data:`PAGE_SEPARATOR` moves the offsets automatically — there is no second place that
        encodes the separator's length, and no way to compute the two from different sources."""
        parts: list[str] = []
        page_map: list[int] = []
        last = len(self.pages) - 1
        for i, page in enumerate(self.pages):
            emitted = page.text if i == last else page.text + PAGE_SEPARATOR
            parts.append(emitted)
            page_map.append(len(emitted))  # the length of what was ACTUALLY appended
        return RenderedText(text="".join(parts), page_map=page_map)

    def layout_overlay(self) -> LayoutOverlay:
        """The pages' raw engine markup as one stable, JSON-serialisable overlay.

        Deliberately NOT part of :meth:`render`: that returns text and its page map together
        because neither is usable without the other, and the overlay is a THIRD, independent
        concern that no text consumer needs. Growing `render` to return it would hand every
        caller of the indexed text a structure it has no use for.

        A page whose `markup` is None is ABSENT here (its engine emits no markup); a page whose
        markup is `""` is present with `""`. So "no markup" and "empty markup" stay
        distinguishable, at page level and at document level — a document where no page carried
        markup yields an overlay with no pages at all.

        `extractor` names the extractor(s) of the markup pages only, not the document's (the
        :attr:`extractor` property): on a mixed document the native-text pages are not in here
        and must not be credited with the markup."""
        pages = tuple(
            LayoutPage(
                page_no=p.page_no,
                markup=p.markup,
                markup_format=p.markup_format,
                degradations=p.degradations,
            )
            for p in self.pages
            if p.markup is not None
        )
        extractor = _META_SEP.join(sorted({p.extractor for p in self.pages if p.markup is not None}))
        return LayoutOverlay(extractor=extractor, pages=pages)

    @property
    def extractor(self) -> str:
        """The extractor(s) that produced this document — comma-joined when pages disagree,
        so a partially-degraded document can never report a single clean extractor."""
        return _META_SEP.join(sorted({p.extractor for p in self.pages}))

    @property
    def capabilities(self) -> tuple[str, ...]:
        """What was recovered from EVERY page — the weakest page decides. A document is only
        table-capable if no page lost tables."""
        if not self.pages:
            return ()
        common = set(self.pages[0].capabilities).intersection(*(set(p.capabilities) for p in self.pages))
        return tuple(sorted(common))

    @property
    def degradations(self) -> tuple[str, ...]:
        """Every reason code seen on any page."""
        return tuple(sorted({reason for p in self.pages for reason in p.degradations}))

    @property
    def min_confidence(self) -> float:
        """Weakest page confidence (FR-B1). A document with no pages asserts nothing, so it
        reports full confidence — it carries no text to be unconfident about."""
        return min((p.confidence for p in self.pages), default=1.0)

    @property
    def table_pages(self) -> tuple[int, ...]:
        """Page numbers a table was actually recovered from, in order."""
        return tuple(p.page_no for p in self.pages if p.tables_found)

    @property
    def table_page_runs(self) -> int:
        """How many maximal runs of CONSECUTIVE table pages the document has.

        ``len(table_pages) / table_page_runs`` is the mean run length. 1.0 means every table
        page is isolated, so no table can be crossing a break; above 1.0 means tables occupy
        adjacent pages. It is an UPPER bound on cross-page tables — two unrelated tables on
        neighbouring pages are indistinguishable from one table split across them, because the
        renderer discards where on the page a table sat."""
        pages = self.table_pages
        return sum(1 for i, page_no in enumerate(pages) if i == 0 or page_no != pages[i - 1] + 1)

    @property
    def table_char_share(self) -> float:
        """Share of the document's source characters that table detection claimed.

        Summed from the per-page COUNTS, not averaged from the per-page ratios: a mean of
        ratios would give a sparse page the same weight as a dense one, so a single mostly-
        empty page misread as a table could dominate a long document's number."""
        source = sum(p.source_chars for p in self.pages)
        return sum(p.table_chars for p in self.pages) / source if source else 0.0


def _uniform_pages(
    texts: list[str],
    *,
    extractor: str,
    capabilities: tuple[str, ...],
    degradations: tuple[str, ...] = (),
    tables_found: bool = False,
) -> list[PageRecord]:
    """Build page records for an extractor that treated every page the same way. Page numbers
    come from the position in `texts`, so they cannot drift from the text they label."""
    return [
        PageRecord(
            page_no=i,
            text=text,
            extractor=extractor,
            capabilities=capabilities,
            degradations=degradations,
            tables_found=tables_found,
        )
        for i, text in enumerate(texts, start=1)
    ]


def provenance_tags(extraction: Extraction) -> dict[str, str]:
    """Render an :class:`Extraction`'s provenance as document metadata tags.

    Every key is ALWAYS emitted (``extraction_degradation`` as ``""`` when clean) so a
    re-extraction can never leave a stale degradation stamp behind, and so "no tag" means
    "ingested before provenance existed" rather than "clean"."""
    table_pages = extraction.table_pages
    return {
        META_EXTRACTOR: extraction.extractor,
        META_CAPABILITIES: _META_SEP.join(sorted(extraction.capabilities)),
        META_DEGRADED: "true" if extraction.degradations else "false",
        META_DEGRADATION: _META_SEP.join(sorted(extraction.degradations)),
        META_TABLES_FOUND: "true" if table_pages else "false",
        META_TABLE_PAGES: str(len(table_pages)),
        META_TABLE_RUNS: str(extraction.table_page_runs),
        # Rounded because it is a screening signal, not a measurement: an operator sorts the
        # corpus by it to find documents where detection ate the prose, and full float
        # precision would only make the stored tag noisier to read and to group by.
        META_TABLE_CHAR_SHARE: str(round(extraction.table_char_share, _TABLE_SHARE_DP)),
        META_BOILERPLATE_LINES: str(sum(p.boilerplate_lines for p in extraction.pages)),
        META_OCR_APPLIED: "true" if any(p.ocr_reason for p in extraction.pages) else "false",
        META_OCR_REASONS: _META_SEP.join(sorted({p.ocr_reason for p in extraction.pages if p.ocr_reason})),
        META_OCR_PAGES: str(sum(1 for p in extraction.pages if p.ocr_reason)),
        META_OCR_IMAGES: str(sum(p.ocr_images for p in extraction.pages)),
    }


@runtime_checkable
class OCREngine(Protocol):
    """Turns page images / scanned bytes into text with confidence (FR-B1/B3)."""

    async def ocr(self, data: bytes, mime: str) -> Extraction: ...


class NoOpOCR:
    """Dev fallback: refuses scanned docs clearly rather than pretending to OCR."""

    async def ocr(self, data: bytes, mime: str) -> Extraction:  # pragma: no cover
        raise NotImplementedError(
            "OCR engine not configured in this dev image. Set RAG_OCR_BACKEND=tesseract "
            "(with the tesseract binary + pytesseract/pdf2image installed) or wire PaddleOCR/"
            "docTR on the GPUaaS pool for scanned documents (FR-B1)."
        )


def _otsu_threshold(gray: Any) -> int:
    """Otsu's method on a grayscale image histogram — the split that best separates ink from
    paper. Pure Python over the 256-bin histogram, so no numpy/OpenCV dependency."""
    hist = gray.histogram()[:256]
    total = sum(hist)
    if total == 0:
        return 128
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_b = 0.0
    weight_b = 0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > max_var:
            max_var = between
            threshold = t
    return threshold


def preprocess_for_ocr(img: Any, *, binarize: bool = False) -> Any:
    """Clean a page image so Tesseract can read it — the biggest accuracy lever on faint
    thermal receipts, halftone newspaper print and skewed phone photos. CPU-only (Pillow).

    The default is GENTLE and tuned for Tesseract 5's LSTM engine, which reads clean
    *grayscale* best: EXIF auto-orient -> grayscale -> upscale small scans (LSTM wants
    ~300 DPI) -> autocontrast (fights faint ink) -> optional OpenCV deskew. We deliberately
    do NOT hard-binarize or blur by default — that destroys thin anti-aliased strokes and
    HURTS accuracy on clean-ish inputs (verified). ``binarize=True`` adds Otsu thresholding
    for the rare legacy case (very noisy bitonal fax scans) where it helps.

    Never raises: on any failure it returns the original image so OCR still runs — but it
    REPORTS that, so the page it produced can be told apart from a properly cleaned one."""
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415

        img = ImageOps.exif_transpose(img) or img  # honour phone-camera rotation
        gray = img.convert("L")
        long_edge = max(gray.size)
        if long_edge < _OCR_UPSCALE_TO_PX:
            scale = _OCR_UPSCALE_TO_PX / long_edge
            gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.Resampling.LANCZOS)
        gray = ImageOps.autocontrast(gray, cutoff=_OCR_AUTOCONTRAST_CUTOFF)
        deskewed = _maybe_deskew(gray)
        gray = deskewed.image
        if binarize:
            threshold = _otsu_threshold(gray)
            gray = gray.point(lambda p: 255 if p > threshold else 0).convert("L")
        return PreparedImage(image=gray, degradations=deskewed.degradations)
    except Exception as exc:  # preprocessing must never break OCR
        _log.warning("ocr_preprocess_failed", error_type=type(exc).__name__, error=str(exc), exc_info=True)
        return PreparedImage(image=img, degradations=(REASON_OCR_PREPROCESS_FAILED,))


def _maybe_deskew(gray: Any) -> PreparedImage:
    """Deskew via OpenCV if it is installed (prod/GPU image); a no-op otherwise. Skew is
    common in phone photos and flatbed scans and hurts Tesseract badly.

    The two ways this can not-happen are NOT the same event and are handled separately:
    OpenCV absent is a deployment choice, uniform across every page of every document, and
    discoverable from the image; cv2 raising on a particular page is a per-page accuracy loss
    that nothing else will ever surface. Only the second is a degradation."""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return PreparedImage(image=gray)  # not a failure: this image simply has no OpenCV
    try:
        from PIL import Image  # noqa: PLC0415

        arr = np.array(gray)
        coords = np.column_stack(np.where(arr < _INK_LEVEL))
        if coords.shape[0] < _MIN_INK_PIXELS_FOR_DESKEW:
            return PreparedImage(image=gray)
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < _DESKEW_ANGLE_FLOOR else -angle
        if abs(angle) < _MIN_DESKEW_DEGREES:  # negligible skew — don't resample for nothing
            return PreparedImage(image=gray)
        h, w = arr.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        return PreparedImage(image=Image.fromarray(rotated))
    except Exception as exc:
        _log.warning("ocr_deskew_failed", error_type=type(exc).__name__, error=str(exc), exc_info=True)
        return PreparedImage(image=gray, degradations=(REASON_OCR_DESKEW_FAILED,))


OCR_EXTRACTOR_TESSERACT = "tesseract"


class TesseractOCR:
    """Sovereign, CPU OCR via Tesseract (FR-B1). Handles scanned PDFs (rasterised per page)
    and image bytes; ``lang`` accepts tesseract codes incl. Indic (e.g. ``eng+hin``). The
    heavy libs (pytesseract/pdf2image/Pillow) are imported on use so the dev image without
    them still boots; a clear error is raised only when an OCR call is actually made.

    Each page is preprocessed (see :func:`preprocess_for_ocr`) and read with
    ``image_to_data`` so we get a REAL per-word confidence (not a hardcoded guess) and can
    flag low-confidence pages for human review, while preserving line layout for receipts."""

    def __init__(
        self, lang: str = "eng", *, preprocess: bool = True, binarize: bool = False, psm: int = 3, oem: int = 1
    ) -> None:
        self._lang = lang
        self._preprocess = preprocess
        self._binarize = binarize
        self._config = f"--oem {oem} --psm {psm}"

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on prod image
            raise NotImplementedError(
                "RAG_OCR_BACKEND=tesseract needs pytesseract + Pillow (+ pdf2image for PDFs) "
                "and the tesseract binary installed in the image."
            ) from exc

        if "pdf" in mime:
            from pdf2image import convert_from_bytes  # noqa: PLC0415

            images = convert_from_bytes(data, dpi=300)  # 300 DPI raster — Tesseract's sweet spot
        else:
            images = [Image.open(io.BytesIO(data))]

        # text, confidence AND the preprocessing outcome stay in ONE list, so none of the three
        # can drift from the page it describes.
        read: list[tuple[str, float, tuple[str, ...]]] = []
        for img in images:
            prepared = (
                preprocess_for_ocr(img, binarize=self._binarize) if self._preprocess else PreparedImage(image=img)
            )
            text, conf = self._read_page(pytesseract, prepared.image)
            read.append((text, conf, prepared.degradations))
        normalized = _normalize_pages([text for text, _, _ in read])
        pages = [
            PageRecord(
                page_no=i,
                text=page.text,
                extractor=OCR_EXTRACTOR_TESSERACT,
                capabilities=(CAP_TEXT,),  # word-level OCR: no table structure recovered
                confidence=conf,
                degradations=degradations,
                boilerplate_lines=page.boilerplate_lines,
            )
            # strict=True: _normalize_pages must return one entry per page. If it ever stops
            # doing so, fail here rather than silently re-number pages.
            for i, (page, (_, conf, degradations)) in enumerate(zip(normalized, read, strict=True), start=1)
        ]
        extraction = Extraction(pages=pages, modality=Modality.TEXT)
        _log.info(
            "ocr_tesseract",
            pages=len(pages),
            chars=sum(len(p.text) for p in pages),
            lang=self._lang,
            preprocess=self._preprocess,
            min_confidence=round(extraction.min_confidence, 3),
        )
        return extraction

    def _read_page(self, pytesseract: Any, img: Any) -> tuple[str, float]:
        """OCR one page: reconstruct text line-by-line (so receipts keep their rows) and
        return the mean per-word confidence in [0,1]."""
        data = pytesseract.image_to_data(img, lang=self._lang, config=self._config, output_type=pytesseract.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for i, raw_word in enumerate(data["text"]):
            word = (raw_word or "").strip()
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf >= 0:
                confs.append(conf)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items())).strip()
        mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, mean_conf


# Tesseract lang codes -> PaddleOCR lang codes (PaddleOCR uses ISO-ish / script names).
_PADDLE_LANG_MAP = {"eng": "en", "en": "en", "hin": "devanagari", "hi": "devanagari"}
# One PaddleOCR reader per (lang, model_dir) — building it loads det+rec+cls models (slow),
# so we cache and reuse across documents. Module-level so it survives per-request wiring.
_PADDLE_READERS: dict[tuple[str, str], _PaddleReader] = {}
_PADDLE_LINE_BUCKET_PX = 12  # detections whose top-y is within this are treated as one row


@dataclass(frozen=True, slots=True)
class _PaddleReader:
    """A built PaddleOCR reader and anything the build had to give up to get it. Paired for
    the same reason as :class:`PreparedImage`: the caller stamps the pages, and cannot know
    what was dropped unless it is told."""

    reader: Any
    degradations: tuple[str, ...] = ()


# Which constructor kwargs are a GUARANTEE and which are tuning. PaddleOCR releases have
# removed kwargs before, so the build has to be able to retry — but not by throwing away the
# whole request. The model directories are why an air-gapped deployment can promise it never
# reaches the internet for weights; angle classification and log verbosity are preferences.
_PADDLE_MODEL_DIR_KWARGS = ("det_model_dir", "rec_model_dir", "cls_model_dir")
_PADDLE_TUNING_KWARGS = ("use_angle_cls", "show_log")


def _paddle_reader(lang: str, model_dir: str) -> _PaddleReader:
    """Lazily build + cache a PaddleOCR reader. ``model_dir`` (optional) points at
    pre-mirrored det/rec/cls model folders for a fully air-gapped/sovereign deployment;
    empty ⇒ PaddleOCR resolves its own models (one-time bootstrap, like TEI/llama).

    A release that rejects a kwarg is retried WITHOUT the tuning options and with the model
    directories intact. If it cannot take those either, the build fails rather than falling
    back to PaddleOCR fetching its own weights — for an air-gapped install that fallback is
    not a lesser outcome, it is the exact thing ``model_dir`` exists to prevent."""
    key = (lang, model_dir)
    cached = _PADDLE_READERS.get(key)
    if cached is not None:
        return cached

    from paddleocr import PaddleOCR  # noqa: PLC0415

    essential: dict[str, Any] = {"lang": lang}
    if model_dir:
        import os  # noqa: PLC0415

        for sub, kwarg in zip(("det", "rec", "cls"), _PADDLE_MODEL_DIR_KWARGS, strict=True):
            path = os.path.join(model_dir, sub)
            if os.path.isdir(path):
                essential[kwarg] = path

    try:
        built = _PaddleReader(reader=PaddleOCR(**essential, use_angle_cls=True, show_log=False))
    except TypeError as exc:
        _log.warning(
            "paddle_tuning_kwargs_dropped",
            dropped=list(_PADDLE_TUNING_KWARGS),
            detail="this PaddleOCR release rejected an accuracy-tuning option; retrying with the "
            "mirrored models and language only",
            error=str(exc),
        )
        try:
            built = _PaddleReader(reader=PaddleOCR(**essential), degradations=(REASON_OCR_TUNING_DROPPED,))
        except TypeError as retry_exc:
            if not model_dir:
                raise  # nothing sovereign at stake — a broken install, not a silent downgrade
            raise ConfigError(
                "the installed PaddleOCR release will not accept the pre-mirrored model "
                f"directories under {model_dir!r} ({retry_exc}). Retrying without them would let "
                "PaddleOCR fetch its own weights, which is what RAG_OCR_MODEL_DIR exists to "
                "prevent on an air-gapped deployment — pin a PaddleOCR version that accepts "
                "det/rec/cls model directories, or clear the setting to accept its bootstrap.",
                details={"model_dir": model_dir, "lang": lang},
            ) from retry_exc

    _PADDLE_READERS[key] = built  # the reason is cached WITH the reader: it describes every
    return built  # document this process reads, not just the first one


OCR_EXTRACTOR_PADDLE = "paddleocr"


class PaddleOCREngine:
    """Sovereign PP-OCRv4 engine (PaddleOCR) — markedly better than Tesseract on low-quality
    scans, receipts, rotated text and multilingual/Indic scripts, and it runs on CPU. The
    heavy libs (paddleocr + paddlepaddle) are the ``[paddleocr]`` extra, imported on use so
    the base image still boots; a clear error is raised only when an OCR call is actually made.

    PaddleOCR does its own detection/binarization, so we feed it the ORIGINAL image (no
    Tesseract-style preprocessing) and reconstruct reading order from the returned line boxes,
    with a real mean confidence from PP-OCR's per-line scores."""

    def __init__(self, lang: str = "en", *, model_dir: str = "") -> None:
        self._lang = _PADDLE_LANG_MAP.get(lang, lang)
        self._model_dir = model_dir

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        try:
            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the [paddleocr] extra
            raise NotImplementedError(
                "RAG_OCR_BACKEND=paddleocr needs the [paddleocr] extra (paddleocr + paddlepaddle). "
                "It is the recommended CPU upgrade for low-quality scans and Indic scripts; install "
                "the extra (and pre-mirror the PP-OCR models for an air-gapped deployment), then it "
                "activates with no call-site change."
            ) from exc

        if "pdf" in mime:
            from pdf2image import convert_from_bytes  # noqa: PLC0415

            images = convert_from_bytes(data, dpi=200)
        else:
            images = [Image.open(io.BytesIO(data))]

        built = _paddle_reader(self._lang, self._model_dir)
        # (text, confidence) stay in ONE list so a page's confidence cannot drift from its text.
        read: list[tuple[str, float]] = [self._read_page(built.reader, np, img) for img in images]
        normalized = _normalize_pages([text for text, _ in read])
        pages = [
            PageRecord(
                page_no=i,
                text=page.text,
                extractor=OCR_EXTRACTOR_PADDLE,
                capabilities=(CAP_TEXT,),  # line-level OCR: no table structure recovered
                confidence=conf,
                # Whatever the engine build had to give up applies to every page it reads.
                degradations=built.degradations,
                boilerplate_lines=page.boilerplate_lines,
            )
            for i, (page, (_, conf)) in enumerate(zip(normalized, read, strict=True), start=1)
        ]
        extraction = Extraction(pages=pages, modality=Modality.TEXT)
        _log.info(
            "ocr_paddle",
            pages=len(pages),
            chars=sum(len(p.text) for p in pages),
            lang=self._lang,
            min_confidence=round(extraction.min_confidence, 3),
        )
        return extraction

    def _read_page(self, reader: Any, np: Any, img: Any) -> tuple[str, float]:
        """Run PP-OCR on one page and rebuild top-to-bottom, left-to-right reading order."""
        arr = np.array(img.convert("RGB"))
        raw = reader.ocr(arr, cls=True) if hasattr(reader, "ocr") else reader.predict(arr)
        return reconstruct_paddle_page(raw)


def reconstruct_paddle_page(raw: Any) -> tuple[str, float]:
    """Turn one PaddleOCR page result into (text, mean_confidence) — pure, so it is unit-
    testable without the native paddle wheel. PaddleOCR returns one entry per input image;
    each line is ``[box(4 points), (text, score)]``. We restore reading order by bucketing
    detections into visual rows (top-y) and ordering left-to-right within a row."""
    lines = raw[0] if raw and isinstance(raw[0], list) else raw
    rows: list[tuple[int, float, str]] = []
    confs: list[float] = []
    for entry in lines or []:
        box, (line_text, score) = entry[0], entry[1]
        top = min(pt[1] for pt in box)
        left = min(pt[0] for pt in box)
        rows.append((round(top / _PADDLE_LINE_BUCKET_PX), left, line_text))
        confs.append(float(score))
    rows.sort(key=lambda r: (r[0], r[1]))
    text = "\n".join(t for _, _, t in rows).strip()
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return text, mean_conf


class TrOCREngine:
    """Optional handwriting/scene-text engine (Microsoft TrOCR via transformers). Tesseract is
    a PRINTED-text engine and cannot read handwriting; this transformer model can. Import-
    guarded (``[trocr]`` extra: transformers + torch); GPU strongly recommended. Raises a clear
    error until wired — handwriting recognition is a provisioning decision, not a silent default."""

    def __init__(self, model: str = "microsoft/trocr-base-handwritten") -> None:
        self._model = model

    async def ocr(self, data: bytes, mime: str) -> Extraction:  # pragma: no cover - optional dep
        raise NotImplementedError(
            "RAG_OCR_BACKEND=trocr needs the [trocr] extra (transformers + torch) and a "
            "handwriting model (e.g. microsoft/trocr-base-handwritten). This is the path for "
            "handwritten documents; provision a GPU model host, then it activates via config."
        )


OCR_EXTRACTOR_CHANDRA = "chandra"

_CHANDRA_OCR_PROMPT = (
    "Transcribe ALL text from this document image exactly, preserving reading order. "
    "Render tables as Markdown tables, keep headings and lists, and include any labels, "
    "stamps, handwriting, or figure captions you can read. "
    # Without this the model flattens raised and lowered characters to ordinary digits, and the
    # meaning goes with them: an algebraic chart indexed as "a2 - b2 = (a-b)(a+b)", and in a
    # pharma corpus "H2SO4" and "NO3-" are different substances from H₂SO₄ and NO₃⁻. The rest of
    # the pipeline already carries these characters through unchanged (sanitize._SCRIPT_CHARS
    # guards them across the NFKC fold, verified end to end), so the only place they were being
    # lost was here, by not being asked for.
    "Keep superscripts and subscripts as superscript and subscript characters (a², x³, H₂SO₄, "
    "NO₃⁻) rather than flattening them to ordinary digits. "
    "Output only the transcription."
)


class ChandraOCREngine:
    """OCR via Chandra (Datalab) — the same VLM that backs the vision gateway (one model, two
    roles). Strong on tables/forms/handwriting/reading-order and Indic scripts. Served over the
    OpenAI-compatible vision chat API (self-hosted vLLM or a hosted endpoint), API-key ready.

    A chat-based VLM does not return per-word confidence, so pages are recorded at confidence
    1.0 (trust a strong model); the ``ocr_min_confidence`` gate is therefore a no-op for this
    engine by design. Tables are declared as a capability (Chandra preserves row/column
    structure), so downstream table-aware handling applies."""

    def __init__(
        self, url: str = "", model: str = "", *, api_key: str | None = None, api_key_header: str | None = None
    ) -> None:
        from rag_clients.chandra import ChandraClient  # noqa: PLC0415 — lib dep, lazy to keep import light

        self._client = ChandraClient(
            url=url or None, model=model or None, api_key=api_key, api_key_header=api_key_header
        )

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        images = self._to_images(data, mime)
        pages: list[PageRecord] = []
        # Bounded per call: a page the model cannot read loops until its budget is spent, and at
        # the platform's general budget that outlives the model gateway's own timeout — the ingest
        # then fails on a page nothing could have read. See Settings.ocr_max_tokens.
        max_tokens = get_settings().ocr_max_tokens
        for i, (img_bytes, img_mime) in enumerate(images, start=1):
            raw = await self._client.run(_CHANDRA_OCR_PROMPT, [img_bytes], mime=img_mime, max_tokens=max_tokens)
            # Chandra answers in HTML however it is asked; index what a reader sees.
            text, looped = trim_degenerate_tail(normalize_ocr_markup(raw.strip()))
            if looped:
                _log.warning("ocr_degenerate_output", page_no=i, kept_chars=len(text))
            pages.append(
                PageRecord(
                    page_no=i,
                    text=text,
                    extractor=OCR_EXTRACTOR_CHANDRA,
                    capabilities=(CAP_TEXT, CAP_TABLES),  # Chandra preserves table structure
                    degradations=(REASON_OCR_DEGENERATE,) if looped else (),
                    # A page the model looped on was not read; saying 1.0 there would assert a
                    # confidence in text that does not exist (see the class docstring's caveat).
                    confidence=0.0 if looped and not text else 1.0,
                    # KEEP the HTML Chandra already answered with, unflattened, as the layout
                    # overlay — `normalize_ocr_markup` above consumed it to build `text` and the
                    # markup was then dropped on the floor. Stored as returned (post
                    # `ChandraClient._sanitize`, pre-normalisation); a page the model LOOPED on
                    # still records its markup, with the degradation already on this record.
                    markup=raw,
                    # HTML, not Markdown: this engine ASKS for Markdown and Chandra answers in
                    # HTML anyway (`normalize_ocr_markup`'s docstring, measured on the live
                    # corpus). Declared rather than inferred from the extractor id, which the
                    # hosted engine below shares while emitting a different format.
                    markup_format="html",
                )
            )
        return Extraction(pages=pages, modality=Modality.OCR)

    @staticmethod
    def _to_images(data: bytes, mime: str) -> list[tuple[bytes, str]]:
        """One (bytes, mime) per page. PDFs are rasterised (pdf2image); an image is sent as-is."""
        if "pdf" not in mime:
            return [(data, mime or "image/png")]
        try:
            from pdf2image import convert_from_bytes  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on prod image
            raise NotImplementedError(
                "RAG_OCR_BACKEND=chandra needs pdf2image (+ poppler) to rasterise PDF pages."
            ) from exc
        out: list[tuple[bytes, str]] = []
        for page in convert_from_bytes(data, dpi=200):
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            out.append((buf.getvalue(), "image/png"))
        return out


class DatalabOCREngine:
    """OCR via the HOSTED Datalab Chandra API (async /convert). For testing against the cloud
    endpoint before a self-hosted GPU exists; the self-hosted vLLM path is ``ChandraOCREngine``.

    Datalab returns transcribed markdown (tables preserved) for the whole document in one call,
    so this records a single page — no per-page confidence (the /convert API gives none), hence
    confidence 1.0 (trust the model). Same CAP_TABLES capability as the self-hosted engine.

    It stamps the SAME extractor id as the self-hosted engine (same model, two hosting modes) but
    its markup is MARKDOWN, not the self-hosted path's HTML — the /convert API takes an explicit
    ``output_format``, which the chat path has no equivalent of. That is why `markup_format` is
    recorded per page: a consumer keying off ``extractor == "chandra"`` would be wrong here."""

    def __init__(self, url: str = "") -> None:
        from rag_clients.datalab import DatalabClient  # noqa: PLC0415 — lib dep, lazy

        self._client = DatalabClient(url=url or None)

    async def ocr(self, data: bytes, mime: str) -> Extraction:
        text = await self._client.convert(data, mime or "image/png", output_format="markdown")
        page = PageRecord(
            page_no=1,
            text=normalize_ocr_markup(text.strip()),
            extractor=OCR_EXTRACTOR_CHANDRA,
            capabilities=(CAP_TEXT, CAP_TABLES),
            confidence=1.0,
            # KEEP what already arrives, same as the self-hosted engine: `text` above is this
            # string flattened by `normalize_ocr_markup` — the very function whose existence is
            # the evidence this model emits markup — so discarding it here would throw away the
            # only structural signal a `chandra_api` deployment ever receives. Post the client's
            # `_sanitize` (NUL/C0 stripped), pre-normalisation.
            markup=text,
            markup_format="markdown",
        )
        return Extraction(pages=[page], modality=Modality.OCR)


def build_ocr_engine(
    backend: str,
    *,
    lang: str = "eng",
    preprocess: bool = True,
    binarize: bool = False,
    psm: int = 3,
    oem: int = 1,
    model_dir: str = "",
) -> OCREngine:
    """Select the OCR engine from config (FR-B1). ``noop`` refuses scans; ``tesseract`` OCRs on
    CPU (sovereign default); ``paddleocr``/``trocr`` are optional stronger/handwriting engines;
    ``chandra`` is the self-hosted VLM (also backs the vision gateway); ``chandra_api`` is the same
    Chandra model on the hosted Datalab cloud API (async /convert) for testing before self-hosting."""
    if backend == "tesseract":
        return TesseractOCR(lang=lang, preprocess=preprocess, binarize=binarize, psm=psm, oem=oem)
    if backend == "paddleocr":
        return PaddleOCREngine(lang=lang.split("+", 1)[0], model_dir=model_dir)
    if backend == "trocr":
        return TrOCREngine()
    if backend == "chandra":
        return ChandraOCREngine()
    if backend == "chandra_api":
        return DatalabOCREngine()
    return NoOpOCR()


def _dehyphenate(text: str) -> str:
    """Join words split across line breaks ("infor-\nmation" -> "information").

    A SERIALIZED TABLE ROW is exempt. Rows are newline-separated and may legitimately end in a
    hyphen — measured in the HR Handbook, "Reviewed By = HR Operations- Head" wraps so that a row
    ends "= Operations-", and joining it to the next row produced "OperationsPrepared Date = 2024":
    two records welded into one field. The newline after a row ends a record, never a word."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


# A word broken across a line break — but not when the next line opens a labelled table field
# ("Prepared Date = 2024"), which is a new row rather than the rest of the word.
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?![^\n|]{1,60}" + re.escape(LABEL_SEP) + r")(\w)")


_DIGITS_RE = re.compile(r"\d+")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_NUMERIC_LINE = "\x00numeric"  # the key every wordless edge line shares
_PAGE_LABEL_LINE = "\x00pagelabel"  # the key every "Page N"-shaped edge line shares
# A line whose only WORDS are page words, around a number: "Page | 6", "Page 3 of 9", "- pg 12 -".
# Narrow on purpose — the words are enumerated, so no heading can fall into it by accident.
_PAGE_LABEL_RE = re.compile(
    r"[^\w]*(?:page|pg|p)[^\w]*\d+(?:[^\w]*(?:of|/)[^\w]*\d+)?[^\w]*",
    re.IGNORECASE,
)


def _furniture_key(line: str) -> str:
    """What makes two edge lines the SAME piece of furniture.

    Identical text is the safe case and stays exact. The one variation worth generalising is a
    line that carries no words at all — "3", "- 3 -", "3/9". Those never repeat literally, so
    exact counting never reached any threshold and the footer survived on every page: measured
    on the real corpus, that left a bare number on 114 of 147 pages of one handbook and 42 of
    56 of another. A line without letters cannot be prose, so collapsing them together cannot
    eat content.

    The gap that used to be left open here was the worded footer with a changing number —
    "Page | 6", "Page 3 of 9". Collapsing digits inside ANY line with words would catch it and
    would just as happily merge "Article 5." with "Article 6.", stripping the clause number off
    every page of a legal document: losing a footer is a nuisance, losing a clause number is a
    wrong answer. The rule that separates them is that a page label's ONLY words are page words.
    "Article", "Chapter" and "Section" are not among them, so a numbered heading still keys to
    itself and is never collapsed; "Page | 6" and "Page | 7" now share one key and the repeated
    edge counter can finally see them. Measured on Acme's HR Handbook, that footer was sitting
    inside 324 of 554 indexed chunks.

    STILL NOT CAUGHT, deliberately: a footer of other words plus a number ("HR Handbook 4.0 |
    6"). Recognising that needs evidence this rule does not have — that the words are furniture
    rather than content — and guessing it wrong deletes real text from every page."""
    line = line.strip()
    if _PAGE_LABEL_RE.fullmatch(line):
        return _PAGE_LABEL_LINE
    return _NUMERIC_LINE if not _ALPHA_RE.search(line) else line


def _edge_block(edges: list[list[str]], threshold: int) -> list[str]:
    """How deep the repeated block runs at one edge, as the furniture key per level.

    Returns one key per level, outermost first. Stops at the first level where no single key
    recurs often enough, so a two-line masthead is found without a depth guess — the previous
    implementation removed exactly one line and left the rest of the block on every page."""
    block: list[str] = []
    while len(block) < _MAX_EDGE_BLOCK_LINES:
        depth = len(block)
        counts: dict[str, int] = {}
        for lines in edges:
            if depth < len(lines):
                key = _furniture_key(lines[depth])
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            break
        key, n = max(counts.items(), key=lambda kv: kv[1])
        if n < threshold:
            break
        block.append(key)
    return block


def _strip_repeated_edges(pages: list[str]) -> list[tuple[str, int]]:
    """Remove running headers/footers, returning each page with the number of lines removed.

    Furniture is text that sits at a page EDGE *and* recurs across most pages. Both conditions
    are load-bearing: a line repeated mid-page is content (a section heading, a total), and a
    line at an edge that appears once is content too. A document too short to establish what
    recurs is left alone entirely."""
    if len(pages) < _MIN_PAGES_FOR_EDGE_STRIP:
        return [(p, 0) for p in pages]

    stripped = [[ln for ln in p.splitlines() if ln.strip()] for p in pages]
    threshold = max(_MIN_PAGES_FOR_EDGE_STRIP, int(len(pages) * _FURNITURE_PAGE_SHARE))
    head = _edge_block(stripped, threshold)
    tail = _edge_block([list(reversed(lines)) for lines in stripped], threshold)

    out: list[tuple[str, int]] = []
    for lines in stripped:
        start, end = 0, len(lines)
        # Only remove a line that actually MATCHES the block at its level: a page without the
        # masthead must keep its first line of content.
        for key in head:
            if start < end and _furniture_key(lines[start]) == key:
                start += 1
            else:
                break
        for key in tail:
            if end > start and _furniture_key(lines[end - 1]) == key:
                end -= 1
            else:
                break
        out.append(("\n".join(lines[start:end]), (start) + (len(lines) - end)))
    return out


# Glyphs a PDF's font can't map to Unicode surface as "(cid:NNN)" (typically bullets/symbols).
# Left in, they pollute the indexed text and the cited "sources you can prove" view (QA ISS-010).
_CID_RE = re.compile(r"\(cid:\d+\)\s?")


@dataclass(frozen=True, slots=True)
class _NormalizedPage:
    """A normalized page and what normalization took off it.

    The count travels WITH the text so a caller cannot report one page's content beside
    another page's edit — the same reason a page's text and confidence are one value."""

    text: str
    boilerplate_lines: int


def normalize_extracted_text(text: str) -> str:
    """Strip CID artefacts, heal words broken across a line break, collapse blank runs.

    PUBLIC because it is half of an INVARIANT, not a formatting pass. The page text is normalised
    HERE and masked afterwards, so the masker sees "PAN ABCDE1234F" where the page printed
    "PAN ABCDE-\n1234F". Anything that masks text which has NOT been through this sees the raw
    broken form, matches nothing, and indexes the identifier in clear — which is exactly what
    happened to the layout-aware chunker, whose blocks come from the PRE-normalisation overlay.
    `DocIntelligenceService.prepare_for_index` is the shared entry point; this is its first step.
    """
    return re.sub(r"\n{3,}", "\n\n", _dehyphenate(_CID_RE.sub("", text)))


def _normalize_pages(pages: list[str]) -> list[_NormalizedPage]:
    return [
        _NormalizedPage(text=normalize_extracted_text(text).strip(), boilerplate_lines=removed)
        for text, removed in _strip_repeated_edges(pages)
    ]


@dataclass(frozen=True, slots=True)
class _DetectedTable:
    """One table pdfplumber found and the serializer accepted — its box and its own cells.

    ``cells`` is the grid as ``table.extract()`` returned it, kept rather than re-derived:
    :func:`_page_tables` already calls it to render the rows into the text, so this is the SAME
    grid the text was built from, not a second extraction that could disagree with it. Cells may
    be ``None`` (an empty cell), exactly as pdfplumber reports them."""

    bbox: tuple[float, float, float, float]
    cells: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True, slots=True)
class _TableBand:
    """A rendered table plus the vertical band it occupies.

    Coordinates live HERE, inside the extractor, only long enough to order one page. They are
    never stored on a :class:`PageRecord` — the record keeps scalar facts, and anything with
    coordinates or spans belongs in an OVERLAY (see :data:`MARKUP_FORMAT_LAYOUT_JSON`, which is
    built from these bands rather than from a second look at the page).

    ``tables`` are the individual detected tables this band is made of — usually one, more when
    two vertically overlapping tables were merged so the splice could order them as a unit."""

    top: float
    bottom: float
    body: str
    tables: tuple[_DetectedTable, ...] = ()


def _page_tables(page: Any) -> tuple[list[_TableBand], list[tuple[float, float, float, float]]]:
    """Every table on the page that actually renders, ordered down the page.

    Returns the merged vertical BANDS (used to splice the rows into the narrative in reading
    order) and the individual table BBOXES (used to decide which characters the tables own).
    They are separate on purpose: merging two side-by-side tables into one band is right for
    ordering, but their union bbox would swallow any narrative sitting between them.

    A candidate that does not render (too few rows — layout noise) contributes NEITHER, so its
    text stays in the narrative instead of being excluded into a table that is never emitted."""
    found: list[tuple[float, float, float, float, str, _DetectedTable]] = []
    for table in page.find_tables():
        cells = table.extract()
        body = serialize_table(cells)
        if body:
            x0, top, x1, bottom = table.bbox
            detected = _DetectedTable(bbox=(x0, top, x1, bottom), cells=tuple(tuple(row) for row in cells))
            found.append((top, x0, bottom, x1, body, detected))

    bands: list[_TableBand] = []
    for top, _x0, bottom, _x1, body, detected in sorted(found, key=lambda f: f[:5]):
        if bands and top < bands[-1].bottom:  # vertically overlaps the previous band
            previous = bands[-1]
            bands[-1] = _TableBand(
                previous.top,
                max(previous.bottom, bottom),
                f"{previous.body}\n\n{body}",
                (*previous.tables, detected),
            )
        else:
            bands.append(_TableBand(top, bottom, body, (detected,)))
    return bands, [(x0, top, x1, bottom) for top, x0, bottom, x1, _, _ in found]


def _owned_by_table(obj: dict[str, Any], bboxes: list[tuple[float, float, float, float]]) -> bool:
    """Does this object belong to a table? Decided ONCE, by its centre point, so every
    character has exactly one owner — a table or the narrative. Testing containment of the
    centre rather than of the whole box matters: pdfplumber's crop CLIPS rather than drops, so
    a character straddling a boundary would otherwise be emitted on both sides."""
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


def _band_crop(page: Any, top: float, bottom: float) -> Any | None:
    """One horizontal slice of a page as its own cropped page, or None when the slice is empty.

    The crop is returned rather than only its text because the layout overlay has to be read
    off the SAME crop the text came from — see :func:`_narrative_blocks`. Two crops built from
    the same numbers would still be two chances to drift apart."""
    _, page_top, _, page_bottom = page.bbox
    top, bottom = max(top, page_top), min(bottom, page_bottom)
    if bottom <= top:
        return None
    return page.crop((page.bbox[0], top, page.bbox[2], bottom))


def _band_text(crop: Any | None) -> str:
    """Text of one horizontal slice of a page, in pdfplumber's own reading order."""
    return "" if crop is None else str(crop.extract_text() or "")


_PDF_PRIMARY = "pdfplumber"  # layout-aware, captures tables (FR-B2)
_PDF_PRIMARY_CAPS = (CAP_TEXT, CAP_TABLES)
_PDF_FALLBACK = "pypdf"  # text layer only — NO table extraction
_PDF_FALLBACK_CAPS = (CAP_TEXT,)
_PDF_LOST_CAPS = sorted(set(_PDF_PRIMARY_CAPS) - set(_PDF_FALLBACK_CAPS))  # derived, not asserted


@dataclass(slots=True)
class _ReadPage:
    """One page as read, BEFORE normalization — text plus who produced it.

    Kept as a single list so a page's text can never drift from its provenance; the
    :class:`PageRecord`s are built from it and the normalized texts in one zip."""

    text: str
    extractor: str
    capabilities: tuple[str, ...]
    degradations: tuple[str, ...]
    has_image_content: bool | None
    tables_found: bool
    table_chars: int = 0
    source_chars: int = 0
    # Defaults to False for readers that expose no vector geometry (the pypdf fallback). Absence
    # of the signal is not evidence of blankness -- has_image_content carries what they do know.
    has_vector_ink: bool = False
    # The page's `layout-json` overlay, or None. Stays None on the pypdf fallback: that reader
    # exposes no per-glyph geometry, so there is nothing to measure and fabricating a bbox-less
    # layout block would claim this path can do what it cannot.
    markup: str | None = None


class _PypdfPages:
    """Lazy pypdf view of the same bytes, built only when a page actually needs the fallback.

    A document where pdfplumber handles every page never constructs a PdfReader at all."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._reader: Any = None

    @property
    def _pages(self) -> Any:
        if self._reader is None:
            from pypdf import PdfReader  # noqa: PLC0415 - heavy dep, imported on use

            self._reader = PdfReader(io.BytesIO(self._data))
        return self._reader.pages

    def count(self) -> int:
        return len(self._pages)

    def text(self, index: int) -> str:
        return str(self._pages[index].extract_text() or "")

    def has_image_content(self, index: int) -> bool | None:
        """Does this page draw any image XObject? Reads the resource dictionary only — no
        decoding, no rasterising. ``None`` means undetermined, which routing treats as
        "might need OCR" rather than silently assuming the page is blank."""
        try:
            xobjects = self._pages[index].get("/Resources", {}).get("/XObject")
            if xobjects is None:
                return False
            return any(ref.get_object().get("/Subtype") == "/Image" for ref in xobjects.get_object().values())
        except Exception as exc:
            _log.warning("pdf_page_image_probe_failed", page=index + 1, error_type=type(exc).__name__, error=str(exc))
            return None


# --- the born-digital layout overlay -------------------------------------------------------
# A THIRD markup language on `PageRecord.markup`, alongside the two Chandra answers with. It is
# neither "html" nor "markdown" because it is not a flattened rendering of a page: it is the
# per-block MEASUREMENTS a born-digital PDF can give and a chat-shaped VLM cannot — a bounding
# box and the typography of every line, read off `page.chars` (`fontname`, `size`, `x0/x1/
# top/bottom` per glyph).
#
# This is the path that matters for this corpus. `application/pdf` classifies as
# `DocKind.NATIVE_PDF`, and a born-digital page only reaches an OCR engine when
# `page_needs_ocr` fires — so a tender or policy PDF produced by a word processor never sees
# Chandra, produced no markup, and got no blocks at all. Measured on the live corpus: 1 of 6
# documents had Chandra markup.
#
# WHY A DECLARED NEW FORMAT RATHER THAN SYNTHESISED HTML: emitting `<h1>`/`<p>` here would mean
# deciding, in the extractor, which lines are headings — a verdict that needs the document's own
# body-size distribution and belongs to the parser (Phase 2c-2). Worse, it would throw away the
# coordinates and the font metrics, which are the only reason this path is worth having. So the
# overlay carries EVIDENCE and the parser draws the conclusion.
#
# COMPLIANCE (same position as the raw bytes and the Chandra overlay, and for the same reasons —
# see `svc_control_plane.pipeline._persist_layout_overlay`): this is captured from the UNMASKED
# page, BEFORE PII masking, so it may contain personal data. It is written under the tenant's own
# object namespace, at the same `layout_object_key(doc_id)` by the same helper, so it is purged by
# the same `delete_document` (which DPDP erasure and KB delete both cascade through) and the same
# failed-ingest rollback — both of which delete that key UNCONDITIONALLY, so the artifact cannot
# outlive the document or the flag. It is never chunked, embedded or indexed, and `text` is never
# rebuilt from it.
MARKUP_FORMAT_LAYOUT_JSON = "layout-json"
# Bumped when the per-page payload's SHAPE changes. Separate from `LAYOUT_OVERLAY_VERSION`, which
# versions the document-level envelope: that one is the writer's contract with the artifact store,
# this one is the writer's contract with whatever parses a page's blocks.
LAYOUT_JSON_VERSION = 1

# `block_type` values. Deliberately the same STRINGS as `rag_core.layout`'s block vocabulary
# where this path can honestly claim one, duplicated rather than imported because doc-intelligence
# does not depend on rag-core.
#
# The asymmetry is the point. A table is a table BY DETECTION — pdfplumber found ruling lines and
# the serializer accepted the grid — so `table` is an observation. A line of narrative text is
# only ever a line: whether it is a heading, a paragraph or a list item is a judgement about its
# typography RELATIVE to the rest of the document, which is why this path reports `line` plus the
# measurements and leaves the classification to the parser. Inventing `heading` here would be the
# same guess `sections.py` already refuses to make from numbering, with a nicer-looking label.
_BLOCK_LINE = "line"
_BLOCK_TABLE = "table"

# A line counts as bold when MORE than this share of its glyphs are in a bold face. A share, not
# "any bold glyph": a paragraph with a bolded lead-in ("**Note:** the tenderer shall…") is not a
# bold line, and a heading with one accidentally-roman glyph still is. Majority-of-glyphs is an
# order statistic, so no single glyph can move it.
_BOLD_CHAR_SHARE = 0.6
# Substrings that mark a PostScript font name as bold. Lowercased before matching, so this covers
# "Arial-BoldMT", "TimesNewRoman,Bold", "Calibri-Semibold" and "Helvetica-Black".
_BOLD_FONT_MARKERS = ("bold", "black", "heavy")
_BBOX_DP = 2  # stored precision for coordinates, in PDF points — ~0.35 µm, far below any glyph


@dataclass(frozen=True, slots=True)
class _LayoutBlock:
    """One block of a born-digital page's layout overlay.

    ``bbox`` is ``(x0, top, x1, bottom)`` in PDF points in pdfplumber's own coordinate space —
    origin at the page's TOP-left, ``top`` growing downwards — the same space the page's width
    and height are reported in, so a consumer can place the block in the page (which is what the
    plan's "recurs in the same page region" header/footer rule needs).

    ``font_size`` and ``bold`` are None on a ``table`` block: a table band spans many lines in
    whatever faces its cells use, and reporting one of them as "the" typography of the band would
    be a claim about nothing. ``cells`` is None on a ``line`` block for the mirror reason."""

    block_type: str
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float | None = None
    bold: bool | None = None
    cells: tuple[tuple[str | None, ...], ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The stable, JSON-serialisable form. Every key is ALWAYS present (null where it does
        not apply) so a consumer branches on `block_type`, never on which keys it happens to
        find."""
        return {
            "block_type": self.block_type,
            "text": self.text,
            "page": self.page,
            "bbox": [round(v, _BBOX_DP) for v in self.bbox],
            "font_size": self.font_size,
            "bold": self.bold,
            "cells": None if self.cells is None else [list(row) for row in self.cells],
        }


def _line_typography(chars: list[dict[str, Any]]) -> tuple[float | None, bool | None]:
    """A line's font size and weight, from the glyphs that actually drew it.

    MEDIAN size, not maximum or mean: a superscript footnote marker, a trailing registered-trade-
    mark sign or a dropped capital would each drag a maximum (and, on a short line, a mean) away
    from the size a reader would call the line's. The median is an order statistic — it takes
    more than half the line's glyphs to move it.

    Returns (None, None) for a line with no glyph metrics at all, so "not measured" never reads
    as "size 0, not bold"."""
    sizes = [float(c["size"]) for c in chars if c.get("size") is not None]
    if not sizes:
        return None, None
    fonts = [str(c.get("fontname") or "") for c in chars]
    bold = sum(1 for f in fonts if any(m in f.lower() for m in _BOLD_FONT_MARKERS))
    return round(statistics.median(sizes), _BBOX_DP), bold > _BOLD_CHAR_SHARE * len(fonts)


def _narrative_blocks(crop: Any, page_no: int) -> list[_LayoutBlock]:
    """One block per text line of a narrative slice, in pdfplumber's own reading order.

    ``crop`` is the EXACT cropped page whose ``extract_text()`` produced this slice's text (see
    :func:`_band_crop`), so the lines are the lines of that text: the table characters are
    already filtered out of it by the same ownership test that spliced the tables elsewhere, and
    no second detection decides anything here."""
    blocks: list[_LayoutBlock] = []
    for line in crop.extract_text_lines():
        text = str(line.get("text") or "")
        if not text:
            continue
        size, bold = _line_typography(list(line.get("chars") or []))
        blocks.append(
            _LayoutBlock(
                block_type=_BLOCK_LINE,
                text=text,
                page=page_no,
                bbox=(float(line["x0"]), float(line["top"]), float(line["x1"]), float(line["bottom"])),
                font_size=size,
                bold=bold,
            )
        )
    return blocks


def _table_block(band: _TableBand, page_no: int) -> _LayoutBlock:
    """One block for one spliced table band — ``text`` is the band's body VERBATIM.

    Per BAND, not per detected table, and that is what makes the overlay and the text
    unable to disagree: the splice emits exactly one body per band, so one block per band means
    the overlay's table count, order and text are the text's own. ``cells`` is the concatenation
    of the band's tables' own grids (a band holds more than one only when two overlapping tables
    were merged to be ordered as a unit)."""
    x0 = min((t.bbox[0] for t in band.tables), default=0.0)
    x1 = max((t.bbox[2] for t in band.tables), default=0.0)
    cells = tuple(row for table in band.tables for row in table.cells)
    return _LayoutBlock(
        block_type=_BLOCK_TABLE,
        text=band.body,
        page=page_no,
        bbox=(x0, band.top, x1, band.bottom),
        cells=cells,
    )


def _layout_markup(page: Any, page_no: int, blocks: list[_LayoutBlock]) -> str:
    """One page's blocks as the stored ``layout-json`` payload.

    The page's own size travels with the blocks because a bbox is meaningless without it: "this
    block sits in the top 8% of the page" is the question the header/footer rule asks, and it
    cannot be answered from coordinates alone."""
    return json.dumps(
        {
            "format": MARKUP_FORMAT_LAYOUT_JSON,
            "version": LAYOUT_JSON_VERSION,
            "page_no": page_no,
            "page_width": round(float(page.width), _BBOX_DP),
            "page_height": round(float(page.height), _BBOX_DP),
            "blocks": [b.as_dict() for b in blocks],
        },
        ensure_ascii=False,
    )


@dataclass(frozen=True, slots=True)
class _Piece:
    """One piece of the page's text, with whatever produced it still attached.

    This is what keeps the overlay honest. ``text`` is what went into the page's text, and the
    ``band`` or ``crop`` beside it is the thing the splice used to produce that text — so the
    blocks are read off the splice's own decisions rather than from a second pass over the page
    that could disagree with them about where a table was."""

    text: str
    band: _TableBand | None = None
    crop: Any | None = None


@dataclass(frozen=True, slots=True)
class _PrimaryPage:
    """What the table-aware path recovered from one page, and how much of it a table claimed.

    One value, because all three are decided by the SAME ownership test — separating them
    would let the flag, the text and the share drift from one another."""

    text: str
    tables_found: bool
    table_chars: int = 0
    source_chars: int = 0
    # The `layout-json` overlay for this page, or None when it was not asked for (see
    # `_read_primary_page`'s `capture_layout`). An OVERLAY on `text`, never an input to it.
    markup: str | None = None


def _read_primary_page(page: Any, *, page_no: int, capture_layout: bool = False) -> _PrimaryPage:
    """One page's text in reading order, with each table represented EXACTLY ONCE.

    ``page.extract_text()`` already returns every table cell, so appending the rendered rows
    on top of it put each table in the text twice. The structured form is the one that
    survives: the table's own characters are removed from the narrative and its rows are
    spliced back in at the table's real vertical position. The narrative around it is still
    pdfplumber's own reading order, just taken band by band.

    The same per-character ownership test that performs the splice also counts what it
    claimed, so the share reported can never disagree with the text produced.

    ``capture_layout`` additionally derives the ``layout-json`` overlay (see
    :data:`MARKUP_FORMAT_LAYOUT_JSON`) FROM THE PIECES THIS SPLICE JUST USED — the same bands,
    the same table-ownership filter, the same crops — rather than detecting anything a second
    time. It is off by default and gated on ``v2_ingest`` at the service seam, because
    unlike Chandra's markup (which already arrived in the response and was being discarded) a
    layout pass is real work per page. ``text`` is byte-identical either way, and a failure
    while deriving the overlay costs the page its overlay, never its text."""
    bands, bboxes = _page_tables(page)
    chars = page.chars
    if not bands:
        # No table on the page, so the narrative IS the page and `page` is its own "crop".
        pieces = [_Piece(text=str(page.extract_text() or ""), crop=page)]
        table_chars = 0
    else:
        narrative = page.filter(lambda obj: not _owned_by_table(obj, bboxes))
        pieces = []
        cursor = page.bbox[1]
        for band in bands:
            crop = _band_crop(narrative, cursor, band.top)
            above = _band_text(crop)
            if above:
                pieces.append(_Piece(text=above, crop=crop))
            pieces.append(_Piece(text=band.body, band=band))
            cursor = band.bottom
        tail_crop = _band_crop(narrative, cursor, page.bbox[3])
        tail = _band_text(tail_crop)
        if tail:
            pieces.append(_Piece(text=tail, crop=tail_crop))
        # Same predicate as the filter above — this is a count of what was spliced out, not a
        # second opinion about where the tables were.
        table_chars = sum(1 for c in chars if _owned_by_table(c, bboxes))

    markup: str | None = None
    if capture_layout:
        try:
            blocks: list[_LayoutBlock] = []
            for piece in pieces:
                if piece.band is not None:
                    blocks.append(_table_block(piece.band, page_no))
                elif piece.crop is not None:
                    blocks.extend(_narrative_blocks(piece.crop, page_no))
            markup = _layout_markup(page, page_no, blocks)
        except Exception as exc:
            # The overlay is additive; losing it must never cost a good page its text, so this
            # does NOT fall through to the pypdf fallback (which would drop the page's tables).
            _log.warning(
                "pdf_layout_capture_failed",
                page=page_no,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )

    # Blank lines around the table, so a page break is not the only place the clause chunker
    # can split a table-bearing page.
    return _PrimaryPage(
        text="\n\n".join(p.text for p in pieces),
        tables_found=bool(bands),
        table_chars=table_chars,
        source_chars=len(chars),
        markup=markup,
    )


def _unreadable(doc_id: str, fmt: str, *, primary: Exception, fallback: Exception | None = None) -> ValidationError:
    """The error raised once every extractor for a format has failed on a file.

    Kept narrow on purpose. Only "we have run out of extractors" becomes a 422 — a statement
    about the upload. Anything else still propagates as itself, because converting our own
    bugs into 422s would blame the user for them and hide the fault."""
    causes = [f"{type(primary).__name__}: {primary}"]
    if fallback is not None:
        causes.append(f"{type(fallback).__name__}: {fallback}")
    _log.warning(
        "file_unreadable",
        doc_id=doc_id,
        format=fmt,
        causes=causes,
        detail="every extractor for this format failed to parse the bytes",
    )
    return ValidationError(
        f'"{doc_id}" could not be read as a {fmt}: the file appears to be corrupt, truncated, '
        "or not actually that format. Re-export it and upload again.",
        details={"doc_id": doc_id, "format": fmt, "causes": causes},
    )


# "Is there anything on this page for OCR to read?" -- the input to page_needs_ocr's blank-page
# test, which skips a page that has neither text nor ink so a duplex back or separator sheet does
# not cost an OCR call.
#
# This used to be ``bool(page.images)``, i.e. raster images only, and that quietly lost two thirds
# of a document. A scanned SOP guidance PDF came through a converter that traced its glyphs into
# VECTOR OUTLINES: every page had zero characters and zero images, but thousands of curves. Pages
# 1 and 2 were therefore classified blank and skipped; page 3 happened to retain two raster images
# and was the only page OCR'd. The document indexed at 4,375 characters -- its numbering section,
# its font table and both department-code tables simply were not there.
#
# The failure mode that makes this worth a guard rather than a footnote: nothing was reported. The
# textless refusal did not fire (page 3 had text), so retrieval was honest, faithfulness scored
# 1.00, and the answer was confidently drawn from the only sections that survived. A silent
# partial extraction reads exactly like a complete one.
#
# The threshold keeps the original intent. A blank page has no vector objects at all; a page with
# a ruled header line has a handful; a page of outlined text has thousands.
_MIN_VECTOR_OBJECTS = 12


# An image only counts for the pdf_image_pages policy if it is big enough to plausibly BE the
# content. That policy exists to catch a figure, table or chart rendered as a picture, and it
# sends the whole page through OCR to get it. A letterhead logo is not a figure -- but it is a
# raster image on every page, so a 47-page SOP with branding had all 47 pages routed to OCR,
# discarding a perfectly good 87,000-character text layer and replacing it with a slower,
# lossier reading of the same words.
#
# 5% of page area is about 3cm x 3cm on A4: far above any logo, icon or rule, and far below any
# real table or chart. Raise it for corpora whose figures are large; lower it if small but
# meaningful diagrams are being missed.
_MIN_IMAGE_AREA_SHARE = 0.05


def _page_has_substantive_image(page: Any) -> bool:
    """True when the page carries a raster image large enough to be content rather than branding."""
    images = getattr(page, "images", None)
    if not images:
        return False
    try:
        page_area = float(page.width) * float(page.height)
    except Exception:
        return True  # no geometry to judge by -- assume it matters rather than skip it
    if page_area <= 0:
        return True
    for image in images:
        width = abs(float(image.get("x1", 0)) - float(image.get("x0", 0)))
        height = abs(float(image.get("bottom", 0)) - float(image.get("top", 0)))
        if (width * height) / page_area >= _MIN_IMAGE_AREA_SHARE:
            return True
    return False


def _page_has_vector_ink(page: Any) -> bool:
    """True when the page carries enough VECTOR drawing to be real content rather than blank.

    Deliberately says nothing about raster images -- that is ``has_image_content``, and the two
    are consumed by different decisions (see the field comment on :class:`PageRecord`).
    """
    vectors = len(page.curves) + len(page.rects) + len(page.lines)
    return vectors >= _MIN_VECTOR_OBJECTS


def _layout_capture_allowed(page_count: int, doc_id: str) -> bool:
    """Whether to capture layout for a document of this many pages.

    A SECOND pdfplumber pass per page (`_narrative_blocks` re-clusters every crop band's chars) and
    one JSON object holding every line's bbox and font metrics are both linear in page count, with
    no bound. Measured on a tender-shaped fixture (29 blocks/page): the capture adds 0-20% to
    extraction time — notably NOT the ~2x a review assumed — and the overlay runs 6.7 KiB/page, so
    a 2000-page tender is ~13 MiB of JSON and ~58k blocks, serialized, written to MinIO, fetched
    back and re-parsed into one dataclass per line, inside the single-api event loop.

    So the cap is a GUARD against a pathological document, not a size policy: the default is
    deliberately far above any real tender, because a 2000-page corrigendum is precisely the
    document this feature exists for and quietly excluding it would be the wrong failure. Over the
    cap, capture is skipped for the WHOLE document rather than the first N pages — a partial overlay
    would make `_chunkable_layout` decline the document anyway (its block stream must cover every
    page that produced text), so a partial capture would be pure cost for the same outcome.

    NOT recorded as a `degradations` reason, deliberately. Every reason in that channel means the
    extracted TEXT lost fidelity — a fallback extractor, a failed OCR pass — and it drives
    `extraction_degraded` in the document's provenance. Here the text is byte-identical and only an
    optional structural overlay is absent, so stamping it would make a perfectly-extracted document
    report itself degraded. The WARN below is the record, exactly as it is for the sibling case
    `pdf_layout_capture_failed`, which is also logged and also non-fatal.
    """
    limit = get_settings().layout_capture_max_pages
    if page_count <= limit:
        return True
    _log.warning("pdf_layout_capture_page_cap", doc_id=doc_id, pages=page_count, limit=limit)
    return False


def extract_pdf_text(data: bytes, *, doc_id: str, capture_layout: bool = False) -> Extraction:
    """Extract text per page from a born-digital PDF — layout-aware with table capture
    (pdfplumber, FR-B2), normalized (de-hyphenation, running header/footer removal), with a
    pypdf fallback PER PAGE when pdfplumber cannot read one.

    The fallback is a real loss of fidelity (pypdf has no table extraction), so it is
    CONTAINED: only the page that failed is re-read and stamped, and the pages that succeeded
    keep their tables. A failure to open the file at all is not attributable to any page and
    degrades every page instead. ``doc_id`` is required so the WARN is actionable.

    ``capture_layout`` additionally records each pdfplumber-read page's ``layout-json`` overlay
    on its :class:`PageRecord` (see :data:`MARKUP_FORMAT_LAYOUT_JSON`), subject to
    :func:`_layout_capture_allowed`. Default OFF, and the
    default is load-bearing: with it off not one extra pdfplumber call is made and ``text`` is
    byte-identical, which is the whole of the safety case for the overlay being additive. The
    overlay is PRE-normalization, exactly as Chandra's markup is — ``_normalize_pages`` runs over
    the joined page string (de-hyphenation, running header/footer removal), so a block's text is
    what the page drew, not necessarily a substring of the page's final ``text``."""
    fallback = _PypdfPages(data)
    read: list[_ReadPage] = []
    open_failure: Exception | None = None

    try:
        import pdfplumber  # noqa: PLC0415 - heavy dep, imported on use

        opened = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:
        open_failure = exc
    else:
        with opened as pdf:
            # Decided ONCE, from the page count, before any page is read — so an over-cap document
            # never pays for a single extra pass.
            capture_layout = capture_layout and _layout_capture_allowed(len(pdf.pages), doc_id)
            for index, page in enumerate(pdf.pages):
                try:
                    primary = _read_primary_page(page, page_no=index + 1, capture_layout=capture_layout)
                    read.append(
                        _ReadPage(
                            text=primary.text,
                            extractor=_PDF_PRIMARY,
                            capabilities=_PDF_PRIMARY_CAPS,
                            degradations=(),
                            has_image_content=_page_has_substantive_image(page),
                            has_vector_ink=_page_has_vector_ink(page),
                            tables_found=primary.tables_found,
                            table_chars=primary.table_chars,
                            source_chars=primary.source_chars,
                            markup=primary.markup,
                        )
                    )
                except Exception as exc:
                    _log.warning(
                        "pdf_page_extraction_failed",
                        doc_id=doc_id,
                        page=index + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        from_extractor=_PDF_PRIMARY,
                        to_extractor=_PDF_FALLBACK,
                        lost_capabilities=_PDF_LOST_CAPS,
                        exc_info=True,
                    )
                    read.append(_fallback_page(fallback, index))

    if open_failure is not None:
        # No page could be read by the primary — every page falls back. Attributing this to a
        # page would be an invention: pdfplumber never got far enough to have a current page.
        try:
            read = [_fallback_page(fallback, index) for index in range(fallback.count())]
        except Exception as exc:
            # Both extractors are out. That is a statement about the FILE, so say so with a
            # 422 rather than letting pypdf's own exception surface as a server fault.
            raise _unreadable(doc_id, "PDF", primary=open_failure, fallback=exc) from exc

    normalized = _normalize_pages([r.text for r in read])
    pages = [
        PageRecord(
            page_no=i,
            text=page.text,
            extractor=r.extractor,
            capabilities=r.capabilities,
            degradations=r.degradations,
            has_image_content=r.has_image_content,
            has_vector_ink=r.has_vector_ink,
            tables_found=r.tables_found,
            table_chars=r.table_chars,
            source_chars=r.source_chars,
            boilerplate_lines=page.boilerplate_lines,
            markup=r.markup,
            # The pairing `PageRecord.__post_init__` enforces, satisfied at the ONE site that can:
            # a format is declared exactly when there is markup to declare it for, derived from
            # the same value rather than set beside it.
            markup_format=MARKUP_FORMAT_LAYOUT_JSON if r.markup is not None else "",
        )
        # strict=True: _normalize_pages must return one entry per page, or page numbers would
        # silently shift away from the text they label.
        for i, (page, r) in enumerate(zip(normalized, read, strict=True), start=1)
    ]

    degraded = [p.page_no for p in pages if REASON_EXTRACTOR_FALLBACK in p.degradations]
    if degraded:
        _log.warning(
            "pdf_extraction_degraded",
            doc_id=doc_id,
            pages_total=len(pages),
            pages_degraded=len(degraded),
            degraded_pages=degraded,
            from_extractor=_PDF_PRIMARY,
            to_extractor=_PDF_FALLBACK,
            lost_capabilities=_PDF_LOST_CAPS,
            **(
                {"error_type": type(open_failure).__name__, "error": str(open_failure)}
                if open_failure is not None
                else {}
            ),
        )
    _log.info("pdf_extracted", doc_id=doc_id, pages=len(pages), chars=sum(len(p.text) for p in pages))
    return Extraction(pages=pages, modality=Modality.TEXT)


def _fallback_page(fallback: _PypdfPages, index: int) -> _ReadPage:
    return _ReadPage(
        text=fallback.text(index),
        extractor=_PDF_FALLBACK,
        capabilities=_PDF_FALLBACK_CAPS,
        degradations=(REASON_EXTRACTOR_FALLBACK,),
        has_image_content=fallback.has_image_content(index),
        # The fallback reader exposes no vector geometry; absence is not evidence of blankness,
        # and has_image_content already carries whatever it does know.
        tables_found=False,  # pypdf has no table extraction — nothing to observe
    )


def split_pdf_page(data: bytes, page_no: int) -> bytes:
    """Extract ONE 1-based page as a standalone single-page PDF.

    Structural copy only — no rasterising, no re-encoding of content streams (measured ~5 ms
    and ~1 KB per page). This is what lets a single page be sent to an OCR engine through the
    unchanged ``OCREngine.ocr`` seam: poppler then parses a one-page file instead of the whole
    document, so memory stays at one page rather than every page at OCR DPI."""
    from pypdf import PdfReader, PdfWriter  # noqa: PLC0415 - heavy dep, imported on use

    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(data)).pages[page_no - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_BODY_PART = "word/document.xml"  # the archive member that identifies a .docx
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
EXTRACTOR_DOCX = "docx"


def extract_docx_text(data: bytes, *, doc_id: str) -> Extraction:
    """Extract paragraphs + tables from a .docx in document order (stdlib zip + safe XML
    via defusedxml — a .docx is untrusted user input)."""
    import zipfile  # noqa: PLC0415

    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read(_DOCX_BODY_PART)
        root = SafeET.fromstring(xml)
    except Exception as exc:
        # A broken archive, a missing part, or unparseable XML are all the same statement to
        # the uploader: this is not a readable .docx. The bare KeyError this used to raise
        # named an internal archive path the uploader has never heard of.
        raise _unreadable(doc_id, "Word document", primary=exc) from exc
    body = root.find(f"{_DOCX_NS}body")
    parts: list[str] = []
    tables_found = False  # observed like the PDF path, so a corpus count spans both formats
    if body is not None:
        for child in body:
            if child.tag == f"{_DOCX_NS}p":
                text = "".join(t.text or "" for t in child.iter(f"{_DOCX_NS}t"))
                if text.strip():
                    parts.append(text.strip())
            elif child.tag == f"{_DOCX_NS}tbl":
                # Same serializer as the PDF path — a w:tbl row and a pdfplumber row are the
                # same problem, so header detection and escaping cannot diverge between formats.
                rendered = serialize_table(
                    [
                        [
                            "".join(t.text or "" for t in tc.iter(f"{_DOCX_NS}t")).strip()
                            for tc in tr.iter(f"{_DOCX_NS}tc")
                        ]
                        for tr in child.iter(f"{_DOCX_NS}tr")
                    ],
                    # A w:tbl is a table by declaration, not by inference — a one-row one is
                    # one row of real data, and was being silently dropped.
                    declared=True,
                )
                if rendered:
                    parts.append(rendered)
                    tables_found = True  # the same value that reached the text decides the flag
    text = "\n".join(parts)
    _log.info("docx_extracted", chars=len(text), blocks=len(parts), tables_found=tables_found)
    # A .docx has no fixed pagination (it is reflowed at render time), so the whole document
    # is one record. Page-accurate citation for Office formats is a separate problem.
    return Extraction(
        pages=_uniform_pages(
            [text],
            extractor=EXTRACTOR_DOCX,
            capabilities=(CAP_TEXT, CAP_TABLES),  # w:tbl rows are recovered as pipe-separated lines
            tables_found=tables_found,
        ),
        modality=Modality.TEXT,
    )


# --- The other two OOXML families ------------------------------------------------------------
# Same approach as .docx: stdlib zip + defusedxml, no new dependency. openpyxl/python-pptx would
# each pull a tree of transitive deps to read what is, in both cases, a handful of XML parts.
_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PRES_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_DRAW_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_XLSX_BOOK_PART = "xl/workbook.xml"
_PPTX_DECK_PART = "ppt/presentation.xml"
EXTRACTOR_XLSX = "xlsx"
EXTRACTOR_PPTX = "pptx"

_COLUMN_LETTERS = re.compile(r"^([A-Z]+)")


def _rels_for(z: Any, part: str) -> dict[str, str]:
    """Relationship id -> target part, resolved relative to ``part``'s own directory.

    OOXML never states document order as a filename convention; it states it as a list of
    relationship ids. Reading the parts in filename order happens to work for files written
    by Office and silently reorders documents written by anything else."""
    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    directory = part.rsplit("/", 1)[0]
    rels_part = f"{directory}/_rels/{part.rsplit('/', 1)[1]}.rels"
    root = SafeET.fromstring(z.read(rels_part))
    out: dict[str, str] = {}
    for rel in root.iter(f"{_PKG_REL_NS}Relationship"):
        target = rel.get("Target", "")
        out[rel.get("Id", "")] = target if target.startswith("/") else f"{directory}/{target}"
    return out


def _ordered_parts(z: Any, root_part: str, list_tag: str, item_tag: str) -> list[tuple[str, str]]:
    """The document's own ordering of its child parts, as ``[(label, part name), ...]``.

    ``label`` is the sheet name where the format carries one, and empty otherwise."""
    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    rels = _rels_for(z, root_part)
    root = SafeET.fromstring(z.read(root_part))
    parts: list[tuple[str, str]] = []
    for container in root.iter(list_tag):
        for item in container:
            if item.tag != item_tag:
                continue
            target = rels.get(item.get(f"{_REL_NS}id", ""))
            if target:
                parts.append((item.get("name", ""), target))
    return parts


def _shared_strings(z: Any) -> list[str]:
    """Excel keeps cell text in one table and puts an INDEX in the cell. Without this, a
    workbook indexes as integers."""
    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    try:
        root = SafeET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []  # a workbook of only numbers/formulas has no shared-string table
    return ["".join(t.text or "" for t in si.iter(f"{_SHEET_NS}t")) for si in root.iter(f"{_SHEET_NS}si")]


def _column_index(ref: str) -> int:
    """``"C7"`` -> 2. Cells are addressed, not positional: a row that skips a column omits the
    cell entirely, so reading them in sequence would shift every later value left."""
    match = _COLUMN_LETTERS.match(ref.upper())
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _sheet_rows(z: Any, part: str, strings: list[str]) -> list[list[str]]:
    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    root = SafeET.fromstring(z.read(part))
    rows: list[list[str]] = []
    for row in root.iter(f"{_SHEET_NS}row"):
        cells: dict[int, str] = {}
        for cell in row.iter(f"{_SHEET_NS}c"):
            if cell.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in cell.iter(f"{_SHEET_NS}t"))
            else:
                raw = cell.findtext(f"{_SHEET_NS}v") or ""
                value = strings[int(raw)] if cell.get("t") == "s" and raw.isdigit() and int(raw) < len(strings) else raw
            cells[_column_index(cell.get("r", ""))] = value.strip()
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def extract_xlsx_text(data: bytes, *, doc_id: str) -> Extraction:
    """Extract a workbook as one record per SHEET.

    A sheet is not a printed page, but it is the unit a reader names ("the Leave sheet"), so
    it is the useful thing for a citation to point at — the same judgement the .docx path makes
    in the other direction when it says a reflowed document has no pages at all.

    Rows go through the SHARED serializer, so a spreadsheet row and a PDF table row are the
    same shape in the index and header detection cannot diverge between the two."""
    import zipfile  # noqa: PLC0415

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            strings = _shared_strings(z)
            sheets = _ordered_parts(z, _XLSX_BOOK_PART, f"{_SHEET_NS}sheets", f"{_SHEET_NS}sheet")
            # `declared`: a sheet IS a grid — the minimum-row guard is for inferred tables.
            rendered = [(name, serialize_table(_sheet_rows(z, part, strings), declared=True)) for name, part in sheets]
    except Exception as exc:
        raise _unreadable(doc_id, "spreadsheet", primary=exc) from exc

    texts = [f"{name}\n{body}" if name else body for name, body in rendered]
    found = any(body for _, body in rendered)
    _log.info("xlsx_extracted", doc_id=doc_id, sheets=len(texts), chars=sum(len(t) for t in texts))
    return Extraction(
        pages=_uniform_pages(
            texts or [""], extractor=EXTRACTOR_XLSX, capabilities=(CAP_TEXT, CAP_TABLES), tables_found=found
        ),
        modality=Modality.TEXT,
    )


def extract_pptx_text(data: bytes, *, doc_id: str) -> Extraction:
    """Extract a deck as one record per SLIDE — a deck genuinely paginates, so a citation can
    name a slide number the reader will recognise.

    Text only: slide tables (``a:tbl``) contribute their cell text through the same ``a:t``
    elements, but their structure is NOT recovered, so CAP_TABLES is deliberately not claimed."""
    import zipfile  # noqa: PLC0415

    from defusedxml import ElementTree as SafeET  # noqa: PLC0415

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            slides = _ordered_parts(z, _PPTX_DECK_PART, f"{_PRES_NS}sldIdLst", f"{_PRES_NS}sldId")
            texts = []
            for _, part in slides:
                root = SafeET.fromstring(z.read(part))
                paragraphs = [
                    "".join(t.text or "" for t in para.iter(f"{_DRAW_NS}t")).strip()
                    for para in root.iter(f"{_DRAW_NS}p")
                ]
                texts.append("\n".join(p for p in paragraphs if p))
    except Exception as exc:
        raise _unreadable(doc_id, "presentation", primary=exc) from exc

    _log.info("pptx_extracted", doc_id=doc_id, slides=len(texts), chars=sum(len(t) for t in texts))
    return Extraction(
        pages=_uniform_pages(texts or [""], extractor=EXTRACTOR_PPTX, capabilities=(CAP_TEXT,)),
        modality=Modality.TEXT,
    )


def pptx_slide_parts(data: bytes) -> list[str]:
    """The deck's slide parts in PRESENTATION order — the order ``extract_pptx_text`` pages them.

    Shared with the OCR check so text read from a screenshot pasted on slide 4 is attached to
    page 4, not to whichever slide part happens to be named slide4.xml. Empty for an unreadable
    deck: the extractor has already raised the actionable error for that case."""
    import zipfile  # noqa: PLC0415

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return [part for _, part in _ordered_parts(z, _PPTX_DECK_PART, f"{_PRES_NS}sldIdLst", f"{_PRES_NS}sldId")]
    except Exception:
        return []


def extract_office_text(data: bytes, *, doc_id: str) -> Extraction:
    """Route a zip container to the extractor its CONTENTS call for.

    One entry point rather than three call sites, so the decision lives beside the marker
    table that made it. An archive we have no extractor for is named rather than pushed
    through the nearest parser to die on a missing part."""
    kind = sniff_archive(data)
    if kind is DocKind.OFFICE_DOC:
        return extract_docx_text(data, doc_id=doc_id)
    if kind is DocKind.OFFICE_SHEET:
        return extract_xlsx_text(data, doc_id=doc_id)
    if kind is DocKind.OFFICE_SLIDES:
        return extract_pptx_text(data, doc_id=doc_id)
    _log.warning("archive_format_unsupported", doc_id=doc_id)
    raise ValidationError(
        f'"{doc_id}" is a zip archive, but not one this platform can read. Word (.docx), Excel '
        "(.xlsx) and PowerPoint (.pptx) are supported; OpenDocument, EPUB and plain .zip are "
        "not — export it to one of those formats, or upload the documents inside it separately.",
        details={"doc_id": doc_id, "kind": kind.value},
    )


EXTRACTOR_PLAINTEXT = "plaintext"


def extract_plain_text(data: bytes) -> Extraction:
    text = data.decode("utf-8", errors="replace")
    return Extraction(
        pages=_uniform_pages([text], extractor=EXTRACTOR_PLAINTEXT, capabilities=(CAP_TEXT,)),
        modality=Modality.TEXT,
    )


class _HTMLTextParser(HTMLParser):
    """Visible-text extractor: skips script/style/nav chrome, keeps headings/paragraphs."""

    _SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "template", "head", "svg"})
    _BREAK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)


EXTRACTOR_HTML = "html"


def extract_html_text(data: bytes) -> Extraction:
    """Extract the visible text of an HTML page (web source, FR-A3) — no tags, no scripts."""
    parser = _HTMLTextParser()
    parser.feed(data.decode("utf-8", errors="replace"))
    # Collapse runs of whitespace but keep paragraph breaks for clause chunking.
    lines = [" ".join(chunk.split()) for chunk in "".join(parser.parts).split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    _log.info("html_extracted", chars=len(text))
    return Extraction(
        pages=_uniform_pages(
            [text],
            extractor=EXTRACTOR_HTML,
            capabilities=(CAP_TEXT,),  # <table> cells become flat text — structure is not preserved
        ),
        modality=Modality.TEXT,
    )
