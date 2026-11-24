"""Layout parser (RAG v2 Phase 2, "Structure tree") — pure logic, no I/O, no config.

Turns the per-page layout overlay the extractor persists (Phase 2a: raw engine markup, one
entry per page that carried any, stored in MinIO at ``documents.minio_layout_key``) into a
Document -> Section -> Subsection -> Block tree with breadcrumbs and the Phase-1 derived ids.

TWO INPUT PATHS, one tree. A page declares its own ``markup_format`` and is read by the branch
that format names; both branches feed ONE block stream in page order, so a scanned-and-appended
tender whose pages declare different formats is one document (see :func:`parse_layout`):

* ``html`` — Chandra's markup (Phase 2a/2b). A chat-shaped VLM: heading LEVEL is explicit in
  the tag, and there are no coordinates and no font metrics at all.
* ``layout-json`` — the born-digital PDF overlay (Phase 2c-1), one payload per page carrying
  per-line ``bbox`` + ``font_size`` + ``bold`` and pdfplumber's own table ``cells``. Its
  ``block_type`` is only ``line`` or ``table``: a table is an OBSERVATION (ruling lines were
  found), while heading-vs-paragraph is a JUDGEMENT about typography, and that judgement is
  made here (Phase 2c-2) — see :func:`_classify_headings`.

WHY THIS AND NOT :mod:`rag_core.sections`: ``sections.py`` infers headings from NUMBERING
in the text layer, so it has to be narrow — single-level numbering is refused outright because
on real corpora "1." is far more often a list item than a heading, and a wrong section path is
worse than none. The overlay carries stronger evidence: the tag says the level outright on the
``html`` path, and the type scale says it on the ``layout-json`` path. ``sections.py`` is
untouched and stays the authority for the plain-text path (:func:`_text_sections`) AND for a
measured page whose typography turns out to carry no signal at all (:func:`_number_headings`).

Where this deviates from `RAG_UPGRADE_PLAN.md` §"Phase 2", stated once here rather than at
every site:

* **``bbox`` exists only on the ``layout-json`` path.** The plan's input is "JSON/HTML with
  block types, page, bbox". Chandra answers HTML and returns NO coordinates (discovery §3,
  corrected 2026-10-19), so an ``html`` block carries no bounding box and no font metrics; its
  heading level comes from the tag alone. A ``layout-json`` block carries both.
* **"the same page REGION" means two different things, per path.** The plan says "drop blocks
  whose normalised text recurs in the same page REGION on >50% of pages". On a ``layout-json``
  page the region is TRUE GEOMETRY — a fraction of ``page_height`` at the top and at the
  bottom (:data:`_REGION_PAGE_SHARE`). On an ``html`` page there are no coordinates, so region
  is approximated by position in the page's block stream: up to :data:`_REGION_BLOCKS` blocks
  at each end, capped at half the page so the two can never meet. Both kinds of page vote in
  ONE recurrence count keyed on ``(region, slot, text)``, so a running footer that appears on
  pages of both kinds pools its evidence across the boundary. See :func:`_region_keys` and
  :func:`_strip_running_blocks`.
* **Cross-page table merge is built here, not reused.** The plan says to reuse an existing
  merge; discovery §3 found none exists — only an upper-bound METRIC (``ocr.py``'s
  ``table_page_runs``, whose own docstring says so). See :func:`_merge_cross_page_tables` for
  what it merges and, more importantly, what it refuses to. It merges across a format boundary
  too: the pages either side of the break need not have been read by the same branch.

THE OVERLAY IS PRE-NORMALISATION, both paths. ``ocr._normalize_pages`` strips running
headers/footers and de-hyphenates the JOINED page ``text`` AFTER the overlay is captured, so a
running header is still a block here although it is gone from ``text`` — which is precisely why
it can be stripped properly, with geometry, on this side. The consequence is that block text
and page text are NOT interchangeable: a consumer that needs the document's text must use the
``text`` it was given (see :func:`parse_layout`'s ``text`` argument), never a join of blocks.

Honest limits, so a later phase does not rediscover them as bugs. Each says which path it is
about, because the two no longer share them:

* ``html`` ONLY: a running header carrying a page number ("Page 3 of 9") does not recur as the
  same TEXT and is therefore kept. Digit masking would catch it but would also merge genuinely
  different short lines, and position-in-the-block-stream is too weak a region to contain that
  risk. On ``layout-json`` the region is strict geometry, which does contain it, so digit-
  insensitive matching IS performed there and ONLY there — see :func:`_region_keys`. That is
  also strictly stronger than the TEXT layer's own running-header removal, whose
  ``ocr._furniture_key`` generalises a "Page N of M"-shaped line but names "a footer of other
  words plus a number ('HR Handbook 4.0 | 6')" as the case it must leave alone, because
  recognising it "needs evidence this rule does not have — that the words are furniture rather
  than content". A geometric band is that evidence.
* ``html`` ONLY: headings are never stripped as running text, so an extractor that marks its
  page header up as ``<h1>`` on every page yields one section per page rather than one section
  for the document. Keeping a duplicate heading costs a duplicate section; dropping a real one
  costs every block beneath it its breadcrumb. On ``layout-json`` the question does not arise:
  running-text removal runs BEFORE typography is classified, so page furniture is gone before
  it can be promoted to a heading (and before it can vote on the document's body size).
* ``html`` ONLY: **on a 2- or 3-block page the region cap cannot protect real content.** The cap
  stops the header and footer regions meeting, but on a page holding only two or three blocks
  there is barely a middle left to protect: a tender page whose blocks are "Tenderer to initial
  each page." and "Signature: ____" has both of them in a region, and both recur on every page,
  so both are stripped although they are form content the document means. Separating them from
  furniture needs the coordinates this path does not have — and a ``layout-json`` page has
  them, so there the block count is irrelevant and only the bands at the page edges are region.
* ``html`` ONLY: ``colspan`` / ``rowspan`` are not interpreted: a spanned cell occupies one cell
  in the linearised row, so a row under a spanning header can be labelled one column out. The
  retained ``html`` carries the truth; the derived ``text`` does not.
* ``layout-json`` ONLY: a block's ``html`` is ``None``. There is no engine markup on this path,
  and synthesising a ``<table>`` from the cells would store provenance the extractor never
  produced — the same objection already recorded for a merged table's markup.
* ``layout-json`` ONLY: a non-heading line becomes a ``paragraph``. ``list_item`` is not
  inferred: a bullet glyph is a typographic convention this path would have to guess at, and
  the plan's block vocabulary is not worth a second guess in the same slice.
* ``layout-json`` ONLY: heading DEPTH is a rank over the document's distinct heading sizes, so
  it is a relative scale, not the absolute one ``<h1>``..``<h6>`` gives. On a mixed document
  both kinds of level land in the same :func:`_nest` stream, where only the ORDER of levels
  matters — but a ``layout-json`` "level 1" and an ``html`` ``<h1>`` are not the same claim.
* A page whose ``markup_format`` is neither ``html`` nor ``layout-json`` is recorded in
  ``skipped_pages`` and parsed by nothing. Reading Markdown would mean a third block parser
  with its own tokenisation edges, which this slice does not budget for; such a document still
  gets a section tree from the plain-text fallback, losing only its blocks.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import Any

from rag_core.chunking import block_id, normalise_breadcrumb, section_id, text_hash
from rag_core.sections import Section, detect_sections

# The plan's breadcrumb separator. Deliberately not `sections.py`'s " > ": that one labels v1
# chunks and is matched by stored metadata, so the two must not quietly become each other.
BREADCRUMB_SEP = " › "  # noqa: RUF001 - the plan names this exact character

# Block types, the closed vocabulary of `blocks.block_type`. A string rather than an enum to
# match how `level` / `block_type` already reach the Phase-1 id helpers.
BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_TABLE = "table"
BLOCK_TABLE_ROW = "table_row"
BLOCK_LIST_ITEM = "list_item"

# Overlay shapes this parser understands. `ocr.LAYOUT_OVERLAY_VERSION` is the writer's side of
# the same contract; an overlay of any other version is REFUSED rather than parsed on the
# assumption that its unknown shape happens to be compatible (plan: the version exists so a
# stored artifact is "detectable rather than silently misparsed").
#
# v1 IS DELIBERATELY NOT SUPPORTED, although this parser was first written against it. A v1
# overlay carries no per-page `markup_format`, so reading one means inferring the markup language
# from the extractor id — and that inference is precisely what is wrong: both Chandra engines
# stamp `extractor == "chandra"` while the self-hosted path answers HTML and the hosted /convert
# path answers Markdown, which is why `ocr.PageRecord.__post_init__` refuses to let markup be
# captured without a declared format. Guessing would silently misparse a Markdown page as HTML,
# the exact failure the version field exists to make impossible. Nothing has ever been persisted
# under v1 (the write is gated on `v2_ingest`, default false, set in no deployment), so there
# is no artifact to strand — only a branch to not write.
SUPPORTED_SCHEMA_VERSIONS = frozenset({2})

# Per-page markup languages this parser reads. Any other declared format is RECORDED as skipped
# rather than guessed at (see `LayoutTree.skipped_pages`).
MARKUP_FORMAT_HTML = "html"
# The born-digital PDF overlay (`ocr.MARKUP_FORMAT_LAYOUT_JSON`) — structured blocks with
# coordinates and per-line typography, which the HTML path has neither of.
MARKUP_FORMAT_LAYOUT_JSON = "layout-json"
# `ocr.LAYOUT_JSON_VERSION` is the writer's side of this contract, and it versions the PER-PAGE
# PAYLOAD rather than the document envelope (`SUPPORTED_SCHEMA_VERSIONS` above does that). A
# payload declaring any other version is refused for the same reason the envelope is: its unknown
# shape would otherwise be parsed on the assumption that it happens to be compatible.
SUPPORTED_LAYOUT_JSON_VERSIONS = frozenset({1})
# `blocks.block_type` in the payload. Deliberately NOT this module's block vocabulary: `line` is
# what the extractor can honestly claim, and turning it into `heading` or `paragraph` is this
# parser's job (see `_classify_headings`).
_JSON_BLOCK_LINE = "line"
_JSON_BLOCK_TABLE = "table"
_BBOX_PARTS = 4  # [x0, top, x1, bottom], pdfplumber space: origin top-left, `top` grows downward

_HEADING_TAG = re.compile(r"h([1-6])")
# Tags that end the run of text before them. `div`/`section` are in here because the extractor
# uses them as paragraph wrappers, not because they are semantically paragraphs.
_PARAGRAPH_TAGS = frozenset({"p", "div", "section", "article", "blockquote", "pre", "figcaption"})
_SOFT_BREAK_TAGS = frozenset({"br", "hr"})

# --- header/footer removal knobs ---------------------------------------------------------------
# How many blocks at each end of a page stand in for "the header region" / "the footer region".
# Two, not one: a header is often a title line plus a document number, and a footer a rule line
# plus a page marker. Wider would start reaching into the body.
_REGION_BLOCKS = 2
# Below this many pages, recurrence is not evidence. A 1-page document's every block recurs on
# 100% of its pages, and stripping on that reading would delete the document; on 2 pages a
# legitimately repeated short paragraph also reads as 100%. Three is the smallest page count at
# which ">50%" can be wrong about a block and still be outvoted by the rest of the document.
_MIN_PAGES_FOR_REPEAT = 3
# The plan's threshold, verbatim: "> 50% of pages".
_REPEAT_PAGE_SHARE = 0.5
# A running header is a LINE. Anything longer that repeats is the document repeating itself (a
# standard clause, a boilerplate paragraph) and is content, not furniture.
_MAX_RUNNING_CHARS = 80
# `layout-json` ONLY: the header and footer BANDS, as a fraction of `page_height` at each edge.
# The plan's actual rule ("the same page region"), finally answerable. 8% of an A4/Letter page is
# ~63 pt — under a one-inch margin, so the band holds the furniture a word processor puts in the
# margin and not the first or last line of the type area. The two bands cannot meet at any share
# below 0.5, so the half-page cap the block-stream approximation needs has no counterpart here.
_REGION_PAGE_SHARE = 0.08
# Digit RUNS, not digits: "Page 1 of 12" and "Page 9 of 12" collapse to the same pattern, while
# "IEC 62368-1" keeps its shape ("IEC #-#") and so cannot collide with a page marker.
_DIGIT_RUN = re.compile(r"\d+")
_DIGIT_MASK = "#"

# --- cross-page table merge knobs --------------------------------------------------------------
# A table narrower than this has no column structure worth comparing: "same column count" is
# nearly vacuous at one column, so the merge test would be no test at all. Refusing here is the
# conservative half of "a wrong merge fuses two unrelated tables".
_MIN_MERGE_COLUMNS = 2

# --- typography heading detection knobs (`layout-json` only) -----------------------------------
# A heading is a title, not a sentence. The bound and the number are `sections._MAX_HEADING_CHARS`
# verbatim, because it is the same judgement about the same corpora: a long line that happens to
# be set large is a pull quote, a banner or a one-line paragraph, not a section name. Duplicated
# rather than imported because it is that module's private calibration, not a shared constant.
_MAX_HEADING_CHARS = 80
# ...and a heading is a NAME, not a fragment. The lower bound has no counterpart in `sections.py`
# because that module reaches a title only through a number+title grammar, which cannot match a
# bare glyph; typography can, and on a real document it did.
#
# MEASURED on the AEGCL SOC/NOC Corrigendum-V. Its DRS tables stack the "S.No" column header one
# glyph per line — "S." / "N" / "o" — each set larger than the body, so each was promoted to a
# section heading. The consequence was not cosmetic: the 16,668 characters of the Video Wall
# requirements became the body of a section whose name is the single letter "o", and the real
# "13. Video Wall Technical Requirements" heading was left holding 37 characters. A section
# nobody can name is a section nobody can retrieve, and this cost that document two of the
# fourteen section groups the Phase-8 acceptance asks for.
#
# TWO conditions, because either alone lets the observed case through: a minimum length (so "o"
# and "S." are out) and a minimum of two LETTERS (so "1.", "(a)" and "—" are out, which are
# list markers and rules, never section names).
_MIN_HEADING_CHARS = 3
_MIN_HEADING_LETTERS = 2

_WHITESPACE_RUN = re.compile(r"\s+")


class UnsupportedLayoutVersionError(ValueError):
    """A layout overlay whose ``schema_version`` this build does not understand."""


# --- public tree --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayoutBlock:
    """One unit of layout in reading order.

    ``ordinal`` is the position within the OWNING SECTION, not within the document: it is a hash
    input to :func:`rag_core.chunking.block_id`, so its scope is the blast radius of an
    edit. Section-scoped is not zero blast radius — inserting a block, or a row into a table,
    re-points the ids of the blocks AFTER it in that same section. What it does buy is that the
    damage stops at the section boundary: a change on page 1 leaves page 9's block ids alone,
    where a document-wide counter would move every one of them.

    ``pages`` is normally just ``(page,)``; it differs only for a table merged across a page
    break, where ``page`` is where the table STARTS."""

    block_id: str
    block_type: str
    text: str
    page: int
    ordinal: int
    pages: tuple[int, ...] = ()
    # The engine's own markup, for ``table`` only (the plan: "Tables: keep HTML"). A merged table
    # holds its parts' raw slices joined by a newline — not a synthesised single ``<table>``,
    # because that would store markup the extractor never emitted.
    #
    # ALWAYS None on the `layout-json` path, including for its tables, for exactly that reason:
    # pdfplumber emits no markup at all, so there is nothing to keep, and building a `<table>`
    # from the cells would put markup in the extractor's mouth. The structure is not lost — it is
    # in the `cells`-derived `table_row` blocks, which is where a consumer should read it.
    html: str | None = None
    # The ``table`` a ``table_row`` came from. Without it a row is addressable but orphaned, and
    # Phase 2b has to re-derive the relation from adjacency.
    parent_block_id: str | None = None


@dataclass(frozen=True, slots=True)
class LayoutSection:
    """One node of the structure tree: a heading, its own blocks, and its subsections.

    ``ordinal`` counts earlier occurrences of this same normalised breadcrumb in this document —
    the role :func:`rag_core.chunking.section_id` documents for it ("separates genuine
    repeats of the same breadcrumb"), not a sibling index. ``breadcrumb`` is stored in the
    normalised form that was hashed, so the column and the id can never disagree."""

    section_id: str
    breadcrumb: str
    title: str
    depth: int
    ordinal: int
    text: str
    text_hash: str
    heading_level: int | None = None  # the <hN> level; None for an implicit root (no heading)
    page: int | None = None  # of the heading; None on the plain-text fallback path
    blocks: tuple[LayoutBlock, ...] = ()
    children: tuple[LayoutSection, ...] = ()


@dataclass(frozen=True, slots=True)
class PageDegradation:
    """What a page's extraction lost, carried through from the overlay.

    Surfaced rather than dropped: a page the model looped on (``ocr_degenerate``) parses into a
    perfectly well-formed block tree, and nothing in that tree says the page is untrustworthy."""

    page: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkippedPage:
    """A page whose markup this parser does not read, and why.

    The hosted Chandra ``/convert`` endpoint answers Markdown while the self-hosted one answers
    HTML, under the same extractor id. Rather than guess, a non-HTML page is skipped and SAID to
    be skipped: a consumer can see that the page's blocks are missing for a known reason instead
    of inferring from a gap that the page was blank."""

    page: int
    markup_format: str
    reason: str


@dataclass(frozen=True, slots=True)
class SectionMerge:
    """One repeated-page-title merge that was performed.

    An OBSERVATION, for the same reason :class:`TableMerge` is one: a wrong merge fuses sections
    that the document kept apart, and silent fusion is as unreviewable as silent deletion.
    ``pages`` are the pages whose copies of the title were folded into the first."""

    breadcrumb: str
    pages: tuple[int, ...]
    children_merged: int


@dataclass(frozen=True, slots=True)
class TableMerge:
    """One cross-page table merge that was performed.

    An OBSERVATION, for exactly the reason ``removed_running_text`` is one: a wrong merge
    fabricates column labels for real rows, and silent fusion is as unreviewable as silent
    deletion. ``header_declared`` records whether EITHER side declared a ``<th>`` — false means
    the merge rested on the shape check in :func:`_continues` alone, which is the weaker
    evidence and the case worth auditing first."""

    pages: tuple[int, ...]
    columns: int
    rows_added: int
    header_declared: bool


@dataclass(frozen=True, slots=True)
class LayoutTree:
    """A document's structure tree plus what the parse itself observed."""

    doc_id: str
    sections: tuple[LayoutSection, ...]
    # "markup" — parsed from the overlay; "text" — the plain-text fallback, so sections carry no
    # blocks. A consumer that needs blocks must check this rather than read an empty tree as a
    # document without structure.
    source: str
    extractor: str = ""
    pages: tuple[int, ...] = ()  # the pages that carried markup, in overlay order
    degradations: tuple[PageDegradation, ...] = ()
    # Normalised texts identified as running furniture and dropped from every header/footer
    # region they appeared in. An OBSERVATION, for the same reason `ocr.PageRecord` records its
    # own stripped lines: silent deletion is unreviewable.
    removed_running_text: tuple[str, ...] = ()
    # SHA-256 of the raw bytes the overlay was extracted from, carried straight through from the
    # overlay (schema v2). Here so a consumer can check this tree against the stored raw object
    # without re-reading the overlay it came from.
    content_hash: str = ""
    skipped_pages: tuple[SkippedPage, ...] = ()
    merged_tables: tuple[TableMerge, ...] = ()
    merged_sections: tuple[SectionMerge, ...] = ()


# --- parse entry point --------------------------------------------------------------------------


def parse_layout(overlay: Mapping[str, Any], doc_id: str, *, text: str = "") -> LayoutTree:
    """Parse a stored layout overlay into a section/block tree.

    ``overlay`` is the plain JSON mapping as persisted (``{"schema_version", "extractor",
    "content_hash", "pages": [{"page_no", "markup", "markup_format", "degradations"}]}``) — a
    Mapping, not the extractor's dataclass, because rag-core must not import service types. Only
    pages that carried markup appear in it, each with its real 1-based ``page_no``, so index and
    page do not line up.

    A page is parsed when its ``markup_format`` is ``"html"`` (Chandra's markup) or
    ``"layout-json"`` (the born-digital PDF overlay). Any other declared format — the hosted
    Chandra ``/convert`` path answers Markdown under the same extractor id — is recorded in
    ``skipped_pages`` and contributes no blocks. The branch is on the FORMAT, never on the
    extractor, because the extractor does not determine the language.

    MIXED DOCUMENTS ARE THE REALISTIC CASE and are the reason the two branches extend ONE list:
    a tender with scanned annexures appended has born-digital pages declaring ``layout-json``
    and OCR'd pages declaring ``html``, and the result must be one tree in page order, not two.
    Everything after the read loop therefore runs over the whole stream — running-text removal
    (whose region means geometry on a measured page and reading order on a Chandra one, pooled
    in one vote), typography classification, and the cross-page table merge, which can join a
    table split across a format boundary.

    ``text`` is the extracted plain text, used ONLY when the markup yields no blocks at all (no
    pages, every page answered with ``""``, or every page skipped as non-HTML). That path defers
    to :func:`rag_core.sections.detect_sections` and produces sections with no blocks: a
    document whose markup this parser cannot read still gets a real tree rather than an empty
    one, and ``sections.py``'s measured narrowness is neither changed nor second-guessed.

    Raises :class:`UnsupportedLayoutVersionError` on an unrecognised ``schema_version``."""
    version = overlay.get("schema_version")
    # `isinstance(True, int)` and `True == 1`, so a bool would pass a bare membership test. For
    # the one field whose whole job is strictness, a bool is a malformed overlay, not a version.
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedLayoutVersionError(
            f"layout overlay schema_version {version!r} is not supported "
            f"(this build parses {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    pages: list[int] = []
    degradations: list[PageDegradation] = []
    skipped: list[SkippedPage] = []
    raw: list[_Raw] = []
    for page in _page_entries(overlay):
        page_no = _page_no(page)
        pages.append(page_no)
        reasons = tuple(str(r) for r in _as_sequence(page.get("degradations")))
        if reasons:
            degradations.append(PageDegradation(page=page_no, reasons=reasons))
        markup = str(page.get("markup") or "")
        markup_format = str(page.get("markup_format") or "")
        if not markup.strip():
            continue  # the engine answered this page with nothing — a real answer, not a skip
        if markup_format == MARKUP_FORMAT_HTML:
            raw.extend(_parse_markup(markup, page_no))
            continue
        if markup_format == MARKUP_FORMAT_LAYOUT_JSON:
            blocks, reason = _parse_layout_json(markup, page_no)
            raw.extend(blocks)
            if reason:
                skipped.append(SkippedPage(page=page_no, markup_format=markup_format, reason=reason))
            continue
        skipped.append(
            SkippedPage(
                page=page_no,
                markup_format=markup_format,
                reason="unreadable_markup_format" if markup_format else "no_markup_format_declared",
            )
        )

    # Runs before the merge on purpose (see `_strip_running_blocks`); a no-op when `raw` is empty,
    # which is the plain-text fallback below.
    kept, removed = _strip_running_blocks(raw)
    # And before typography is classified, so page furniture is gone before it can be promoted to
    # a heading or vote on the document's body size — see `_classify_headings`.
    _type_measured_lines(kept)
    merged, merges = _merge_cross_page_tables(kept)
    sections = _tree(doc_id, merged) if raw else _text_sections(doc_id, text)
    sections, section_merges = _merge_repeated_page_titles(sections)
    return LayoutTree(
        doc_id=doc_id,
        sections=sections,
        source="markup" if raw else "text",
        extractor=str(overlay.get("extractor") or ""),
        pages=tuple(pages),
        degradations=tuple(degradations),
        removed_running_text=removed,
        content_hash=str(overlay.get("content_hash") or ""),
        skipped_pages=tuple(skipped),
        merged_tables=tuple(merges),
        merged_sections=tuple(section_merges),
    )


# --- overlay reading ----------------------------------------------------------------------------


def _as_sequence(value: Any) -> Sequence[Any]:
    """A JSON list, or empty for anything else. The overlay is read back from object storage, so
    its shape is asserted here rather than assumed."""
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _page_entries(overlay: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [p for p in _as_sequence(overlay.get("pages")) if isinstance(p, Mapping)]


def _page_no(page: Mapping[str, Any]) -> int:
    try:
        return int(page["page_no"])
    except (KeyError, TypeError, ValueError):
        return 0  # an overlay page with no usable number: keep its blocks, do not claim a page


# --- intermediate (mutable) block ---------------------------------------------------------------


@dataclass(slots=True)
class _Row:
    page: int
    cells: list[str]


@dataclass(slots=True)
class _Raw:
    """A block before header/footer removal, table merging and id assignment.

    Mutable and private: merging rewrites a table in place, and the public blocks are frozen, so
    the two cannot be the same type."""

    kind: str
    text: str
    page: int
    level: int = 0  # <hN> level (html) or the size rank (layout-json), headings only
    html: str | None = None
    header: list[str] | None = None  # the header row's labels
    header_from_th: bool = False  # True when <th> said so; False when we took the first row
    rows: list[_Row] = field(default_factory=list)  # DATA rows only (header excluded)
    pages: list[int] = field(default_factory=list)
    # Is `text` the extractor's OWN rendering of this table, to be kept verbatim, or is it ours
    # to derive from `header`/`rows`? True only for a `layout-json` table band, whose body is the
    # exact string that reached the document's text (see `_table_text`). A flag rather than
    # "`text` is non-empty", because an empty verbatim body and a derivable one are different
    # things and a cross-format merge has to be able to tell them apart.
    text_verbatim: bool = False
    # --- `layout-json` only; all None on the html path, which measures nothing ---
    # The line's median glyph size and whether a majority of its glyphs were in a bold face, as
    # the extractor measured them. None on a table block there too: a band spans many faces.
    size: float | None = None
    bold: bool | None = None
    # The block's vertical extent and the page's own height, all in pdfplumber points from the
    # page's TOP edge. The page height travels with the block because a coordinate means nothing
    # without it — "in the bottom 8% of the page" is the question the region rule asks.
    top: float | None = None
    bottom: float | None = None
    page_height: float | None = None


def _collapse(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _separates_words(tag: str) -> bool:
    """Does this start/end tag put a word boundary between the text either side of it?

    Every block-level tag does, plus ``<br>``/``<hr>``. Used in two places: inside a table cell,
    where such a tag must not start a block but must still separate ("700<br>nits"), and as the
    signal that an unclosed ``<table>`` has been left behind. Table structural tags are
    deliberately absent — a ``<td>`` where a row should be is ordinary table markup, not evidence
    of an unclosed table — so :func:`_separates_in_cell` adds them for the first use only."""
    return tag in _SOFT_BREAK_TAGS or tag in _PARAGRAPH_TAGS or tag == "li" or bool(_HEADING_TAG.fullmatch(tag))


def _separates_in_cell(tag: str) -> bool:
    """As :func:`_separates_words`, plus a nested table's own cell and row boundaries: those land
    inside the OUTER cell, where "<td>B</td><td>C</td>" must not become "BC"."""
    return _separates_words(tag) or tag in ("td", "th", "tr")


# --- markup -> blocks ---------------------------------------------------------------------------


class _MarkupBlocks(HTMLParser):
    """The engine's HTML as a stream of typed blocks, in reading order.

    Built on the stdlib parser for the same reason ``ocr_check._MarkupToText`` is (the in-repo
    precedent for reading this exact markup): the overlay is engine output, not a document we
    control, and a tolerant parser that ignores what it does not recognise beats a dependency
    that rejects it. Unlike that one, this keeps the structure instead of rendering it away."""

    def __init__(self, markup: str, page: int) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_Raw] = []
        self._page = page
        self._markup = markup
        # Line start offsets, so `getpos()` can be turned back into an index into `markup` and a
        # table's raw slice taken verbatim rather than re-serialised from the parse.
        self._line_starts = [0]
        for line in markup.splitlines(keepends=True):
            self._line_starts.append(self._line_starts[-1] + len(line))
        self._buf: list[str] = []
        self._kind = BLOCK_PARAGRAPH
        self._level = 0
        self._table: _Raw | None = None
        self._table_at = 0
        self._table_depth = 0
        self._row: list[str] | None = None
        self._row_has_th = False
        self._cell: list[str] | None = None

    # -- buffered text blocks

    def _flush(self) -> None:
        text = _collapse("".join(self._buf))
        self._buf = []
        kind, level = self._kind, self._level
        self._kind, self._level = BLOCK_PARAGRAPH, 0
        if text:
            self.blocks.append(_Raw(kind=kind, text=text, page=self._page, level=level, pages=[self._page]))

    def _offset(self) -> int:
        line, col = self.getpos()
        if not 1 <= line < len(self._line_starts):
            return 0
        return self._line_starts[line - 1] + col

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._open_table()
            return
        if self._cell is not None:
            # Inside a cell, block markup only SEPARATES WORDS — it never starts a block, and a
            # nested table's own tags fall here too (see `_open_table`). Without the separator
            # "<td>700<br>nits</td>" embeds as "700nits".
            if _separates_in_cell(tag):
                self._cell.append(" ")
            return
        if tag == "tr" and self._table_depth == 1:
            self._row, self._row_has_th = [], False
            return
        if tag in ("td", "th") and self._row is not None and self._table_depth == 1:
            self._cell = []
            self._row_has_th = self._row_has_th or tag == "th"
            return
        if self._table is not None and self._row is None and _separates_words(tag):
            # A block element where a row should be: the markup never closed its <table>. Closing
            # it here costs only the table; without this the whole rest of the page — headings,
            # and therefore every section and breadcrumb under them — was dropped by the
            # between-rows rule in `handle_data`.
            self._close_table(explicit=False)
        if tag in _SOFT_BREAK_TAGS:
            self._buf.append(" ")
        elif tag == "li":
            self._flush()
            self._kind = BLOCK_LIST_ITEM
        elif heading := _HEADING_TAG.fullmatch(tag):
            self._flush()
            self._kind, self._level = BLOCK_HEADING, int(heading.group(1))
        elif tag in _PARAGRAPH_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            self._close_table(explicit=True)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None and self._table_depth == 1:
            self._row.append(_collapse("".join(self._cell)))
            self._cell = None
        elif self._cell is not None:
            if _separates_in_cell(tag):
                self._cell.append(" ")
        elif tag == "tr" and self._row is not None and self._table_depth == 1:
            self._close_row()
        elif tag == "li" or tag in _PARAGRAPH_TAGS or _HEADING_TAG.fullmatch(tag):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif self._table is None:
            # Text between a table's rows belongs to no cell and no paragraph; dropping it is
            # preferable to emitting a paragraph that interrupts the table in reading order.
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        if self._table is not None:  # markup whose <table> ran to the end of the page
            self._close_table(explicit=False)
        self._flush()

    # -- tables

    def _open_table(self) -> None:
        self._table_depth += 1
        if self._table_depth > 1:
            # A NESTED table. Its tags are ignored and its text reaches `handle_data` while the
            # OUTER cell is still open, so its content lands in that cell — one cell holding
            # "B C" rather than a second table. The outer row keeps its own cells either side,
            # and no inner value can be mistaken for the outer table's header.
            return
        self._flush()
        self._table = _Raw(kind=BLOCK_TABLE, text="", page=self._page, pages=[self._page])
        self._table_at = self._offset()

    def _close_row(self) -> None:
        cells, has_th = self._row or [], self._row_has_th
        self._row, self._row_has_th = None, False
        table = self._table
        if table is None:
            return
        # "Detect the header row (<th>, else the first row)." A <th> row is authoritative; absent
        # one, the first row is taken as labels but remembered as a GUESS (`header_from_th`), so a
        # cross-page continuation can hand it back as data when it turns out to be one.
        is_header = table.header is None and (has_th or not table.rows)
        if not any(cells) and not (is_header and has_th):
            # An all-blank DATA row is a spacer and is dropped, as is an all-blank first row that
            # only our own first-row guess called a header. An all-blank <th> row is kept: the
            # table declared a header that happens to carry no labels, and that declaration is
            # still what fixes the column count. Its columns are then named by position.
            return
        if is_header:
            table.header, table.header_from_th = cells, has_th
        else:
            table.rows.append(_Row(page=self._page, cells=cells))

    def _close_table(self, *, explicit: bool) -> None:
        """Close the outermost open table. ``explicit`` distinguishes a real ``</table>`` from an
        implied one, which decides where the retained raw HTML ends: at the closing tag, or at
        the start of whatever element implied the close (so the slice cannot swallow it)."""
        self._table_depth = max(0, self._table_depth - 1)
        if self._table_depth or self._table is None:
            return
        if explicit:
            end = self._markup.find(">", self._offset())
            end = end + 1 if end >= 0 else len(self._markup)
        else:
            end = self._offset() or len(self._markup)
        self._table.html = self._markup[self._table_at : end].rstrip() or None
        if self._row is not None:  # an unclosed <tr> still holds real cells
            self._close_row()
        self.blocks.append(self._table)
        self._table = None
        self._table_depth = 0


def _parse_markup(markup: str, page: int) -> list[_Raw]:
    """One page's markup as blocks. An empty markup string ("the engine answered this page with
    nothing") yields no blocks, which is exactly what it means."""
    if not markup.strip():
        return []
    parser = _MarkupBlocks(markup, page)
    parser.feed(markup)
    parser.close()
    return parser.blocks


# --- layout-json -> blocks ----------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """A JSON number, or None for anything else. A bool is NOT a number here: ``True`` would
    otherwise read as a font size of 1.0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _bbox_span(entry: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """``(top, bottom)`` of a block's bbox, or ``(None, None)`` if it has no usable one.

    Only the vertical extent is read. The horizontal one is in the payload and is deliberately
    unused: nothing in this slice asks a question about columns, and reading it would imply it
    had been thought about."""
    bbox = _as_sequence(entry.get("bbox"))
    if len(bbox) != _BBOX_PARTS:
        return None, None
    top, bottom = _as_float(bbox[1]), _as_float(bbox[3])
    if top is None or bottom is None:
        return None, None
    return top, bottom


def _grid(entry: Mapping[str, Any]) -> list[list[str]]:
    """A table block's ``cells`` as a rectangular-ish grid of strings.

    pdfplumber's own ``table.extract()`` output, carried through 2c-1 unchanged — so a cell is a
    cell rather than something re-parsed out of the rendered ``text``, which is the whole reason
    the payload carries it. A ``None`` cell is an empty one (that is what pdfplumber means by it),
    and an all-blank row is dropped as a spacer, exactly as the html path drops one."""
    rows: list[list[str]] = []
    for row in _as_sequence(entry.get("cells")):
        cells = [_collapse(str(c)) if c is not None else "" for c in _as_sequence(row)]
        if any(cells):
            rows.append(cells)
    return rows


def _parse_layout_json(markup: str, page: int) -> tuple[list[_Raw], str]:
    """One born-digital page's payload as blocks, plus "" or the reason it was not read.

    A failure here costs the PAGE its blocks and says so in ``skipped_pages`` — the same
    position the unreadable-format branch takes, and for the same reason: the payload is read
    back out of object storage, so its shape is asserted rather than assumed, and a document
    with one malformed page keeps the other pages' structure.

    Every line arrives as a ``paragraph`` carrying its measurements. Which of them are headings
    is decided later, over the whole document, by :func:`_classify_headings` — a per-page or
    per-block decision cannot see the document's body size, which is the only thing a size is
    evidence relative to."""
    try:
        payload = json.loads(markup)
    except ValueError:
        return [], "unparseable_layout_json"
    if not isinstance(payload, Mapping):
        return [], "unparseable_layout_json"
    version = payload.get("version")
    if isinstance(version, bool) or version not in SUPPORTED_LAYOUT_JSON_VERSIONS:
        return [], "unreadable_layout_json_version"

    height = _as_float(payload.get("page_height"))
    out: list[_Raw] = []
    for entry in _as_sequence(payload.get("blocks")):
        if not isinstance(entry, Mapping):
            continue
        top, bottom = _bbox_span(entry)
        kind = str(entry.get("block_type") or "")
        if kind == _JSON_BLOCK_LINE:
            text = _collapse(str(entry.get("text") or ""))
            if not text:
                continue
            out.append(
                _Raw(
                    kind=BLOCK_PARAGRAPH,
                    text=text,
                    page=page,
                    pages=[page],
                    size=_as_float(entry.get("font_size")),
                    bold=entry.get("bold") if isinstance(entry.get("bold"), bool) else None,
                    top=top,
                    bottom=bottom,
                    page_height=height,
                )
            )
        elif kind == _JSON_BLOCK_TABLE:
            grid = _grid(entry)
            out.append(
                _Raw(
                    kind=BLOCK_TABLE,
                    # The band body VERBATIM — this exact string is what reached the document's
                    # text (2c-1 pins that), so re-deriving it from the cells would let the
                    # block and the text disagree about a table they came from together.
                    text=str(entry.get("text") or "").strip(),
                    text_verbatim=True,
                    page=page,
                    pages=[page],
                    # "the header row (<th>, else the first row)", the html path's rule, with the
                    # same honesty about which it was: pdfplumber declares no header, so the
                    # first row is a GUESS and `header_from_th` stays False — which is what lets
                    # a cross-page continuation hand its own first row back as data.
                    header=grid[0] if grid else None,
                    rows=[_Row(page=page, cells=cells) for cells in grid[1:]],
                    top=top,
                    bottom=bottom,
                    page_height=height,
                )
            )
        # Any other `block_type` is from a writer this build does not know. Skipping the block
        # rather than guessing is the same position the version check above takes; the page keeps
        # the blocks that ARE in the vocabulary.
    return out, ""


# --- typography -> headings (`layout-json` only) ------------------------------------------------


def _body_size(weighted: list[tuple[float, int]]) -> float:
    """The document's BODY font size — the modal size over its measured lines, each line weighted
    by how many characters it carries.

    DOCUMENT-WIDE, not per page, and that is the whole point: a title page set in 24 pt would
    otherwise declare 24 pt to be its own body and every genuine heading on it a paragraph. A
    tender is overwhelmingly body text, so the mode over the document is the body.

    MODE, not mean or median: a type scale is categorical (12 pt body, 14 pt and 18 pt headings),
    and an average over categories lands between two of them, belonging to neither.

    WEIGHTED BY CHARACTERS, not one vote per line, because a size's share of the document's TEXT
    is what makes it the body — a short document's furniture and headings are many short lines
    and can outnumber the body's lines while carrying a fraction of its words. Measured on the
    fixtures: a 2-page tender whose 9 pt page markers survived the recurrence guard (2 pages is
    below :data:`_MIN_PAGES_FOR_REPEAT`) had as many 9 pt lines as 12 pt ones, and an unweighted
    mode made 9 pt the body and every 12 pt paragraph in the document a heading.

    Ties break to the SMALLER size: when two sizes carry equally much text, the smaller is the
    one the document reads as normal and the larger is what stands out from it."""
    counts: Counter[float] = Counter()
    for size, weight in weighted:
        counts[size] += weight
    most = max(counts.values())
    return min(size for size, n in counts.items() if n == most)


def _type_measured_lines(blocks: list[_Raw]) -> None:
    """Decide which MEASURED lines are headings, in place. See :func:`_classify_headings`.

    Typography WINS over numbering wherever both speak, and the fallback is only reached when
    typography says nothing at all about the whole document. Why that way round:

    * A size and a weight are DIRECT evidence of how the document presents a line — the author
      made it stand out. Numbering is an inference from the line's own words, and
      ``sections.py``'s whole design is an admission of how weak that inference is: it refuses
      single-level numbering outright because "1." is more often a list item than a heading.
    * They produce incompatible depth scales. Numbering depth comes from the number ("6.1.1" is
      three levels deep whether or not the document ever wrote a "6"), while typography depth is
      a rank over the sizes this document actually uses. Interleaving them would nest a 14 pt
      "Scope" under a 12 pt "6.1.1 Entry Criteria" or the reverse, depending on which detector
      happened to fire — a structure claim neither signal supports.

    So the numbering fallback runs only when NO block in the whole stream ended up a heading: a
    single-font tender PDF, where every line is 12 pt and nothing is bold, carries no typographic
    signal and inventing a hierarchy from one would be exactly the wrong-heading-is-worse-than-no-
    heading failure ``sections.py`` is built to avoid. The condition is deliberately over the
    whole stream, not over the measured pages: on a mixed document whose Chandra pages already
    yielded real ``<hN>`` headings, adding numbering-derived levels from the born-digital pages
    would mix the two scales again."""
    _classify_headings(blocks)
    # Between the two, deliberately: it removes furniture the typography could not tell from a
    # heading, and a document whose ONLY candidates were furniture should then get the numbering
    # fallback's turn rather than an empty tree.
    _demote_repeated_candidates(blocks)
    if not any(block.kind == BLOCK_HEADING for block in blocks):
        _number_headings(blocks)


def _nameable(text: str) -> bool:
    """Whether ``text`` could be a section NAME, by length alone.

    The one judgement both bounds encode: a section's name has to be something a breadcrumb can
    say and a reader can recognise. A sentence is too much (it is a paragraph that happens to be
    set large) and a glyph is too little (it is a table header, a list marker or a rule). Neither
    bound looks at typography — that is the caller's job, and this is the filter it applies
    first, so a line that cannot be a name never competes for a depth rank either."""
    stripped = text.strip()
    return (
        _MIN_HEADING_CHARS <= len(stripped) <= _MAX_HEADING_CHARS
        and sum(1 for ch in stripped if ch.isalpha()) >= _MIN_HEADING_LETTERS
    )


#: A numbered-heading prefix: "4.", "17.3", "1.2.3", followed by a title. DELIBERATELY NARROW --
#: it is a COMPATIBILITY EXCEPTION for one adjudicated case, not a general licence for any line
#: starting with a digit. The number must be a bare clause number (digits and dots only), must be
#: followed by whitespace and then real text, and must sit at the very start of the line.
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def _reads_as_sentence(text: str) -> bool:
    """Whether a heading CANDIDATE is really prose — a paragraph that happens to be set large.

    SEPARATE FROM :func:`_nameable`, which commits in its own docstring to deciding "by length
    alone". This is a different question and it gets its own predicate rather than being smuggled
    into that one.

    THE RULE: a candidate ending in a sentence terminator is prose, unless it is
    numbered-heading-shaped.

    WHY A TERMINATOR. Measured counterfactually across 18 documents in three families (the AEGCL
    SOC/NOC corrigendum, the Acme COE SOP, 16 RAG as a Service documents): 1,359 lines passed the
    typographic test, 873 were promoted, and this rejects 190 of them -- every one adjudicated by
    hand as prose. Those are wrapped sentence tails set in heading type: "system.", "revocable.",
    "force-merged.", "wrap every layer.". The defect is systemic, not one document's quirk -- the
    FRS promotes 10 prose fragments of its 45 headings, the Architecture document 8 of 28. The
    190 figure is the PRECISION of the rejected set, not recall over every prose fragment in the
    corpora.

    WHY THE NUMBERING EXCEPTION. The same calibration found one counterexample:
    "4. NG-SIEM / SOAR / UEBA Solution — License Details." is a genuine section heading of the
    AEGCL corrigendum that ends in a full stop. Rejecting every terminator-ending candidate would
    knowingly destroy it. Across all 18 PDFs this exception fires ZERO times, so it changes no
    other classification -- its whole population is that one heading.

    AND ITS EVIDENCE IS LIMITED, recorded here rather than rounded up. That heading is known to be
    genuine from the document's STORED SECTIONS, not from its PDF, which is not available to the
    calibration -- so its original typography and geometry were never independently verified. This
    is text-level evidence for a layout-level rule. The accepted residual risk is SHORT numbered
    prose (inside :data:`_MAX_HEADING_CHARS`, terminator-ending, with enough contrast to be a
    candidate): no document in the population contains such a line, so the exception could be
    neither validated nor falsified against real text. `test_layout_d3_continuation.py` holds
    synthetic adversarial cases for it, labelled as safeguards rather than corpus evidence.

    Re-runnable as new document families arrive: ``scripts/calibrate_heading_rule.py``.
    """
    stripped = text.strip()
    if not stripped.endswith((".", "!", "?")):
        return False
    return not _NUMBERED_HEADING.match(stripped)


def _demote_repeated_candidates(blocks: list[_Raw]) -> None:
    """Demote heading candidates whose text RECURS in the document, in place.

    A document's section names are distinct; the things that repeat are furniture. MEASURED on
    the real AEGCL SOC/NOC Corrigendum-V, where typography alone cannot tell them apart: its
    letterhead sets five lines ("Assam Electricity Grid Corporation Limited", the registered
    office, the CIN) above body size on each of its first three pages, and its DRS tables set
    the two halves of an "Offered by the bidder." column header the same way. All seven texts
    were promoted to sections. Of the document's 72 heading candidates, 60 were distinct and
    EVERY repeated one was furniture — so this rule removes 19 occurrences and leaves all 53
    unique headings untouched.

    THE DAMAGE IT UNDOES IS NOT COSMETIC. A section the document never named owns the content
    that follows it, so the 16,668 characters of Video Wall requirements became the body of a
    section called "the bidder.", and the corrigendum's own Sl.7-9 clauses — including the
    withdrawal of the D.3 confidentiality declaration — became the body of one called
    "CIN: U40101AS2003SGC007238". Neither is findable by a question about what it contains.

    THE ONE EXEMPTION IS A REPEATED PAGE TITLE — the page's LEADING heading, sitting in its
    heading — and it exists because
    :func:`_merge_repeated_page_titles` already handles those and its docstring rejects deleting
    them outright: a tender that sets "Attachment-1" in heading type at the top of every page
    means it as a heading, and demoting it would take the breadcrumb of everything beneath with
    it. So a candidate is spared when EVERY one of its occurrences leads its page from the header
    region — which a letterhead's second through fifth lines do not, and a mid-page table header
    cannot.

    REGION IS :func:`_region_keys`, the same definition the running-text rule uses, deliberately:
    two notions of "the top of the page" in one module is how they come to disagree.

    WHY NOT THE RUNNING-TEXT RULE INSTEAD. That rule is the plan's own ">50% of pages" vote and
    both of these fall under it correctly — the letterhead is on 3 of 80 pages, and a table
    header sits mid-page where there is no region to recur in. Widening the vote to catch them
    would start stripping body text that a document legitimately repeats. This is narrower: it
    only ever demotes a HEADING to a paragraph, so the text and its content survive and join the
    section above, which is where the document put them."""
    # MEASURED CANDIDATES ONLY — this rule second-guesses THIS MODULE'S inference, never an
    # extractor's claim. A `layout-json` line became a heading because `_classify_headings`
    # judged its typography (and that judgement is what the AEGCL letterhead defeats); an `html`
    # block became one because the extractor wrote `<h1>`, which is a statement about the
    # document and not a guess about it. The distinction is load-bearing and not a nicety:
    # `<h1>Annexure</h1>` repeated on three pages, each owning its own prose, is three real
    # sections (`_merge_repeated_page_titles` has the test), and demoting those would weld
    # unrelated content onto whatever preceded them. `size is not None` is exactly "this block
    # was measured" — the html path measures nothing.
    heads = [block for block in blocks if block.kind == BLOCK_HEADING and block.size is not None]
    if not heads:
        return
    counts = Counter(_collapse(block.text).casefold() for block in heads)
    repeated = {text for text, n in counts.items() if n > 1}
    if not repeated:
        return

    # A PAGE TITLE IS THE PAGE'S LEADING HEADING. A letterhead's second through fifth lines are
    # not, so they go; its first line is kept, and `_merge_repeated_page_titles` then folds it
    # across the pages into one front-matter section, which is what it is.
    #
    # THE HEADER REGION IS DELIBERATELY NOT PART OF THIS TEST, though it looks like it should be.
    # On a measured page the region is true geometry (`_geometric_band`), and a page title set
    # large does NOT fit inside the head band — which is exactly what stops the furniture rule
    # eating it, and is stated in that rule's own test. Requiring the region here demoted every
    # real repeated page title on this path.
    page_title: set[int] = set()  # id() of the blocks that look like a page title
    by_page: dict[int, list[_Raw]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        leading = next((b for b in page_blocks if b.kind == BLOCK_HEADING), None)
        if leading is not None:
            page_title.add(id(leading))

    for text in repeated:
        occurrences = [b for b in heads if _collapse(b.text).casefold() == text]
        if all(id(b) in page_title for b in occurrences):
            continue  # a page title — `_merge_repeated_page_titles` folds these
        for block in occurrences:
            block.kind = BLOCK_PARAGRAPH
            block.level = 0


def _classify_headings(blocks: list[_Raw]) -> None:
    """Promote measured lines that stand above the document's body type to headings, in place.

    THE RULE: a line is a heading when it is set LARGER than the body, or at body size in BOLD.
    Depth is the rank of its size among the distinct heading sizes, descending — the biggest size
    is depth 1, the next depth 2 — so the document's own type scale becomes nesting the way
    ``<h1>``..``<h6>`` does on the other path. Body-size bold ranks BELOW every larger size: it
    is the weakest heading signal in the document, so it is also the deepest.

    THE GUARDS, because a wrong heading is worse than no heading (``sections.py``'s stated
    principle, and the reason this is conservative rather than clever):

    * :func:`_nameable` — a heading is a title: long enough to be a name
      (:data:`_MIN_HEADING_CHARS`, :data:`_MIN_HEADING_LETTERS`), short enough not to be a clause
      that happens to be numbered (:data:`_MAX_HEADING_CHARS`).
    * :func:`_reads_as_sentence` — and it is not prose. The length bound alone let a 69-character
      sentence through, which split a real document's sentence across a section boundary and lost
      a contractual disposition entirely; see that function for the calibration.
    * Bold at BODY size counts only where bold is actually distinguishing, i.e. where some
      body-size line is NOT bold. A document set entirely in a bold face would otherwise have
      every one of its lines promoted, which is the degenerate reading of "stands out".
    * A document with no size variation and no bold variation yields NO candidates, so nothing
      is invented for it and :func:`_number_headings` gets its turn.
    * The size ranking is computed over the ACCEPTED candidates only. A size used exclusively by
      long lines (a 14 pt pull quote) therefore does not consume a depth rank it never names a
      heading at, which would push every genuine heading one level deeper."""
    measured: list[tuple[_Raw, float]] = [
        (b, b.size) for b in blocks if b.kind == BLOCK_PARAGRAPH and b.size is not None
    ]
    if not measured:
        return
    body = _body_size([(size, len(line.text)) for line, size in measured])
    # Bold is evidence only by contrast with the body text around it.
    bold_speaks = any(size == body and not line.bold for line, size in measured)
    candidates = [
        (line, size)
        for line, size in measured
        if _nameable(line.text)
        and not _reads_as_sentence(line.text)
        and (size > body or (size == body and line.bold and bold_speaks))
    ]
    if not candidates:
        return
    ranked = sorted({size for _, size in candidates if size > body}, reverse=True)
    depth = {size: rank for rank, size in enumerate(ranked, start=1)}
    for line, size in candidates:
        line.kind = BLOCK_HEADING
        line.level = depth.get(size, len(ranked) + 1)


def _number_headings(blocks: list[_Raw]) -> None:
    """The fallback for a measured document whose typography carries no signal: hand its lines to
    :func:`rag_core.sections.detect_sections` and take the headings IT finds.

    Used as-is, deliberately — it is tuned on two real corpora, it refuses single-level numbering
    unless the document's numbers ascend, and this slice has no business second-guessing either.
    Depth is the numbering depth (``len(levels)``), which is what the plain-text fallback
    (:func:`_text_sections`) already uses, so the two agree.

    A heading ``detect_sections`` finds MID-LINE (its trailing-heading rule, for a heading PDF
    extraction ran onto the end of the previous paragraph) is ignored here: a block on this path
    is one measured line with one bbox, and splitting it would produce two blocks whose
    coordinates are a fiction. That case belongs to the text layer, where it is already handled."""
    lines = [b for b in blocks if b.kind == BLOCK_PARAGRAPH and b.size is not None]
    if not lines:
        return
    at: dict[int, _Raw] = {}
    offset = 0
    for line in lines:
        at[offset] = line
        offset += len(line.text) + 1  # the "\n" joined below
    for section in detect_sections("\n".join(line.text for line in lines)):
        found = at.get(section.offset)
        if found is None:
            continue
        found.kind = BLOCK_HEADING
        found.level = len(section.levels)


# --- header/footer removal ----------------------------------------------------------------------


_Key = tuple[str, int, str]  # (region, slot, normalised text) — what recurrence is counted over


def _geometric_band(block: _Raw) -> str | None:
    """``"head"``, ``"foot"``, ``""`` (measured, but between the bands) or None (not measured).

    The plan's region, as geometry: a band of :data:`_REGION_PAGE_SHARE` of ``page_height`` at
    each edge. The block must sit ENTIRELY inside the band — its whole box, not just the edge
    nearest the margin — so a line that merely reaches into the band from the type area is body
    text, which is what it looks like on the page."""
    if block.top is None or block.bottom is None or not block.page_height:
        return None
    band = block.page_height * _REGION_PAGE_SHARE
    if block.bottom <= band:
        return "head"
    if block.top >= block.page_height - band:
        return "foot"
    return ""


def _mask_digits(norm: str) -> str:
    return _DIGIT_RUN.sub(_DIGIT_MASK, norm)


def _region_keys(page_blocks: list[_Raw]) -> list[tuple[int, _Key, _Key | None]]:
    """``(index into page_blocks, recurrence key, digit-insensitive key or None)`` for the blocks
    in this page's header/footer REGIONS.

    WHAT "REGION" MEANS depends on what the page was read from, and only that:

    * A MEASURED page (``layout-json``) uses :func:`_geometric_band` — true geometry, the plan's
      rule as written. The block stream is not consulted at all, so the block COUNT is irrelevant
      here and the html path's half-page cap has no counterpart.
    * A Chandra page (``html``) has no coordinates, so region is approximated by position in the
      page's stream: the first and last :data:`_REGION_BLOCKS`, capped at HALF the page so the
      two can never meet. That cap matters on a short page: with two blocks either end, a
      3-block page would otherwise be entirely "region" and the paragraph between its header and
      its footer would be judged as furniture. Measured on the fixtures: without the cap, a body
      paragraph repeated on 2 of 3 pages was stripped. A 1-block page has no region at all — its
      one block is the page's content.

    The key pins the SLOT as well as the text, which is the part that separates a running header
    from a short paragraph the document genuinely repeats: furniture sits in the same place on
    every page, repeated content does not. The slot is kept on the measured path too, where
    geometry has already pinned the region, for ONE reason: it makes the two paths' keys the same
    shape, so a footer appearing on both kinds of page in a mixed document POOLS its votes
    instead of each kind failing the >50% test alone.

    THE DIGIT-INSENSITIVE KEY is the second return value, and it is produced ONLY inside a
    geometric band. The html path documents why it refuses digit masking: "Page 3 of 9" does not
    recur as the same text, and masking would also merge genuinely different short lines, with
    only position-in-the-stream to contain the damage. Inside a strict geometric band that
    objection is much weaker — a line in the bottom 8% of every page whose text differs only in
    its digits is a page marker — so it is matched there and nowhere else. It is bounded three
    ways: the band (body text is not in it), :data:`_MAX_RUNNING_CHARS`, and the unchanged
    >50%-of-pages vote, which a clause citing "IEC 62368-1" on one page cannot pass.

    Headings, tables and table rows are never candidates — see the module docstring."""
    keys: list[tuple[int, _Key, _Key | None]] = []
    region = min(_REGION_BLOCKS, len(page_blocks) // 2)
    for index, block in enumerate(page_blocks):
        if block.kind not in (BLOCK_PARAGRAPH, BLOCK_LIST_ITEM):
            continue
        norm = normalise_breadcrumb(block.text).casefold()
        if not norm or len(norm) > _MAX_RUNNING_CHARS:
            continue
        from_end = len(page_blocks) - 1 - index
        band = _geometric_band(block)
        if band is not None:
            if band:
                key = (band, index if band == "head" else from_end, norm)
                masked = _mask_digits(norm)
                keys.append((index, key, (band, key[1], masked) if masked != norm else None))
            continue
        if index < region:
            keys.append((index, ("head", index, norm), None))
        if from_end < region:
            keys.append((index, ("foot", from_end, norm), None))
    return keys


def _strip_running_blocks(blocks: list[_Raw]) -> tuple[list[_Raw], tuple[str, ...]]:
    """Drop running headers/footers. Returns ``(kept blocks, the normalised texts dropped)``.

    The plan's rule is "> 50% of pages"; what "region" means per path (:func:`_region_keys`) and
    the guards
    are in the module docstring. Runs BEFORE :func:`_merge_cross_page_tables` on purpose: the
    page header is precisely what sits between a table's two halves, so a merge attempted first
    would see the header, not the continuation, as the next block.

    The page count is derived from the blocks, so it counts pages that actually PRODUCED
    CONTENT. The overlay's own page list is the wrong denominator twice over: a page whose
    markup is ``""`` is a documented, normal shape, and counting one of those both inflated the
    ">50% of pages" share and — worse — let a single empty page satisfy the
    :data:`_MIN_PAGES_FOR_REPEAT` guard on a 2-page document, which is the one thing that guard
    exists to prevent.

    The slot is the EVIDENCE but not the action. A key recurring in the same slot on >50% of
    pages is what identifies furniture; once identified, every occurrence of that TEXT anywhere
    in a page's header/footer region goes. Keying the removal on the slot too left the same
    header standing on the pages where it had shifted by one block, while
    ``removed_running_text`` reported it as gone — an observation field that lied.

    ``removed_running_text`` may therefore report a DIGIT-MASKED pattern ("page # of #") rather
    than any text the document literally contains — that is the honest record of what the rule
    matched, and it is also how a reviewer can tell the two kinds of match apart. It reports what
    was ACTUALLY matched rather than every key that won the vote, because a header recurring
    identically wins under both its plain and its masked key and is one piece of furniture."""
    by_page: dict[int, list[_Raw]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    if len(by_page) < _MIN_PAGES_FOR_REPEAT:
        return blocks, ()

    pages_per_key: dict[_Key, set[int]] = {}
    for page, page_blocks in by_page.items():
        for _, key, masked in _region_keys(page_blocks):
            pages_per_key.setdefault(key, set()).add(page)
            if masked is not None:
                pages_per_key.setdefault(masked, set()).add(page)

    threshold = len(by_page) * _REPEAT_PAGE_SHARE
    running = {key[2] for key, pages in pages_per_key.items() if len(pages) > threshold}
    if not running:
        return blocks, ()

    drop: set[int] = set()
    removed: set[str] = set()
    for page_blocks in by_page.values():
        for index, key, masked in _region_keys(page_blocks):
            # The plain text first, so a header that recurs IDENTICALLY is reported as itself and
            # not as its digit-masked pattern. Both keys win the vote for such a header, and
            # reporting both would list one piece of furniture twice under two names.
            if key[2] in running:
                matched = key[2]
            elif masked is not None and masked[2] in running:
                matched = masked[2]
            else:
                continue
            drop.add(id(page_blocks[index]))
            removed.add(matched)
    return [b for b in blocks if id(b) not in drop], tuple(sorted(removed))


# --- cross-page table merge ---------------------------------------------------------------------


def _columns(table: _Raw) -> int:
    return max((len(r.cells) for r in [*([_Row(0, table.header)] if table.header else []), *table.rows]), default=0)


def _continues(head: _Raw, tail: _Raw) -> bool:
    """Is ``tail`` (first block of page N+1) the rest of ``head`` (last block of page N)?

    Deliberately refused, because a wrong merge fuses two unrelated tables — worse than leaving
    one table split, which only costs a row its neighbours:

    * a continuation with a ``<th>`` header row of its OWN: a table that re-declares its columns
      is announcing itself, not continuing;
    * a different column count;
    * fewer than :data:`_MIN_MERGE_COLUMNS` columns, where "same column count" proves nothing;
    * a continuation with nothing in it;
    * a continuation whose first row does not have the SHAPE of the head's data in the same
      columns — see :func:`_shapes_agree`.

    That last one is the only guard with teeth when an extractor never emits ``<th>`` at all,
    which is the normal case: the ``<th>`` test above is then vacuous, every continuation's first
    row is a guessed header, and two plainly unrelated tables either side of a page break merged
    — the second table's header row re-labelled under the first table's columns. The row was not
    eaten, which is what the earlier reasoning claimed; it was mislabelled, which is worse,
    because a fabricated column name is indistinguishable from a real one downstream.

    The alternative guard — requiring that at least one side declared ``<th>`` — was rejected: it
    would switch cross-page merging off entirely for every extractor that does not emit ``<th>``,
    i.e. for most real tables, trading a wrong merge for no merge at all on the common path. The
    shape check costs merges only where the evidence genuinely points the other way.

    NOT refused: a ``head`` that is a header row and nothing else, i.e. a table whose labels were
    stranded at the foot of page N. Leaving that split is not the cheap outcome it looks like —
    the continuation has no ``<th>``, so the header heuristic would consume its first DATA row as
    labels, and the merge is what gives those rows the names the document actually wrote. It is
    also the one case the shape check cannot judge (no head data to compare against), so it rests
    on position alone and is the weakest merge this function allows; :class:`TableMerge` records
    ``header_declared`` so exactly these are reviewable.

    Refused by the CALLER, not here, because they are about position rather than shape: pages
    that are not consecutive by ``page_no`` (the overlay omits pages with no markup, so adjacent
    entries can be pages 1 and 3), and anything other than a table at the end of page N and the
    very start of page N+1 — an intervening heading or paragraph means the document moved on."""
    if head.kind != BLOCK_TABLE or tail.kind != BLOCK_TABLE:
        return False
    if tail.header_from_th:
        return False
    head_columns = _columns(head)
    if head_columns < _MIN_MERGE_COLUMNS or head_columns != _columns(tail):
        return False
    if not (tail.rows or tail.header):
        return False
    return tail.header is None or _shapes_agree(head, tail.header)


def _has_digit(cell: str) -> bool:
    return any(c.isdigit() for c in cell)


def _shapes_agree(head: _Raw, candidate: list[str]) -> bool:
    """Does ``candidate`` read as more of ``head``'s data, or as another table's header row?

    The signal is per-column NUMERACY, which is the cheapest property that actually separates the
    two on real tables: a value column ("700 nits", "4000:1", "Rs. 19.44 Crores") carries digits
    and its label ("Value", "Revised amount") does not. So for each column where both sides say
    something, the head's data and the candidate must agree on whether that column holds digits.
    One disagreeing column is enough to refuse — a wrong merge is worse than a missed one.

    Blank cells are skipped: they are absence of evidence, not evidence of difference. A head
    with no data rows yields nothing to compare and the check passes, which is the stranded-header
    case :func:`_continues` documents.

    Deliberately NOT a similarity score over the head's header text: a continuation's first row
    is data, so comparing it to labels would answer a different question, and any threshold would
    be a number with no corpus behind it."""
    for column, cell in enumerate(candidate):
        if not cell.strip():
            continue
        seen = [r.cells[column] for r in head.rows if column < len(r.cells) and r.cells[column].strip()]
        if not seen:
            continue
        if _has_digit(cell) != (sum(_has_digit(c) for c in seen) * 2 > len(seen)):
            return False
    return True


def _absorb(head: _Raw, tail: _Raw) -> TableMerge:
    """Fold a continuation into the table it continues, each row keeping its own page.

    Returns the observation of what was done — a merge is a judgement call about two tables
    being one, and an unrecorded judgement call cannot be reviewed."""
    before = len(head.rows)
    if tail.header is not None and not tail.header_from_th:
        # It was never a header — only the first row of a table with no <th>, which is what the
        # header heuristic falls back to. Hand it back as data or the merge would eat a row.
        head.rows.append(_Row(page=tail.page, cells=tail.header))
    head.rows.extend(tail.rows)
    head.pages.extend(p for p in tail.pages if p not in head.pages)
    if head.html and tail.html:
        head.html = f"{head.html}\n{tail.html}"
    if head.text_verbatim and tail.text_verbatim:
        # Both sides carry the extractor's own band body (`layout-json`), so the continuation's
        # body is joined on or the merged table's text would stop at the page break while its
        # ROWS went on past it. Same join as the retained markup above, and for the same reason:
        # each page's body is kept as the extractor wrote it rather than re-rendered as one.
        head.text = f"{head.text}\n{tail.text}" if head.text else tail.text
    elif head.text_verbatim or tail.text_verbatim:
        # A merge ACROSS A FORMAT BOUNDARY (a scanned page continuing onto a born-digital one, or
        # the reverse). One side's body is the extractor's own and the other side has none, so
        # there is no single verbatim body for the merged table: fall back to deriving the text
        # from the merged rows, which is true of both halves. Keeping one half's verbatim body
        # would silently describe the merged table by only the part it came from.
        head.text, head.text_verbatim = "", False
    return TableMerge(
        pages=tuple(head.pages),
        columns=_columns(head),
        rows_added=len(head.rows) - before,
        header_declared=head.header_from_th or tail.header_from_th,
    )


def _merge_repeated_page_titles(
    sections: tuple[LayoutSection, ...],
) -> tuple[tuple[LayoutSection, ...], list[SectionMerge]]:
    """Fold a page TITLE repeated on consecutive pages into one section.

    A tender sets its attachment title in heading type at the top of every page ("Attachment-1",
    18 pt). Each copy is a real heading, so the furniture rule correctly refuses to touch it — a
    heading is never a running-text candidate, because "stripping by text alone would take the
    heading and every breadcrumb beneath it", which is the Phase-2 decision that makes the plan's
    acceptance breadcrumb reachable at all. The consequence was that the tree grew ONE ROOT PER
    PAGE: three identical `Attachment-1` sections at ordinals 0/1/2, each owning only that page's
    subsections.

    That is not a retrieval defect — the breadcrumbs are already right, and a container heading
    emits no chunk of its own — it is a STRUCTURE defect, and it lands on the phases that navigate
    the tree rather than search it: a per-page `section_id` means "everything under Attachment-1"
    returns three disconnected subtrees, and Phase 4 would summarise the same logical section three
    times.

    DELETING the title was the other option and is rejected, not deferred: it is the furniture rule
    this function exists to work around, and it would take the breadcrumb with it.

    Merged conservatively, on the same principle the cross-page table merge states — a wrong merge
    fuses things the document kept apart, which is worse than leaving them split. All four must
    hold:

    * identical breadcrumbs, so only true repeats are candidates;
    * each occurrence on a DIFFERENT page, consecutive by page number — a page title repeats by
      page, so a repeat within one page is something else;
    * no occurrence owns any block but its heading. This is the load-bearing guard: a page title is
      a CONTAINER, whereas two genuinely distinct same-named sections ("Annexure" ... "Annexure")
      each carry their own prose, and fusing those would weld unrelated content into one section;
    * the run spans at least :data:`_MIN_PAGES_FOR_REPEAT` pages — the same vote the furniture rule
      takes, so a title appearing twice is left alone exactly as a two-page document's footer is.

    Applied to siblings at EVERY depth, not just roots: the breadcrumb includes the parent path, so
    only true siblings can ever compare equal, and a repeated sub-heading has the same signature and
    the same fix. Recurses into the result, so a merged parent's children are themselves considered.
    """
    merges: list[SectionMerge] = []
    out: list[LayoutSection] = []
    group: list[LayoutSection] = []

    def flush() -> None:
        if not group:
            return
        if len(group) >= _MIN_PAGES_FOR_REPEAT:
            head = group[0]
            children = tuple(child for section in group for child in section.children)
            out.append(replace(head, children=children))
            merges.append(
                SectionMerge(
                    breadcrumb=head.breadcrumb,
                    pages=tuple(s.page for s in group[1:] if s.page is not None),
                    children_merged=sum(len(s.children) for s in group[1:]),
                )
            )
        else:
            out.extend(group)
        group.clear()

    for section in sections:
        if group and _continues_page_title(group[-1], section):
            group.append(section)
            continue
        flush()
        group.append(section) if _is_page_title_candidate(section) else out.append(section)
    flush()

    final: list[LayoutSection] = []
    for section in out:
        children, child_merges = _merge_repeated_page_titles(section.children)
        final.append(replace(section, children=children) if children != section.children else section)
        merges.extend(child_merges)
    return tuple(final), merges


def _is_page_title_candidate(section: LayoutSection) -> bool:
    """A container heading on a known page — owning nothing but its own heading block."""
    return (
        section.page is not None
        and bool(section.children)
        and all(b.block_type == BLOCK_HEADING for b in section.blocks)
    )


def _continues_page_title(previous: LayoutSection, candidate: LayoutSection) -> bool:
    """Whether ``candidate`` is the NEXT page's copy of ``previous``'s title."""
    return (
        _is_page_title_candidate(candidate)
        and candidate.breadcrumb == previous.breadcrumb
        and previous.page is not None
        and candidate.page is not None
        and candidate.page == previous.page + 1
    )


def _merge_cross_page_tables(blocks: list[_Raw]) -> tuple[list[_Raw], list[TableMerge]]:
    """Join tables split by a page break into one table whose rows span both pages.

    Returns the merged blocks and one :class:`TableMerge` per merge performed."""
    by_page: dict[int, list[_Raw]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)

    out: list[_Raw] = []
    merges: dict[int, TableMerge] = {}  # keyed by table identity, so a 3-page chain is ONE record
    open_table: _Raw | None = None  # a table that ran to the end of its page
    open_page: int | None = None
    for page in sorted(by_page):
        page_blocks = by_page[page]
        if open_table is not None and open_page == page - 1 and _continues(open_table, page_blocks[0]):
            merge = _absorb(open_table, page_blocks[0])  # already emitted; mutated in place
            prior = merges.get(id(open_table))
            merges[id(open_table)] = (
                merge if prior is None else replace(merge, rows_added=prior.rows_added + merge.rows_added)
            )
            page_blocks = page_blocks[1:]
        out.extend(page_blocks)
        if page_blocks:
            last = page_blocks[-1]
            open_table = last if last.kind == BLOCK_TABLE else None
        # An empty remainder means this page held nothing but the continuation, so the table is
        # still the open one and a third page can continue it.
        open_page = page
    return out, list(merges.values())


# --- tree assembly ------------------------------------------------------------------------------


# (title, <hN> level, page, the blocks the heading owns) — what `_nest` orders into a tree.
_Heading = tuple[str, int, int | None, list["_Raw"]]


@dataclass(slots=True)
class _Node:
    title: str
    level: int
    page: int | None
    blocks: list[_Raw] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)


def _nest(headings: list[_Heading]) -> list[_Node]:
    """Nest ``(title, level, page, blocks)`` by level: deeper than its predecessor nests under
    it, shallower closes back to the right ancestor, equal becomes a sibling.

    A heading whose level SKIPS a level (``<h1>`` then ``<h3>``) still nests one deeper rather
    than inventing the missing ``<h2>`` — the same rule ``sections.section_path_at`` states for
    numbering: showing a level the document never wrote would be a claim about its structure
    that is not true."""
    roots: list[_Node] = []
    stack: list[_Node] = []
    for title, level, page, blocks in headings:
        node = _Node(title=title, level=level, page=page, blocks=blocks)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _safe_title(text: str) -> str:
    """A heading's text with the breadcrumb separator neutralised, so only this parser's own
    joins can create a path level.

    A heading that itself contains the separator ("A SEP B", writing SEP for the character)
    otherwise FORGES one: it produces a section whose breadcrumb is "A SEP B", identical to
    the breadcrumb of a genuine ``<h2>B</h2>`` nested under
    ``<h1>A</h1>``. The forgery takes ``ordinal=0``, so the real nested section's
    ``section_id`` — and every block id beneath it — comes to depend on an unrelated heading
    elsewhere in the document. That is the same class of drift :func:`normalise_breadcrumb`
    exists to prevent, and it does not cover this: U+203A is not a control character.

    ``>`` rather than deletion: it keeps what the document visually said, where stripping would
    run "A SEP B" together as "A B". The heading BLOCK keeps the text verbatim either way, so
    nothing is lost — this is only the form that goes into a path and an id."""
    return _collapse(text.replace(BREADCRUMB_SEP.strip(), ">"))


def _nodes(blocks: list[_Raw]) -> list[_Node]:
    """The block stream as a forest of heading nodes.

    Blocks before the first heading — and EVERY block of a document whose markup has no heading
    at all — go into one implicit root rather than being discarded: the heading is what names a
    section, not what makes its content real."""
    implicit: _Node | None = None
    headings: list[_Heading] = []
    for block in blocks:
        if block.kind == BLOCK_HEADING:
            headings.append((_safe_title(block.text), block.level, block.page, [block]))
        elif headings:
            headings[-1][3].append(block)
        else:
            if implicit is None:
                implicit = _Node(title="", level=0, page=block.page)
            implicit.blocks.append(block)
    return ([implicit] if implicit is not None else []) + _nest(headings)


def _tree(doc_id: str, blocks: list[_Raw]) -> tuple[LayoutSection, ...]:
    return _sections(doc_id, _nodes(blocks), (), 0, {})


def _sections(
    doc_id: str,
    nodes: list[_Node],
    trail: tuple[str, ...],
    depth: int,
    seen: dict[str, int],
) -> tuple[LayoutSection, ...]:
    """Build frozen sections in pre-order, assigning ids as we go.

    Pre-order because ``ordinal`` counts earlier occurrences of the same breadcrumb and must
    therefore be assigned in document order; ``seen`` is that counter, shared down the walk."""
    out: list[LayoutSection] = []
    for node in nodes:
        titles = (*trail, node.title) if node.title else trail
        breadcrumb = normalise_breadcrumb(BREADCRUMB_SEP.join(titles))
        ordinal = seen.get(breadcrumb, 0)
        seen[breadcrumb] = ordinal + 1
        sid = section_id(doc_id, breadcrumb, ordinal)
        blocks = _blocks(sid, node.blocks, breadcrumb)
        text = "\n\n".join(b.text for b in blocks)
        out.append(
            LayoutSection(
                section_id=sid,
                breadcrumb=breadcrumb,
                title=node.title,
                depth=depth,
                ordinal=ordinal,
                text=text,
                text_hash=text_hash(text),
                heading_level=node.level or None,
                page=node.page,
                blocks=blocks,
                children=_sections(doc_id, node.children, titles, depth + 1, seen),
            )
        )
    return tuple(out)


# --- blocks -------------------------------------------------------------------------------------


def _label(header: list[str] | None, index: int) -> str:
    """A column's name — its header label, or its POSITION when the table gave none. Inventing a
    name ("Amount") would put a word in the document's mouth; "Column 2" says only where the
    value sat, which is true."""
    if header is not None and index < len(header) and header[index]:
        return header[index]
    return f"Column {index + 1}"


def _row_text(breadcrumb: str, header: list[str] | None, cells: list[str]) -> str:
    """``<breadcrumb>`` + :data:`BREADCRUMB_SEP` + ``Table: <col1>=<v1> | <col2>=<v2>``.

    The plan's linearisation of a table row.

    The header names and the breadcrumb are IN the row's text so the row stands alone: retrieved
    on its own, "19.44" says nothing, and the embedder cannot recover either from elsewhere."""
    pairs = " | ".join(f"{_label(header, i)}={cell}" for i, cell in enumerate(cells))
    body = f"Table: {pairs}"
    return f"{breadcrumb}{BREADCRUMB_SEP}{body}" if breadcrumb else body


def _table_text(table: _Raw) -> str:
    """A table as pipe-delimited lines, header first. Its rows are separately addressable
    blocks; this is the whole table as one retrievable unit.

    Only for a table that does not already HAVE text. An html ``<table>`` does not — the markup
    is the structure and the rendering is ours to choose — while a ``layout-json`` table block
    arrives with the band body the extractor already spliced into the document's text, and
    re-rendering that would let the block and the text disagree about a table they came from
    together."""
    rows = [*([table.header] if table.header else []), *(r.cells for r in table.rows)]
    return "\n".join(" | ".join(cells) for cells in rows)


def _blocks(section: str, raw: list[_Raw], breadcrumb: str) -> tuple[LayoutBlock, ...]:
    """A section's blocks, with ids, plus one ``table_row`` per data row of every table."""
    out: list[LayoutBlock] = []
    for block in raw:
        text = block.text if block.text_verbatim or block.kind != BLOCK_TABLE else _table_text(block)
        ordinal = len(out)
        bid = block_id(section, block.kind, ordinal, text_hash(text))
        out.append(
            LayoutBlock(
                block_id=bid,
                block_type=block.kind,
                text=text,
                page=block.page,
                ordinal=ordinal,
                pages=tuple(block.pages or [block.page]),
                html=block.html if block.kind == BLOCK_TABLE else None,
            )
        )
        if block.kind != BLOCK_TABLE:
            continue
        for row in block.rows:
            row_text = _row_text(breadcrumb, block.header, row.cells)
            row_ordinal = len(out)
            out.append(
                LayoutBlock(
                    block_id=block_id(section, BLOCK_TABLE_ROW, row_ordinal, text_hash(row_text)),
                    block_type=BLOCK_TABLE_ROW,
                    text=row_text,
                    page=row.page,
                    ordinal=row_ordinal,
                    pages=(row.page,),
                    parent_block_id=bid,
                )
            )
    return tuple(out)


# --- plain-text fallback ------------------------------------------------------------------------


def _text_sections(doc_id: str, text: str) -> tuple[LayoutSection, ...]:
    """The tree for a document with no markup, from :func:`rag_core.sections.detect_sections`.

    Sections carry NO blocks: the text layer has no block types, and manufacturing paragraphs
    from blank lines would produce units the plan's block stream promises coordinates and types
    for. ``detect_sections`` is used exactly as it is — it is tuned on two real corpora and the
    v1 indexer depends on its current behaviour."""
    found: list[Section] = detect_sections(text)
    if not found:
        return _sections(doc_id, [_Node(title="", level=0, page=None)], (), 0, {})
    headings: list[_Heading] = [(_safe_title(s.label), len(s.levels), None, []) for s in found]
    return _sections(doc_id, _nest(headings), (), 0, {})
