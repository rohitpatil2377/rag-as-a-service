"""The layout parser: a stored overlay -> a section/block tree (RAG v2 Phase 2).

THE FIXTURES BELOW STAND IN FOR A DOCUMENT THIS REPO DOES NOT HAVE. The plan's Phase-2
acceptance names the AEGCL corrigendum ("Corrigendum clauses Sl.1-9 and Attachment-1 sections
1-23 ... e.g. `Attachment-1 > 13. Video Wall Technical Requirements > Standards and
Compliance`. 'Attachment-1' page header no longer appears inside block text"). That PDF is not
in the repository and is not reproduced here. These fixtures encode the same SHAPES the
acceptance describes — the repeated page header, the nested numbered breadcrumb, the table
split by a page break — at the smallest size that still exercises them, and should be replaced
by (not merely joined to) a fixture built from the real document once it is supplied.

Everything here goes through the public `parse_layout`; the overlay is written as the literal
JSON mapping that is persisted to MinIO, because that mapping is the contract this parser has
with the extractor, not the extractor's dataclass.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag_core.layout import (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TABLE_ROW,
    BREADCRUMB_SEP,
    MARKUP_FORMAT_HTML,
    MARKUP_FORMAT_LAYOUT_JSON,
    LayoutBlock,
    LayoutSection,
    SkippedPage,
    TableMerge,
    UnsupportedLayoutVersionError,
    parse_layout,
)

_DOC = "fixture-corrigendum-shapes.pdf"
# The plan's acceptance breadcrumb, verbatim down to its separator — the one literal here, so a
# change to `BREADCRUMB_SEP` cannot quietly make these tests agree with themselves.
_PLAN_BREADCRUMB = "Attachment-1 › 13. Video Wall Technical Requirements › Standards and Compliance"  # noqa: RUF001
_ATTACHMENT_13 = _PLAN_BREADCRUMB.rsplit(BREADCRUMB_SEP, 1)[0]


# The overlay with no pages at all, in the persisted v2 shape.
_EMPTY_OVERLAY: dict[str, Any] = {"schema_version": 2, "extractor": "", "content_hash": "", "pages": []}


def _overlay(
    *pages: tuple[int, str],
    extractor: str = "chandra",
    markup_format: str = "html",
    content_hash: str = "",
) -> dict[str, Any]:
    """An overlay holding exactly these ``(page_no, markup)`` pages — only pages that carried
    markup appear, so page_no and position deliberately need not line up. Written as the literal
    persisted JSON (schema v2), because that mapping is the contract, not the writer's dataclass."""
    return {
        "schema_version": 2,
        "extractor": extractor,
        "content_hash": content_hash,
        "pages": [
            {"page_no": no, "markup": markup, "markup_format": markup_format, "degradations": []}
            for no, markup in pages
        ],
    }


# --- the born-digital (`layout-json`) overlay ---------------------------------------------------
#
# Written as the literal persisted payload for the same reason the html overlay is: that mapping
# is the contract with the extractor. A page is 612 x 792 pt (Letter), pdfplumber's coordinate
# space — origin TOP-left, `top` growing downwards — so `top=40` is near the top of the page and
# `top=750` near the bottom, and the header/footer bands are the outer 8% at each edge.

_PAGE_H = 792.0
_BODY_SIZE = 12.0
# Inside the geometric bands: a header's whole box above 63.36 pt, a footer's whole box below
# 728.64 pt. Chosen to sit where a word processor puts page furniture — in the margin.
_IN_HEAD = 30.0
_IN_FOOT = 745.0
_MID_PAGE = 300.0


def _line(
    text: str,
    *,
    page: int = 1,
    size: float = _BODY_SIZE,
    bold: bool = False,
    top: float = _MID_PAGE,
) -> dict[str, Any]:
    """One measured text line. ``cells`` is None — the mirror of a table's None typography."""
    return {
        "block_type": "line",
        "text": text,
        "page": page,
        "bbox": [72.0, top, 400.0, top + size],
        "font_size": size,
        "bold": bold,
        "cells": None,
    }


def _table(cells: list[list[str | None]], *, text: str, page: int = 1, top: float = _MID_PAGE) -> dict[str, Any]:
    """One spliced table band: pdfplumber's own ``cells`` grid, and the band body VERBATIM as
    ``text`` — the exact string that reached the document's text. No typography, by design."""
    return {
        "block_type": "table",
        "text": text,
        "page": page,
        "bbox": [72.0, top, 400.0, top + 20.0 * len(cells)],
        "font_size": None,
        "bold": None,
        "cells": cells,
    }


def _payload(page: int, blocks: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "format": "layout-json",
            "version": 1,
            "page_no": page,
            "page_width": 612.0,
            "page_height": _PAGE_H,
            "blocks": blocks,
        }
    )


def _json_overlay(*pages: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
    return _overlay(
        *((no, _payload(no, blocks)) for no, blocks in pages),
        extractor="pdfplumber",
        markup_format="layout-json",
    )


_LINES_ONE_HEADING = [
    _line("5.1 Process", size=18.0, bold=True, top=60.0),
    _line("Entry to the site is controlled by the officer on duty.", top=100.0),
]


def _walk(sections: tuple[LayoutSection, ...]) -> list[LayoutSection]:
    out: list[LayoutSection] = []
    for section in sections:
        out.append(section)
        out.extend(_walk(section.children))
    return out


def _blocks(sections: tuple[LayoutSection, ...]) -> list[LayoutBlock]:
    return [block for section in _walk(sections) for block in section.blocks]


def _by_crumb(sections: tuple[LayoutSection, ...], breadcrumb: str) -> LayoutSection:
    found = [s for s in _walk(sections) if s.breadcrumb == breadcrumb]
    assert found, f"no section {breadcrumb!r} in {[s.breadcrumb for s in _walk(sections)]}"
    return found[0]


# --- the plan's acceptance shapes ---------------------------------------------------------------

# A three-page attachment: page 1 carries the real heading tree, pages 2 and 3 repeat
# "Attachment-1" as a running page header, and a parameter table runs from page 1 onto page 2.
_PAGE_1 = """
<h1>Attachment-1</h1>
<h2>13. Video Wall Technical Requirements</h2>
<h3>Standards and Compliance</h3>
<p>The controller shall comply with IEC 62443-4-2 and IEC 61000-4-2.</p>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Brightness</td><td>700 nits</td></tr>
</table>
"""
_PAGE_2 = """
<p>Attachment-1</p>
<table>
  <tr><td>Contrast ratio</td><td>4000:1</td></tr>
  <tr><td>Bezel width</td><td>0.88 mm</td></tr>
</table>
"""
_PAGE_3 = """
<p>Attachment-1</p>
<h3>Warranty and Support</h3>
<ul><li>Five years comprehensive on-site warranty.</li></ul>
"""


def _attachment() -> tuple[LayoutSection, ...]:
    return parse_layout(_overlay((1, _PAGE_1), (2, _PAGE_2), (3, _PAGE_3)), _DOC).sections


def test_a_repeated_page_header_does_not_appear_in_any_block_text() -> None:
    tree = parse_layout(_overlay((1, _PAGE_1), (2, _PAGE_2), (3, _PAGE_3)), _DOC)
    paragraphs = [b.text for b in _blocks(tree.sections) if b.block_type != BLOCK_HEADING]
    assert "Attachment-1" not in paragraphs
    assert tree.removed_running_text == ("attachment-1",)


def test_the_genuine_heading_survives_the_page_header_that_repeats_its_words() -> None:
    """The running header and the real ``<h1>`` carry the SAME text. Stripping by text alone
    would take the heading with it, and every breadcrumb beneath it."""
    sections = _attachment()
    assert [s.breadcrumb for s in sections] == ["Attachment-1"]
    assert sections[0].blocks[0].block_type == BLOCK_HEADING
    assert sections[0].blocks[0].text == "Attachment-1"


def test_nested_headings_give_the_breadcrumb_of_the_plans_shape() -> None:
    section = _by_crumb(_attachment(), _PLAN_BREADCRUMB)
    assert (section.depth, section.heading_level, section.page) == (2, 3, 1)
    assert section.title == "Standards and Compliance"


def test_a_shallower_heading_closes_back_to_the_right_ancestor() -> None:
    """``<h3>Warranty and Support</h3>`` on page 3 follows an ``<h3>`` on page 1, so it is that
    heading's SIBLING under ``13. ...``, not a child of it."""
    sections = _attachment()
    parent = _by_crumb(sections, _ATTACHMENT_13)
    assert [c.title for c in parent.children] == ["Standards and Compliance", "Warranty and Support"]


def test_a_heading_deeper_than_its_predecessor_nests_under_it() -> None:
    sections = _attachment()
    assert [s.title for s in sections[0].children] == ["13. Video Wall Technical Requirements"]
    assert sections[0].depth == 0 and sections[0].children[0].depth == 1


def test_a_list_item_is_its_own_block_type() -> None:
    blocks = _by_crumb(_attachment(), f"{_ATTACHMENT_13}{BREADCRUMB_SEP}Warranty and Support")
    assert [b.block_type for b in blocks.blocks] == [BLOCK_HEADING, "list_item"]


# --- cross-page table merge ---------------------------------------------------------------------


def _merged_table(sections: tuple[LayoutSection, ...]) -> tuple[LayoutBlock, list[LayoutBlock]]:
    tables = [b for b in _blocks(sections) if b.block_type == BLOCK_TABLE]
    assert len(tables) == 1, [t.text for t in tables]
    rows = [b for b in _blocks(sections) if b.block_type == BLOCK_TABLE_ROW]
    return tables[0], rows


def test_a_table_split_by_a_page_break_comes_back_as_one_table() -> None:
    table, rows = _merged_table(_attachment())
    assert table.page == 1
    assert table.pages == (1, 2)
    assert [r.text.rsplit("Table: ", 1)[1] for r in rows] == [
        "Parameter=Brightness | Value=700 nits",
        "Parameter=Contrast ratio | Value=4000:1",
        "Parameter=Bezel width | Value=0.88 mm",
    ]


def test_every_row_of_a_merged_table_keeps_its_own_page() -> None:
    _, rows = _merged_table(_attachment())
    assert [r.page for r in rows] == [1, 2, 2]


def test_a_merged_tables_rows_all_point_back_at_the_one_table() -> None:
    table, rows = _merged_table(_attachment())
    assert {r.parent_block_id for r in rows} == {table.block_id}


def test_a_continuation_with_a_header_row_of_its_own_is_not_merged() -> None:
    """A table that re-declares its columns is announcing itself, not continuing. A wrong merge
    fuses two unrelated tables, which is worse than leaving one split."""
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tail = "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_a_continuation_with_a_different_column_count_is_not_merged() -> None:
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tail = "<table><tr><td>3</td><td>4</td><td>5</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_a_single_column_table_is_not_merged() -> None:
    """With one column, "same column count" proves nothing, so the merge test is no test."""
    head = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
    tail = "<table><tr><td>2</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_tables_on_non_consecutive_pages_are_not_merged() -> None:
    """The overlay omits pages with no markup, so two ADJACENT entries can be pages 1 and 3.
    Adjacency in the overlay is not adjacency in the document."""
    table = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tail = "<table><tr><td>3</td><td>4</td></tr></table>"
    tree = parse_layout(_overlay((1, table), (3, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_a_table_is_not_merged_across_an_intervening_heading() -> None:
    """A heading on page N+1 before the table means the document moved on."""
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tail = "<h2>14. Something Else</h2><table><tr><td>3</td><td>4</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_a_table_that_does_not_end_its_page_is_not_merged() -> None:
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table><p>Note below.</p>"
    tail = "<table><tr><td>3</td><td>4</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2


def test_a_table_can_be_merged_across_three_pages() -> None:
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    mid = "<table><tr><td>3</td><td>4</td></tr></table>"
    end = "<table><tr><td>5</td><td>6</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, mid), (3, end)), _DOC)
    table, rows = _merged_table(tree.sections)
    assert table.pages == (1, 2, 3)
    assert [r.page for r in rows] == [1, 2, 3]


# --- table rows as addressable units ------------------------------------------------------------


def test_a_row_is_linearised_with_its_breadcrumb_and_header_names() -> None:
    """Retrieved alone, "4000:1" says nothing and the embedder cannot recover either the column
    it sat in or the section it came from."""
    _, rows = _merged_table(_attachment())
    expected = f"{_PLAN_BREADCRUMB}{BREADCRUMB_SEP}Table: Parameter=Contrast ratio | Value=4000:1"
    assert rows[1].text == expected


def test_the_first_row_is_the_header_when_the_table_declares_no_th() -> None:
    markup = (
        "<table><tr><td>Clause</td><td>Revised value</td></tr><tr><td>MAAT</td><td>Rs. 19.44 Crores</td></tr></table>"
    )
    tree = parse_layout(_overlay((1, markup)), _DOC)
    rows = [b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]
    assert [r.text for r in rows] == ["Table: Clause=MAAT | Revised value=Rs. 19.44 Crores"]


def test_a_table_with_no_header_labels_names_its_columns_positionally() -> None:
    """Inventing a name would put a word in the document's mouth; a position is only true."""
    markup = "<table><tr><th></th><th></th></tr><tr><td>MAAT</td><td>Rs. 19.44 Crores</td></tr></table>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    rows = [b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]
    assert [r.text for r in rows] == ["Table: Column 1=MAAT | Column 2=Rs. 19.44 Crores"]


def test_a_table_block_keeps_the_engines_own_html() -> None:
    markup = "<p>Before.</p><table><tr><th>A</th></tr><tr><td>1</td></tr></table><p>After.</p>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    table = next(b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE)
    assert table.html == "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"


def test_a_merged_tables_html_is_its_parts_markup_and_not_a_synthesised_table() -> None:
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tail = "<table><tr><td>3</td><td>4</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    table, _ = _merged_table(tree.sections)
    assert table.html == f"{head}\n{tail}"


# --- documents the markup path must not lose ----------------------------------------------------


def test_a_document_with_no_heading_yields_one_implicit_root_holding_every_block() -> None:
    markup = "<p>First paragraph.</p><p>Second paragraph.</p><ul><li>An item.</li></ul>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    assert len(tree.sections) == 1
    root = tree.sections[0]
    assert (root.breadcrumb, root.title, root.heading_level, root.depth) == ("", "", None, 0)
    assert [b.text for b in root.blocks] == ["First paragraph.", "Second paragraph.", "An item."]


def test_blocks_before_the_first_heading_are_kept_not_discarded() -> None:
    markup = "<p>Issued under cover of letter no. 12.</p><h1>Corrigendum</h1><p>Clause 1 is revised.</p>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    assert [s.breadcrumb for s in tree.sections] == ["", "Corrigendum"]
    assert [b.text for b in tree.sections[0].blocks] == ["Issued under cover of letter no. 12."]


def test_header_footer_removal_does_not_eat_a_single_page_document() -> None:
    """On one page every block recurs on 100% of pages. Stripping on that reading would delete
    the document."""
    markup = "<p>Attachment-1</p><h1>Corrigendum</h1><p>Clause 1 is revised.</p><p>Page 1 of 1</p>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    assert [b.text for b in _blocks(tree.sections)] == [
        "Attachment-1",
        "Corrigendum",
        "Clause 1 is revised.",
        "Page 1 of 1",
    ]
    assert tree.removed_running_text == ()


_REPEATED = "Subject to clause 4."


def test_a_short_paragraph_repeated_in_the_body_of_every_page_is_kept() -> None:
    """A page's header/footer regions are capped at half of it, so the middle of a page is never
    furniture however often it repeats."""
    pages = tuple(
        (n, f"<h2>Section {n}</h2><p>Opening {n}.</p><p>{_REPEATED}</p><p>Detail {n}.</p><p>Closing {n}.</p>")
        for n in (1, 2, 3)
    )
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ()
    assert sum(1 for b in _blocks(tree.sections) if b.text == _REPEATED) == 3


def test_a_short_paragraph_repeated_at_a_different_slot_on_each_page_is_kept() -> None:
    """Furniture sits in the same place on every page; content that repeats does not. Here the
    sentence lands in the head region on page 1 and the foot region on page 3, so neither slot
    passes the >50%-of-pages test even though the TEXT recurs on all three."""
    pages = (
        (1, f"<h2>One</h2><p>{_REPEATED}</p><p>A.</p><p>B.</p><p>C.</p>"),
        (2, f"<h2>Two</h2><p>D.</p><p>{_REPEATED}</p><p>E.</p><p>F.</p>"),
        (3, f"<h2>Three</h2><p>G.</p><p>H.</p><p>{_REPEATED}</p><p>I.</p>"),
    )
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ()
    assert sum(1 for b in _blocks(tree.sections) if b.text == _REPEATED) == 3


def test_the_middle_of_a_three_block_page_is_not_treated_as_a_region() -> None:
    """Two blocks either end would make a 3-block page entirely "region". Measured: without the
    half-page cap, "Body." below was stripped from pages 2 and 3."""
    pages = tuple((n, f"<h2>Section {n}</h2><p>Body.</p><p>Footer {n}.</p>") for n in (1, 2, 3))
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ()
    assert sum(1 for b in _blocks(tree.sections) if b.text == "Body.") == 3


def test_a_running_footer_is_removed_from_the_foot_of_every_page() -> None:
    pages = tuple((n, f"<h2>Section {n}</h2><p>Body {n}.</p><p>Confidential</p>") for n in (1, 2, 3))
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ("confidential",)
    assert [b.text for b in _blocks(tree.sections) if b.block_type == BLOCK_PARAGRAPH] == [
        "Body 1.",
        "Body 2.",
        "Body 3.",
    ]


def test_a_long_repeated_paragraph_is_content_not_furniture() -> None:
    boilerplate = (
        "<p>This document is the property of the issuing authority and may not be reproduced, "
        "transmitted or disclosed to any third party without prior written consent.</p>"
    )
    pages = tuple((n, f"<h2>Section {n}</h2>{boilerplate}") for n in (1, 2, 3))
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ()
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_PARAGRAPH]) == 3


# --- empty and absent markup --------------------------------------------------------------------


def test_a_page_whose_markup_is_empty_contributes_no_blocks_but_is_still_a_page() -> None:
    """``markup == ""`` means the engine answered this page with nothing — distinct from a page
    that is absent because its engine emits no markup at all."""
    tree = parse_layout(_overlay((1, _PAGE_1), (2, "")), _DOC)
    assert tree.pages == (1, 2)
    assert tree.source == "markup"
    assert {b.page for b in _blocks(tree.sections)} == {1}


def test_an_overlay_with_no_pages_falls_back_to_the_text_tree() -> None:
    overlay = {"schema_version": 2, "extractor": "", "pages": []}
    tree = parse_layout(overlay, _DOC, text="5.1 Process\nEntry is controlled.\n")
    assert tree.source == "text"
    assert tree.pages == ()
    assert [s.breadcrumb for s in _walk(tree.sections)] == ["5.1 Process"]


def test_an_overlay_whose_every_page_is_empty_falls_back_to_the_text_tree() -> None:
    tree = parse_layout(_overlay((1, ""), (2, "   ")), _DOC, text="5.1 Process\nEntry is controlled.\n")
    assert tree.source == "text"
    assert [s.breadcrumb for s in _walk(tree.sections)] == ["5.1 Process"]


# --- plain-text fallback ------------------------------------------------------------------------

_TEXT = """Introduction paragraph with no heading above it.

5.1 Process
The process is owned by the content team.

5.1.1 Entry Criteria
A request must be logged before work starts.

5.2 Review
Two reviewers are required.
"""


def test_the_text_fallback_nests_the_sections_sections_py_detects() -> None:
    tree = parse_layout({"schema_version": 2, "extractor": "", "pages": []}, _DOC, text=_TEXT)
    assert [s.breadcrumb for s in _walk(tree.sections)] == [
        "5.1 Process",
        f"5.1 Process{BREADCRUMB_SEP}5.1.1 Entry Criteria",
        "5.2 Review",
    ]


def test_text_fallback_sections_carry_no_blocks() -> None:
    """The text layer has no block types; manufacturing them would invent structure."""
    tree = parse_layout({"schema_version": 2, "extractor": "", "pages": []}, _DOC, text=_TEXT)
    assert all(s.blocks == () for s in _walk(tree.sections))
    assert all(s.page is None for s in _walk(tree.sections))


def test_text_with_no_detectable_heading_still_yields_one_root_section() -> None:
    tree = parse_layout({"schema_version": 2, "extractor": "", "pages": []}, _DOC, text="Just prose, no headings.")
    assert len(tree.sections) == 1
    assert tree.sections[0].breadcrumb == ""


def test_an_overlay_with_no_pages_and_no_text_still_yields_a_tree() -> None:
    tree = parse_layout({"schema_version": 2, "extractor": "", "pages": []}, _DOC)
    assert len(tree.sections) == 1 and tree.sections[0].blocks == ()


# --- degradations -------------------------------------------------------------------------------


def test_a_degraded_page_is_surfaced_rather_than_parsed_as_if_clean() -> None:
    """A page the model looped on parses into a perfectly well-formed tree, and nothing in that
    tree says the page is untrustworthy."""
    overlay = {
        "schema_version": 2,
        "extractor": "chandra",
        "content_hash": "",
        "pages": [
            {"page_no": 1, "markup": "<p>Clean.</p>", "markup_format": "html", "degradations": []},
            {
                "page_no": 2,
                "markup": "<p>Looped.</p>",
                "markup_format": "html",
                "degradations": ["ocr_degenerate"],
            },
        ],
    }
    tree = parse_layout(overlay, _DOC)
    assert [(d.page, d.reasons) for d in tree.degradations] == [(2, ("ocr_degenerate",))]
    assert tree.extractor == "chandra"


# --- version handling ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 3, 0, "2", None, True, False])
def test_an_unrecognised_schema_version_is_refused_explicitly(version: object) -> None:
    """The overlay carries a version so a stored artifact written by another build is detectable
    rather than silently misparsed."""
    with pytest.raises(UnsupportedLayoutVersionError):
        parse_layout({"schema_version": version, "extractor": "", "pages": []}, _DOC)


def test_an_overlay_with_no_version_at_all_is_refused() -> None:
    with pytest.raises(UnsupportedLayoutVersionError):
        parse_layout({"extractor": "", "pages": []}, _DOC)


# --- ids ----------------------------------------------------------------------------------------
#
# That the Phase-1 helpers are deterministic is proved in `test_structure_ids.py` and is not
# re-proved here. What is tested is that THIS parser feeds them deterministically — the same
# overlay must produce the same ids on a re-run, or every stored citation moves on re-ingest.


def _ids(sections: tuple[LayoutSection, ...]) -> list[str]:
    return [s.section_id for s in _walk(sections)] + [b.block_id for b in _blocks(sections)]


def test_the_same_overlay_yields_the_same_ids_on_a_re_run() -> None:
    overlay = _overlay((1, _PAGE_1), (2, _PAGE_2), (3, _PAGE_3))
    assert _ids(parse_layout(overlay, _DOC).sections) == _ids(parse_layout(overlay, _DOC).sections)


def test_a_breadcrumb_whose_whitespace_drifts_does_not_move_the_section_id() -> None:
    """Heading text comes out of OCR, where a re-run can legitimately differ by a double space
    or a stray newline. Unnormalised, that would re-point a whole subtree of stored ids."""
    tidy = "<h1>Attachment-1</h1><h2>13. Video Wall Technical Requirements</h2><p>Body.</p>"
    drifted = "<h1>\n  Attachment-1 </h1><h2>13.  Video Wall\n Technical  Requirements\t</h2><p>Body.</p>"
    one = parse_layout(_overlay((1, tidy)), _DOC).sections
    two = parse_layout(_overlay((1, drifted)), _DOC).sections
    assert [s.section_id for s in _walk(one)] == [s.section_id for s in _walk(two)]
    assert [s.breadcrumb for s in _walk(one)] == [s.breadcrumb for s in _walk(two)]


def test_the_stored_breadcrumb_is_the_normalised_form_that_was_hashed() -> None:
    from rag_core.chunking import normalise_breadcrumb, section_id

    section = _by_crumb(_attachment(), _PLAN_BREADCRUMB)
    assert section.breadcrumb == normalise_breadcrumb(section.breadcrumb)
    assert section.section_id == section_id(_DOC, section.breadcrumb, section.ordinal)


def test_a_repeated_breadcrumb_gets_a_distinct_ordinal_and_so_a_distinct_id() -> None:
    markup = "<h1>Annexure</h1><p>One.</p><h1>Annexure</h1><p>Two.</p>"
    sections = parse_layout(_overlay((1, markup)), _DOC).sections
    assert [s.ordinal for s in sections] == [0, 1]
    assert sections[0].section_id != sections[1].section_id


def test_a_section_carries_the_hash_of_its_own_block_text() -> None:
    from rag_core.chunking import text_hash

    section = _by_crumb(_attachment(), _PLAN_BREADCRUMB)
    assert section.text_hash == text_hash(section.text)
    assert section.text.startswith("Standards and Compliance")


def test_block_ordinals_are_scoped_to_their_section() -> None:
    """A document-wide counter would re-point every id after an edit on an earlier page."""
    for section in _walk(_attachment()):
        assert [b.ordinal for b in section.blocks] == list(range(len(section.blocks)))


def test_a_header_stranded_at_the_foot_of_a_page_merges_with_its_rows() -> None:
    """Unmerged, the continuation has no ``<th>``, so its first DATA row would be consumed as
    column labels. The merge is what gives those rows the names the document wrote."""
    head = "<p>Parameters follow.</p><table><tr><th>Parameter</th><th>Value</th></tr></table>"
    tail = "<table><tr><td>Brightness</td><td>700 nits</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    table, rows = _merged_table(tree.sections)
    assert table.pages == (1, 2)
    assert [r.text for r in rows] == ["Table: Parameter=Brightness | Value=700 nits"]
    assert [r.page for r in rows] == [2]


# --- overlay schema v2 --------------------------------------------------------------------------
#
# Slice A bumped the persisted overlay to v2 (top-level `content_hash`, per-page `markup_format`)
# after this parser was first written against v1. v1 is now REFUSED rather than supported: a v1
# overlay declares no markup language, so reading one would mean inferring it from the extractor
# id — and both Chandra engines stamp "chandra" while emitting different languages. Nothing was
# ever persisted under v1.


def test_a_v2_overlay_is_parsed_and_its_content_hash_is_carried_onto_the_tree() -> None:
    """So a consumer can check this tree against the stored raw object without re-reading the
    overlay it came from."""
    tree = parse_layout(_overlay((1, _PAGE_1), content_hash="sha256:deadbeef"), _DOC)
    assert tree.source == "markup"
    assert tree.content_hash == "sha256:deadbeef"


def test_a_v1_overlay_is_refused_because_it_declares_no_markup_format() -> None:
    overlay = {"schema_version": 1, "extractor": "chandra", "pages": [{"page_no": 1, "markup": "<p>x</p>"}]}
    with pytest.raises(UnsupportedLayoutVersionError):
        parse_layout(overlay, _DOC)


def test_a_markdown_page_is_recorded_as_skipped_rather_than_parsed_as_html() -> None:
    """The hosted Chandra ``/convert`` path answers Markdown under the same extractor id, so the
    branch must be on the FORMAT. Parsing "# Heading" as HTML would silently yield one paragraph
    and no section."""
    tree = parse_layout(_overlay((1, "# 5.1 Process\n\nBody."), markup_format="markdown"), _DOC)
    assert tree.skipped_pages == (SkippedPage(page=1, markup_format="markdown", reason="unreadable_markup_format"),)
    assert _blocks(tree.sections) == []


def test_a_document_of_only_markdown_pages_still_gets_a_tree_from_the_text_fallback() -> None:
    """Skipped is not the same as lost: the document keeps its sections and loses only blocks."""
    tree = parse_layout(_overlay((1, "# x"), markup_format="markdown"), _DOC, text=_TEXT)
    assert tree.source == "text"
    assert next(s.breadcrumb for s in _walk(tree.sections)) == "5.1 Process"
    assert len(tree.skipped_pages) == 1


def test_a_born_digital_layout_json_page_is_read_as_blocks_not_skipped() -> None:
    """Phase 2c-1 declared a THIRD format — structured blocks with bounding boxes and per-line
    typography, which Chandra cannot give — and this parser skipped it with a named reason until
    it learned to read it. It does now (2c-2): the page contributes real blocks and is not in
    ``skipped_pages``."""
    tree = parse_layout(_json_overlay((1, _LINES_ONE_HEADING)), _DOC)

    assert tree.skipped_pages == ()
    assert tree.source == "markup"
    assert [(b.block_type, b.text) for b in _blocks(tree.sections)] == [
        (BLOCK_HEADING, "5.1 Process"),
        (BLOCK_PARAGRAPH, "Entry to the site is controlled by the officer on duty."),
    ]


def test_markup_with_no_declared_format_is_skipped_and_said_to_be() -> None:
    tree = parse_layout(_overlay((1, "<p>x</p>"), markup_format=""), _DOC)
    assert [(s.page, s.reason) for s in tree.skipped_pages] == [(1, "no_markup_format_declared")]


def test_an_empty_page_is_not_reported_as_skipped() -> None:
    """``markup == ""`` is the engine answering with nothing — a real answer, not a format this
    parser declined to read."""
    tree = parse_layout(_overlay((1, _PAGE_1), (2, "")), _DOC)
    assert tree.skipped_pages == ()


def test_the_html_format_name_is_the_one_the_writer_stamps() -> None:
    assert MARKUP_FORMAT_HTML == "html"


# --- tokenisation edges -------------------------------------------------------------------------


def test_an_unclosed_table_costs_only_the_table() -> None:
    """The rest of the page used to be dropped by the between-rows rule — not just its text but
    the SECTION, and so every breadcrumb under it."""
    markup = "<table><tr><td>x</td></tr><h2>13. Real Heading</h2><p>Its body.</p>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    assert [(b.block_type, b.text) for b in _blocks(tree.sections)] == [
        (BLOCK_TABLE, "x"),
        (BLOCK_HEADING, "13. Real Heading"),
        (BLOCK_PARAGRAPH, "Its body."),
    ]
    assert _by_crumb(tree.sections, "13. Real Heading").depth == 0


def test_an_implicitly_closed_tables_html_stops_at_the_tag_that_closed_it() -> None:
    markup = "<table><tr><td>x</td></tr><h2>Next</h2>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    table = next(b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE)
    assert table.html == "<table><tr><td>x</td></tr>"


def test_a_nested_table_lands_in_its_outer_cell_and_loses_no_outer_cell() -> None:
    """'A' and 'C' used to be dropped and the inner cell 'B' became the OUTER table's header, so
    a real data row was labelled with another table's value — a fabricated column name, which is
    worse than a bad merge because nothing downstream can tell it from a real one."""
    markup = (
        "<table><tr><td>A</td><td><table><tr><td>B</td></tr></table></td><td>C</td></tr>"
        "<tr><td>D</td><td>E</td><td>F</td></tr></table>"
    )
    tree = parse_layout(_overlay((1, markup)), _DOC)
    table, rows = _merged_table(tree.sections)
    assert table.text == "A | B | C\nD | E | F"
    assert [r.text for r in rows] == ["Table: A=D | B=E | C=F"]


def test_a_nested_tables_cells_are_separated_inside_the_outer_cell() -> None:
    markup = "<table><tr><td>Outer</td><td><table><tr><td>B</td><td>C</td></tr></table></td></tr></table>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    table = next(b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE)
    assert table.text == "Outer | B C"


def test_a_break_inside_a_cell_separates_the_words_either_side() -> None:
    """``<br>`` in a table cell is ordinary VLM output, and ``text`` is what gets embedded even
    though ``html`` keeps the truth."""
    markup = "<table><tr><th>P</th><th>V</th></tr><tr><td>Brightness</td><td>700<br>nits</td></tr></table>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    rows = [b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]
    assert [r.text for r in rows] == ["Table: P=Brightness | V=700 nits"]


def test_a_block_element_inside_a_cell_separates_without_starting_a_block() -> None:
    markup = "<table><tr><th>P</th><th>V</th></tr><tr><td>Brightness</td><td><p>700</p><p>nits</p></td></tr></table>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    kinds = [b.block_type for b in _blocks(tree.sections)]
    assert kinds == [BLOCK_TABLE, BLOCK_TABLE_ROW]
    assert _blocks(tree.sections)[1].text == "Table: P=Brightness | V=700 nits"


def test_an_all_blank_data_row_is_dropped_as_a_spacer() -> None:
    """The complement of the all-blank ``<th>`` row, which is kept because it still declares the
    column count."""
    markup = "<table><tr><th>A</th><th>B</th></tr><tr><td></td><td></td></tr><tr><td>1</td><td>2</td></tr></table>"
    tree = parse_layout(_overlay((1, markup)), _DOC)
    rows = [b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]
    assert [r.text for r in rows] == ["Table: A=1 | B=2"]


# --- merge guard where no <th> exists anywhere ---------------------------------------------------


def test_unrelated_tables_are_not_merged_when_neither_declares_a_th() -> None:
    """The ``<th>`` test is vacuous for an extractor that never emits one — the normal case. Both
    tables' first rows are then guessed headers, and these two merged, the second table's header
    re-labelled under the first table's columns."""
    head = "<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Brightness</td><td>700 nits</td></tr></table>"
    tail = (
        "<table><tr><td>Clause</td><td>Revised amount</td></tr><tr><td>MAAT</td><td>Rs. 19.44 Crores</td></tr></table>"
    )
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert len([b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE]) == 2
    assert tree.merged_tables == ()
    rows = [b.text for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]
    assert rows == [
        "Table: Parameter=Brightness | Value=700 nits",
        "Table: Clause=MAAT | Revised amount=Rs. 19.44 Crores",
    ]


def test_a_continuation_whose_shape_matches_the_heads_data_still_merges_without_any_th() -> None:
    head = "<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Brightness</td><td>700 nits</td></tr></table>"
    tail = "<table><tr><td>Contrast ratio</td><td>4000:1</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    _, rows = _merged_table(tree.sections)
    assert [r.text for r in rows] == [
        "Table: Parameter=Brightness | Value=700 nits",
        "Table: Parameter=Contrast ratio | Value=4000:1",
    ]


def test_every_merge_is_recorded_on_the_tree() -> None:
    """Silent fusion is as unreviewable as silent deletion, which is why
    ``removed_running_text`` exists."""
    tree = parse_layout(_overlay((1, _PAGE_1), (2, _PAGE_2), (3, _PAGE_3)), _DOC)
    assert tree.merged_tables == (TableMerge(pages=(1, 2), columns=2, rows_added=2, header_declared=True),)


def test_a_merge_resting_on_shape_alone_is_recorded_as_undeclared() -> None:
    head = "<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Brightness</td><td>700 nits</td></tr></table>"
    tail = "<table><tr><td>Contrast ratio</td><td>4000:1</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, tail)), _DOC)
    assert [m.header_declared for m in tree.merged_tables] == [False]


def test_a_three_page_merge_is_one_record_not_two() -> None:
    head = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    mid = "<table><tr><td>3</td><td>4</td></tr></table>"
    end = "<table><tr><td>5</td><td>6</td></tr></table>"
    tree = parse_layout(_overlay((1, head), (2, mid), (3, end)), _DOC)
    assert tree.merged_tables == (TableMerge(pages=(1, 2, 3), columns=2, rows_added=2, header_declared=True),)


# --- header/footer: the right denominator and the right action ----------------------------------


def _three_pages_with_furniture() -> tuple[tuple[int, str], ...]:
    return tuple((n, f"<h2>Section {n}</h2><p>Body {n}.</p><p>Acme Ltd</p>") for n in (1, 2, 3))


def test_an_empty_page_does_not_make_up_the_three_pages_the_guard_requires() -> None:
    """``markup == ""`` is a documented, normal shape. Counting one of those let a single empty
    page satisfy the guard on a 2-page document — the one thing the guard exists to prevent."""
    two = _three_pages_with_furniture()[:2]
    assert parse_layout(_overlay(*two), _DOC).removed_running_text == ()
    assert parse_layout(_overlay(*two, (9, "")), _DOC).removed_running_text == ()
    assert parse_layout(_overlay(*_three_pages_with_furniture()), _DOC).removed_running_text == ("acme ltd",)


def test_a_skipped_page_does_not_make_up_the_three_pages_either() -> None:
    two = _three_pages_with_furniture()[:2]
    overlay = _overlay(*two)
    overlay["pages"].append({"page_no": 9, "markup": "# x", "markup_format": "markdown", "degradations": []})
    assert parse_layout(overlay, _DOC).removed_running_text == ()


def test_furniture_is_removed_from_every_region_it_appears_in_even_where_its_slot_shifted() -> None:
    """The slot is the EVIDENCE, not the action. Keying removal on the slot too left the header
    standing on the pages where it had shifted, while ``removed_running_text`` reported it gone —
    an observation field that lied."""
    pages = (
        (1, "<p>Acme Ltd</p><h2>S1</h2><p>Body 1.</p><p>Tail 1.</p>"),
        (2, "<p>Acme Ltd</p><h2>S2</h2><p>Body 2.</p><p>Tail 2.</p>"),
        (3, "<h2>S3</h2><p>Acme Ltd</p><p>Body 3.</p><p>Tail 3.</p>"),
    )
    tree = parse_layout(_overlay(*pages), _DOC)
    assert tree.removed_running_text == ("acme ltd",)
    assert [b.text for b in _blocks(tree.sections) if b.text == "Acme Ltd"] == []


# --- a heading may not forge a path level -------------------------------------------------------


def test_a_heading_containing_the_separator_cannot_forge_a_path_level() -> None:
    """``<h1>A SEP B</h1>`` used to produce the breadcrumb a genuine ``<h2>B</h2>`` under
    ``<h1>A</h1>`` produces, taking ``ordinal=0`` from it — so the real section's id came to
    depend on an unrelated heading elsewhere in the document."""
    forged = f"<h1>A{BREADCRUMB_SEP}B</h1><p>one</p>"
    genuine = "<h1>A</h1><h2>B</h2><p>two</p>"
    tree = parse_layout(_overlay((1, forged), (2, genuine)), _DOC)
    crumbs = [(s.breadcrumb, s.ordinal) for s in _walk(tree.sections)]
    assert crumbs == [("A > B", 0), ("A", 0), (f"A{BREADCRUMB_SEP}B", 0)]


def test_the_genuine_sections_id_does_not_depend_on_a_forged_heading_elsewhere() -> None:
    genuine = "<h1>A</h1><h2>B</h2><p>two</p>"
    alone = parse_layout(_overlay((1, genuine)), _DOC).sections
    with_forgery = parse_layout(_overlay((1, f"<h1>A{BREADCRUMB_SEP}B</h1><p>one</p>"), (2, genuine)), _DOC).sections
    nested = _by_crumb(with_forgery, f"A{BREADCRUMB_SEP}B")
    assert nested.section_id == _by_crumb(alone, f"A{BREADCRUMB_SEP}B").section_id


def test_a_heading_block_keeps_its_text_verbatim_even_when_the_title_is_sanitised() -> None:
    """Only the form that goes into a path and an id is rewritten; the document's own words are
    still in the block."""
    tree = parse_layout(_overlay((1, f"<h1>A{BREADCRUMB_SEP}B</h1><p>one</p>")), _DOC)
    heading = next(b for b in _blocks(tree.sections) if b.block_type == BLOCK_HEADING)
    assert heading.text == f"A{BREADCRUMB_SEP}B"
    assert tree.sections[0].title == "A > B"


# --- the Phase-1 guarantee this parser depends on -----------------------------------------------


def test_a_control_character_in_a_heading_does_not_reach_a_section_id() -> None:
    """``\\x1f`` is ``chunking._ID_PART_SEP`` itself: hashed verbatim it would re-introduce the
    ambiguity the separator exists to prevent. `normalise_breadcrumb` strips it, and this pins
    that the parser routes heading text through it."""
    clean = parse_layout(_overlay((1, "<h1>Annexure A</h1><p>x</p>")), _DOC).sections
    dirty = parse_layout(_overlay((1, "<h1>Annexure\x1fA</h1><p>x</p>")), _DOC).sections
    assert dirty[0].breadcrumb == "Annexure A"
    assert dirty[0].section_id == clean[0].section_id


# --- the born-digital path: typography -> headings ----------------------------------------------
#
# The judgement Phase 2 could not make. Chandra returns no coordinates and no font metrics, so
# the plan's "typography cues from layout (font size / bold)" heading step and its
# "recurs in the same page REGION" header/footer rule both had to be approximated. On this path
# the measurements are real, and these tests pin what is done with them — and what is refused.


def test_heading_depth_follows_the_size_ranking_descending() -> None:
    """The document's own type scale becomes nesting, the way <h1>..<h6> does on the other path:
    biggest size depth 1, next depth 2, and the body below both."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("SECTION 1 SCOPE", size=18.0, bold=True, top=60.0),
                    _line("1.1 Standards and Compliance", size=14.0, bold=True, top=90.0),
                    _line("The controller shall comply with the standards listed.", top=120.0),
                ],
            )
        ),
        _DOC,
    )
    crumbs = [(s.breadcrumb, s.depth, s.heading_level) for s in _walk(tree.sections)]

    assert crumbs == [
        ("SECTION 1 SCOPE", 0, 1),
        (f"SECTION 1 SCOPE{BREADCRUMB_SEP}1.1 Standards and Compliance", 1, 2),
    ]
    assert [b.text for b in _by_crumb(tree.sections, crumbs[1][0]).blocks] == [
        "1.1 Standards and Compliance",
        "The controller shall comply with the standards listed.",
    ]


def test_three_sizes_give_three_depths_and_a_shallower_one_closes_back() -> None:
    """The ranking is over the document's DISTINCT heading sizes, so an 18/14/12.5 scale nests
    three deep and the next 14 pt line closes back to the 18 pt heading — exactly `_nest`'s rule
    for an <h3> following an <h2>."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("PART A", size=18.0, bold=True, top=60.0),
                    _line("A.1 Scope", size=14.0, bold=True, top=90.0),
                    _line("A.1.1 Standards", size=12.5, bold=True, top=120.0),
                    _line("The body of the clause runs here in ordinary type.", top=150.0),
                    _line("A.2 Delivery", size=14.0, bold=True, top=180.0),
                ],
            )
        ),
        _DOC,
    )

    assert [(s.title, s.depth) for s in _walk(tree.sections)] == [
        ("PART A", 0),
        ("A.1 Scope", 1),
        ("A.1.1 Standards", 2),
        ("A.2 Delivery", 1),
    ]


def test_a_body_size_bold_line_is_a_heading_at_the_deepest_rank() -> None:
    """ "Larger, OR body-size-and-bold." Bold at body size is the weakest heading signal the
    document has, so it ranks below every larger size — it nests under them, never over."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("SECTION 1 SCOPE", size=18.0, bold=True, top=60.0),
                    _line("Standards and Compliance", bold=True, top=90.0),
                    _line("The controller shall comply with the standards listed.", top=120.0),
                ],
            )
        ),
        _DOC,
    )

    assert [(s.title, s.depth, s.heading_level) for s in _walk(tree.sections)] == [
        ("SECTION 1 SCOPE", 0, 1),
        ("Standards and Compliance", 1, 2),
    ]


def test_a_document_set_entirely_in_bold_has_no_headings_invented_from_bold() -> None:
    """The degenerate reading of "stands out": with nothing NOT bold, bold distinguishes nothing,
    and every line in the document would otherwise be promoted."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("Clause one of the agreement is set out below.", bold=True, top=60.0),
                    _line("Clause two of the agreement is set out below.", bold=True, top=90.0),
                ],
            )
        ),
        _DOC,
    )

    assert [b.block_type for b in _blocks(tree.sections)] == [BLOCK_PARAGRAPH, BLOCK_PARAGRAPH]
    assert [s.breadcrumb for s in _walk(tree.sections)] == [""]


def test_a_long_line_set_large_is_not_a_heading() -> None:
    """A heading is a title, not a sentence — `sections.py`'s own bound, same number. A banner or
    a one-line paragraph set large is not a section name."""
    long_line = "This document and the information in it are the property of the issuing authority."
    tree = parse_layout(
        _json_overlay((1, [_line(long_line, size=18.0, bold=True, top=60.0), _line("Ordinary body text.", top=90.0)])),
        _DOC,
    )

    assert len(long_line) > 80
    assert [b.block_type for b in _blocks(tree.sections)] == [BLOCK_PARAGRAPH, BLOCK_PARAGRAPH]


def test_a_size_used_only_by_long_lines_does_not_consume_a_depth_rank() -> None:
    """The ranking is over ACCEPTED candidates. A 14 pt pull quote that fails the length bound
    must not push the document's real headings one level deeper than they are."""
    quote = "The parties acknowledge that nothing in this clause limits any statutory right at all."
    body = [_line(f"Clause {n} of the agreement is set out in full here.", top=200.0 + 30 * n) for n in range(5)]
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("PART A", size=18.0, bold=True, top=60.0),
                    _line(quote, size=14.0, top=90.0),
                    _line("A.1 Scope", size=13.0, bold=True, top=140.0),
                    *body,
                ],
            )
        ),
        _DOC,
    )

    assert len(quote) > 80
    assert [(s.title, s.depth, s.heading_level) for s in _walk(tree.sections)] == [
        ("PART A", 0, 1),
        ("A.1 Scope", 1, 2),
    ]


def test_the_body_size_is_the_documents_mode_not_the_page_s() -> None:
    """A title page set entirely in 24 pt would declare 24 pt its own body and every heading on
    it a paragraph. The mode is taken over the whole document, where the body text lives."""
    body = [_line(f"Clause {n} of the agreement is set out in full here.", page=2, top=60.0 + 30 * n) for n in range(6)]
    tree = parse_layout(
        _json_overlay(
            (1, [_line("TENDER DOCUMENT", size=24.0, bold=True, top=200.0)]),
            (2, [_line("1. Scope of Work", page=2, size=18.0, bold=True, top=40.0), *body]),
        ),
        _DOC,
    )

    assert [(s.title, s.depth) for s in _walk(tree.sections)] == [("TENDER DOCUMENT", 0), ("1. Scope of Work", 1)]


# --- the born-digital path: no typographic signal at all ----------------------------------------


def test_a_single_font_document_invents_no_headings_and_falls_back_to_numbering() -> None:
    """A single-font tender PDF carries NO typography signal: one size, nothing bold. Inventing a
    hierarchy from that is the wrong-heading-is-worse-than-no-heading failure, so the numbering
    evidence gets its turn — `sections.detect_sections`, used exactly as it is."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("5.1 Process", top=60.0),
                    _line("Entry to the site is controlled by the officer on duty.", top=90.0),
                    _line("5.1.1 Entry Criteria", top=120.0),
                    _line("A visitor shall produce photographic identification.", top=150.0),
                ],
            )
        ),
        _DOC,
    )

    assert [(s.breadcrumb, s.depth, s.heading_level) for s in _walk(tree.sections)] == [
        ("5.1 Process", 0, 2),
        (f"5.1 Process{BREADCRUMB_SEP}5.1.1 Entry Criteria", 1, 3),
    ]


def test_typography_wins_where_both_signals_speak() -> None:
    """The numbering fallback is reached only when typography says nothing about the whole
    document. Here the sizes speak, so the depths are the SIZE ranks (1, 2) and not the numbering
    depths (2, 3) the same lines would have produced on their own."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("5.1 Process", size=18.0, bold=True, top=60.0),
                    _line("5.1.1 Entry Criteria", size=14.0, bold=True, top=90.0),
                    _line("A visitor shall produce photographic identification.", top=120.0),
                ],
            )
        ),
        _DOC,
    )

    assert [(s.title, s.heading_level) for s in _walk(tree.sections)] == [
        ("5.1 Process", 1),
        ("5.1.1 Entry Criteria", 2),
    ]


def test_a_single_font_document_with_no_numbering_gets_one_implicit_root() -> None:
    """Neither signal speaks, so nothing is claimed: one root holding every block, which is the
    same answer the html path gives a document with no heading markup."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("Entry to the site is controlled by the officer on duty.", top=60.0),
                    _line("A visitor shall produce photographic identification.", top=90.0),
                ],
            )
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == [""]
    assert len(_blocks(tree.sections)) == 2
    assert all(b.block_type == BLOCK_PARAGRAPH for b in _blocks(tree.sections))


def test_a_heading_the_numbering_fallback_finds_mid_line_is_not_split_out() -> None:
    """`sections.py` also catches a heading PDF extraction ran onto the end of the previous
    paragraph. A block here is ONE measured line with ONE bbox, so splitting it would produce two
    blocks whose coordinates are a fiction — that case stays the text layer's."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("The agency distributes only after sign-off. 6.2.3 Social Media", top=60.0),
                    _line("Every post shall be approved before publication.", top=90.0),
                ],
            )
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == [""]
    assert [b.block_type for b in _blocks(tree.sections)] == [BLOCK_PARAGRAPH, BLOCK_PARAGRAPH]


# --- the born-digital path: header/footer by TRUE region ----------------------------------------


def _bands(*, head: str = "", foot: str = "", body: str, pages: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """A document whose every page carries the same body line, optionally under a header and over
    a footer; `{n}` in any of them is replaced by the page number."""
    built: list[tuple[int, list[dict[str, Any]]]] = []
    for page in pages:
        blocks: list[dict[str, Any]] = []
        if head:
            blocks.append(_line(head.format(n=page), page=page, top=_IN_HEAD))
        blocks.append(_line(body.format(n=page), page=page, top=_MID_PAGE))
        if foot:
            blocks.append(_line(foot.format(n=page), page=page, size=9.0, top=_IN_FOOT))
        built.append((page, blocks))
    return _json_overlay(*built)


def test_a_running_header_in_the_top_band_is_stripped_by_geometry() -> None:
    tree = parse_layout(_bands(head="Attachment-1", body="Clause {n} of the agreement."), _DOC)

    assert tree.removed_running_text == ("attachment-1",)
    assert [b.text for b in _blocks(tree.sections)] == [f"Clause {n} of the agreement." for n in (1, 2, 3)]


def test_a_line_repeated_mid_page_is_never_furniture_however_often_it_recurs() -> None:
    """The band is the region, and the middle of the page is not in it. No block count is
    involved, so unlike the html approximation this holds on a 1-block page too."""
    tree = parse_layout(_bands(body="This page is intentionally left blank."), _DOC)

    assert tree.removed_running_text == ()
    assert len(_blocks(tree.sections)) == 3


def test_a_page_marker_differing_only_in_its_digits_is_stripped_from_the_foot_band() -> None:
    """The capability the html path documents as deliberately absent. "Page 1 of 3" does not
    recur as the same TEXT, and the html path refuses digit masking because position in the block
    stream is too weak a region to contain the risk. A strict geometric band is not: a line in
    the bottom 8% of every page whose text differs only in its digits is a page marker.

    The observation reports the MASKED pattern, which is the honest record of what matched."""
    tree = parse_layout(_bands(foot="Page {n} of 3", body="Clause {n} of the agreement."), _DOC)

    assert tree.removed_running_text == ("page # of #",)
    assert [b.text for b in _blocks(tree.sections)] == [f"Clause {n} of the agreement." for n in (1, 2, 3)]


def test_a_body_line_citing_a_standard_survives_although_it_contains_digits() -> None:
    """The first negative case. Mid-page, so it is not in a band at all — digit-insensitive
    matching is scoped to the bands and cannot reach it."""
    tree = parse_layout(
        _bands(foot="Page {n} of 3", body="The controller shall comply with IEC 62368-1."),
        _DOC,
    )

    assert tree.removed_running_text == ("page # of #",)
    assert [b.text for b in _blocks(tree.sections)] == ["The controller shall comply with IEC 62368-1."] * 3


def test_a_clause_that_falls_inside_the_foot_band_on_a_short_page_is_kept() -> None:
    """The second negative case, and the one the band alone does not answer: on a short page a
    genuine clause can sit low enough to be IN the footer band. What keeps it is the unchanged
    >50%-of-pages vote — it appears on one page, the page marker on all three."""
    pages: list[tuple[int, list[dict[str, Any]]]] = []
    for page in (1, 2, 3):
        blocks = [
            _line(f"Clause {page} of the agreement.", page=page, top=_MID_PAGE),
            _line(f"Page {page} of 3", page=page, size=9.0, top=_IN_FOOT),
        ]
        if page == 2:  # the short page: its last clause lands in the band
            blocks.insert(1, _line("Note: see IEC 62368-1 clause 4.2.", page=page, top=_IN_FOOT - 20.0))
        pages.append((page, blocks))
    tree = parse_layout(_json_overlay(*pages), _DOC)

    assert tree.removed_running_text == ("page # of #",)
    assert "Note: see IEC 62368-1 clause 4.2." in [b.text for b in _blocks(tree.sections)]


def test_two_pages_of_page_markers_are_not_stripped_at_all() -> None:
    """`_MIN_PAGES_FOR_REPEAT` is unchanged and applies to both paths: on two pages a
    legitimately repeated line also reads as 100%, so recurrence is not yet evidence. A 2-page
    document's "Page 1 of 2" / "Page 2 of 2" therefore stays, geometry or not."""
    tree = parse_layout(_bands(foot="Page {n} of 2", body="Clause {n}.", pages=(1, 2)), _DOC)

    assert tree.removed_running_text == ()
    assert sorted(b.text for b in _blocks(tree.sections)) == ["Clause 1.", "Clause 2.", "Page 1 of 2", "Page 2 of 2"]


def test_digit_insensitive_matching_does_not_leak_onto_the_html_path() -> None:
    """Scoped to the geometric band, so an html page — which has no bands — keeps its page
    marker, exactly as the module docstring's honest limit says it does."""
    pages = tuple((n, f"<h2>Section {n}</h2><p>Body {n}.</p><p>Page {n} of 3</p>") for n in (1, 2, 3))
    tree = parse_layout(_overlay(*pages), _DOC)

    assert tree.removed_running_text == ()
    kept = [b.text for b in _blocks(tree.sections) if b.text.startswith("Page ")]
    assert kept == [f"Page {n} of 3" for n in (1, 2, 3)]


def test_a_running_header_is_gone_before_it_can_become_a_heading() -> None:
    """Running-text removal runs BEFORE typography is classified, so a bold page header is
    furniture rather than a section per page — the html path's documented limit, which this path
    does not share. It also keeps the header out of the body-size mode."""
    pages: list[tuple[int, list[dict[str, Any]]]] = [
        (
            page,
            [
                _line("ACME LTD CONFIDENTIAL", page=page, bold=True, top=_IN_HEAD),
                _line(f"Clause {page} of the agreement is set out in full here.", page=page, top=_MID_PAGE),
            ],
        )
        for page in (1, 2, 3)
    ]
    tree = parse_layout(_json_overlay(*pages), _DOC)

    assert tree.removed_running_text == ("acme ltd confidential",)
    assert [s.breadcrumb for s in _walk(tree.sections)] == [""]


# --- the born-digital path: tables --------------------------------------------------------------

_GRID: list[list[str | None]] = [["Item", "Standard"], ["Video wall", "BIS 616"], ["Controller", None]]
_BAND_BODY = "Item: Video wall | Standard: BIS 616\nItem: Controller"


def test_a_tables_rows_come_from_pdfplumbers_own_cells_and_carry_the_header_names() -> None:
    """`cells` is `table.extract()`'s own grid, so a cell is a cell rather than something
    re-parsed out of the rendered text. Its first row is the header — the html path's rule ("<th>,
    else the first row"), and pdfplumber declares no <th>, so it is taken as a guess."""
    tree = parse_layout(_json_overlay((1, [_table(_GRID, text=_BAND_BODY, top=100.0)])), _DOC)
    blocks = _blocks(tree.sections)

    assert [b.block_type for b in blocks] == [BLOCK_TABLE, BLOCK_TABLE_ROW, BLOCK_TABLE_ROW]
    assert [b.text for b in blocks[1:]] == [
        "Table: Item=Video wall | Standard=BIS 616",
        "Table: Item=Controller | Standard=",
    ]
    assert all(b.parent_block_id == blocks[0].block_id for b in blocks[1:])


def test_a_born_digital_table_block_has_no_html_and_keeps_the_band_body_verbatim() -> None:
    """There is no engine markup on this path: pdfplumber emits none, and synthesising a <table>
    from the cells would store provenance the extractor never produced — the objection already
    recorded for a merged table's markup. `text` is the band body VERBATIM, the exact string that
    reached the document's text, so the block and the text cannot disagree about the table."""
    tree = parse_layout(_json_overlay((1, [_table(_GRID, text=_BAND_BODY, top=100.0)])), _DOC)
    table = _blocks(tree.sections)[0]

    assert table.html is None
    assert table.text == _BAND_BODY


def test_an_all_blank_cells_row_is_dropped_as_a_spacer() -> None:
    """The same rule the html path applies to an all-blank data row, so the rows kept here line
    up with the rows the serialized band body actually rendered."""
    grid: list[list[str | None]] = [["Item", "Standard"], [None, ""], ["Video wall", "BIS 616"]]
    tree = parse_layout(_json_overlay((1, [_table(grid, text=_BAND_BODY, top=100.0)])), _DOC)
    rows = [b for b in _blocks(tree.sections) if b.block_type == BLOCK_TABLE_ROW]

    assert [b.text for b in rows] == ["Table: Item=Video wall | Standard=BIS 616"]


def test_a_born_digital_table_merges_across_a_page_break() -> None:
    """The merge is the html path's, unchanged, and it reads `header`/`rows` — which this path
    fills from `cells`. The continuation's guessed header is handed back as data, and both bands'
    verbatim bodies are joined rather than re-rendered as one."""
    head = _table([["Item", "Standard"], ["Video wall", "BIS 616"]], text="Item: Video wall | Standard: BIS 616")
    tail = _table(
        [["Controller", "IEC 62368"], ["Cabling", "IS 694"]],
        text="Controller | IEC 62368\nCabling | IS 694",
        page=2,
    )
    tree = parse_layout(_json_overlay((1, [head]), (2, [tail])), _DOC)
    blocks = _blocks(tree.sections)

    assert [b.block_type for b in blocks] == [BLOCK_TABLE, *[BLOCK_TABLE_ROW] * 3]
    assert [b.text for b in blocks[1:]] == [
        "Table: Item=Video wall | Standard=BIS 616",
        "Table: Item=Controller | Standard=IEC 62368",
        "Table: Item=Cabling | Standard=IS 694",
    ]
    assert blocks[0].pages == (1, 2)
    assert blocks[0].text == "Item: Video wall | Standard: BIS 616\nController | IEC 62368\nCabling | IS 694"
    assert [(m.pages, m.columns, m.rows_added) for m in tree.merged_tables] == [((1, 2), 2, 2)]


# --- mixed documents ----------------------------------------------------------------------------


def _mixed(*pages: tuple[int, str, str]) -> dict[str, Any]:
    """An overlay whose pages declare DIFFERENT formats — ``(page_no, markup, markup_format)``.
    A tender with scanned annexures appended is exactly this, and it is one document."""
    return {
        "schema_version": 2,
        "extractor": "chandra,pdfplumber",
        "content_hash": "",
        "pages": [
            {"page_no": no, "markup": markup, "markup_format": fmt, "degradations": []} for no, markup, fmt in pages
        ],
    }


def test_a_mixed_html_and_layout_json_document_parses_into_one_tree_in_page_order() -> None:
    """Both branches feed ONE block stream in page order, so the born-digital pages' sections and
    the scanned pages' sections are one hierarchy — not two trees, and not a tree per format."""
    born = [
        _line("SECTION 1 SCOPE", size=18.0, bold=True, top=60.0),
        _line("The tenderer shall supply the items listed in the schedule.", top=100.0),
    ]
    scanned = "<h2>Annexure A</h2><p>The scanned annexure follows the clauses above.</p>"
    tree = parse_layout(
        _mixed(
            (1, _payload(1, born), "layout-json"),
            (2, scanned, "html"),
        ),
        _DOC,
    )

    assert tree.skipped_pages == ()
    assert [(s.breadcrumb, s.depth, s.page) for s in _walk(tree.sections)] == [
        ("SECTION 1 SCOPE", 0, 1),
        (f"SECTION 1 SCOPE{BREADCRUMB_SEP}Annexure A", 1, 2),
    ]
    assert [(b.page, b.text) for b in _blocks(tree.sections)] == [
        (1, "SECTION 1 SCOPE"),
        (1, "The tenderer shall supply the items listed in the schedule."),
        (2, "Annexure A"),
        (2, "The scanned annexure follows the clauses above."),
    ]


def test_a_running_footer_pools_its_votes_across_a_format_boundary() -> None:
    """The region means geometry on a measured page and reading order on a Chandra one, but the
    recurrence KEY is the same shape either way — so a footer present on pages of both kinds is
    counted once over the whole document instead of failing the >50% test twice."""
    born = [_line("Clause 1.", top=_MID_PAGE), _line("Confidential", size=9.0, top=_IN_FOOT)]
    tree = parse_layout(
        _mixed(
            (1, _payload(1, born), "layout-json"),
            (2, "<p>Clause 2.</p><p>Confidential</p>", "html"),
            (3, "<p>Clause 3.</p><p>Confidential</p>", "html"),
        ),
        _DOC,
    )

    assert tree.removed_running_text == ("confidential",)
    assert [b.text for b in _blocks(tree.sections)] == ["Clause 1.", "Clause 2.", "Clause 3."]


def test_a_table_merges_across_a_format_boundary_and_says_its_text_is_derived() -> None:
    """A born-digital table continuing onto a scanned page. The merge itself is position-and-shape
    based and does not care which branch read either half; what cannot survive is a single
    VERBATIM body, since only one half has one — so the merged table's text is derived from the
    merged rows, which is true of both halves."""
    head = _table([["Item", "Standard"], ["Video wall", "BIS 616"]], text="Item: Video wall | Standard: BIS 616")
    tail = "<table><tr><td>Controller</td><td>IEC 62368</td></tr></table>"
    tree = parse_layout(_mixed((1, _payload(1, [head]), "layout-json"), (2, tail, "html")), _DOC)
    blocks = _blocks(tree.sections)

    assert [(m.pages, m.rows_added) for m in tree.merged_tables] == [((1, 2), 1)]
    assert blocks[0].text == "Item | Standard\nVideo wall | BIS 616\nController | IEC 62368"
    # And `html` goes the same way, by the merge's existing rule that a merged table keeps markup
    # only when BOTH halves had some: here one half never had any, so markup that describes half
    # the table is not labelled as the table's.
    assert blocks[0].html is None


# --- the born-digital payload is asserted, not assumed ------------------------------------------


def test_a_payload_of_an_unknown_version_is_skipped_rather_than_guessed() -> None:
    """The per-page payload carries its own version, separate from the envelope's, for the same
    reason the envelope has one: an unknown shape must be detectable rather than parsed on the
    assumption that it happens to be compatible."""
    markup = json.dumps({"format": "layout-json", "version": 2, "page_no": 1, "page_height": 792.0, "blocks": []})
    tree = parse_layout(_overlay((1, markup), markup_format=MARKUP_FORMAT_LAYOUT_JSON), _DOC, text=_TEXT)

    assert tree.skipped_pages == (
        SkippedPage(page=1, markup_format=MARKUP_FORMAT_LAYOUT_JSON, reason="unreadable_layout_json_version"),
    )
    assert tree.source == "text", "and the document still gets its sections from the text layer"


def test_a_payload_that_is_not_json_costs_the_page_its_blocks_and_says_so() -> None:
    """The payload is read back out of object storage, so its shape is asserted rather than
    assumed. A malformed page costs the page, never the document: the other pages keep theirs."""
    tree = parse_layout(
        _overlay(
            (1, "{not json at all"),
            (2, _payload(2, [_line("Entry to the site is controlled.", page=2, top=60.0)])),
            markup_format=MARKUP_FORMAT_LAYOUT_JSON,
        ),
        _DOC,
    )

    assert [(s.page, s.reason) for s in tree.skipped_pages] == [(1, "unparseable_layout_json")]
    assert [(b.page, b.text) for b in _blocks(tree.sections)] == [(2, "Entry to the site is controlled.")]


def test_a_block_type_outside_the_payloads_vocabulary_is_skipped_not_guessed() -> None:
    """A `block_type` from a writer this build does not know is not a line and not a table, and
    neither assumption is free. The page keeps the blocks that ARE in the vocabulary."""
    unknown = {"block_type": "figure", "text": "Figure 1", "page": 1, "bbox": [72.0, 200.0, 400.0, 300.0]}
    tree = parse_layout(
        _json_overlay((1, [unknown, _line("Entry to the site is controlled.", top=60.0)])),
        _DOC,
    )

    assert [b.text for b in _blocks(tree.sections)] == ["Entry to the site is controlled."]


def test_a_block_with_no_usable_bbox_still_contributes_its_text() -> None:
    """Geometry is what the band rule needs, not what a block needs to exist. A block with a
    malformed bbox loses its region membership — it is simply never a header/footer candidate —
    and keeps everything else."""
    broken = {"block_type": "line", "text": "Entry is controlled.", "page": 1, "bbox": [72.0, None], "font_size": 12.0}
    tree = parse_layout(_json_overlay((1, [broken])), _DOC)

    assert [b.text for b in _blocks(tree.sections)] == ["Entry is controlled."]


# --- a page title repeated on every page ---------------------------------------------------------


def _titled(*pages: str) -> tuple[tuple[int, str], ...]:
    return tuple((i, markup) for i, markup in enumerate(pages, start=1))


def test_a_page_title_repeated_on_consecutive_pages_becomes_one_section() -> None:
    """A tender sets its attachment title in heading type at the top of EVERY page. Each copy is a
    real heading, so the furniture rule correctly refuses it — a heading is never a running-text
    candidate, because stripping by text alone would take the heading and every breadcrumb beneath
    it, which is what makes the plan's acceptance breadcrumb reachable at all.

    The cost was one ROOT PER PAGE: three identical `Attachment-1` sections at ordinals 0/1/2, each
    owning only its own page's subsections. Not a retrieval defect — the breadcrumbs were already
    right and a container heading emits no chunk — but a structure one, landing on the phases that
    NAVIGATE the tree: a per-page `section_id` makes "everything under Attachment-1" three
    disconnected subtrees, and Phase 4 would summarise one logical section three times."""
    tree = parse_layout(
        _overlay(
            *_titled(
                "<h1>Attachment-1</h1><h2>13. Video Wall</h2><p>ALPHA. Comply with IEC 62443-4-2.</p>",
                "<h1>Attachment-1</h1><h2>14. Control Room</h2><p>BETA. Reverberation under 0.6 s.</p>",
                "<h1>Attachment-1</h1><h2>15. Power</h2><p>GAMMA. Dual feed changeover.</p>",
            )
        ),
        _DOC,
    )

    assert len(tree.sections) == 1
    root = tree.sections[0]
    assert root.breadcrumb == "Attachment-1"
    assert [c.breadcrumb.split(BREADCRUMB_SEP)[-1] for c in root.children] == [
        "13. Video Wall",
        "14. Control Room",
        "15. Power",
    ]
    # The title is KEPT, which is the whole point of merging rather than stripping.
    assert all(c.breadcrumb.startswith("Attachment-1") for c in root.children)
    # And the merge is an OBSERVATION, because silent fusion is as unreviewable as silent deletion.
    assert [(m.breadcrumb, m.pages, m.children_merged) for m in tree.merged_sections] == [("Attachment-1", (2, 3), 2)]


def test_a_title_on_only_two_pages_is_left_alone() -> None:
    """The same vote the furniture rule takes. A two-page document's repeated line is not evidence
    of furniture, and `_MIN_PAGES_FOR_REPEAT` is the one guard that must not be weakened."""
    tree = parse_layout(
        _overlay(
            *_titled(
                "<h1>Attachment-1</h1><h2>A</h2><p>Alpha.</p>",
                "<h1>Attachment-1</h1><h2>B</h2><p>Beta.</p>",
            )
        ),
        _DOC,
    )

    assert len(tree.sections) == 2
    assert tree.merged_sections == ()


def test_two_distinct_sections_sharing_a_title_are_not_fused() -> None:
    """The load-bearing guard, and the reason this merge is safe at all. A page title is a
    CONTAINER; two genuinely distinct same-named sections each carry their own prose, and fusing
    those would weld unrelated content into one section — worse than leaving them split, which is
    the principle the cross-page table merge already states."""
    tree = parse_layout(
        _overlay(
            *_titled(
                "<h1>Annexure</h1><p>First annexure text.</p><h2>A</h2><p>Alpha.</p>",
                "<h1>Annexure</h1><p>Second annexure text.</p><h2>B</h2><p>Beta.</p>",
                "<h1>Annexure</h1><p>Third annexure text.</p><h2>C</h2><p>Gamma.</p>",
            )
        ),
        _DOC,
    )

    assert len(tree.sections) == 3, "each owns its own prose, so these are three real sections"
    assert tree.merged_sections == ()


def test_a_run_broken_by_another_title_does_not_merge_across_the_gap() -> None:
    """A page title repeats by PAGE, consecutively. Merging across a gap would join sections the
    document separated with something else."""
    tree = parse_layout(
        _overlay(
            *_titled(
                "<h1>Attachment-1</h1><h2>A</h2><p>Alpha.</p>",
                "<h1>Attachment-2</h1><h2>B</h2><p>Beta.</p>",
                "<h1>Attachment-1</h1><h2>C</h2><p>Gamma.</p>",
            )
        ),
        _DOC,
    )

    assert len(tree.sections) == 3
    assert tree.merged_sections == ()


def test_a_repeated_sub_heading_merges_on_the_same_evidence() -> None:
    """Applied to siblings at every depth, not only roots: the breadcrumb carries the parent path so
    only true siblings can compare equal, and a repeated sub-heading has the same signature."""
    tree = parse_layout(
        _overlay(
            *_titled(
                "<h1>Tender</h1><h2>Schedule</h2><h3>A</h3><p>Alpha.</p>",
                "<h1>Tender</h1><h2>Schedule</h2><h3>B</h3><p>Beta.</p>",
                "<h1>Tender</h1><h2>Schedule</h2><h3>C</h3><p>Gamma.</p>",
            )
        ),
        _DOC,
    )

    assert len(tree.sections) == 1
    schedules = tree.sections[0].children
    assert len(schedules) == 1, "the repeated sub-heading collapsed too"
    assert [c.breadcrumb.split(BREADCRUMB_SEP)[-1] for c in schedules[0].children] == ["A", "B", "C"]


def test_a_single_glyph_set_large_is_not_a_heading() -> None:
    """MEASURED on the AEGCL SOC/NOC Corrigendum-V. Its DRS tables stack the "S.No" column
    header one glyph per line — "S." / "N" / "o" — each set larger than the body, so each was
    promoted to a section. The damage is not cosmetic: the 16,668 characters of the Video Wall
    requirements became the body of a section named "o", and the real heading kept 37.

    A section nobody can name is a section nobody can retrieve."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("13. Video Wall Technical Requirements", size=14.0, bold=True, top=60.0),
                    _line("S.", size=13.0, bold=True, top=90.0),
                    _line("N", size=13.0, bold=True, top=105.0),
                    _line("o", size=13.0, bold=True, top=120.0),
                    _line("The video wall shall comply with BIS and IEC 62368-1.", top=150.0),
                ],
            )
        ),
        _DOC,
    )
    crumbs = [s.breadcrumb for s in _walk(tree.sections)]

    assert crumbs == ["13. Video Wall Technical Requirements"]
    body = _by_crumb(tree.sections, crumbs[0]).blocks
    assert "The video wall shall comply with BIS and IEC 62368-1." in [b.text for b in body]


def test_a_bare_number_or_rule_set_large_is_not_a_heading() -> None:
    """The second bound. "1." and "(a)" are list markers and "——" is a rule; each can be set
    large, and none of them names a section."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("Eligibility Criteria", size=16.0, bold=True, top=60.0),
                    _line("1.", size=14.0, bold=True, top=90.0),
                    _line("(a)", size=14.0, bold=True, top=105.0),
                    _line("———", size=14.0, bold=True, top=120.0),
                    _line("The bidder shall hold a valid registration.", top=150.0),
                ],
            )
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == ["Eligibility Criteria"]


def test_a_short_real_heading_is_still_a_heading() -> None:
    """The bound is three characters and two letters, not "long enough to look important" — a
    genuinely terse section name must survive it."""
    tree = parse_layout(
        _json_overlay(
            (
                1,
                [
                    _line("Fee", size=16.0, bold=True, top=60.0),
                    _line("The processing fee is non-refundable.", top=90.0),
                ],
            )
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == ["Fee"]


# --- repeated candidates: furniture typography cannot distinguish from a heading -----------------


def _page(no: int, *lines: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    return (no, list(lines))


def test_a_repeated_mid_page_candidate_is_demoted() -> None:
    """MEASURED on the real AEGCL corrigendum: its DRS tables set the two halves of an
    "Offered by the bidder." column header in heading type, and both became sections — the
    second then owning the 16,668 characters of Video Wall requirements that followed it.

    A document's section names are distinct; the things that repeat are furniture. Of that
    document's 72 candidates, 60 were distinct and EVERY repeated one was furniture."""
    tree = parse_layout(
        _json_overlay(
            _page(
                1,
                _line("13. Video Wall", size=16.0, bold=True, top=60.0),
                _line("Offered by", size=13.0, bold=True, top=300.0),
                _line("The wall shall comply with BIS and IEC 62368-1.", top=330.0),
            ),
            _page(
                2,
                _line("14. Workstation", size=16.0, bold=True, top=60.0),
                _line("Offered by", size=13.0, bold=True, top=300.0),
                _line("The workstation OEM shall supply an HBOM.", top=330.0),
            ),
        ),
        _DOC,
    )

    crumbs = [s.breadcrumb for s in _walk(tree.sections)]
    assert crumbs == ["13. Video Wall", "14. Workstation"], "the repeated table header is gone"
    body = [b.text for b in _by_crumb(tree.sections, "13. Video Wall").blocks]
    assert "The wall shall comply with BIS and IEC 62368-1." in body, "its content joined the real section"


def test_a_repeated_page_title_is_spared() -> None:
    """The exemption, and the reason it exists: `_merge_repeated_page_titles` folds these and its
    docstring rejects deleting them — a tender that sets "Attachment-1" in heading type at the
    top of every page MEANS it as a heading, and demoting it takes the breadcrumb of everything
    beneath with it."""
    tree = parse_layout(
        _json_overlay(
            _page(
                1,
                _line("Attachment-1", size=18.0, bold=True, top=40.0),
                _line("A. Scope", size=14.0, bold=True, top=90.0),
                _line("The scope of this attachment covers the supply and commissioning works.", top=120.0),
            ),
            _page(
                2,
                _line("Attachment-1", size=18.0, bold=True, top=40.0),
                _line("B. Supply", size=14.0, bold=True, top=90.0),
                _line("The bidder shall supply every item listed in the schedule of rates.", top=120.0),
            ),
        ),
        _DOC,
    )

    assert any(s.breadcrumb.startswith("Attachment-1") for s in _walk(tree.sections))


def test_only_the_leading_candidate_of_a_repeated_block_survives() -> None:
    """A letterhead is several repeated candidates in a row, and only its first line leads the
    page. The rest go — which is what frees the body text they had captured."""

    def letterhead(no: int, tail: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        return _page(
            no,
            _line("Assam Electricity Grid Corporation Limited", size=14.0, bold=True, top=40.0),
            _line("Regd. Office: Bijulee Bhawan", size=13.0, bold=True, top=60.0),
            _line("CIN: U40101AS2003SGC007238", size=13.0, bold=True, top=80.0),
            tail,
        )

    tree = parse_layout(
        _json_overlay(
            letterhead(1, _line("The D.3 declaration is withdrawn.", top=200.0)),
            letterhead(2, _line("The turnover floor is Rs. 19.44 Crores.", top=200.0)),
        ),
        _DOC,
    )

    crumbs = [s.breadcrumb for s in _walk(tree.sections)]
    assert not any("CIN:" in c for c in crumbs), "a letterhead line is not a section name"
    assert not any("Regd. Office" in c for c in crumbs)
    assert any("Assam Electricity" in c for c in crumbs), "the leading line is spared as a page title"


def test_a_declared_html_heading_is_never_demoted() -> None:
    """The rule second-guesses THIS MODULE'S typographic inference, never an extractor's claim.
    `<h1>Annexure</h1>` repeated on three pages, each owning its own prose, is three real
    sections — `_merge_repeated_page_titles` has the test, and this must not undo it."""
    tree = parse_layout(
        _overlay(
            (1, "<h1>Annexure</h1><p>First.</p>"),
            (2, "<h1>Annexure</h1><p>Second.</p>"),
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == ["Annexure", "Annexure"]


def test_a_unique_candidate_is_untouched() -> None:
    tree = parse_layout(
        _json_overlay(
            _page(
                1,
                _line("21. Switches", size=16.0, bold=True, top=60.0),
                _line("22. Firewalls", size=16.0, bold=True, top=200.0),
                _line("Each firewall ships with an HBOM.", top=230.0),
            )
        ),
        _DOC,
    )

    assert [s.breadcrumb for s in _walk(tree.sections)] == ["21. Switches", "22. Firewalls"]
