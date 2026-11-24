"""A born-digital PDF, end to end: real bytes -> the `layout-json` overlay -> a structure tree.

Phase 2c-1 taught the extractor to MEASURE a word-processor PDF (per-line bbox, font size and
weight, pdfplumber's own table cells) and 2c-2 taught :func:`rag_core.layout.parse_layout`
to read those measurements. The unit-level semantics of that reading — the size ranking, the
guards, the geometric bands, the digit-insensitive matching — are pinned against literal
payloads in ``libs/rag-core/tests/test_layout.py``, where the coordinates can be stated exactly.

What these tests add is that the two halves actually meet on a REAL PDF: the numbers pdfplumber
reports for a page laid out like a tender are numbers the parser's rules fire on. They go
through :func:`pdf_fixtures.make_pdf` (``styles`` takes the PDF RESOURCE name — ``"F2"`` is
Helvetica-Bold — and pdfplumber then reports ``fontname='Helvetica-Bold'``), then through the
real ``extract_pdf_text`` and the real ``layout_overlay()``, so nothing here is hand-written
except the document.
"""

from __future__ import annotations

from typing import Any

from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import extract_pdf_text

from rag_core.layout import (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TABLE_ROW,
    BREADCRUMB_SEP,
    LayoutBlock,
    LayoutSection,
    LayoutTree,
    parse_layout,
)

_HEADING = "SECTION 1 SCOPE OF WORK"
_SUBHEADING = "1.1 Standards and Compliance"
_BODY = "The tenderer shall supply the items listed in the schedule below."
_TAIL = "All rates quoted are inclusive of taxes, freight and delivery."
_PAGE_2 = "Delivery shall be completed within sixty days of the award of contract."
_PAGE_3 = "The successful tenderer shall furnish a performance bank guarantee."
_ROWS = [["Item", "Standard"], ["Video wall", "BIS 616"], ["Controller", "IEC 62368"]]


def _tree(pdf: bytes, *, doc_id: str = "tender.pdf") -> LayoutTree:
    """The real pipeline: extract with the layout capture on, hand the parser the overlay AS
    PERSISTED plus the extracted text, exactly as ``pipeline._layout_tree`` does."""
    extraction = extract_pdf_text(pdf, doc_id=doc_id, capture_layout=True)
    return parse_layout(extraction.layout_overlay().as_dict(), doc_id, text=extraction.render().text)


def _walk(sections: tuple[LayoutSection, ...]) -> list[LayoutSection]:
    out: list[LayoutSection] = []
    for section in sections:
        out.append(section)
        out.extend(_walk(section.children))
    return out


def _blocks(tree: LayoutTree) -> list[LayoutBlock]:
    return [block for section in _walk(tree.sections) for block in section.blocks]


def _tender_pdf(**extra: Any) -> bytes:
    """A three-page tender: an 18 pt bold heading over a 14 pt bold subheading over 12 pt body, a
    ruled table mid-page, and a page marker in every page's bottom margin."""
    return make_pdf(
        [[_HEADING, _SUBHEADING, _BODY, _TAIL], [_PAGE_2], [_PAGE_3]],
        tables={1: _ROWS},
        table_after_line={1: 3},
        styles={1: {0: ("F2", 18.0), 1: ("F2", 14.0)}},
        footers={n: f"Page {n} of 3" for n in (1, 2, 3)},
        **extra,
    )


# --- typography becomes a hierarchy ---------------------------------------------------------


def test_a_tender_shaped_pdf_gives_the_hierarchy_its_typography_implies() -> None:
    """The plan's heading-detection step 3, "typography cues from layout (font size / bold)",
    which the html path had to document as permanently unavailable. 18 pt ranks above 14 pt
    ranks above the 12 pt body, so the document's own type scale is the nesting — and the body,
    the table and the following pages' prose all land under the deepest heading that opened."""
    tree = _tree(_tender_pdf())

    assert tree.source == "markup"
    assert tree.skipped_pages == ()
    assert [(s.breadcrumb, s.depth, s.heading_level, s.page) for s in _walk(tree.sections)] == [
        (_HEADING, 0, 1, 1),
        (f"{_HEADING}{BREADCRUMB_SEP}{_SUBHEADING}", 1, 2, 1),
    ]


def test_the_blocks_under_those_headings_are_the_pages_own_in_page_order() -> None:
    tree = _tree(_tender_pdf())
    deep = _walk(tree.sections)[1]

    assert [(b.block_type, b.page) for b in deep.blocks] == [
        (BLOCK_HEADING, 1),
        (BLOCK_PARAGRAPH, 1),
        (BLOCK_TABLE, 1),
        (BLOCK_TABLE_ROW, 1),
        (BLOCK_TABLE_ROW, 1),
        (BLOCK_PARAGRAPH, 1),  # the paragraph below the table
        (BLOCK_PARAGRAPH, 2),
        (BLOCK_PARAGRAPH, 3),
    ]
    assert [b.text for b in deep.blocks if b.block_type == BLOCK_PARAGRAPH] == [_BODY, _TAIL, _PAGE_2, _PAGE_3]


def test_heading_depth_follows_the_size_ranking_on_a_real_pdf() -> None:
    """Three sizes, three depths, and the second 14 pt heading closes back to the 18 pt one —
    the rank is over the document's distinct heading sizes, not over the order they appear in."""
    pdf = make_pdf(
        [["PART A", "A.1 Scope", "A.1.1 Standards", _BODY, "A.2 Delivery", _TAIL]],
        styles={1: {0: ("F2", 18.0), 1: ("F2", 14.0), 2: ("F2", 12.5), 4: ("F2", 14.0)}},
    )

    assert [(s.title, s.depth) for s in _walk(_tree(pdf).sections)] == [
        ("PART A", 0),
        ("A.1 Scope", 1),
        ("A.1.1 Standards", 2),
        ("A.2 Delivery", 1),
    ]


def test_a_single_font_tender_invents_no_headings_and_falls_back_to_numbering() -> None:
    """One size, nothing bold: a single-font tender PDF carries no typographic signal at all, and
    a hierarchy invented from none would be the wrong-heading-is-worse-than-no-heading failure.
    So the numbering evidence gets its turn — ``sections.detect_sections``, tuned on real
    corpora, which is why the depths here are the NUMBERING depths (2 and 3) rather than ranks."""
    pdf = make_pdf([["5.1 Process", _BODY, "5.1.1 Entry Criteria", _TAIL]])
    tree = _tree(pdf)

    assert [(s.breadcrumb, s.depth, s.heading_level) for s in _walk(tree.sections)] == [
        ("5.1 Process", 0, 2),
        (f"5.1 Process{BREADCRUMB_SEP}5.1.1 Entry Criteria", 1, 3),
    ]


def test_a_single_font_document_with_no_numbering_claims_no_structure() -> None:
    """Neither signal speaks, so nothing is claimed: one implicit root holding every block."""
    tree = _tree(make_pdf([[_BODY, _TAIL], [_PAGE_2]]))

    assert [s.breadcrumb for s in _walk(tree.sections)] == [""]
    assert [b.text for b in _blocks(tree)] == [_BODY, _TAIL, _PAGE_2]


# --- the geometric header/footer region -----------------------------------------------------


def test_a_page_marker_is_stripped_although_every_page_spells_it_differently() -> None:
    """The plan's header/footer rule as written — "recurs in the same page REGION" — now that
    there is a region. "Page 1 of 3" never recurs as the same TEXT, so the html path keeps it by
    documented design; inside the bottom 8% of the page, a line that differs only in its digits
    is a page marker, and the observation reports the MASKED pattern it matched."""
    tree = _tree(_tender_pdf())

    assert tree.removed_running_text == ("page # of #",)
    assert [b.text for b in _blocks(tree) if "Page" in b.text] == []


def test_the_overlay_is_pre_normalisation_so_the_tree_must_strip_the_marker_itself() -> None:
    """BLOCK TEXT AND PAGE TEXT ARE NOT INTERCHANGEABLE, and this is where it bites.
    ``_normalize_pages`` already removed the page marker from every page's ``text`` (it is a
    "Page N of M" shaped edge line, which its own furniture key does generalise), but the overlay
    is captured BEFORE that — so the marker is still a block, and the structure tree has to
    decide about it on its own evidence. Convenient rather than awkward: the parser gets to make
    that decision with geometry, which is better evidence than the text layer has."""
    extraction = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True)

    assert all(page.boilerplate_lines == 1 for page in extraction.pages), "gone from the text"
    assert all("Page" not in page.text for page in extraction.pages)
    assert all(page.markup is not None and "Page" in page.markup for page in extraction.pages), "still a block"


def test_a_worded_footer_with_a_changing_number_is_caught_here_and_not_in_the_text_layer() -> None:
    """``ocr._furniture_key`` names this as its own deliberate gap: "a footer of other words plus
    a number ('HR Handbook 4.0 | 6')" is left alone because recognising it "needs evidence this
    rule does not have — that the words are furniture rather than content".

    A strict geometric band IS that evidence. The line sits in the bottom 8% of every page, so
    here it can be matched digit-insensitively without the risk that stopped the text layer."""
    pdf = make_pdf(
        [[_BODY], [_TAIL], [_PAGE_2]],
        footers={n: f"Tender Document 4.0 | {n}" for n in (1, 2, 3)},
    )
    extraction = extract_pdf_text(pdf, doc_id="tender.pdf", capture_layout=True)
    tree = parse_layout(extraction.layout_overlay().as_dict(), "tender.pdf", text=extraction.render().text)

    assert all("Tender Document 4.0" in page.text for page in extraction.pages), "the text layer kept it"
    assert tree.removed_running_text == ("tender document #.# | #",)
    assert [b.text for b in _blocks(tree)] == [_BODY, _TAIL, _PAGE_2]


def test_a_body_line_citing_a_standard_is_kept_although_it_contains_digits() -> None:
    """The first negative case: digit-insensitive matching is scoped to the bands, and the middle
    of the page is not one. The clause recurs on every page and still survives."""
    clause = "The controller shall comply with IEC 62368-1 and IEC 61000-4-2."
    tree = _tree(make_pdf([[clause], [clause], [clause]], footers={n: f"Page {n} of 3" for n in (1, 2, 3)}))

    assert tree.removed_running_text == ("page # of #",)
    assert [b.text for b in _blocks(tree)] == [clause] * 3


def test_a_clause_that_falls_inside_the_foot_band_on_a_short_page_is_kept() -> None:
    """The second negative case, and the one the band alone cannot answer: a genuine clause CAN
    sit in the footer band on a short page. What keeps it is the unchanged >50%-of-pages vote —
    it is in the band on one page, the page marker on all three.

    Drawing it with ``footers`` is the point: the fixture puts it exactly where furniture goes,
    so the band cannot be what distinguishes them."""
    clause = "Note: see IEC 62368-1 clause 4.2."
    tree = _tree(
        make_pdf(
            [[_BODY], [_TAIL], [_PAGE_2]],
            footers={1: f"Page 1 of 3 {clause}", 2: "Page 2 of 3", 3: "Page 3 of 3"},
        )
    )

    assert tree.removed_running_text == ("page # of #",)
    assert [b.text for b in _blocks(tree)] == [_BODY, f"Page 1 of 3 {clause}", _TAIL, _PAGE_2]


# --- tables ---------------------------------------------------------------------------------


def test_a_tables_rows_come_from_pdfplumbers_cells_and_carry_their_header_names() -> None:
    """``cells`` is ``table.extract()``'s own grid, carried through 2c-1 unchanged, so a cell is a
    cell rather than something re-parsed out of the rendered band body. Its first row is the
    header — the html path's rule ("<th>, else the first row") — and each row is linearised with
    the breadcrumb and those names so it stands alone when retrieved."""
    tree = _tree(_tender_pdf())
    rows = [b for b in _blocks(tree) if b.block_type == BLOCK_TABLE_ROW]
    table = next(b for b in _blocks(tree) if b.block_type == BLOCK_TABLE)
    crumb = f"{_HEADING}{BREADCRUMB_SEP}{_SUBHEADING}{BREADCRUMB_SEP}"

    assert [b.text for b in rows] == [
        f"{crumb}Table: Item=Video wall | Standard=BIS 616",
        f"{crumb}Table: Item=Controller | Standard=IEC 62368",
    ]
    assert all(b.parent_block_id == table.block_id for b in rows)


def test_a_born_digital_table_has_no_html_and_keeps_the_band_body_verbatim() -> None:
    """``html`` is None on this path and that is deliberate: pdfplumber emits no markup, so
    synthesising a ``<table>`` from the cells would store provenance the extractor never
    produced. ``text`` is the band body VERBATIM — the exact string that reached the page's text,
    so the block and the text cannot disagree about a table they came from together."""
    extraction = extract_pdf_text(_tender_pdf(), doc_id="tender.pdf", capture_layout=True)
    tree = parse_layout(extraction.layout_overlay().as_dict(), "tender.pdf", text=extraction.render().text)
    table = next(b for b in _blocks(tree) if b.block_type == BLOCK_TABLE)

    assert table.html is None
    assert table.text in extraction.pages[0].text


# --- the repeated page title, inferred from TYPOGRAPHY --------------------------------------------


def test_a_page_title_in_heading_type_on_every_page_yields_one_section() -> None:
    """The case that motivated the merge, on the path where it actually bites.

    This path has no declared `<h1>` — "Attachment-1" is a heading because it is 18 pt against a
    10 pt body. It also cannot be removed as furniture: at 18 pt the block does not fit ENTIRELY
    inside the 8% head band, so it is never a running-text candidate, and even if it were, a
    heading is exempt by design because stripping it by text would take every breadcrumb beneath
    it. Measured before the merge: three identical roots at ordinals 0/1/2, one per page.
    """
    bodies = [
        ("13. Video Wall Technical Requirements", "Standards and Compliance", "ALPHA. Comply with IEC 62443-4-2."),
        ("14. Control Room Requirements", "Acoustic Treatment", "BETA. Reverberation under 0.6 seconds."),
        ("15. Power Requirements", "Redundancy", "GAMMA. Dual feed with automatic changeover."),
    ]
    pages, styles = [], {}
    for i, (h2, h3, body) in enumerate(bodies, start=1):
        pages.append(["Attachment-1", h2, h3, body])
        styles[i] = {0: ("F2", 18.0), 1: ("F2", 14.0), 2: ("F2", 12.0), 3: ("F1", 10.0)}

    tree = _tree(make_pdf(pages, styles=styles), doc_id="attachment.pdf")

    assert tree.source == "markup"
    assert len(tree.sections) == 1, "one logical Attachment-1, not one per page"
    root = tree.sections[0]
    assert root.breadcrumb == "Attachment-1"
    assert len(root.children) == 3, "every page's real section hangs off the single title"
    # The plan's acceptance breadcrumb, from typography alone, with the title intact.
    deepest = [s.breadcrumb for s in _walk(tree.sections) if not s.children]
    expected = BREADCRUMB_SEP.join(
        ("Attachment-1", "13. Video Wall Technical Requirements", "Standards and Compliance")
    )
    assert expected in deepest
    assert [m.breadcrumb for m in tree.merged_sections] == ["Attachment-1"]
