"""One representation of a table, where the table actually was (FR-B2).

`extract_text()` already returns every table cell in reading order, and the rendered
pipe-rows used to be APPENDED on top of that — so each table was in the indexed text twice,
and a mid-page table was moved below narrative that originally followed it.

The structured form is the one that survives; the table's own characters are removed from the
narrative text and the rows are spliced in at the table's real position. See the module
docstring of svc_doc_intelligence.ocr for why structure wins over the inline form.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import extract_pdf_text

_ABOVE = "Paragraph above the table describing the entitlement rules."
_BELOW = "Paragraph below the table covering the notice period."
_ROWS = [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]


def _page_text(lines: list[str], rows: list[list[str]] | None, after: int | None) -> str:
    pdf = make_pdf(
        [lines],
        tables={1: rows} if rows else None,
        table_after_line={1: after} if after is not None else None,
    )
    return extract_pdf_text(pdf, doc_id="doc.pdf").pages[0].text


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9]+", text))


# --- one representation, in the right place --------------------------------------------------


def test_a_mid_page_table_appears_once_and_between_the_paragraphs() -> None:
    text = _page_text([_ABOVE, _BELOW], _ROWS, after=1)

    # Every DATA cell exactly once — not once inline and again as a serialized row. Header
    # words are the deliberate exception: they are on every row now, which is the whole point
    # of a self-describing row (see test_tables.py).
    for cell in ("Officer", "24", "Staff", "18"):
        assert _tokens(text)[cell] == 1, f"{cell!r} appears {_tokens(text)[cell]} times in:\n{text}"

    # ...and positioned where it was, not appended after everything.
    assert text.index(_ABOVE) < text.index("Grade = Officer") < text.index(_BELOW)
    assert "Grade = Officer | Days = 24" in text  # the structured form is the one that survived


def test_repositioning_the_table_loses_no_content() -> None:
    """Removing the inline copy must remove ONLY the duplicate: every token of the original
    page still has to be present."""
    lines = [_ABOVE, _BELOW]
    with_table = _tokens(_page_text(lines, _ROWS, after=1))
    for cell in ("Officer", "24", "Staff", "18"):
        assert with_table[cell] == 1
    for header in ("Grade", "Days"):
        assert with_table[header] == len(_ROWS) - 1  # once per body row, by design
    for word in re.findall(r"[A-Za-z0-9]+", _ABOVE + " " + _BELOW):
        assert with_table[word] >= 1


def test_a_multi_line_cell_stays_on_its_row() -> None:
    """The inline form breaks a wrapped cell across lines, orphaning its tail from the row
    label. The structured form keeps the row intact — the reason structure wins."""
    rows = [["Grade", "Entitlement"], ["Officer", "Twenty four working days"], ["Staff", "Eighteen days"]]
    text = _page_text([_ABOVE, _BELOW], rows, after=1)
    # All-text table -> no header identified -> positional, but the row is still one line.
    assert "Officer | Twenty four working days" in text


# --- guards -----------------------------------------------------------------------------------


def test_a_page_with_no_table_is_untouched() -> None:
    text = _page_text([_ABOVE, _BELOW], None, None)
    assert text.splitlines() == [_ABOVE, _BELOW]


def test_a_candidate_that_does_not_render_keeps_its_text_in_the_narrative() -> None:
    """_MIN_TABLE_ROWS drops a one-row candidate. Its text must then stay in the narrative —
    excluding a region whose table is never emitted would silently DELETE that content."""
    text = _page_text([_ABOVE, _BELOW], [["Grade", "Days"]], after=1)
    assert "Grade" in text and "Days" in text
    assert "|" not in text and ":" not in text  # nothing was rendered as a table


def test_tables_found_still_matches_what_is_in_the_text() -> None:
    """The flag and the content are decided by the same value, so they cannot disagree."""
    pdf = make_pdf([[_ABOVE, _BELOW], [_ABOVE, _BELOW]], tables={2: _ROWS}, table_after_line={2: 1})
    pages = extract_pdf_text(pdf, doc_id="doc.pdf").pages
    for page in pages:
        assert page.tables_found == ("|" in page.text)
    assert [p.tables_found for p in pages] == [False, True]


def test_a_table_split_across_a_chunk_boundary_stays_interpretable() -> None:
    """The failure this format exists to prevent: a chunk that begins mid-table must still say
    what each value measures. Long enough that FixedChunker cannot hold it in one window."""
    from datetime import UTC, datetime

    from rag_core.chunking import FixedChunker
    from rag_schemas import DocumentEnvelope
    from rag_schemas.common import SourceKind

    rows = [["Grade", "Days"]] + [[f"Officer Level {i}", str(20 + i)] for i in range(1, 60)]
    text = _page_text([_ABOVE, _BELOW], rows, after=1)
    doc = DocumentEnvelope(
        tenant_id="acme",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime.now(UTC),
    )
    chunks = FixedChunker().chunk(doc, text)
    carrying = [c for c in chunks if "Officer Level" in c.text]
    assert len(carrying) > 1, "fixture is not long enough to actually split"

    # EVERY chunk holding table rows must carry the column meaning, not just the first.
    #
    # A fixed-size window cuts at a character offset, so it can slice THROUGH a row at either
    # end of the chunk — the first line may be missing its label and the last its final field.
    # No serialization prevents that; only row-aligned chunking would, and that is separate
    # work. What the format guarantees is that every row the window contained WHOLE is
    # self-describing, wherever the chunk began. Assert exactly that, and no more: the previous
    # version happened to hold for the first line too, purely because the old separator's
    # length made the windows land on row boundaries.
    for chunk in carrying:
        row_lines = [ln for ln in chunk.text.splitlines() if "Officer Level" in ln]
        assert row_lines
        whole = [ln for ln in row_lines if ln.startswith("Grade = ") and "Days = " in ln]
        assert whole, f"no complete row survived in:\n{chunk.text[:200]}"
        assert len(whole) >= len(row_lines) - 2, chunk.text[:200]  # at most one cut at each end


@pytest.mark.parametrize("after", [0, 1, 2])
def test_the_table_lands_at_its_own_position_wherever_that_is(after: int) -> None:
    """Top of page, mid-page, bottom of page — the rows go where the table was drawn."""
    lines = [_ABOVE, _BELOW]
    text = _page_text(lines, _ROWS, after=after)
    positions = [text.index(line) for line in lines[:after]]
    table_at = text.index("Grade = Officer")
    assert all(p < table_at for p in positions)
    assert all(text.index(line) > table_at for line in lines[after:])
