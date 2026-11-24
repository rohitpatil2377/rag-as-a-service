"""Observed table extraction (FR-B2).

``CAP_TABLES`` says the extractor WOULD capture a table — a property of the extraction path.
``PageRecord.tables_found`` says one actually WAS captured — a property of the content. They
are different facts and are recorded separately; see the argument pinned in
``test_capability_is_not_dropped_when_a_page_has_no_tables``.

The rollup tags exist to answer two corpus questions without re-extracting anything:
how much of the corpus has tables at all, and whether tables sit on consecutive pages
(the observable proxy for a table crossing a page break).
"""

from __future__ import annotations

import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import (
    CAP_TABLES,
    META_CAPABILITIES,
    META_TABLE_PAGES,
    META_TABLE_RUNS,
    META_TABLES_FOUND,
    Extraction,
    extract_pdf_text,
    provenance_tags,
)
from test_ocr_provenance import fail_pdfplumber_on

_ROWS = [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]


def _pages(n: int) -> list[list[str]]:
    return [[f"Narrative paragraph for page {i} of this handbook."] for i in range(1, n + 1)]


def _extract(n: int, table_pages: list[int]) -> Extraction:
    pdf = make_pdf(_pages(n), tables=dict.fromkeys(table_pages, _ROWS))
    return extract_pdf_text(pdf, doc_id="handbook.pdf")


# --- observed, not declared -----------------------------------------------------------------


def test_tables_found_is_observed_per_page() -> None:
    ex = _extract(3, table_pages=[2])
    assert [p.tables_found for p in ex.pages] == [False, True, False]
    # ...and the observation matches what actually reached the indexed text.
    assert "Grade = Officer | Days = 24" in ex.pages[1].text
    assert all("Officer" not in ex.pages[i].text for i in (0, 2))


def test_a_single_row_table_is_not_counted_as_one() -> None:
    """_MIN_TABLE_ROWS drops a one-row 'table' as layout noise, so nothing is appended to the
    text. The observation must agree with the text, not with what pdfplumber merely detected."""
    ex = _extract(1, table_pages=[])
    one_row = extract_pdf_text(make_pdf(_pages(1), tables={1: [["Grade", "Days"]]}), doc_id="d.pdf")
    assert ex.pages[0].tables_found is False
    assert one_row.pages[0].tables_found is False
    assert "|" not in one_row.pages[0].text


def test_a_page_that_fell_back_reports_no_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """pypdf cannot extract tables at all, so a fallback page observes none — and says so via
    its capabilities too, which is the separate fact."""
    fail_pdfplumber_on(monkeypatch, {2})
    pdf = make_pdf(_pages(3), tables={2: _ROWS, 3: _ROWS})
    ex = extract_pdf_text(pdf, doc_id="handbook.pdf")

    assert ex.pages[1].tables_found is False  # the table on page 2 was lost with the fallback
    assert CAP_TABLES not in ex.pages[1].capabilities
    assert ex.pages[2].tables_found is True
    assert CAP_TABLES in ex.pages[2].capabilities


# --- capability is NOT observation ----------------------------------------------------------


def test_capability_is_not_dropped_when_a_page_has_no_tables() -> None:
    """A page with no table keeps CAP_TABLES, deliberately.

    Document capabilities are an INTERSECTION across pages. If an observed-empty page dropped
    CAP_TABLES, then any document without a table on EVERY page would lose it — which is
    essentially the whole corpus — and `extraction_capabilities` would stop meaning "extracted
    by a table-capable path" and start meaning "has a table on every page", silently, under
    the same field name and the same saved queries."""
    ex = _extract(3, table_pages=[2])
    assert all(CAP_TABLES in p.capabilities for p in ex.pages)
    assert CAP_TABLES in ex.capabilities  # the intersection survives
    assert CAP_TABLES in provenance_tags(ex)[META_CAPABILITIES]
    # The two facts disagree for this document, and that is the point of having both.
    assert provenance_tags(ex)[META_TABLES_FOUND] == "true"
    assert not all(p.tables_found for p in ex.pages)


def test_a_document_with_no_tables_still_reports_table_capability() -> None:
    ex = _extract(2, table_pages=[])
    assert CAP_TABLES in ex.capabilities  # it WOULD have captured one
    tags = provenance_tags(ex)
    assert tags[META_TABLES_FOUND] == "false"  # there simply were none
    assert tags[META_TABLE_PAGES] == "0"
    assert tags[META_TABLE_RUNS] == "0"


# --- the rollup that answers the two corpus questions ----------------------------------------


@pytest.mark.parametrize(
    ("table_pages", "expected_pages", "expected_runs"),
    [
        ([], "0", "0"),
        ([2], "1", "1"),  # one isolated table
        ([2, 3], "2", "1"),  # one RUN of two consecutive table pages -> may be one table
        ([1, 3, 5], "3", "3"),  # three isolated tables, no run
        ([1, 2, 3, 6, 7], "5", "2"),  # two runs: pages 1-3 and 6-7
    ],
)
def test_table_rollup_counts_pages_and_consecutive_runs(
    table_pages: list[int], expected_pages: str, expected_runs: str
) -> None:
    """pages/runs is the mean consecutive-table-page run length. 1.0 means tables never sit on
    adjacent pages; above 1.0 means some do, which is the upper bound on tables crossing a
    break (two unrelated tables on adjacent pages look the same from here)."""
    tags = provenance_tags(_extract(8, table_pages=table_pages))
    assert tags[META_TABLE_PAGES] == expected_pages
    assert tags[META_TABLE_RUNS] == expected_runs
    assert tags[META_TABLES_FOUND] == ("true" if table_pages else "false")


def test_rollup_tags_are_always_emitted() -> None:
    """Like every other provenance tag: always present, so a re-extraction cannot leave a stale
    value behind and "absent" unambiguously means "ingested before this existed"."""
    for keys in (provenance_tags(_extract(2, [])), provenance_tags(_extract(2, [1]))):
        assert {META_TABLES_FOUND, META_TABLE_PAGES, META_TABLE_RUNS} <= keys.keys()
