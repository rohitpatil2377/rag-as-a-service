"""How much of a page was claimed by table detection (P45).

Once a detected table's characters are REMOVED from the narrative and replaced by rendered
rows, a false positive stops being noise beside the prose and starts restructuring it. There
is no way to tell from the output text that it happened.

The extractor already knows: it decides ownership character by character to splice the table
in. So it can say what share of the page it claimed, without a classifier and without keeping
any coordinates. A page that is 4% table is a table on a page; a page reported as 95% table
that reads like prose is a detection failure, and now it is findable with a query.
"""

from __future__ import annotations

from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import META_TABLE_CHAR_SHARE, extract_pdf_text, provenance_tags

_ROWS = [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]
_PROSE = [
    "The annual leave entitlement is set out in the table below.",
    "Applications must be submitted through the portal in advance.",
]


def test_a_page_with_no_table_claims_nothing() -> None:
    ex = extract_pdf_text(make_pdf([_PROSE]), doc_id="prose.pdf")
    assert ex.pages[0].table_chars == 0
    assert ex.pages[0].table_char_share == 0.0


def test_a_page_with_a_table_reports_the_share_it_claimed() -> None:
    ex = extract_pdf_text(make_pdf([_PROSE], tables={1: _ROWS}), doc_id="mixed.pdf")
    page = ex.pages[0]
    assert page.tables_found is True
    # A real table on a real page of prose: some of the page, not all of it.
    assert 0.0 < page.table_char_share < 1.0
    assert page.table_chars < page.source_chars


def test_the_document_share_is_derived_from_the_counts_not_averaged() -> None:
    """Averaging per-page ratios would weight a sparse page the same as a dense one. The
    counts aggregate exactly, which is why both are kept rather than the ratio alone."""
    ex = extract_pdf_text(make_pdf([_PROSE, _PROSE], tables={2: _ROWS}), doc_id="two.pdf")
    expected = sum(p.table_chars for p in ex.pages) / sum(p.source_chars for p in ex.pages)
    assert ex.table_char_share == expected
    assert 0.0 < ex.table_char_share < 1.0


def test_the_share_is_always_stamped_so_absence_means_pre_dates_it() -> None:
    """Every provenance tag is emitted on every document, so a missing tag means "ingested
    before this existed" rather than "nothing was claimed"."""
    tags = provenance_tags(extract_pdf_text(make_pdf([_PROSE]), doc_id="prose.pdf"))
    assert tags[META_TABLE_CHAR_SHARE] == "0.0"
