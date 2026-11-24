"""Running headers and footers (P7).

Page furniture repeats on every page, so it lands in every chunk, dilutes every embedding and
gives duplicate-looking evidence to the generator. The existing stripper caught the easy case
— one identical line at an edge — and measurement on the real corpus showed what it did not:

* **A footer that IS a page number never repeats literally.** "1", "2", "3" are three
  different strings, so the counter never reached its threshold. On Acme_HR_POLICY_HANDBOOK
  that left a bare number line on 114 of 147 pages; on Abstract, 42 of 56.
* **Only ONE line was removed per edge.** A two- or three-line masthead lost its first line
  and kept the rest.

The rule stays conservative in the way that matters: something is furniture only if it sits at
a page EDGE and recurs across most pages. Both conditions are needed — a number on its own
line in the middle of a page is data.
"""

from __future__ import annotations

from svc_doc_intelligence.ocr import _normalize_pages


def _texts(pages: list[str]) -> list[str]:
    return [p.text for p in _normalize_pages(pages)]


def test_a_bare_page_number_line_is_stripped() -> None:
    """The case that survived on the real corpus: the footer IS the number, so it never
    repeats literally and exact counting never reached any threshold."""
    pages = [f"Body text of page {i} goes here.\n{i}" for i in range(1, 7)]
    out = _texts(pages)
    assert [p.strip() for p in out] == [f"Body text of page {i} goes here." for i in range(1, 7)]


def test_a_decorated_page_number_is_stripped_too() -> None:
    """ "- 3 -", "3/9", "[4]" — decoration around a number is still not prose."""
    pages = [f"Body of page {i}.\n- {i} -" for i in range(1, 6)]
    assert all("-" not in p.splitlines()[-1] for p in _texts(pages))


def test_a_numbered_heading_is_never_mistaken_for_a_footer() -> None:
    """The refusal that keeps this safe. Any rule that collapses digits INSIDE worded lines
    would merge "Article 5." with "Article 6." and strip the clause number off every page of a
    statute. Losing a footer is a nuisance; losing a clause number is a wrong answer."""
    pages = [f"Article {i}.\nThe provisions of this section apply as follows." for i in range(1, 8)]
    out = _texts(pages)
    assert [p.splitlines()[0] for p in out] == [f"Article {i}." for i in range(1, 8)]


def test_a_multi_line_masthead_is_removed_whole() -> None:
    """One line per edge left the rest of the block behind on every page."""
    head = "GOVERNMENT OF INDIA\nMinistry of Corporate Affairs\nCircular No. 12/2026"
    pages = [f"{head}\nParagraph {i} of the circular." for i in range(1, 6)]
    out = _texts(pages)
    assert all("GOVERNMENT OF INDIA" not in p and "Ministry of Corporate" not in p for p in out)
    assert all("Circular No" not in p for p in out)
    assert [p.strip() for p in out] == [f"Paragraph {i} of the circular." for i in range(1, 6)]


def test_content_that_merely_looks_alike_is_kept() -> None:
    """A line has to recur on MOST pages. Two out of ten is a coincidence, not furniture."""
    pages = ["Shared opening line.\nUnique body A."] * 2 + [f"Opening {i}.\nUnique body {i}." for i in range(3, 11)]
    assert sum("Shared opening line." in p for p in _texts(pages)) == 2


def test_a_number_inside_the_page_is_not_furniture() -> None:
    """Edge AND recurrence, both required — otherwise every figure in a numeric table on a
    consistent layout would be read as a page number."""
    pages = [f"Heading {i}\n42\nClosing remark {i}." for i in range(1, 6)]
    assert all("42" in p for p in _texts(pages))


def test_short_documents_are_left_alone() -> None:
    """Two pages cannot establish what recurs; stripping on that evidence would eat content."""
    pages = ["SAME HEADER\nBody one.", "SAME HEADER\nBody two."]
    assert all("SAME HEADER" in p for p in _texts(pages))


def test_what_was_removed_is_counted() -> None:
    """Stripping is an aggressive edit to stored text. It is recorded as an OBSERVATION rather
    than a degradation: it succeeded, and marking every page of every long document 'degraded'
    would bury the pages that genuinely lost something."""
    pages = [f"MINISTRY OF FINANCE\nContent {i}.\n{i}" for i in range(1, 6)]
    assert [p.boilerplate_lines for p in _normalize_pages(pages)] == [2] * 5
    assert all(p.boilerplate_lines == 0 for p in _normalize_pages(["a", "b", "c"]))


def test_the_document_total_reaches_the_provenance_tags() -> None:
    """Always emitted, like every other provenance tag, so a missing tag means "ingested
    before this existed" rather than "nothing was stripped"."""
    from pdf_fixtures import make_pdf
    from svc_doc_intelligence.ocr import META_BOILERPLATE_LINES, extract_pdf_text, provenance_tags

    pdf = make_pdf([["MINISTRY OF FINANCE", f"Paragraph {i} of the report."] for i in range(1, 6)])
    tags = provenance_tags(extract_pdf_text(pdf, doc_id="report.pdf"))
    assert tags[META_BOILERPLATE_LINES] == "5"  # one masthead line removed from each page
