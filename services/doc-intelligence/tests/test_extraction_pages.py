"""Per-page extraction records: joined text and page map derived in one pass.

Two things are pinned here.

**No fused tokens at a page break.** Joining pages with "" welds the last word of one page
to the first word of the next ("flight hoursSection 4"), producing tokens that match nothing
in BM25 and pollute the embedding, and leaving the clause splitter no whitespace to split on.

**Page attribution.** ``test_page_attribution`` is the guard on the "grounded, verifiable
citations" claim: a known string on a known page must be reported as being on that page,
through the SAME offset->page function the chunker uses. It is fixture-driven (add a row to
``_ATTRIBUTION_CASES``) so it can be extended as extraction gains per-page fallback, per-page
OCR routing and PII offset handling, and must keep passing through every one of those.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import pytest
from pdf_fixtures import make_pdf
from svc_doc_intelligence.ocr import PAGE_SEPARATOR, RenderedText, extract_pdf_text
from test_ocr_provenance import fail_pdfplumber_on

from rag_core.chunking import _CLAUSE_RE, ClauseChunker, FixedChunker, _page_for_offset
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id="handbook.pdf",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime.now(UTC),
    )


# --- the fusion bug -----------------------------------------------------------------------


def test_page_break_mid_sentence_does_not_fuse_tokens() -> None:
    """A sentence continuing across a page break must not weld into "hoursSection"."""
    pdf = make_pdf(
        [
            ["Total accrued flight hours"],  # page 1 ends mid-sentence, no trailing punctuation
            ["Section 4 covers maintenance and grounding rules."],
        ]
    )
    text = extract_pdf_text(pdf, doc_id="handbook.pdf").render().text

    assert "hoursSection" not in text  # the fused token that matches nothing in BM25
    assert "hours" in text and "Section 4" in text
    junction = text.index("hours") + len("hours")
    assert text[junction : junction + len(PAGE_SEPARATOR)] == PAGE_SEPARATOR


def test_page_separator_creates_a_clause_boundary() -> None:
    """The separator is chosen so page breaks become natural chunk boundaries. This pins the
    coupling to _CLAUSE_RE: a single newline would NOT split (verified in the same assert)."""
    assert len(_CLAUSE_RE.split(f"page one text{PAGE_SEPARATOR}page two text")) == 2
    assert len(_CLAUSE_RE.split("page one text\npage two text")) == 1  # why one \n is not enough


# --- page attribution (the citations guard) -----------------------------------------------


@dataclass(frozen=True, slots=True)
class AttributionCase:
    name: str
    pages: list[list[str]]  # pages[i] = the lines of page i+1
    needle: str  # a string known to appear on...
    expected_page: int  # ...this 1-based page
    # 1-based pages on which pdfplumber is made to fail, so the record list is produced by a
    # MIX of extractors. Attribution must not care which extractor produced a given page.
    fallback_pages: frozenset[int] = frozenset()
    # 1-based page -> a table drawn after that many of the page's lines. A mid-page table is
    # LIFTED OUT of the narrative and spliced back in, so the page's text is rebuilt in
    # pieces — offsets must still land on the right page afterwards.
    tables: dict[int, list[list[str]]] = field(default_factory=dict)
    table_after_line: dict[int, int] = field(default_factory=dict)


def _extract(case: AttributionCase, monkeypatch: pytest.MonkeyPatch) -> RenderedText:
    """Run a case, forcing the per-page fallback where the case asks for it."""
    if case.fallback_pages:
        fail_pdfplumber_on(monkeypatch, set(case.fallback_pages))
    pdf = make_pdf(case.pages, tables=case.tables or None, table_after_line=case.table_after_line or None)
    return extract_pdf_text(pdf, doc_id="doc.pdf").render()


_ATTRIBUTION_CASES = [
    AttributionCase(
        name="first_page",
        pages=[["Leave entitlement is 24 days."], ["Notice period is 60 days."]],
        needle="Leave entitlement",
        expected_page=1,
    ),
    AttributionCase(
        name="second_page",
        pages=[["Leave entitlement is 24 days."], ["Notice period is 60 days."]],
        needle="Notice period",
        expected_page=2,
    ),
    AttributionCase(
        name="last_of_many",
        pages=[["Alpha clause one."], ["Bravo clause two."], ["Charlie clause three."]],
        needle="Charlie",
        expected_page=3,
    ),
    AttributionCase(  # the offset arithmetic must survive several separators before the hit
        name="after_several_separators",
        pages=[["Alpha one."], ["Bravo two."], ["Charlie three."], ["Delta four."], ["Echo five."]],
        needle="Echo five",
        expected_page=5,
    ),
    AttributionCase(  # a page whose text layer is empty must still hold its page number
        name="page_number_survives_an_empty_page",
        pages=[["Alpha clause one."], [], ["Charlie clause three."]],
        needle="Charlie",
        expected_page=3,
    ),
    AttributionCase(  # pages 2 and 4 come from pypdf, the rest from pdfplumber
        name="mixed_extractors_before_the_needle",
        pages=[["Alpha one."], ["Bravo two."], ["Charlie three."], ["Delta four."], ["Echo five."]],
        needle="Echo five",
        expected_page=5,
        fallback_pages=frozenset({2, 4}),
    ),
    AttributionCase(  # the needle itself sits on a page produced by the fallback extractor
        name="needle_on_a_fallback_page",
        pages=[["Alpha one."], ["Bravo two."], ["Charlie three."]],
        needle="Charlie",
        expected_page=3,
        fallback_pages=frozenset({3}),
    ),
    AttributionCase(  # the needle follows a table that was lifted out and spliced back in
        name="needle_after_a_mid_page_table",
        pages=[["Alpha one."], ["Bravo intro line.", "Bravo tail after the table."], ["Charlie three."]],
        needle="Bravo tail after the table",
        expected_page=2,
        tables={2: [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]},
        table_after_line={2: 1},
    ),
    AttributionCase(  # ...and a page AFTER the rebuilt one still resolves correctly
        name="page_after_a_mid_page_table",
        pages=[["Alpha one."], ["Bravo intro line.", "Bravo tail after the table."], ["Charlie three."]],
        needle="Charlie",
        expected_page=3,
        tables={2: [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]},
        table_after_line={2: 1},
    ),
    AttributionCase(  # a citation landing INSIDE a serialized table body, not in narrative
        name="needle_inside_a_table_body",
        pages=[["Alpha one."], ["Bravo intro line.", "Bravo tail after the table."], ["Charlie three."]],
        needle="Grade = Zulu Officer | Days = 24",
        expected_page=2,
        tables={2: [["Grade", "Days"], ["Zulu Officer", "24"], ["Staff", "18"]]},
        table_after_line={2: 1},
    ),
]


@pytest.mark.parametrize("case", _ATTRIBUTION_CASES, ids=lambda c: c.name)
def test_page_attribution(case: AttributionCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ground truth in -> same page number out, via the chunker's own offset->page function."""
    rendered = _extract(case, monkeypatch)
    offset = rendered.text.index(case.needle)
    assert _page_for_offset(offset, rendered.page_map) == case.expected_page


def _padded(pages: list[list[str]], *, lead_chars: int) -> tuple[int, list[list[str]]]:
    """Prepend filler to each page so the needle sits more than one chunk-window into its page.

    Without this the assertion below would be false for a reason that is NOT a bug: a fixed
    window that STARTS on page N legitimately reports page N even where it runs on into page
    N+1, so a needle within one window of the page top can land in a chunk that began on the
    previous page. Filler lines are numbered per page because ``_strip_repeated_edges`` deletes
    a first line that repeats across most pages — identical filler would be eaten as a header."""
    line = "Filler sentence {}.{} carrying bulk so the page exceeds one chunk window."
    per_page = lead_chars // len(line) + 2
    return per_page, [
        [line.format(page_no, j) for j in range(per_page)] + lines for page_no, lines in enumerate(pages, start=1)
    ]


@pytest.mark.parametrize("case", _ATTRIBUTION_CASES, ids=lambda c: c.name)
def test_page_attribution_survives_the_default_chunker(case: AttributionCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same guarantee at the level that actually ships a citation: the chunk carrying the
    needle reports the right page. FIXED with its shipped defaults is the ingest default
    (see pipeline.ingest_document)."""
    chunker = FixedChunker()
    # The filler goes BEFORE the page's own lines, so a table anchored after line N must move
    # with it — otherwise the padding would push the table to the top of the page and undo
    # exactly the geometry this padding exists to create.
    per_page, pages = _padded(case.pages, lead_chars=chunker.size)
    padded = replace(
        case,
        pages=pages,
        table_after_line={p: n + per_page for p, n in case.table_after_line.items()},
    )
    rendered = _extract(padded, monkeypatch)
    chunks = chunker.chunk(_doc(), rendered.text, page_map=rendered.page_map)
    carrying = [c for c in chunks if case.needle in c.text]
    assert carrying, f"no chunk contains {case.needle!r}"
    assert all(c.page == case.expected_page for c in carrying)


def test_page_attribution_survives_the_clause_chunker() -> None:
    """Was a strict xfail: the chunker rebuilt offsets with ``cursor += len(part) + 1``,
    assuming every clause separator was one character. `_CLAUSE_RE` matches a whitespace RUN —
    two characters at every page boundary, since PAGE_SEPARATOR is "\\n\\n" — so offsets drifted
    one character per page and a citation's page number got further from the truth the deeper
    into the document it sat. Offsets now come from the match objects that decided the split."""
    pages = [[f"Clause {i}: the entitlement is {i * 4} days."] for i in range(1, 9)]
    pages.append(["Zulu final clause naming the appendix."])
    rendered = extract_pdf_text(make_pdf(pages), doc_id="doc.pdf").render()
    chunks = ClauseChunker(max_chars=20).chunk(_doc(), rendered.text, page_map=rendered.page_map)
    carrying = [c for c in chunks if "Zulu" in c.text]
    assert carrying

    # EVERY chunk carrying the needle names its real page. `Chunk.page` is a single number,
    # so a chunk that OWNS content from two pages cannot be right about both — the chunker
    # therefore flushes at a page boundary. Overlap still crosses it, because carried context
    # is not citable content.
    assert all(c.page == len(pages) for c in carrying)


# --- the map is derived, not reconstructed ------------------------------------------------


@pytest.mark.parametrize("case", _ATTRIBUTION_CASES, ids=lambda c: c.name)
def test_page_map_accounts_for_every_character(case: AttributionCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """sum(page_map) == len(text) is what makes accumulation by any consumer correct. If a
    separator were emitted but not counted, this is the assert that catches it."""
    rendered = _extract(case, monkeypatch)
    assert sum(rendered.page_map) == len(rendered.text)
    assert len(rendered.page_map) == len(case.pages)


def test_rendered_text_rejects_a_page_map_that_does_not_match() -> None:
    """Hand-constructing the pair must fail loudly rather than ship silent offset drift."""
    with pytest.raises(ValueError, match="page_map"):
        RenderedText(text="abcdef", page_map=[3])


def test_offsets_follow_the_separator_automatically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Change the separator, and the page map must follow with no second edit anywhere."""
    from svc_doc_intelligence import ocr

    pdf = make_pdf([["Alpha one."], ["Bravo two."], ["Charlie three."]])
    monkeypatch.setattr(ocr, "PAGE_SEPARATOR", "\n\n\n\n<<<PAGE>>>\n\n\n\n")
    rendered = extract_pdf_text(pdf, doc_id="doc.pdf").render()

    assert sum(rendered.page_map) == len(rendered.text)
    assert _page_for_offset(rendered.text.index("Charlie"), rendered.page_map) == 3


# --- per-page records ---------------------------------------------------------------------


def test_pages_carry_their_own_number_and_provenance() -> None:
    ex = extract_pdf_text(make_pdf([["Alpha one."], ["Bravo two."]]), doc_id="doc.pdf")
    assert [p.page_no for p in ex.pages] == [1, 2]
    assert [p.text for p in ex.pages] == ["Alpha one.", "Bravo two."]
    assert all(p.extractor == "pdfplumber" for p in ex.pages)
    assert all("tables" in p.capabilities for p in ex.pages)


def test_document_provenance_is_derived_from_the_pages() -> None:
    """Document-level provenance must not be a second, independently-set value: capabilities
    are the weakest page's (a doc is only table-capable if EVERY page was), degradations the
    union. This is what lets per-page fallback slot in without touching the tagging code."""
    from svc_doc_intelligence.ocr import Extraction, PageRecord

    from rag_schemas.common import Modality

    ex = Extraction(
        pages=[
            PageRecord(page_no=1, text="a", extractor="pdfplumber", capabilities=("tables", "text")),
            PageRecord(page_no=2, text="b", extractor="pypdf", capabilities=("text",), degradations=("fell_back",)),
        ],
        modality=Modality.TEXT,
    )
    assert ex.capabilities == ("text",)  # page 2 had no tables -> the DOCUMENT had no tables
    assert ex.degradations == ("fell_back",)
    assert ex.extractor == "pdfplumber,pypdf"  # honest about a mixed document
