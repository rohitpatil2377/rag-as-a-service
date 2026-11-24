"""PII masking must not silently invalidate the page map (P4).

Masking rewrites spans to a DIFFERENT length — "officer@acme.example" (18 chars) becomes
"[REDACTED:EMAIL]" (16). The page map was derived from the pre-mask text and then handed on
unchanged beside the post-mask text, so every offset after the first redaction on a page was
wrong by the accumulated length delta. Nothing failed; citations just pointed slightly off,
and increasingly so down a document.

`RenderedText.__post_init__` would have caught this immediately, which is exactly why
`ExtractedDoc` was carrying `text` and `page_map` as two loose fields instead of the type that
guarantees they match. The fix is not to rescale the map afterwards — that re-derives offsets
from assumed lengths, the very bug the type exists to prevent. Masking returns SPANS, and one
pass rebuilds the text and the map together.
"""

from __future__ import annotations

import pytest
from svc_doc_intelligence.ocr import PAGE_SEPARATOR, Extraction, PageRecord, RenderedText

from rag_schemas.common import Modality


def _rendered(pages: list[str]) -> RenderedText:
    return Extraction(
        pages=[
            PageRecord(page_no=i, text=t, extractor="test", capabilities=("text",))
            for i, t in enumerate(pages, start=1)
        ],
        modality=Modality.TEXT,
    ).render()


def test_replacements_keep_the_map_matching_the_text() -> None:
    """The invariant the type enforces: every character belongs to exactly one page."""
    rendered = _rendered(["Contact officer@acme.example now.", "Second page body."])
    start = rendered.text.index("officer@acme.example")
    out = rendered.apply_replacements([(start, start + len("officer@acme.example"), "[REDACTED:EMAIL]")])

    assert "[REDACTED:EMAIL]" in out.text
    assert "officer@acme.example" not in out.text
    assert sum(out.page_map) == len(out.text)  # __post_init__ would raise otherwise


def test_a_shortening_replacement_moves_the_page_boundary() -> None:
    """A page that loses characters must lose them from ITS OWN entry, or the following page's
    text starts being attributed to it."""
    page_one = "Reach officer@acme.example today."
    rendered = _rendered([page_one, "Leave is 24 days."])
    before = rendered.page_map[0]

    start = rendered.text.index("officer@acme.example")
    out = rendered.apply_replacements([(start, start + 18, "[REDACTED:EMAIL]")])

    assert out.page_map[0] == before - (18 - len("[REDACTED:EMAIL]"))
    assert out.page_map[1] == rendered.page_map[1]  # the untouched page is untouched
    # ...and the second page's text still begins exactly where its entry says it does.
    assert out.text[out.page_map[0] :].strip() == "Leave is 24 days."


def test_a_lengthening_replacement_is_handled_the_same_way() -> None:
    """`hash` and `redact` strategies can produce a LONGER token than the value they replace."""
    rendered = _rendered(["PAN ABCDE1234F here.", "Second."])
    start = rendered.text.index("ABCDE1234F")
    long_token = "[REDACTED:PAN:0123456789abcdef]"
    out = rendered.apply_replacements([(start, start + 10, long_token)])

    assert out.page_map[0] == rendered.page_map[0] + (len(long_token) - 10)
    assert sum(out.page_map) == len(out.text)


def test_replacements_on_several_pages_each_adjust_their_own_page() -> None:
    rendered = _rendered(["Mail a@b.co now.", "Mail c@d.co now.", "Clean page."])
    spans = []
    for needle in ("a@b.co", "c@d.co"):
        at = rendered.text.index(needle)
        spans.append((at, at + len(needle), "[X]"))
    out = rendered.apply_replacements(spans)

    delta = len("a@b.co") - len("[X]")
    assert out.page_map == [rendered.page_map[0] - delta, rendered.page_map[1] - delta, rendered.page_map[2]]
    assert sum(out.page_map) == len(out.text)


def test_the_page_separator_is_never_swallowed() -> None:
    """A replacement ending at a page boundary must not consume the separator that follows it
    — that separator is what makes the page break a chunk boundary (I1)."""
    rendered = _rendered(["Ends with a@b.co", "Next page."])
    at = rendered.text.index("a@b.co")
    out = rendered.apply_replacements([(at, at + len("a@b.co"), "[X]")])
    assert out.text.startswith("Ends with [X]" + PAGE_SEPARATOR)


def test_out_of_order_or_overlapping_spans_are_refused() -> None:
    """Applying them in the wrong order would corrupt both text and map silently."""
    rendered = _rendered(["aaaa bbbb cccc", "second"])
    with pytest.raises(ValueError):
        rendered.apply_replacements([(5, 9, "[X]"), (0, 4, "[Y]")])  # descending
    with pytest.raises(ValueError):
        rendered.apply_replacements([(0, 6, "[X]"), (4, 9, "[Y]")])  # overlapping


async def test_page_attribution_survives_redaction_end_to_end() -> None:
    """The property all of this exists for. A PDF whose first page carries an email, and a
    fact on a LATER page: the fact must still be attributed to its own page after the redaction
    shortened everything ahead of it."""
    from pdf_fixtures import make_pdf
    from svc_doc_intelligence.service import DocIntelligenceService

    from rag_core.chunking import _page_for_offset
    from rag_schemas import DocumentEnvelope
    from rag_schemas.common import SourceKind

    pages = [
        ["Escalate to officer@acme.example without delay."],
        ["Routine matters follow the standard process."],
        ["The night shift allowance is 400 rupees."],
    ]
    from datetime import UTC, datetime

    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="policy.pdf",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    out = await DocIntelligenceService().extract(doc, make_pdf(pages))

    assert out.pii_masked >= 1 and "officer@acme.example" not in out.text
    assert sum(out.page_map) == len(out.text)  # text and map still describe each other
    # The fact is on page 3, and the offset that reaches a Citation must say so.
    assert _page_for_offset(out.text.index("night shift allowance"), out.page_map) == 3


def test_no_replacements_is_the_same_object_shape() -> None:
    rendered = _rendered(["Nothing sensitive.", "Here either."])
    out = rendered.apply_replacements([])
    assert out.text == rendered.text
    assert out.page_map == rendered.page_map
