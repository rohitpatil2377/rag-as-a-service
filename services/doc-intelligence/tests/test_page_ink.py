"""Which pages have INK on them (FR-B2) — the input to the blank-page skip and the
image-page OCR policy. Both misread a real document on this deployment.

These are pure geometry predicates over a pdfplumber-shaped page, so they are tested against a
stub page rather than a fixture PDF: the bug was in the RULE, not in the reader.
"""

from __future__ import annotations

from typing import ClassVar

from svc_doc_intelligence.ocr import _page_has_substantive_image, _page_has_vector_ink


class _Page:
    """The attributes pdfplumber exposes that these two predicates read."""

    def __init__(
        self,
        *,
        images: list[dict[str, float]] | None = None,
        curves: int = 0,
        rects: int = 0,
        lines: int = 0,
        width: float = 612.0,  # US Letter, pdfplumber's units are points
        height: float = 792.0,
    ) -> None:
        self.images = images or []
        self.curves = [object()] * curves
        self.rects = [object()] * rects
        self.lines = [object()] * lines
        self.width = width
        self.height = height


def _image(*, share: float, page: _Page) -> dict[str, float]:
    """A square image covering ``share`` of the page area."""
    side = (share * page.width * page.height) ** 0.5
    return {"x0": 10.0, "x1": 10.0 + side, "top": 10.0, "bottom": 10.0 + side}


# --- blank-page test ------------------------------------------------------------------------
def test_a_truly_blank_page_has_no_vector_ink() -> None:
    assert not _page_has_vector_ink(_Page())


def test_a_page_with_a_rule_or_two_is_still_blank() -> None:
    """A duplex back or separator sheet often carries a header rule. Counting that as content
    would send every one of them to OCR to produce nothing."""
    assert not _page_has_vector_ink(_Page(lines=2, rects=1))


def test_glyphs_traced_into_vector_outlines_are_ink() -> None:
    """The real failure: a converter traced a scanned SOP's glyphs into outlines, so pages had
    zero characters and zero raster images but thousands of curves. They were classed blank and
    skipped -- two thirds of the document went missing with nothing reported."""
    assert _page_has_vector_ink(_Page(curves=1500))


def test_a_ruled_table_counts_as_ink() -> None:
    assert _page_has_vector_ink(_Page(lines=20, rects=14))


# --- image-page OCR policy ------------------------------------------------------------------
def test_a_page_with_no_images_has_no_substantive_image() -> None:
    assert not _page_has_substantive_image(_Page())


def test_a_letterhead_logo_does_not_make_a_page_an_image_page() -> None:
    """The other real failure: branding on all 47 pages of an SOP routed every page to OCR,
    discarding an 87,000-character text layer for a slower, lossier reading of the same words."""
    page = _Page()
    page.images = [_image(share=0.004, page=page)]  # ~2cm x 2cm on Letter
    assert not _page_has_substantive_image(page)


def test_a_full_width_figure_or_chart_is_substantive() -> None:
    page = _Page()
    page.images = [_image(share=0.35, page=page)]
    assert _page_has_substantive_image(page)


def test_one_big_figure_counts_even_beside_small_branding() -> None:
    page = _Page()
    page.images = [_image(share=0.004, page=page), _image(share=0.30, page=page)]
    assert _page_has_substantive_image(page)


def test_a_page_whose_geometry_cannot_be_read_is_investigated_not_skipped() -> None:
    """Absence of a usable signal is not evidence that the image does not matter -- an
    unreadable page is looked at rather than silently written off."""

    class _NoGeometry:
        images: ClassVar[list[dict[str, float]]] = [{"x0": 0.0, "x1": 1.0, "top": 0.0, "bottom": 1.0}]

        @property
        def width(self) -> float:
            raise RuntimeError("this reader exposes no page geometry")

    assert _page_has_substantive_image(_NoGeometry())


def test_a_zero_area_page_is_investigated_not_skipped() -> None:
    page = _Page(width=0.0, height=0.0)
    page.images = [{"x0": 0.0, "x1": 1.0, "top": 0.0, "bottom": 1.0}]
    assert _page_has_substantive_image(page)


def test_the_two_signals_are_independent() -> None:
    """They answer different questions and feed different decisions: "is there anything to read
    at all" versus "is a picture carrying this page's content". Conflating them made every ruled
    table look like a page that needed OCR."""
    outlined_text = _Page(curves=1500)
    assert _page_has_vector_ink(outlined_text)
    assert not _page_has_substantive_image(outlined_text)

    figure_page = _Page()
    figure_page.images = [_image(share=0.4, page=figure_page)]
    assert _page_has_substantive_image(figure_page)
    assert not _page_has_vector_ink(figure_page)


# --- mathematical / chemical notation -------------------------------------------------------
# Raised and lowered characters carry meaning: "a2 - b2" is not the identity, and H2SO4 is not
# the same substance as H₂SO₄. The OCR chain must carry them through untouched, so that asking
# the model for them (see _CHANDRA_OCR_PROMPT) is enough to keep them.
def test_the_ocr_chain_preserves_superscripts_and_subscripts() -> None:
    from svc_doc_intelligence.ocr_check import normalize_ocr_markup, trim_degenerate_tail
    from svc_doc_intelligence.sanitize import normalize_text

    for source in (
        "| (a+b)\u00b2 = a\u00b2 + 2ab + b\u00b2 | a\u00b3 \u2212 b\u00b3 = (a\u2212b)(a\u00b2+ab+b\u00b2) |",
        "H\u2082SO\u2084, NO\u2083\u207b and CO\u2082 at 25\u00b0C",
        "x\u207d\u207f\u207a\u00b9\u207e and the \u208d\u2081\u208e footnote marker",
    ):
        after_ocr, _ = trim_degenerate_tail(normalize_ocr_markup(source))
        assert after_ocr == source, f"OCR normalisation altered notation: {after_ocr!r}"
        # ...and again across the NFKC fold, which is what would otherwise rewrite ² to 2.
        assert normalize_text(after_ocr) == source, "the unicode fold flattened the notation"
