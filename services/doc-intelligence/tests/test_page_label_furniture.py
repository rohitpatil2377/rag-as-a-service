"""The page footer that survived furniture stripping: "Page | 6".

_furniture_key collapses edge lines that carry NO letters ("3", "- 3 -"), because those never
repeat literally. A footer with a WORD in it was deliberately left alone: collapsing digits
inside worded lines would just as happily merge "Article 5." with "Article 6." and strip the
clause number off every page of a legal document.

Measured on Acme's HR Handbook, that gap left "Page | N" inside 324 of 554 indexed chunks — 58%
of the corpus carrying a token that means nothing and is embedded with everything around it.

The rule that tells the two apart: a page label's ONLY words are page words. "Article" is not
one, and no rule here needs to guess what "Article" means.
"""

from __future__ import annotations

from svc_doc_intelligence.ocr import _furniture_key


def test_worded_page_labels_with_different_numbers_share_one_key() -> None:
    """The point: they must collide, or the repeated-edge counter never reaches its threshold."""
    keys = {_furniture_key(line) for line in ("Page | 6", "Page | 7", "Page | 148")}
    assert len(keys) == 1

    assert _furniture_key("Page 3 of 9") == _furniture_key("Page 4 of 9")
    assert _furniture_key("- page 12 -") == _furniture_key("- page 13 -")


def test_a_numbered_clause_heading_is_never_treated_as_a_page_label() -> None:
    """The reason the original rule refused to touch worded lines at all. Losing a footer is a
    nuisance; losing a clause number is a wrong answer."""
    assert _furniture_key("Article 5.") != _furniture_key("Article 6.")
    assert _furniture_key("CHAPTER 2: EMPLOYEE CLASSIFICATION") != _furniture_key("CHAPTER 3: GENERAL DISCIPLINE")
    assert _furniture_key("Section 4 of 9") != _furniture_key("Section 5 of 9")
    assert _furniture_key("12. Salary Advance Policy") != _furniture_key("13. Loan Policy")


def test_the_wordless_and_identical_cases_still_behave() -> None:
    assert _furniture_key("3") == _furniture_key("- 4 -")  # unchanged: no letters at all
    assert _furniture_key("Acme Confidential") == _furniture_key("Acme Confidential")
    assert _furniture_key("Acme Confidential") != _furniture_key("Page | 6")
