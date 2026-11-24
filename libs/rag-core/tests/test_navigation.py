"""A table of contents names every policy in the book and states none of them.

That makes it the strongest lexical match for a great many questions and evidence for none: it
lists "Salary Advance Policy .... 50" without saying anything about salary advances. Measured on
Acme's HR Handbook, 14 of 554 chunks are contents pages, and one has already been observed
winning a citation over the policy text it points at.

Dropped from the evidence ONLY when real evidence is present — a question that genuinely asks
which chapter covers something has nothing else to answer from.
"""

from __future__ import annotations

from rag_core.navigation import drop_navigation, is_navigation
from rag_schemas import RetrievedSpan

_TOC = (
    "INDEX Contents CHAPTER 1: PREAMBLE 8 CHAPTER 2: EMPLOYEE CLASSIFICATION 10 "
    "A. HR Operations .............................................................. 33 "
    "1. Data Protection Policy ...................................................... 33 "
    "2. Dress Code Policy ........................................................... 37"
)
_POLICY = (
    "Salary Advance: an employee who has completed one year of service may request an advance of "
    "up to one month's gross salary, recovered in six equal instalments from the following month."
)


def _span(chunk_id: str, text: str, score: float) -> RetrievedSpan:
    return RetrievedSpan(chunk_id=chunk_id, doc_id="handbook", text=text, score=score, scorer="cross_encoder")


def test_a_contents_listing_is_recognised() -> None:
    assert is_navigation(_TOC) is True


def test_policy_prose_is_not() -> None:
    assert is_navigation(_POLICY) is False


def test_an_ellipsis_in_ordinary_prose_is_not_a_leader() -> None:
    assert is_navigation("The clause reads: 'the employee shall... be reimbursed in full'.") is False


def test_a_numbered_list_of_real_clauses_is_not_navigation() -> None:
    assert is_navigation("1. Wear the badge. 2. Sign the register. 3. Collect the visitor pass.") is False


def test_navigation_is_dropped_when_real_evidence_is_present() -> None:
    spans = [_span("toc", _TOC, 0.91), _span("policy", _POLICY, 0.62)]
    kept = drop_navigation(spans)
    assert [s.chunk_id for s in kept] == ["policy"], "the contents page outranked the policy it points at"


def test_navigation_is_kept_when_it_is_all_there_is() -> None:
    """ "Which chapter covers salary advances?" is answered by the contents page and nothing else.
    Dropping it here would turn a good answer into an abstention."""
    spans = [_span("toc", _TOC, 0.91)]
    assert drop_navigation(spans) == spans


def test_ordering_and_scores_are_untouched() -> None:
    """A filter, not a re-ranking: it may remove spans and must not reorder or rescore them."""
    spans = [_span("a", _POLICY, 0.9), _span("toc", _TOC, 0.8), _span("b", _POLICY + " Two.", 0.7)]
    kept = drop_navigation(spans)
    assert [(s.chunk_id, s.score) for s in kept] == [("a", 0.9), ("b", 0.7)]
