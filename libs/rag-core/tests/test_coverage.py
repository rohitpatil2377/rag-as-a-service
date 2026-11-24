"""Coverage-aware generation detection (pure logic).

``uncovered_spans`` is the whole "the model dropped a point" signal: it must catch a
clearly-relevant span the answer omitted, IGNORE a low-score span the model correctly left
out (forcing that back in would inject noise), and stay silent when the answer is complete.
"""

from __future__ import annotations

from typing import Literal

from rag_core.coverage import uncovered_spans
from rag_core.grounding import build_coverage_reprompt
from rag_schemas import RetrievedSpan

Scorer = Literal["cross_encoder", "lexical", "unscored"]


def _span(chunk_id: str, text: str, *, score: float, scorer: Scorer = "cross_encoder") -> RetrievedSpan:
    return RetrievedSpan(chunk_id=chunk_id, doc_id="d", text=text, score=score, scorer=scorer)


_LEAVE = "Annual leave is twenty four working days each year."
_RTO = "Recovery time objective is two hours after an incident."
_PARKING = "Parking permits cost fifty rupees monthly here."


def _uncovered(answer: str, spans: list[RetrievedSpan]) -> list[RetrievedSpan]:
    return uncovered_spans(answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=6)


def test_detects_a_dropped_high_score_span() -> None:
    """Retrieval surfaced two relevant points; the answer listed only one. The omitted, clearly
    relevant span must be reported so the service can re-prompt for it."""
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    answer = "Annual leave is twenty four working days each year. [1]"
    result = _uncovered(answer, spans)
    assert [s.chunk_id for s in result] == ["b"]


def test_ignores_an_omitted_low_score_span() -> None:
    """A span the reranker scored below the floor is NOT clearly relevant — the model very likely
    omitted it correctly, and dragging it back in would inject noise. It must never be flagged."""
    spans = [
        _span("a", _LEAVE, score=0.9),
        _span("b", _RTO, score=0.9),
        _span("c", _PARKING, score=0.2),  # below min_score — a point the model rightly skipped
    ]
    answer = "Annual leave is twenty four working days each year. [1]"
    result = _uncovered(answer, spans)
    assert [s.chunk_id for s in result] == ["b"]  # only the high-score omission, never "c"


def test_returns_empty_for_a_complete_answer() -> None:
    """When the answer already reflects every relevant span, there is nothing to add and no
    re-prompt should fire — an empty list is the "leave it alone" signal."""
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    answer = (
        "Annual leave is twenty four working days each year. [1] "
        "Recovery time objective is two hours after an incident. [2]"
    )
    assert _uncovered(answer, spans) == []


def test_a_partially_reflected_span_is_considered_covered() -> None:
    """A point the answer paraphrases in part is covered, not dropped — the ratio guards against
    mistaking a rewording for an omission."""
    # Answer carries 3 of the 5 salient RTO terms (recovery, objective, hours) => 0.6 >= 0.4.
    spans = [_span("b", _RTO, score=0.9)]
    answer = "The recovery objective is a few hours. [1]"
    assert _uncovered(answer, spans) == []


def test_highest_score_first_and_capped() -> None:
    """Multiple omissions are returned worst-relevance-first and bounded by max_spans, so one
    re-prompt is never asked to fold in an unbounded list."""
    spans = [
        _span("a", _LEAVE, score=0.6),
        _span("b", _RTO, score=0.95),
        _span("c", _PARKING, score=0.8),
    ]
    answer = "Nothing relevant is stated here about any of them at all."
    result = uncovered_spans(answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=2)
    assert [s.chunk_id for s in result] == ["b", "c"]  # top-2 by score, "a" (0.6) capped out


def test_reprompt_numbers_all_spans_and_flags_the_omitted_one() -> None:
    """The re-prompt must present the FULL span set with build_prompt's numbering (so the prior
    answer's citations stay valid) and point the model at the omitted source number."""
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    uncovered = [spans[1]]
    prompt = build_coverage_reprompt(
        "what are the leave and recovery rules?", "Leave is 24 days. [1]", spans, uncovered
    )
    assert "[1] (" in prompt and "[2] (" in prompt  # both spans numbered
    assert "[2]" in prompt.split("SOURCES:")[0]  # the omitted source is highlighted before SOURCES
    assert "Leave is 24 days." in prompt  # the prior answer is carried in for rewriting
