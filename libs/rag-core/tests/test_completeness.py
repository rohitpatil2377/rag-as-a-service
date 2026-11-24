"""Pure-logic tests for the adaptive completeness guard (Slice 1)."""

from __future__ import annotations

from rag_core.completeness import (
    GuardBudget,
    build_completeness,
    continuation_bound,
    escalated_budget,
    expected_coverage,
    is_truncated,
    looks_truncated,
    stitch_continuation,
)
from rag_schemas import RetrievedSpan

_LEAVE = "Annual leave is twenty four working days each year."
_RTO = "Recovery time objective is two hours or less."


def _span(chunk_id: str, text: str, *, score: float, scorer: str = "cross_encoder") -> RetrievedSpan:
    return RetrievedSpan(chunk_id=chunk_id, doc_id="d", text=text, score=score, scorer=scorer)


# --- truncation detection ---
def test_finish_reason_length_is_authoritative_truncation() -> None:
    # A well-terminated sentence is NOT structurally truncated, but finish_reason=length overrides.
    assert is_truncated("The policy is complete. [1]", "length") is True


def test_finish_reason_stop_defers_to_structure() -> None:
    assert is_truncated("The policy is complete. [1]", "stop") is False
    assert is_truncated("- alpha [1]\n- beta [2]\n-", "stop") is True  # dangling bullet


def test_structural_heuristic_is_conservative_and_only_flags_dangling_enumeration() -> None:
    # Dangling list items ARE flagged (a started-but-unfinished enumeration).
    assert looks_truncated("- alpha [1]\n- beta [2]\n-") is True  # bare trailing bullet
    assert looks_truncated("1. First point.\n2.") is True  # bare trailing number marker
    assert looks_truncated("- The recovery objective is") is True  # bullet item, no terminator
    # A concise, non-list answer WITHOUT terminal punctuation is NOT flagged (avoid false positives).
    assert looks_truncated("See section 3") is False
    assert looks_truncated("There are three points: the first is") is False  # prose, not a list
    assert looks_truncated("- alpha [1]\n- beta [2]") is False  # a completed final bullet
    assert looks_truncated("All done. [1]") is False
    assert looks_truncated("| Item | Detail |") is False  # a complete table row
    assert looks_truncated("") is False  # empty is the empty-generation case, not truncation


# --- adaptive bounds / budget ---
def test_continuation_bound_is_derived_from_query_demand() -> None:
    assert continuation_bound("list every leave rule", cap=4) == 4  # exhaustive demand → full ceiling
    assert continuation_bound("what is the leave rule?", cap=4) == 2  # ordinary → half
    assert continuation_bound("anything", cap=1) == 1  # never below 1


def test_escalated_budget_compounds_and_is_capped() -> None:
    assert escalated_budget(1000, 0, factor=1.5, cap=4000) == 1500
    assert escalated_budget(1000, 1, factor=1.5, cap=4000) == 2250
    assert escalated_budget(1000, 5, factor=1.5, cap=4000) == 4000  # capped
    assert escalated_budget(0, 0, factor=1.5, cap=4000) == 4000  # zero base falls back to cap


# --- adaptive expected coverage ---
def test_expected_coverage_counts_only_spans_above_the_bar() -> None:
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9), _span("c", "noise", score=0.1)]
    assert expected_coverage(spans, min_score=0.5) == 2  # the low-score span is not "expected"


# --- completeness signal ---
def test_complete_when_all_points_covered_and_not_truncated() -> None:
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    answer = f"{_LEAVE} [1] {_RTO} [2]"
    sig = build_completeness(answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=6, truncated=False)
    assert sig.complete is True
    assert sig.reason is None
    assert (sig.covered, sig.expected, sig.truncated) == (2, 2, False)


def test_incomplete_marks_uncovered_point() -> None:
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    answer = f"{_LEAVE} [1]"  # omits the recovery point entirely
    sig = build_completeness(answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=6, truncated=False)
    assert sig.complete is False
    assert sig.reason is not None and "not covered" in sig.reason
    assert (sig.covered, sig.expected) == (1, 2)


def test_incomplete_marks_truncation() -> None:
    spans = [_span("a", _LEAVE, score=0.9)]
    answer = f"{_LEAVE} [1]"
    sig = build_completeness(answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=6, truncated=True)
    assert sig.complete is False
    assert sig.reason is not None and "cut off" in sig.reason
    assert sig.truncated is True


def test_covered_expected_use_the_uncapped_uncovered_total() -> None:
    # Five clearly-relevant spans, all omitted; a small re-prompt cap (max_spans=2) must NOT make
    # the signal claim 3 were covered — the count is measured uncapped.
    spans = [_span(f"s{i}", f"unique point number {i} alpha bravo", score=0.9) for i in range(5)]
    sig = build_completeness(
        "Nothing relevant here. [1]", spans, min_score=0.5, coverage_ratio=0.4, max_spans=2, truncated=False
    )
    assert (sig.covered, sig.expected) == (0, 5)
    assert sig.complete is False


def test_measure_coverage_false_is_truncation_only() -> None:
    spans = [_span("a", _LEAVE, score=0.9), _span("b", _RTO, score=0.9)]
    answer = f"{_LEAVE} [1]"  # omits a point, but coverage is intentionally not measured
    sig = build_completeness(
        answer, spans, min_score=0.5, coverage_ratio=0.4, max_spans=6, truncated=False, measure_coverage=False
    )
    assert sig.complete is True  # omission does not count when coverage is not measured
    assert (sig.covered, sig.expected) == (0, 0)


def test_exhausted_budget_marks_incomplete_with_reason() -> None:
    spans = [_span("a", _LEAVE, score=0.9)]
    sig = build_completeness(
        f"{_LEAVE} [1]", spans, min_score=0.5, coverage_ratio=0.4, max_spans=6, truncated=False, exhausted=True
    )
    assert sig.complete is False
    assert sig.reason is not None and "budget" in sig.reason


# --- stitch_continuation ---
def test_stitch_dedupes_repeated_overlap() -> None:
    prior = "Annual leave is 24 days. [1] Recovery time"
    cont = "Recovery time objective is two hours. [2]"
    assert stitch_continuation(prior, cont) == "Annual leave is 24 days. [1] Recovery time objective is two hours. [2]"


def test_stitch_does_not_seam_mid_token() -> None:
    stitched = stitch_continuation("Recovery ti", "me objective is two hours. [1]")
    assert stitched == "Recovery time objective is two hours. [1]"


def test_stitch_joins_with_space_when_not_mid_token() -> None:
    assert stitch_continuation("First point. [1]", "Second point. [2]") == "First point. [1] Second point. [2]"


def test_stitch_handles_empty_sides() -> None:
    assert stitch_continuation("done. [1]", "") == "done. [1]"
    assert stitch_continuation("", "start [1]") == "start [1]"


# --- GuardBudget ---
def test_guard_budget_grants_then_exhausts() -> None:
    b = GuardBudget(2)
    assert b.take() is True
    assert b.take() is True
    assert b.take() is False  # exhausted
    assert b.remaining == 0


def test_guard_budget_zero_never_grants() -> None:
    assert GuardBudget(0).take() is False
    assert GuardBudget(-5).take() is False  # clamped to 0
