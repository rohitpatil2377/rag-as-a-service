"""Unit tests for the pure threshold-calibration math (feeds scripts/calibrate.py)."""

from __future__ import annotations

from rag_core.calibration import (
    best_threshold,
    candidate_grid,
    confusion_at,
    format_report,
    separation_gap,
    sweep,
)


def test_confusion_at_counts() -> None:
    pos = [0.9, 0.8, 0.4]  # answerable
    neg = [0.1, 0.05, 0.5]  # unanswerable
    c = confusion_at(0.45, pos, neg)
    assert (c.tp, c.fn) == (2, 1)  # 0.9,0.8 pass; 0.4 stopped
    assert (c.fp, c.tn) == (1, 2)  # 0.5 leaks; 0.1,0.05 stopped
    assert abs(c.recall - 2 / 3) < 1e-9
    assert abs(c.specificity - 2 / 3) < 1e-9


def test_candidate_grid_inclusive_and_clean() -> None:
    g = candidate_grid(0.0, 1.0, 0.25)
    assert g == [0.0, 0.25, 0.5, 0.75, 1.0]  # no float drift


def test_best_threshold_finds_clean_separation() -> None:
    pos = [0.7, 0.8, 0.9]  # all >= 0.7
    neg = [0.1, 0.2, 0.3]  # all <= 0.3
    best = best_threshold(sweep(pos, neg, candidate_grid(0.0, 1.0, 0.05)))
    assert best is not None
    # Any threshold in (0.3, 0.7] separates perfectly; tie-break picks the lowest such (0.35).
    assert 0.3 < best.threshold <= 0.7
    assert best.youden_j == 1.0 and best.fp == 0 and best.fn == 0


def test_separation_gap_sign() -> None:
    assert separation_gap([0.6, 0.7], [0.2, 0.4]) == 0.6 - 0.4  # clean, positive
    overlap = separation_gap([0.3, 0.5], [0.2, 0.4])
    assert overlap is not None and overlap < 0  # overlap, negative
    assert separation_gap([], [0.1]) is None


def test_best_threshold_none_on_empty() -> None:
    assert best_threshold([]) is None


def test_format_report_recommends_move_and_is_markdown() -> None:
    # Current threshold 0.9 misses answerable at 0.6/0.7 (2 false abstains); optimum should move down.
    report = format_report(
        "Abstain gate",
        current=0.9,
        positives=[0.6, 0.7, 0.8],
        negatives=[0.1, 0.2],
        candidates=candidate_grid(0.0, 1.0, 0.05),
    )
    assert "### Abstain gate" in report
    assert "Recommendation:" in report and "0.9" in report
    assert "| current |" in report and "optimal" in report


def test_format_report_keeps_optimal_when_already_best() -> None:
    report = format_report(
        "Gate",
        current=0.5,
        positives=[0.6, 0.7],
        negatives=[0.1, 0.2],
        candidates=candidate_grid(0.0, 1.0, 0.1),
    )
    # 0.5 already separates cleanly -> keep.
    assert "keep `0.5`" in report


def test_format_report_no_samples() -> None:
    report = format_report("Gate", current=0.5, positives=[], negatives=[], candidates=[])
    assert "No samples gathered" in report
