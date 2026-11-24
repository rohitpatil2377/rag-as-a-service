"""Unit tests for the pure benchmark math (feeds scripts/benchmark.py)."""

from __future__ import annotations

from rag_core.benchmark import (
    format_comparison_report,
    format_scaling_report,
    format_target_report,
    meets_reduction,
    percentile,
    reduction_pct,
    speedup,
    summarize,
)


def test_percentile_interpolates_and_handles_edges() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 99) == 5.0
    assert percentile([0.0, 10.0], 50) == 5.0  # midpoint
    assert percentile([0.0, 100.0], 95) == 95.0  # linear interp


def test_summarize_distribution() -> None:
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s.count == 4
    assert s.minimum == 10.0 and s.maximum == 40.0
    assert s.mean == 25.0
    assert s.p50 == 25.0  # interp between 20 and 30


def test_summarize_empty_is_zeroed() -> None:
    s = summarize([])
    assert s.count == 0 and s.p95 == 0.0 and s.mean == 0.0


def test_reduction_and_target() -> None:
    assert reduction_pct(100.0, 65.0) == 35.0  # a -35% latency win
    assert reduction_pct(0.0, 5.0) == 0.0  # no baseline -> no claim
    assert meets_reduction(100.0, 65.0, 35.0) is True  # exactly meets
    assert meets_reduction(100.0, 70.0, 35.0) is False  # only -30%


def test_speedup_ratio() -> None:
    assert speedup(200.0, 100.0) == 2.0  # 2x
    assert speedup(200.0, 0.0) == 0.0  # guard div-by-zero


def test_format_scaling_report_is_markdown_table() -> None:
    rows = [(10, summarize([50.0, 60.0])), (50, summarize([200.0, 260.0]))]
    md = format_scaling_report(rows)
    assert "| N peers |" in md and "| 10 |" in md and "| 50 |" in md


def test_format_comparison_names_lower_p95() -> None:
    qdrant = summarize([40.0, 45.0, 50.0])
    milvus = summarize([80.0, 85.0, 90.0])
    md = format_comparison_report("Qdrant", qdrant, "Milvus", milvus)
    assert "Qdrant vs Milvus" in md
    assert "Lower p95:** Qdrant" in md  # qdrant is faster here


def test_format_target_verdict() -> None:
    met = format_target_report("scatter-gather p95 -35%", baseline=100.0, measured=60.0, target_pct=35.0)
    assert "MET" in met and "40.0%" in met
    missed = format_target_report("x", baseline=100.0, measured=80.0, target_pct=35.0)
    assert "NOT met" in missed
