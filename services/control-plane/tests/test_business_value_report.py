"""Business-value report — the pure builder turns measured samples + a baseline into the
cost/latency/quality deltas, and is HONEST (caveats always present; no baseline => no numbers
invented)."""

from __future__ import annotations

from datetime import UTC, datetime

from svc_control_plane.business_value import (
    BizBaseline,
    CostBreakdown,
    build_report,
)

_NOW = datetime(2026, 8, 5, tzinfo=UTC)


def _cost(cpa: float) -> CostBreakdown:
    return CostBreakdown(
        total_cost=cpa * 4,
        cost_per_answer=cpa,
        currency="INR",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        rate_card_indicative=True,
    )


def test_baseline_comparison_computes_reduction_and_speedup() -> None:
    baseline = BizBaseline(
        name="hosted-API centralized RAG",
        cost_per_answer=2.0,
        latency_p50_ms=4000.0,
        latency_p95_ms=9000.0,
        grounded_rate=0.60,
    )
    # Measured: cheaper (1.3 vs 2.0 = 35% off), faster (2000 vs 4000 = 2x), higher grounded-rate.
    r = build_report(
        "kb",
        generated_at=_NOW,
        latencies_ms=[1900.0, 2000.0, 2100.0, 2000.0],
        statuses=["grounded", "grounded", "grounded", "abstained"],
        faithfulness=[0.9, 0.85, 0.95],
        hits=[True, True, False, None],
        cost=_cost(1.3),
        baseline=baseline,
    )
    assert r.workload_size == 4
    c = r.comparison
    assert c is not None
    assert c.cost_reduction_pct == 35.0  # (2.0-1.3)/2.0
    assert c.latency_p50_reduction_pct == 50.0  # (4000-2000)/4000
    assert c.productivity_speedup == 2.0  # 4000/2000 — the 2x, MEASURED
    assert c.grounded_rate_delta == round(0.75 - 0.60, 4)  # grounded 3/4 = 0.75
    assert r.quality.grounded_rate == 0.75
    assert r.quality.hit_rate == round(2 / 3, 4)  # 2 hits of 3 with an expected fact
    assert r.caveats  # always present


def test_no_baseline_means_no_comparison_and_says_so() -> None:
    r = build_report(
        "kb",
        generated_at=_NOW,
        latencies_ms=[1000.0, 1200.0],
        statuses=["grounded", "grounded"],
        faithfulness=[0.9, 0.9],
        hits=[None, None],
        cost=_cost(1.0),
        baseline=None,
    )
    assert r.comparison is None
    assert any("estimate, not a benchmark" in c for c in r.caveats)


def test_a_partial_baseline_only_compares_what_it_provides() -> None:
    # Only a cost baseline given — latency/grounded comparisons stay None (not fabricated).
    r = build_report(
        "kb",
        generated_at=_NOW,
        latencies_ms=[2000.0],
        statuses=["grounded"],
        faithfulness=[0.9],
        hits=[None],
        cost=_cost(1.0),
        baseline=BizBaseline(name="prev quarter", cost_per_answer=2.0),
    )
    assert r.comparison is not None
    assert r.comparison.cost_reduction_pct == 50.0
    assert r.comparison.latency_p50_reduction_pct is None
    assert r.comparison.productivity_speedup is None
