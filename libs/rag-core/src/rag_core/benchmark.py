"""Pure benchmark statistics + report formatting (no I/O) — feeds ``scripts/benchmark.py``.

Turns raw latency samples into the numbers the Federated RAG proposal claims but has not yet
measured: scatter-gather latency as the peer-department count N scales (10, 50), a
Milvus-vs-Qdrant retrieval comparison (ADR-0003, still open), and whether a measured run hits
the proposal's -35% / 2x targets. The math lives here so it is unit-tested; the orchestrator
script only gathers timings from a running stack.
"""

from __future__ import annotations

from dataclasses import dataclass


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]). 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


@dataclass(frozen=True)
class LatencySummary:
    """Latency distribution over a set of samples, all in milliseconds."""

    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float


def summarize(latencies_ms: list[float]) -> LatencySummary:
    """Summarise raw latency samples (ms)."""
    if not latencies_ms:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return LatencySummary(
        count=len(latencies_ms),
        p50=percentile(latencies_ms, 50),
        p95=percentile(latencies_ms, 95),
        p99=percentile(latencies_ms, 99),
        mean=sum(latencies_ms) / len(latencies_ms),
        minimum=min(latencies_ms),
        maximum=max(latencies_ms),
    )


def reduction_pct(baseline: float, measured: float) -> float:
    """Percent REDUCTION of a lower-is-better metric (latency/cost). Positive = improvement.
    0.0 if baseline is non-positive (no basis to compare)."""
    if baseline <= 0:
        return 0.0
    return (baseline - measured) / baseline * 100.0


def meets_reduction(baseline: float, measured: float, target_pct: float) -> bool:
    """True when the measured reduction meets/exceeds the target (e.g. the proposal's 35%)."""
    return reduction_pct(baseline, measured) >= target_pct


def speedup(baseline: float, measured: float) -> float:
    """Higher-is-better ratio (baseline/measured) — a 2x claim is ``speedup == 2.0``. 0.0 if
    measured is non-positive."""
    if measured <= 0:
        return 0.0
    return baseline / measured


def format_scaling_report(rows: list[tuple[int, LatencySummary]]) -> str:
    """Markdown: scatter-gather latency as the peer count N scales."""
    lines = [
        "### Federated scatter-gather latency vs peer count",
        "",
        "| N peers | samples | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for n, s in rows:
        lines.append(f"| {n} | {s.count} | {s.p50:.1f} | {s.p95:.1f} | {s.p99:.1f} | {s.mean:.1f} |")
    lines.append("")
    return "\n".join(lines)


def format_comparison_report(name_a: str, a: LatencySummary, name_b: str, b: LatencySummary) -> str:
    """Markdown: two vector backends head-to-head (e.g. Qdrant vs Milvus, ADR-0003)."""
    faster, slower = (name_a, name_b) if a.p95 <= b.p95 else (name_b, name_a)
    lines = [
        f"### Retrieval latency: {name_a} vs {name_b}",
        "",
        "| backend | samples | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |",
        "|---|---|---|---|---|---|",
        f"| {name_a} | {a.count} | {a.p50:.1f} | {a.p95:.1f} | {a.p99:.1f} | {a.mean:.1f} |",
        f"| {name_b} | {b.count} | {b.p50:.1f} | {b.p95:.1f} | {b.p99:.1f} | {b.mean:.1f} |",
        "",
    ]
    if a.count and b.count:
        lines += [f"**Lower p95:** {faster} (vs {slower}).", ""]
    return "\n".join(lines)


def format_target_report(label: str, *, baseline: float, measured: float, target_pct: float) -> str:
    """Markdown: does a measured run hit a proposal target (e.g. -35% latency)? Pass/fail."""
    red = reduction_pct(baseline, measured)
    verdict = "MET" if red >= target_pct else "NOT met"
    return (
        f"### Target: {label}\n\n"
        f"- baseline: {baseline:.1f} - measured: {measured:.1f}\n"
        f"- reduction: **{red:.1f}%** (target -{target_pct:.0f}%) -> **{verdict}**\n"
    )
