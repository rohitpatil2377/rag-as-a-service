"""Business-value benchmark (FR-H4 + G9) — turn the proposal's ESTIMATES into MEASURED numbers.

The proposal claims "-35% cost / 2x productivity". Those are projections; this runs a real
workload against a real KB and reports what actually happened, compared to a caller-supplied
BASELINE (the "before" the claim is measured against — hosted-API RAG, manual search, whatever):

* **cost** — the platform's own FinOps meter (token totals x the operator rate card), as
  cost-per-answer, and the % reduction vs the baseline.
* **latency** — per-query wall-clock, summarised (p50/p95/p99) via the benchmark harness, and the
  reduction vs the baseline. A latency SPEEDUP is the honest, measurable proxy for "productivity"
  (time-to-cited-answer); a true 2x productivity number needs a user study, flagged in caveats.
* **quality** — grounded-rate, expected-fact hit-rate, faithfulness (the golden-eval signals),
  because a cheaper/faster answer that is wrong is not value.

Every report carries its own CAVEATS: an estimate dressed as a benchmark is worse than an honest
estimate. The pure builder (:func:`build_report`) is I/O-free and unit-tested; the orchestration
(:func:`run_business_value`) drives the real pipeline.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_common import get_settings
from rag_core.benchmark import LatencySummary, reduction_pct, speedup, summarize
from rag_schemas import RetrievalQuery
from rag_schemas.common import AnswerStatus, AuditAction
from svc_control_plane import pipeline, wiring
from svc_control_plane.quality import EvalQuestion

MAX_WORKLOAD = 200


class BizBaseline(BaseModel):
    """The 'before' a claim is measured against. Every field optional — a comparison is computed
    only where a baseline value is given, so a partial baseline is fine."""

    name: str = Field(..., description="What this baseline IS, e.g. 'hosted-API centralized RAG'.")
    cost_per_answer: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    grounded_rate: float | None = None


class LatencyBreakdown(BaseModel):
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, s: LatencySummary) -> LatencyBreakdown:
        return cls(count=s.count, p50=s.p50, p95=s.p95, p99=s.p99, mean=s.mean, minimum=s.minimum, maximum=s.maximum)


class QualityBreakdown(BaseModel):
    grounded: int
    flagged: int
    abstained: int
    with_expected: int
    grounded_rate: float | None = None
    hit_rate: float | None = None
    faithfulness_avg: float | None = None


class CostBreakdown(BaseModel):
    total_cost: float
    cost_per_answer: float
    currency: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    rate_card_indicative: bool = True


class BizComparison(BaseModel):
    baseline_name: str
    cost_reduction_pct: float | None = None
    latency_p50_reduction_pct: float | None = None
    latency_p95_reduction_pct: float | None = None
    grounded_rate_delta: float | None = None
    # Productivity PROXY: baseline_p50 / measured_p50 (time-to-cited-answer speedup). A 2x claim
    # is productivity_speedup == 2.0. A true productivity study (task completion, users) is out
    # of a software benchmark's reach — see the report caveats.
    productivity_speedup: float | None = None


class BusinessValueReport(BaseModel):
    kb_id: str
    generated_at: datetime
    workload_size: int
    latency: LatencyBreakdown
    quality: QualityBreakdown
    cost: CostBreakdown
    baseline: BizBaseline | None = None
    comparison: BizComparison | None = None
    caveats: list[str] = Field(default_factory=list)


def _compare(
    baseline: BizBaseline, cost: CostBreakdown, latency: LatencyBreakdown, quality: QualityBreakdown
) -> BizComparison:
    c = BizComparison(baseline_name=baseline.name)
    if baseline.cost_per_answer is not None:
        c.cost_reduction_pct = round(reduction_pct(baseline.cost_per_answer, cost.cost_per_answer), 2)
    if baseline.latency_p50_ms is not None:
        c.latency_p50_reduction_pct = round(reduction_pct(baseline.latency_p50_ms, latency.p50), 2)
        c.productivity_speedup = round(speedup(baseline.latency_p50_ms, latency.p50), 3)
    if baseline.latency_p95_ms is not None:
        c.latency_p95_reduction_pct = round(reduction_pct(baseline.latency_p95_ms, latency.p95), 2)
    if baseline.grounded_rate is not None and quality.grounded_rate is not None:
        c.grounded_rate_delta = round(quality.grounded_rate - baseline.grounded_rate, 4)
    return c


def _caveats(cost: CostBreakdown, baseline: BizBaseline | None) -> list[str]:
    notes = [
        "Cost uses the operator rate card (finops_cost_per_*); defaults are INDICATIVE — set real "
        "Acme rates for a real number."
        if cost.rate_card_indicative
        else "",
        "Cost is attributed to this run's window on this KB — run on a DEDICATED/quiet KB so no "
        "other traffic inflates the token/query counts.",
        "'productivity_speedup' is a time-to-cited-answer PROXY, not a measured productivity study "
        "(task completion with real users is out of a software benchmark's reach).",
        f"Quality is judged by the configured faithfulness backend ({get_settings().faithfulness_backend}); "
        "thresholds are uncalibrated by design (ADR-0005/0015) until a human golden + stronger judge.",
        "Latency is single-node here; scatter-gather at N=10/50 departments needs the multi-node "
        "deployment (see benchmark.py / the scale-benchmark item).",
    ]
    if baseline is None:
        notes.append(
            "No baseline supplied → no reduction/speedup computed. A number without a "
            "stated 'before' is an estimate, not a benchmark."
        )
    return [n for n in notes if n]


def build_report(
    kb_id: str,
    *,
    generated_at: datetime,
    latencies_ms: list[float],
    statuses: list[str],
    faithfulness: list[float],
    hits: list[bool | None],
    cost: CostBreakdown,
    baseline: BizBaseline | None,
) -> BusinessValueReport:
    """Pure: assemble measured latency/quality/cost (+ baseline comparison) into a report."""
    latency = LatencyBreakdown.of(summarize(latencies_ms))
    n = len(statuses)
    grounded = sum(1 for s in statuses if s == AnswerStatus.GROUNDED.value)
    flagged = sum(1 for s in statuses if s == AnswerStatus.FLAGGED_FOR_REVIEW.value)
    abstained = sum(1 for s in statuses if s == AnswerStatus.ABSTAINED.value)
    with_expected = sum(1 for h in hits if h is not None)
    hit_n = sum(1 for h in hits if h)
    quality = QualityBreakdown(
        grounded=grounded,
        flagged=flagged,
        abstained=abstained,
        with_expected=with_expected,
        grounded_rate=round(grounded / n, 4) if n else None,
        hit_rate=round(hit_n / with_expected, 4) if with_expected else None,
        faithfulness_avg=round(sum(faithfulness) / len(faithfulness), 4) if faithfulness else None,
    )
    comparison = _compare(baseline, cost, latency, quality) if baseline is not None else None
    return BusinessValueReport(
        kb_id=kb_id,
        generated_at=generated_at,
        workload_size=n,
        latency=latency,
        quality=quality,
        cost=cost,
        baseline=baseline,
        comparison=comparison,
        caveats=_caveats(cost, baseline),
    )


def _cost_over(counts: dict[str, int], tokens: dict[str, int], answers: int) -> CostBreakdown:
    """Apply the FinOps rate card (same math as usage.usage_report) to this run's meters."""
    s = get_settings()
    queries = counts.get("query", 0)
    documents = counts.get("ingest", 0)
    total_tokens = tokens.get("total_tokens", 0)
    total = (
        queries * s.finops_cost_per_query
        + documents * s.finops_cost_per_document
        + total_tokens / 1000 * s.finops_cost_per_1k_tokens
    )
    return CostBreakdown(
        total_cost=round(total, 4),
        cost_per_answer=round(total / answers, 6) if answers else 0.0,
        currency=s.finops_currency,
        prompt_tokens=tokens.get("prompt_tokens", 0),
        completion_tokens=tokens.get("completion_tokens", 0),
        total_tokens=total_tokens,
    )


async def run_business_value(
    kb_id: str,
    questions: list[EvalQuestion],
    baseline: BizBaseline | None = None,
    *,
    actor: str = "bench",
    now: datetime | None = None,
) -> BusinessValueReport:
    """Run a workload against ``kb_id``, measuring per-query latency + quality, then price the run
    from the audit meters and compare to ``baseline``. Records a nothing-new audit trail — it
    reuses the normal answer path, so the run's own queries ARE its cost."""
    started = now or datetime.now(UTC)
    latencies_ms: list[float] = []
    statuses: list[str] = []
    faithfulness: list[float] = []
    hits: list[bool | None] = []
    for q in questions[:MAX_WORKLOAD]:
        t0 = time.perf_counter()
        answer = await pipeline.answer_query(
            RetrievalQuery(tenant_id=kb_id, text=q.question, top_k=5, acl=[]), actor=f"bench:{actor}"
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        statuses.append(answer.status.value)
        if answer.faithfulness_score is not None:
            faithfulness.append(answer.faithfulness_score)
        hits.append(q.expected.lower() in answer.answer.lower() if q.expected else None)

    audit = wiring.audit_service()
    counts = await audit.count_actions(kb_id, started)
    tokens = await audit.token_totals(kb_id, started)
    cost = _cost_over(counts, tokens, answers=len(statuses))
    report = build_report(
        kb_id,
        generated_at=started,
        latencies_ms=latencies_ms,
        statuses=statuses,
        faithfulness=faithfulness,
        hits=hits,
        cost=cost,
        baseline=baseline,
    )
    await audit.record(
        tenant_id=kb_id,
        trace_id=started.isoformat(),
        action=AuditAction.EVAL_RUN,
        actor=actor,
        payload={"business_value": report.model_dump(mode="json")},
    )
    return report


def render_markdown(r: BusinessValueReport) -> str:
    """A shareable benchmark result — the estimate becomes a measured table."""
    lines = [
        f"# Business-value benchmark — `{r.kb_id}`",
        f"_{r.generated_at.isoformat()} · workload: {r.workload_size} queries_",
        "",
        "## Cost",
        f"- **{r.cost.cost_per_answer} {r.cost.currency}/answer** "
        f"(total {r.cost.total_cost}, {r.cost.total_tokens} tokens)"
        + ("  ⚠️ indicative rate card" if r.cost.rate_card_indicative else ""),
        "",
        "## Latency (ms)",
        f"- p50 **{r.latency.p50:.0f}** · p95 {r.latency.p95:.0f} · p99 {r.latency.p99:.0f}",
        f"- mean {r.latency.mean:.0f} · max {r.latency.maximum:.0f} ms over {r.latency.count} queries",
        "",
        "## Quality",
        f"- grounded-rate **{r.quality.grounded_rate}** · hit-rate {r.quality.hit_rate} "
        f"· faithfulness {r.quality.faithfulness_avg}",
        f"- grounded {r.quality.grounded} · flagged {r.quality.flagged} · abstained {r.quality.abstained}",
    ]
    if r.comparison is not None:
        c = r.comparison
        lines += [
            "",
            f"## vs baseline: {c.baseline_name}",
            f"- cost reduction: **{c.cost_reduction_pct}%**" if c.cost_reduction_pct is not None else "",
            (
                f"- latency p50 reduction: **{c.latency_p50_reduction_pct}%** "
                f"(productivity proxy: **{c.productivity_speedup}x**)"
            )
            if c.latency_p50_reduction_pct is not None
            else "",
            f"- grounded-rate delta: {c.grounded_rate_delta}" if c.grounded_rate_delta is not None else "",
        ]
    lines += ["", "## Caveats", *[f"- {c}" for c in r.caveats]]
    return "\n".join(line for line in lines if line != "")
