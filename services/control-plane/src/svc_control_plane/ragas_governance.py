"""RAGAS governance gate — the published quality contract that GOVERNS the RaaS.

The platform already *computes* the sovereign RAGAS suite (``ragas_metrics`` /
``ragas_eval`` / ``quality.run_eval``) on its own models. What turns those numbers into
*governance* is a codified, versioned CONTRACT: the floors every knowledge base's answers
must clear, and a deterministic PASS / FAIL / INCONCLUSIVE verdict against them. This module
is that contract.

It is PURE (pydantic + stdlib): it takes an already-scored aggregate (the fields
``quality.EvalReport`` produces) and returns a verdict. The retrieval + answering + scoring
is done upstream by the real pipeline; the HTTP driver that runs it fleet-wide for QA lives
in ``scripts/ragas_gate.py``. Keeping the maths here means the gate is unit-testable with no
stack, and the SAME contract backs the CLI gate, CI, and the Grafana metric.

Design notes:
* A metric is judged only when it was actually scored (``*_avg is not None`` AND at least
  ``min_sample`` questions ran). Certifying "faithfulness 1.0" over ONE question is not
  governance — that is an INCONCLUSIVE run, reported distinctly from a PASS.
* The grounded-or-abstain invariant is enforced at the type level (``GroundedAnswer``); the
  gate additionally asserts the run produced ZERO answers that were grounded yet uncited
  (``integrity_violations``) — a structural must-be-zero, never merely a threshold.
* Floors are defaults here and overridable by env (``RAG_RAGAS_*``) or the CLI, so an
  operator can raise the bar for a regulated tenant but never silently lower it below the
  sovereign floor without it showing in the recorded contract.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.0.0"


class GateStatus(str, Enum):  # noqa: UP042  # (str,Enum) str() differs from StrEnum; keep serialization
    """A KB's (or the fleet's) verdict against the RAGAS contract."""

    PASS = "pass"  # noqa: S105  # enum member value, not a credential
    FAIL = "fail"
    # Scored, but on too few questions to certify — governance must not report this as PASS.
    INCONCLUSIVE = "inconclusive"


class RagasFloors(BaseModel):
    """The published quality floors — the numbers the platform's quality claim is measured
    against. Each is a minimum in [0,1]; a KB's aggregate must be >= the floor to clear it."""

    # The grounded-or-abstain core: how supported answers are by their cited evidence. Highest bar.
    faithfulness_min: float = Field(default=0.80, ge=0.0, le=1.0)
    # Does the answer actually address the question (Ragas answer relevancy)?
    answer_relevancy_min: float = Field(default=0.70, ge=0.0, le=1.0)
    # Are the useful retrieved passages ranked near the top (Ragas context precision)?
    context_precision_min: float = Field(default=0.70, ge=0.0, le=1.0)
    # Did retrieval bring back the evidence the ground truth needs (Ragas context recall)?
    context_recall_min: float = Field(default=0.70, ge=0.0, le=1.0)
    # Governance minimum sample: below this many scored questions a run is INCONCLUSIVE, not PASS.
    min_sample: int = Field(default=8, ge=1)
    # Optional ceiling on the FLAGGED-for-review rate (a very high rate signals judge over-flag or
    # a quality problem). None = informational only, never fails the gate.
    max_flag_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_env(cls) -> RagasFloors:
        """Load floors from ``RAG_RAGAS_*`` env vars, falling back to the sovereign defaults."""

        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None and raw != "" else default

        def _i(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None and raw != "" else default

        max_flag = os.getenv("RAG_RAGAS_MAX_FLAG_RATE")
        return cls(
            faithfulness_min=_f("RAG_RAGAS_FAITHFULNESS_MIN", 0.80),
            answer_relevancy_min=_f("RAG_RAGAS_ANSWER_RELEVANCY_MIN", 0.70),
            context_precision_min=_f("RAG_RAGAS_CONTEXT_PRECISION_MIN", 0.70),
            context_recall_min=_f("RAG_RAGAS_CONTEXT_RECALL_MIN", 0.70),
            min_sample=_i("RAG_RAGAS_MIN_SAMPLE", 8),
            max_flag_rate=float(max_flag) if max_flag is not None and max_flag != "" else None,
        )


class MetricVerdict(BaseModel):
    """One metric's outcome against its floor."""

    metric: str
    value: float | None
    floor: float
    # None when the metric was not scored (excluded from the verdict, not counted as a failure).
    passed: bool | None


class KbGateResult(BaseModel):
    """A single KB's verdict against the contract."""

    kb_id: str
    status: GateStatus
    sample: int = Field(description="Questions actually scored on the RAGAS suite.")
    verdicts: list[MetricVerdict]
    integrity_violations: int = Field(
        default=0, description="Answers that were grounded yet uncited — a structural must-be-zero."
    )
    flag_rate: float | None = None
    breaches: list[str] = Field(default_factory=list, description="Human-readable reasons the KB did not PASS.")
    notes: list[str] = Field(default_factory=list)


class ScoredRun(BaseModel):
    """The minimal shape the gate needs from an upstream RAGAS run (a superset of the fields
    ``quality.EvalReport`` exposes). Decouples the contract from the report model."""

    kb_id: str
    ragas_scored: int = 0
    total: int = 0
    grounded: int = 0
    abstained: int = 0
    flagged: int = 0
    faithfulness_avg: float | None = None
    answer_relevancy_avg: float | None = None
    context_precision_avg: float | None = None
    context_recall_avg: float | None = None
    integrity_violations: int = 0


def evaluate_kb(run: ScoredRun, floors: RagasFloors) -> KbGateResult:
    """Score one KB's run against the contract. Deterministic; no I/O."""
    verdicts: list[MetricVerdict] = []
    breaches: list[str] = []
    notes: list[str] = []

    checks = (
        ("faithfulness", run.faithfulness_avg, floors.faithfulness_min),
        ("answer_relevancy", run.answer_relevancy_avg, floors.answer_relevancy_min),
        ("context_precision", run.context_precision_avg, floors.context_precision_min),
        ("context_recall", run.context_recall_avg, floors.context_recall_min),
    )
    for name, value, floor in checks:
        if value is None:
            verdicts.append(MetricVerdict(metric=name, value=None, floor=floor, passed=None))
            notes.append(f"{name}: not scored (no rows carried this metric)")
            continue
        ok = value >= floor
        verdicts.append(MetricVerdict(metric=name, value=value, floor=floor, passed=ok))
        if not ok:
            breaches.append(f"{name} {value:.3f} < floor {floor:.2f}")

    flag_rate = round(run.flagged / run.total, 4) if run.total else None
    if floors.max_flag_rate is not None and flag_rate is not None and flag_rate > floors.max_flag_rate:
        breaches.append(f"flag_rate {flag_rate:.3f} > ceiling {floors.max_flag_rate:.2f}")

    # Structural: grounded-or-abstain integrity is a must-be-zero, not a threshold.
    if run.integrity_violations > 0:
        breaches.append(f"{run.integrity_violations} grounded-but-uncited answer(s) — invariant breach")

    # Verdict precedence: an integrity breach or a floor breach is always FAIL, even on a small
    # sample (a real defect is a defect regardless of n). A clean run on too few questions is
    # INCONCLUSIVE, not PASS. A clean run on a sufficient sample is PASS.
    if breaches:
        status = GateStatus.FAIL
    elif run.ragas_scored < floors.min_sample:
        status = GateStatus.INCONCLUSIVE
        notes.append(
            f"only {run.ragas_scored} question(s) scored (min {floors.min_sample} to certify) — "
            "expand the golden set to reach a PASS"
        )
    else:
        status = GateStatus.PASS

    return KbGateResult(
        kb_id=run.kb_id,
        status=status,
        sample=run.ragas_scored,
        verdicts=verdicts,
        integrity_violations=run.integrity_violations,
        flag_rate=flag_rate,
        breaches=breaches,
        notes=notes,
    )


class FleetGateReport(BaseModel):
    """The whole-fleet verdict — PASS only when every KB clears the contract."""

    contract_version: str = CONTRACT_VERSION
    floors: RagasFloors
    status: GateStatus
    kbs: list[KbGateResult]
    passed: int = 0
    failed: int = 0
    inconclusive: int = 0


def evaluate_fleet(runs: list[ScoredRun], floors: RagasFloors) -> FleetGateReport:
    """Roll individual KB verdicts into a fleet verdict. The fleet PASSES only when EVERY KB
    passes; any FAIL makes the fleet FAIL; otherwise (some INCONCLUSIVE, none failing) the
    fleet is INCONCLUSIVE — never silently reported as PASS."""
    results = [evaluate_kb(r, floors) for r in runs]
    passed = sum(1 for r in results if r.status is GateStatus.PASS)
    failed = sum(1 for r in results if r.status is GateStatus.FAIL)
    inconclusive = sum(1 for r in results if r.status is GateStatus.INCONCLUSIVE)

    if failed or not results:
        status = GateStatus.FAIL
    elif inconclusive:
        status = GateStatus.INCONCLUSIVE
    else:
        status = GateStatus.PASS

    return FleetGateReport(
        floors=floors,
        status=status,
        kbs=results,
        passed=passed,
        failed=failed,
        inconclusive=inconclusive,
    )


def governance_statement(floors: RagasFloors) -> str:
    """The plain-language quality claim the platform commits to and measures itself against —
    the sentence a QA lead or auditor reads next to the numbers."""
    return (
        "Every knowledge base's answers are continuously measured on the sovereign RAGAS suite "
        "(faithfulness, answer relevancy, context precision, context recall), judged on the "
        "platform's own in-boundary models. A KB is certified only when it clears every floor "
        f"(faithfulness >= {floors.faithfulness_min:.2f}, answer relevancy >= {floors.answer_relevancy_min:.2f}, "
        f"context precision >= {floors.context_precision_min:.2f}, context recall >= {floors.context_recall_min:.2f}) "
        f"over at least {floors.min_sample} questions, with zero grounded-but-uncited answers. "
        f"Contract v{CONTRACT_VERSION}."
    )


def prometheus_text(report: FleetGateReport) -> str:
    """Render the fleet report as Prometheus text-exposition metrics for the textfile collector,
    so the governing numbers are continuously visible on the Grafana quality dashboard."""
    lines: list[str] = [
        "# HELP rag_ragas_gate_pass Whether a KB clears the RAGAS governance contract (1=pass,0=fail/inconclusive).",
        "# TYPE rag_ragas_gate_pass gauge",
        "# HELP rag_ragas_metric A KB's sovereign RAGAS metric aggregate in [0,1].",
        "# TYPE rag_ragas_metric gauge",
        "# HELP rag_ragas_sample Questions scored on the RAGAS suite for a KB.",
        "# TYPE rag_ragas_sample gauge",
        "# HELP rag_ragas_fleet_status Fleet verdict (1=pass,0=not pass).",
        "# TYPE rag_ragas_fleet_status gauge",
    ]
    lines.append(
        f'rag_ragas_fleet_status{{contract="{report.contract_version}"}} {1 if report.status is GateStatus.PASS else 0}'
    )
    for kb in report.kbs:
        lines.append(
            f'rag_ragas_gate_pass{{kb="{kb.kb_id}",status="{kb.status.value}"}} '
            f"{1 if kb.status is GateStatus.PASS else 0}"
        )
        lines.append(f'rag_ragas_sample{{kb="{kb.kb_id}"}} {kb.sample}')
        for v in kb.verdicts:
            if v.value is not None:
                lines.append(f'rag_ragas_metric{{kb="{kb.kb_id}",metric="{v.metric}"}} {v.value}')
    return "\n".join(lines) + "\n"
