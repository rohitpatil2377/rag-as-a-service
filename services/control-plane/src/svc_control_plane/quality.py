"""Answer-quality dashboards + on-demand evaluation (FR-G5).

Two complementary views of quality:

* :func:`quality_report` — what LIVE traffic says: grounded/flagged/abstained rates,
  faithfulness average + distribution, abstain reasons, guardrail blocks, and how humans
  ruled on flagged answers. Aggregated from the tamper-evident audit trail in-store.
* :func:`run_eval` — what a GOLDEN SET says: run the caller's questions through the real
  pipeline and score them (status, expected-fact hit, faithfulness). The per-tenant
  stored golden set + scheduled runs are the follow-up; this runs inline.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_common import get_logger, tenant_scope
from rag_schemas import RetrievalQuery
from svc_control_plane import pipeline, wiring
from svc_control_plane.usage import window_since

_log = get_logger(__name__)

MAX_EVAL_QUESTIONS = 20


class QualityReport(BaseModel):
    """Live-traffic quality over a window, per knowledge base."""

    tenant_id: str
    window: str
    answers: int = Field(description="Faithfulness-checked answers in the window.")
    grounded: int
    flagged: int
    abstained_faithfulness: int
    grounded_rate: float | None = Field(default=None, description="grounded / answers; null when no answers.")
    faithfulness_avg: float | None = None
    faithfulness_buckets: dict[str, int] = Field(default_factory=dict, description="low (<0.4) / mid / high (>=0.7).")
    abstain_reasons: dict[str, int] = Field(default_factory=dict)
    guardrail_blocks: dict[str, int] = Field(default_factory=dict, description="By stage: input / output.")
    reviews_accepted: int = 0
    reviews_modified: int = 0
    reviews_overridden: int = 0
    human_agreement_rate: float | None = Field(
        default=None, description="accepted / (accepted+modified+overridden); how often humans agree with the model."
    )
    feedback_up: int = 0
    feedback_down: int = 0
    satisfaction_rate: float | None = Field(
        default=None, description="up / (up+down) end-user thumbs (FR-I2); null when no feedback in the window."
    )


async def quality_report(tenant_id: str, window: str, *, now: datetime | None = None) -> QualityReport:
    since = window_since(window, now=now or datetime.now(UTC))
    audit = wiring.audit_service()
    stats = await audit.quality_stats(tenant_id, since)
    # KB-scoped union read (KB schema plus org-schema events attributed to this KB) so a multi-KB
    # org's dashboard is not blind to query-time activity that reached the org schema. A no-op for
    # a single-KB org (kb_id == org): the KB schema IS the org schema.
    counts = await audit.count_actions_kb(tenant_id, since)

    outcomes = stats.get("outcomes")
    outcomes = outcomes if isinstance(outcomes, dict) else {}
    grounded = int(outcomes.get("grounded", 0))
    flagged = int(outcomes.get("flagged_for_review", 0))
    abstained = int(outcomes.get("abstained", 0))
    answers = grounded + flagged + abstained

    accepted = counts.get("hitl_accept", 0)
    modified = counts.get("hitl_modify", 0)
    overridden = counts.get("hitl_override", 0)
    ruled = accepted + modified + overridden

    avg = stats.get("faithfulness_avg")
    buckets = stats.get("faithfulness_buckets")
    reasons = stats.get("abstain_reasons")
    stages = stats.get("guardrail_stages")
    fb = stats.get("feedback")
    fb = fb if isinstance(fb, dict) else {}
    up = int(fb.get("up", 0))
    down = int(fb.get("down", 0))
    return QualityReport(
        tenant_id=tenant_id,
        window=window,
        answers=answers,
        grounded=grounded,
        flagged=flagged,
        abstained_faithfulness=abstained,
        grounded_rate=round(grounded / answers, 4) if answers else None,
        faithfulness_avg=avg if isinstance(avg, int | float) else None,
        faithfulness_buckets=buckets if isinstance(buckets, dict) else {},
        abstain_reasons=reasons if isinstance(reasons, dict) else {},
        guardrail_blocks=stages if isinstance(stages, dict) else {},
        reviews_accepted=accepted,
        reviews_modified=modified,
        reviews_overridden=overridden,
        human_agreement_rate=round(accepted / ruled, 4) if ruled else None,
        feedback_up=up,
        feedback_down=down,
        satisfaction_rate=round(up / (up + down), 4) if (up + down) else None,
    )


class EvalQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    expected: str | None = Field(default=None, description="A fact the answer should contain (case-insensitive).")


class EvalRow(BaseModel):
    question: str
    status: str
    hit: bool | None = Field(default=None, description="expected fact found in the answer; null if none given.")
    faithfulness: float | None = None
    answer: str
    # Sovereign RAGAS retrieval-quality metrics (null when not computed for this row).
    answer_relevancy: float | None = Field(default=None, description="How well the answer addresses the question.")
    context_precision: float | None = Field(
        default=None, description="Are useful retrieved contexts ranked near the top."
    )
    context_recall: float | None = Field(
        default=None,
        description="Did retrieval bring back the evidence the expected fact needs; null without an expected.",
    )


class EvalReport(BaseModel):
    """Golden-set run: every question through the REAL pipeline, scored."""

    submitted: int = Field(description="Questions the caller sent, before the cap.")
    total: int = Field(description="Questions actually run (capped at MAX_EVAL_QUESTIONS).")
    truncated: bool = Field(
        default=False,
        description="True when the golden set exceeded the cap and the tail was NOT run. A "
        "truncated run that looks complete reports a real score over incomplete coverage.",
    )
    grounded: int
    abstained: int
    flagged: int
    hits: int = Field(description="Questions whose expected fact appeared in the answer.")
    with_expected: int
    hit_rate: float | None = None
    faithfulness_avg: float | None = None
    # Sovereign RAGAS aggregates (mean over the rows that carry each metric; null when none did).
    answer_relevancy_avg: float | None = None
    context_precision_avg: float | None = None
    context_recall_avg: float | None = None
    ragas_scored: int = Field(default=0, description="Rows scored on the RAGAS retrieval-quality suite.")
    rows: list[EvalRow]


async def _build_ragas_judges(tenant_id: str):  # type: ignore[no-untyped-def]
    """Bind the tenant's OWN self-hosted LLM (Sarvam) as the RAGAS judge and BGE-M3 as the
    embedder — the same sovereign models used elsewhere, so the evaluation never leaves the
    boundary. Returns ``(judge, embed)`` or ``(None, None)`` when the models can't be resolved
    (then the eval still runs faithfulness-only, RAGAS columns stay null)."""
    try:
        gateway = await wiring.resolve_tenant_gateway(tenant_id)
        embedder = await wiring.resolve_tenant_embedder(tenant_id)
    except Exception as exc:
        _log.warning("ragas_models_unavailable", tenant_id=tenant_id, error=str(exc)[:200])
        return None, None

    async def judge(prompt: str) -> str:
        return await gateway.complete(prompt, max_tokens=256)

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embedder.embed(texts)

    return judge, embed


async def run_eval(
    tenant_id: str, questions: list[EvalQuestion], *, acl: list[str], actor: str, ragas: bool = True
) -> EvalReport:
    # Local import breaks the quality <-> ragas_eval import cycle (ragas_eval imports from quality).
    from svc_control_plane import ragas_eval  # noqa: PLC0415
    from svc_control_plane import ragas_metrics as rm  # noqa: PLC0415

    judge = embed = None
    if ragas:
        judge, embed = await _build_ragas_judges(tenant_id)

    rows: list[EvalRow] = []
    scores: list[float] = []
    ragas_scores: list[rm.RagasScores] = []
    for q in questions[:MAX_EVAL_QUESTIONS]:
        answer = await pipeline.answer_query(
            RetrievalQuery(tenant_id=tenant_id, text=q.question, top_k=5, acl=acl), actor=f"eval:{actor}"
        )
        hit: bool | None = None
        if q.expected:
            hit = q.expected.lower() in answer.answer.lower()
        if answer.faithfulness_score is not None:
            scores.append(answer.faithfulness_score)

        rr: rm.RagasScores | None = None
        if judge is not None and embed is not None:
            # Best-effort per question: one RAGAS failure must not void the whole run.
            try:
                with tenant_scope(tenant_id):
                    svc = await wiring.retrieval_service_for(tenant_id)
                    res = await svc.retrieve(RetrievalQuery(tenant_id=tenant_id, text=q.question, top_k=5, acl=acl))
                contexts = [s.text for s in res.spans]
                rr = await ragas_eval.score_question(
                    q.question,
                    contexts,
                    answer.answer,
                    q.expected,
                    judge=judge,
                    embed=embed,
                    faithfulness=answer.faithfulness_score,
                )
                ragas_scores.append(rr)
            except Exception as exc:
                _log.warning("ragas_question_failed", tenant_id=tenant_id, error=str(exc)[:200])
                rr = None

        rows.append(
            EvalRow(
                question=q.question,
                status=answer.status.value,
                hit=hit,
                faithfulness=answer.faithfulness_score,
                answer=answer.answer[:400],
                answer_relevancy=(rr.answer_relevancy if rr and rr.answer_relevancy is not None else 0.0),
                context_precision=(rr.context_precision if rr and rr.context_precision is not None else 0.0),
                context_recall=(rr.context_recall if rr and rr.context_recall is not None else 0.0),
            )
        )
    with_expected = sum(1 for r in rows if r.hit is not None)
    hits = sum(1 for r in rows if r.hit)
    agg = rm.aggregate(ragas_scores) if ragas_scores else None
    return EvalReport(
        submitted=len(questions),
        total=len(rows),
        truncated=len(questions) > MAX_EVAL_QUESTIONS,
        grounded=sum(1 for r in rows if r.status == "grounded"),
        abstained=sum(1 for r in rows if r.status == "abstained"),
        flagged=sum(1 for r in rows if r.status == "flagged_for_review"),
        hits=hits,
        with_expected=with_expected,
        hit_rate=(round(hits / with_expected, 4) if with_expected else 0.0),
        faithfulness_avg=(round(sum(scores) / len(scores), 4) if scores else 0.0),
        answer_relevancy_avg=(agg.answer_relevancy_avg if agg and agg.answer_relevancy_avg is not None else 0.0),
        context_precision_avg=(agg.context_precision_avg if agg and agg.context_precision_avg is not None else 0.0),
        context_recall_avg=(agg.context_recall_avg if agg and agg.context_recall_avg is not None else 0.0),
        ragas_scored=len(ragas_scores),
        rows=rows,
    )
