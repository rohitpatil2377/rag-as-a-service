"""Sovereign Ragas-style evaluation harness — score a KB's golden set on our own models and
record the result in the tamper-evident audit chain.

Complements ``continuous_eval`` (which tracks grounded-rate + faithfulness) with the metrics
that expose RETRIEVAL quality — answer relevancy, context precision, and the blind-spot metric
context recall ("did we retrieve the evidence the ground truth needs?"). The judge is the
tenant's self-hosted LLM (Sarvam) and the embedder is BGE-M3, so the evaluation itself never
leaves the sovereign boundary. Every run is written as an ``EVAL_RUN`` audit event carrying a
``ragas`` payload, so a quality claim is reproducible for audit scrutiny.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from rag_common import get_logger, tenant_scope
from rag_schemas import RetrievalQuery
from rag_schemas.common import AuditAction
from svc_control_plane import ragas_metrics as rm
from svc_control_plane import wiring
from svc_control_plane.quality import MAX_EVAL_QUESTIONS, EvalQuestion

_log = get_logger(__name__)


class RagasRow(BaseModel):
    """One golden question's sovereign RAG-quality scores."""

    question: str
    scores: rm.RagasScores


class RagasEvalReport(BaseModel):
    """A whole golden-set run: per-question rows + the aggregate + what it was over."""

    kb_id: str
    aggregate: rm.RagasAggregate
    rows: list[RagasRow]


async def score_question(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str | None,
    *,
    judge: rm.Judge,
    embed: rm.Embed,
    faithfulness: float | None = None,
) -> rm.RagasScores:
    """Compute the sovereign metric suite for one (question, retrieved-contexts, answer,
    ground-truth) tuple. context_recall is only computed when a ground truth is available."""
    return rm.RagasScores(
        answer_relevancy=await rm.answer_relevancy(question, answer, judge=judge, embed=embed),
        context_precision=await rm.context_precision(question, contexts, judge=judge, reference=answer),
        context_recall=(await rm.context_recall(ground_truth, contexts, judge=judge)) if ground_truth else None,
        faithfulness=faithfulness,
    )


async def run_ragas_eval(
    kb_id: str,
    items: list[tuple[str, list[str], str, str | None, float | None]],
    *,
    judge: rm.Judge,
    embed: rm.Embed,
) -> RagasEvalReport:
    """Score a list of pre-computed ``(question, contexts, answer, ground_truth, faithfulness)``
    tuples. Pure over the injected judge/embed — the retrieval + answering is done by the
    caller, so this core is deterministically testable with fakes."""
    rows: list[RagasRow] = []
    for question, contexts, answer, gt, faith in items:
        scores = await score_question(question, contexts, answer, gt, judge=judge, embed=embed, faithfulness=faith)
        rows.append(RagasRow(question=question, scores=scores))
    return RagasEvalReport(kb_id=kb_id, aggregate=rm.aggregate([r.scores for r in rows]), rows=rows)


async def run_ragas_eval_for_kb(kb_id: str, *, actor: str = "ragas") -> RagasEvalReport:
    """Production entry: load the KB's golden set, run each question through the REAL pipeline
    (retrieval → grounded answer), score it on the tenant's own Sarvam judge + BGE-M3 embedder,
    aggregate, and record an ``EVAL_RUN`` audit event with the ragas scores. Best-effort per
    question — one failure is skipped, not fatal, so a single bad question cannot void the run.
    """
    from svc_control_plane import pipeline  # noqa: PLC0415  # local import: avoid an import cycle at module load

    golden = await _load_golden(kb_id)
    if not golden:
        report = RagasEvalReport(kb_id=kb_id, aggregate=rm.aggregate([]), rows=[])
        await _record(kb_id, report, actor)
        return report

    gateway = await wiring.resolve_tenant_gateway(kb_id)
    embedder = await wiring.resolve_tenant_embedder(kb_id)

    async def judge(prompt: str) -> str:
        return await gateway.complete(prompt, max_tokens=256)

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embedder.embed(texts)

    items: list[tuple[str, list[str], str, str | None, float | None]] = []
    with tenant_scope(kb_id):
        svc = await wiring.retrieval_service_for(kb_id)
        for q in golden[:MAX_EVAL_QUESTIONS]:
            try:
                res = await svc.retrieve(RetrievalQuery(tenant_id=kb_id, text=q.question, top_k=5, acl=[]))
                contexts = [s.text for s in res.spans]
                ans = await pipeline.answer_query(
                    RetrievalQuery(tenant_id=kb_id, text=q.question, top_k=5, acl=[]), actor=actor
                )
                items.append((q.question, contexts, ans.answer, q.expected, None))
            except Exception as exc:
                _log.warning("ragas_question_failed", kb_id=kb_id, error=str(exc)[:200])

    report = await run_ragas_eval(kb_id, items, judge=judge, embed=embed)
    await _record(kb_id, report, actor)
    return report


async def _load_golden(kb_id: str) -> list[EvalQuestion]:
    """The KB's saved golden questions (reuses the same store key continuous_eval reads)."""
    from svc_control_plane.continuous_eval import _load_golden as _load  # noqa: PLC0415

    return await _load(kb_id)


async def _record(kb_id: str, report: RagasEvalReport, actor: str) -> None:
    """Write the run to the tamper-evident audit chain as an EVAL_RUN carrying the ragas
    aggregate — so the quality numbers are reproducible and pullable for audit."""
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.EVAL_RUN,
        actor=actor,
        payload={"eval": "ragas", "kb_id": kb_id, "ragas": report.aggregate.model_dump()},
    )
