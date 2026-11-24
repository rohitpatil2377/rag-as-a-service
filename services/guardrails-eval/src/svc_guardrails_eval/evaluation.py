"""RAG evaluation harness (TDD §4.7 · FR-G5) — the CI release gate.

The gate answers one question: *did quality regress?* It can only do that if every number it
reports is true when the platform is broken, not just when it is healthy. Two of them were not:

* ``citation_coverage`` divided by the count of GROUNDED answers, so a build that abstained on
  everything scored 1.0 — full marks for answering nothing.
* ``context_hit_rate`` was computed and then ignored by ``passes()``, so a build that answered
  every question confidently out of the wrong documents was green.

Metrics here are deliberately arithmetic, not model-judged: a gate that needs an LLM to decide
whether the build passes inherits that model's variance, and a threshold against it means
nothing. Judge-based scoring belongs in the per-KB quality report (``svc_control_plane.quality``,
FR-G9), where a human reads the number rather than a build depending on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_schemas import AnswerStatus, GroundedAnswer


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """One golden Q&A ground-truth record (``sample-data/golden/``).

    ``expected_chunk_ids`` is optional and grown incrementally: an item without it is EXCLUDED
    from recall/MRR rather than counted as a miss, so adding a question never makes the score
    fall."""

    question: str
    expected_answer: str
    expected_doc_ids: list[str]
    expect_abstain: bool = False
    expected_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalReport:
    n: int
    # How many items expect an ANSWER (the rest expect an abstention). Carried because it is
    # what makes "nothing was answered" readable: with answerable items it is the regression
    # this gate exists to catch; with none, it is correct behaviour.
    answerable: int
    grounded: int
    # Structurally 1.0 whenever anything was answered: GroundedAnswer REFUSES a grounded or
    # flagged answer with no citations, so this cannot fail unless something builds an answer
    # outside that type. Kept as defence in depth, and because None (nothing answered at all)
    # is a real and important signal.
    citation_coverage: float | None
    abstain_correctness: float
    # Over the items that expect an answer; 1.0 when none do.
    context_hit_rate: float
    recall_at_k: float | None = None
    mrr: float | None = None

    def passes(
        self,
        *,
        min_citation: float = 1.0,
        min_abstain: float = 1.0,
        min_context: float = 1.0,
    ) -> bool:
        """CI release gate (FR-G5): fail the build if quality regresses.

        ``citation_coverage`` of None means no answer was produced at all. That is a FAIL when
        the golden set contained anything answerable — a build that abstains on everything is
        exactly the regression this gate exists to catch, and the KPI it stands for ("100% of
        answers carry a valid citation") is vacuous rather than satisfied. A set of only
        abstain items answering nothing is correct, not a regression."""
        if self.citation_coverage is None:
            return self.answerable == 0
        return (
            self.citation_coverage >= min_citation
            and self.abstain_correctness >= min_abstain
            and self.context_hit_rate >= min_context
        )


def evaluate(
    items: list[GoldenItem],
    answers: list[GroundedAnswer],
    retrieved: list[list[str]] | None = None,
) -> EvalReport:
    """Compute the gate metrics over aligned (item, answer) pairs.

    ``retrieved`` is the ordered chunk ids each question actually retrieved. Supply it and the
    report carries retrieval recall@k and MRR; omit it and both are None. They are NOT derived
    from citations as a fallback: citations say what the GENERATOR used, recall says what
    RETRIEVAL found, and collapsing the two hides the difference between "retrieval missed it"
    and "the model did not cite it" — different bugs with different fixes."""
    if len(items) != len(answers):
        raise ValueError("items/answers length mismatch")
    if retrieved is not None and len(retrieved) != len(items):
        raise ValueError("items/retrieved length mismatch")

    n = len(items) or 1
    answered = cited = abstain_ok = ctx_ok = ctx_n = 0
    for item, ans in zip(items, answers, strict=True):
        # An answer is one the platform stood behind — grounded OR flagged for review. Both
        # reach a reader, so both owe a citation; only an abstention does not.
        if ans.status is not AnswerStatus.ABSTAINED:
            answered += 1
            if ans.citations:
                cited += 1
        if item.expect_abstain == (ans.status is AnswerStatus.ABSTAINED):
            abstain_ok += 1
        # Scored only where an answer was expected. An item that SHOULD abstain cites nothing
        # by definition, so counting it as a context miss would penalise correct behaviour and
        # make the floor unreachable for any golden set containing an abstain case.
        if not item.expect_abstain:
            ctx_n += 1
            got_docs = {c.doc_id for c in ans.citations}
            if not item.expected_doc_ids or got_docs & set(item.expected_doc_ids):
                ctx_ok += 1

    return EvalReport(
        n=len(items),
        answerable=ctx_n,
        grounded=sum(1 for a in answers if a.status is AnswerStatus.GROUNDED),
        citation_coverage=(cited / answered) if answered else None,
        abstain_correctness=abstain_ok / n,
        context_hit_rate=(ctx_ok / ctx_n) if ctx_n else 1.0,
        **_retrieval_metrics(items, retrieved),
    )


def _retrieval_metrics(items: list[GoldenItem], retrieved: list[list[str]] | None) -> dict[str, float | None]:
    """recall@k and MRR over the items that carry expected chunk ids.

    recall@k is per-question: did ANY wanted chunk come back. MRR uses the rank of the FIRST
    wanted chunk, which is what decides whether it survives the top-k trim into the prompt —
    retrieving the right chunk at rank 40 is not the same as retrieving it at rank 1."""
    if retrieved is None:
        return {"recall_at_k": None, "mrr": None}
    hits: list[float] = []
    reciprocals: list[float] = []
    for item, ids in zip(items, retrieved, strict=True):
        wanted = set(item.expected_chunk_ids)
        if not wanted:
            continue  # not annotated yet — excluded, never counted as a miss
        ranks = [i for i, cid in enumerate(ids, start=1) if cid in wanted]
        hits.append(1.0 if ranks else 0.0)
        reciprocals.append(1 / ranks[0] if ranks else 0.0)
    if not hits:
        return {"recall_at_k": None, "mrr": None}
    return {"recall_at_k": sum(hits) / len(hits), "mrr": sum(reciprocals) / len(reciprocals)}
