"""The release gate's own metrics (FR-G5).

These had no test. A metric that lies is worse than no metric: it reports a green build
while the platform regresses, and it is the number a regulated customer is shown.
"""

from __future__ import annotations

from svc_guardrails_eval import GoldenItem, evaluate

from rag_schemas import AnswerStatus, Citation, GroundedAnswer


def _grounded(doc_id: str = "brief", chunk_id: str = "brief::0") -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="an answer [1]",
        citations=[Citation(chunk_id=chunk_id, doc_id=doc_id, quote="q", score=0.9)],
        confidence=0.9,
    )


def _abstained() -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.ABSTAINED,
        answer="I don't know",
        citations=[],
        confidence=0.0,
        abstain_reason="retrieval_confidence_below_threshold",
    )


def _item(expect_abstain: bool = False, docs: list[str] | None = None) -> GoldenItem:
    return GoldenItem(
        question="q",
        expected_answer="a",
        expected_doc_ids=docs if docs is not None else ["brief"],
        expect_abstain=expect_abstain,
    )


def test_a_build_that_abstains_on_everything_does_not_score_perfect_coverage() -> None:
    """citation_coverage divided by the number of GROUNDED answers, so a build that abstained
    on every question scored 1.0 — perfect marks for answering nothing. The KPI it reports on
    is "100% of answers carry a valid citation"; with no answers that claim is vacuous, not
    satisfied."""
    items = [_item(), _item()]
    report = evaluate(items, [_abstained(), _abstained()])
    assert report.grounded == 0
    assert report.citation_coverage is None, "coverage over zero answers is undefined, not 1.0"
    assert not report.passes()


def test_an_answer_without_a_citation_cannot_even_be_built() -> None:
    """Why coverage can only ever be 1.0 or None: the grounded-or-abstain invariant is enforced
    on the TYPE, so an uncited answer never exists to be measured. The metric stays as defence
    in depth against an answer assembled outside GroundedAnswer — and because None (nothing was
    answered at all) is the signal that actually matters."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GroundedAnswer(
            tenant_id="t",
            query="q",
            status=AnswerStatus.FLAGGED_FOR_REVIEW,
            answer="no citation here",
            citations=[],
            confidence=0.9,
        )

    report = evaluate([_item()], [_grounded()])
    assert report.citation_coverage == 1.0


def test_an_item_expected_to_abstain_is_not_a_context_miss() -> None:
    """It cites nothing by definition. Counting that as a miss would penalise correct behaviour
    and put the floor out of reach for any golden set with an abstain case in it."""
    report = evaluate([_item(expect_abstain=True)], [_abstained()])
    assert report.context_hit_rate == 1.0
    assert report.passes()


def test_citing_the_wrong_document_fails_the_gate() -> None:
    """context_hit_rate was computed and then ignored by passes(), so a build that answered
    every question confidently out of the wrong documents was a green build."""
    report = evaluate([_item(docs=["handbook"])], [_grounded(doc_id="brief")])
    assert report.context_hit_rate == 0.0
    assert not report.passes()


def test_a_clean_run_passes() -> None:
    items = [_item(), _item(expect_abstain=True)]
    report = evaluate(items, [_grounded(), _abstained()])
    assert report.passes(), report


def test_retrieval_recall_is_absent_rather_than_guessed_from_citations() -> None:
    """Citations say what the GENERATOR used; recall says what RETRIEVAL found. Substituting
    one for the other would hide the difference between "retrieval missed it" and "the model
    did not cite it" — which are different bugs with different fixes."""
    report = evaluate([_item()], [_grounded()])
    assert report.recall_at_k is None
    assert report.mrr is None


def test_recall_and_mrr_are_measured_when_retrieval_is_supplied() -> None:
    item = GoldenItem(question="q", expected_answer="a", expected_doc_ids=["brief"], expected_chunk_ids=["brief::7"])
    # The wanted chunk came back third.
    report = evaluate([item], [_grounded()], retrieved=[["brief::1", "brief::4", "brief::7"]])
    assert report.recall_at_k == 1.0
    assert report.mrr == 1 / 3


def test_a_miss_scores_zero_on_both() -> None:
    item = GoldenItem(question="q", expected_answer="a", expected_doc_ids=["brief"], expected_chunk_ids=["brief::7"])
    report = evaluate([item], [_grounded()], retrieved=[["brief::1", "brief::4"]])
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_items_without_expected_chunks_are_excluded_not_counted_as_misses() -> None:
    """A golden set is grown incrementally. Counting un-annotated items as misses would make
    recall fall every time a question is added, punishing exactly the thing that should be
    encouraged."""
    annotated = GoldenItem(
        question="q", expected_answer="a", expected_doc_ids=["brief"], expected_chunk_ids=["brief::7"]
    )
    report = evaluate([annotated, _item()], [_grounded(), _grounded()], retrieved=[["brief::7"], ["brief::1"]])
    assert report.recall_at_k == 1.0
