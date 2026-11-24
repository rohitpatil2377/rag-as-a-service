"""Multi-source citation attribution: a mis-typed [n] marker is re-pointed to the span that
actually supports the sentence, WITHOUT ever weakening grounded-or-abstain (FR-E2/E3).

Background: on a multi-KB / federated answer the reasoning model occasionally attaches a
sentence's [n] to the WRONG numbered source. The faithfulness checker already scores each
sentence against the UNION of retrieved spans, so this does not lower the score — but the
DISPLAYED citation (kept by `_cited_only`) then points at a source that does not support the
sentence. `_realign_citations` fixes the marker; it runs only inside `ground()` (downstream of
the grounding gate) so it can never turn an abstain into a grounded answer.
"""

from __future__ import annotations

from rag_core.grounding import _realign_citations, ground
from rag_schemas import AnswerStatus, RetrievedSpan


def _span(chunk_id: str, text: str) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::", 1)[0],
        text=text,
        score=0.9,
        scorer="cross_encoder",
    )


def test_miscited_marker_is_repointed_to_the_supporting_span() -> None:
    """The claim is about the notice period (span 2), but the model typed [1] (the leave-days span).
    Before: the displayed citation would point at span 1, which does not support the sentence.
    After: the marker is re-pointed to span 2, the span that actually contains the claim."""
    spans = [
        _span("leave::0", "Employees are entitled to 24 days of annual leave each calendar year."),
        _span("notice::0", "The notice period for resignation is 90 days for senior staff."),
    ]
    answer = "The notice period for resignation is 90 days for senior staff. [1]"

    realigned = _realign_citations(answer, spans)
    assert "[2]" in realigned and "[1]" not in realigned

    grounded = ground("t", "q", answer, spans, 0.9)
    assert grounded.status is AnswerStatus.GROUNDED
    assert len(grounded.citations) == 1
    # The kept citation is the notice-period span, not the mis-cited leave span.
    assert grounded.citations[0].chunk_id == "notice::0"


def test_a_correctly_cited_answer_is_left_untouched() -> None:
    """The safety floor: when the cited span already supports the sentence, nothing changes."""
    spans = [
        _span("leave::0", "Employees are entitled to 24 days of annual leave each calendar year."),
        _span("notice::0", "The notice period for resignation is 90 days for senior staff."),
    ]
    answer = "Employees are entitled to 24 days of annual leave each calendar year. [1]"
    assert _realign_citations(answer, spans) == answer
    grounded = ground("t", "q", answer, spans, 0.9)
    assert grounded.citations[0].chunk_id == "leave::0"


def test_a_claim_no_span_supports_is_not_given_a_bogus_citation() -> None:
    """The invariant: re-alignment never fabricates support. A sentence entailed by NO retrieved
    span keeps its original marker (it is NOT re-pointed to some unrelated span), so the
    faithfulness gate still sees an unsupported claim and can still abstain/flag exactly as today."""
    spans = [
        _span("leave::0", "Employees are entitled to 24 days of annual leave each calendar year."),
        _span("notice::0", "The notice period for resignation is 90 days for senior staff."),
    ]
    # A claim about pension vesting — present in neither span.
    answer = "Pension contributions vest fully after seven years of continuous service. [1]"
    assert _realign_citations(answer, spans) == answer  # marker unchanged; no bogus re-point


def test_multi_marker_answer_repoints_only_the_wrong_one() -> None:
    spans = [
        _span("leave::0", "Employees are entitled to 24 days of annual leave each calendar year."),
        _span("notice::0", "The notice period for resignation is 90 days for senior staff."),
    ]
    answer = (
        "Employees are entitled to 24 days of annual leave each calendar year. [1]\n"
        "The notice period for resignation is 90 days for senior staff. [1]"
    )
    realigned = _realign_citations(answer, spans)
    # First sentence keeps [1] (leave), second is re-pointed to [2] (notice).
    assert realigned.count("[1]") == 1
    assert realigned.count("[2]") == 1
