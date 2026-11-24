"""Aggregating verification provenance: the distribution survives, and no score is invented.

WHY THIS IS A UNIT TEST AND NOT A PIPELINE ONE. The mixed-method case was first written against
the map-reduce route, where it would have passed vacuously: probed on the real pipeline, that
route calls the faithfulness checker EXACTLY ONCE over the assembled answer, so its method
distribution can only ever hold one entry. Several batches verified by different methods is the
WHOLE-DOCUMENT route's shape — a real incumbent run produced 13 batch verifications, one of which
fell back to the lexical checker after a `ReadError`. The aggregation rule is the same either way
and belongs with the function that implements it.
"""

from __future__ import annotations

from rag_core import FaithfulnessVerdict, verification_block


def _nli(score: float = 0.95) -> FaithfulnessVerdict:
    return FaithfulnessVerdict(score=score, faithful=True, method="nli")


def _fallback(score: float = 0.80) -> FaithfulnessVerdict:
    return FaithfulnessVerdict(
        score=score,
        faithful=True,
        method="lexical_fallback",
        degraded=True,
        degraded_reason="nli_failed_fallback_used",
        degraded_detail="ReadError: ",
    )


def _unavailable() -> FaithfulnessVerdict:
    return FaithfulnessVerdict(
        score=None,
        faithful=None,
        method="unavailable",
        degraded=True,
        degraded_reason="nli_failed_fallback_failed",
    )


def test_a_clean_run_is_scored_with_its_score() -> None:
    block = verification_block([_nli(0.9), _nli(0.95)])
    assert block is not None
    assert block["outcome"] == "scored"
    assert block["degraded"] is False
    assert block["methods"] == [{"method": "nli", "count": 2}]
    assert block["score"] == 0.925, "one method, so a mean of its own scores is a real number"
    assert "findings" not in block


def test_a_mixed_aggregate_keeps_both_methods_and_manufactures_no_score() -> None:
    """THE INVARIANT. The real run that prompted this had 12 batches scored by NLI and 1 by the
    lexical checker. An aggregate reporting the majority method, or a mean across both, tells the
    reader the whole answer was NLI-verified — and the one batch that was not is precisely the one
    a reviewer needs to look at."""
    block = verification_block([_nli(), _nli(), _fallback()])
    assert block is not None
    assert block["methods"] == [{"method": "nli", "count": 2}, {"method": "lexical_fallback", "count": 1}]
    assert block["degraded"] is True
    assert block["outcome"] == "mixed"
    assert "score" not in block, (
        "an NLI score and a lexical score are not the same measurement; averaging them yields a "
        "number that means nothing and looks authoritative"
    )
    assert len(block["batches"]) == 3, "per-batch provenance is retained, not summarised away"
    assert [f["reason_code"] for f in block["findings"]] == ["nli_failed_fallback_used"]


def test_score_is_present_exactly_when_the_outcome_says_it_is() -> None:
    """THE CONTRACT A CONSUMER BINDS TO, pinned in both directions.

    Omitting the key is NOT a safeguard on its own: in JavaScript an absent property is
    `undefined`, and `block.score ?? 0` yields 0 for `undefined` exactly as it does for `null`, so
    a careless consumer turns "not comparable" into a failing grade either way. `outcome` is the
    thing that makes the misuse detectable -- it must be impossible for a reader that branches on
    it to find a score it should not trust, or to miss one it should."""
    cases = [
        ([_nli(), _nli()], "scored"),
        ([_fallback(), _fallback()], "scored"),
        ([_nli(), _fallback()], "mixed"),
        ([_nli(), _unavailable()], "unavailable"),
        ([_unavailable()], "unavailable"),
    ]
    for verdicts, expected in cases:
        block = verification_block(verdicts)
        assert block is not None
        assert block["outcome"] == expected
        assert ("score" in block) is (expected == "scored"), (
            f"outcome={expected!r} must {'carry' if expected == 'scored' else 'omit'} a score"
        )


def test_every_batch_failing_over_is_scored_and_also_degraded() -> None:
    """Two different facts, and collapsing them loses one. Every batch used the SAME method, so
    its scores are comparable and one number is meaningful -- and the intended mechanism failed
    on every one of them, which a reader has to know. `scored` is about comparability; `degraded`
    is about availability."""
    block = verification_block([_fallback(0.8), _fallback(0.9)])
    assert block is not None
    assert block["outcome"] == "scored"
    assert block["score"] == 0.85
    assert block["degraded"] is True
    assert block["methods"] == [{"method": "lexical_fallback", "count": 1 + 1}]


def test_unavailable_outranks_a_fallback_in_the_outcome() -> None:
    """The outcome takes the WEAKEST case present. A run where one batch fell back and another
    could not be checked at all is not "degraded" — part of it was never examined."""
    block = verification_block([_nli(), _fallback(), _unavailable()])
    assert block is not None
    assert block["outcome"] == "unavailable"
    assert {m["method"] for m in block["methods"]} == {"nli", "lexical_fallback", "unavailable"}
    assert "score" not in block


def test_a_deliberately_lexical_deployment_is_not_degraded() -> None:
    """Running the checker you configured is not a downgrade. Marking it degraded would make a
    deployment choice look like an incident; reporting `unset` would hide it entirely."""
    block = verification_block([FaithfulnessVerdict(score=0.9, faithful=True, method="lexical")])
    assert block is not None
    assert block["outcome"] == "scored"
    assert block["degraded"] is False
    assert block["methods"] == [{"method": "lexical", "count": 1}]
    assert "findings" not in block, "running the configured checker produces no finding to report"


def test_a_checker_that_records_no_provenance_yields_nothing() -> None:
    """`method="unset"` means a checker was never taught to say what it is. Reporting it as a
    method would put a placeholder in a governance surface; inventing `nli` for it would assert
    provenance nobody recorded. Returning None says "no verification provenance here", which is
    true, and the answer's `verification` field stays absent."""
    assert verification_block([FaithfulnessVerdict(score=0.9, faithful=True)]) is None
    assert verification_block([]) is None
