"""A verification DOWNGRADE must be visible in the verdict, not only in a log line.

FOUND IN A REAL RUN, not constructed. `a2incumbent_c1.applog`, one line among 597:

    {"error": "NLI checker at http://localhost:8085 did not answer /predict (ReadError: )",
     "event": "nli_scorer_failed_fallback_lexical", ...}

One of that run's 13 batch verifications was scored by the LEXICAL checker instead of NLI. The
`FaithfulnessVerdict` it produced is indistinguishable in shape from an NLI one -- same score,
same `faithful`, same `unsupported_claims`, no field naming the method -- so the answer, the
audit trail and every governance surface downstream represent a lexically-checked batch as
NLI-verified. The only trace is a warning in a server log that no reviewer reads.

THREE OUTCOMES MUST BE DISTINGUISHABLE, and today there are two shapes for three outcomes:

    NLI completed                       -> verified by NLI
    NLI failed, lexical fallback ran    -> verified by fallback, DEGRADED
    NLI failed and fallback failed      -> verification UNAVAILABLE

A fallback is not necessarily the wrong thing to do. Silently counting it as the thing it
replaced is. This is the same rule the composite-HITL work settled on for withheld evidence:
a missing or weaker measurement must stay visibly missing or weaker, never become a reassuring
success.

THESE WERE THE SPECIFICATION AND ARE NOW THE REGRESSION SUITE. They were written as strict
expected failures while a paired benchmark was running -- changing the verification path
mid-measurement would have meant early and later cycles ran different product code -- and every
one of them flipped to passing when the checker was taught to report provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from rag_clients.nli_faithfulness import EntailFn, NLIFaithfulnessChecker
from rag_core.faithfulness import FaithfulnessVerdict
from rag_schemas import RetrievedSpan


def _span(text: str) -> RetrievedSpan:
    return RetrievedSpan(chunk_id="c1", doc_id="d1", text=text)


def _unreachable_nli() -> EntailFn:
    """An NLI scorer that fails the way the live one did: a read error mid-request."""

    async def _fn(premise: str, hypothesis: str) -> float:
        raise httpx.ReadError("")

    return _fn


@dataclass
class _StubLexical:
    """The lexical checker, standing in. Returns a verdict that LOOKS like any other."""

    score: float = 0.9
    raises: bool = False

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        if self.raises:
            raise RuntimeError("lexical checker also unavailable")
        return FaithfulnessVerdict(score=self.score, faithful=True, unsupported_claims=[])


_ANSWER = "The uptime target is 99.5 percent. [1]"
_SPANS = [_span("The platform uptime target is 99.5 percent at baseline.")]


async def test_a_successful_nli_check_says_it_was_nli() -> None:
    """The baseline the other two are read against. Without this, 'degraded' has nothing to
    contrast with and a reader cannot tell a verified answer from an unverifiable one."""

    async def entail(premise: str, hypothesis: str) -> float:
        return 0.95

    verdict = await NLIFaithfulnessChecker(entail).check(_ANSWER, _SPANS)

    assert verdict.method == "nli"
    assert verdict.degraded is False


async def test_a_lexical_fallback_is_marked_degraded_and_names_the_failure() -> None:
    """The case that actually happened. The score may be perfectly good; what must not survive
    is the claim that NLI produced it. The original failure is retained because 'degraded' with
    no cause cannot be triaged -- a one-off ReadError and a scorer that has been down for an hour
    need different responses, and the distinction is only in the error."""
    checker = NLIFaithfulnessChecker(_unreachable_nli(), fallback=_StubLexical(score=0.9))

    verdict = await checker.check(_ANSWER, _SPANS)

    assert verdict.method == "lexical_fallback"
    assert verdict.degraded is True
    assert verdict.score == 0.9, "the fallback's own score still stands; only its provenance changes"
    assert verdict.degraded_reason == "nli_failed_fallback_used"
    assert verdict.degraded_detail and "ReadError" in verdict.degraded_detail, (
        "the CODE is for routing and alerting; the DETAIL is what tells a one-off read error "
        "from a scorer that has been down for an hour, and only the detail can do that"
    )


async def test_the_two_ways_of_being_unavailable_are_told_apart() -> None:
    """Both report `unavailable`, and they call for different operational actions: a fallback
    that FAILED means two services are down and someone is paged; a fallback that was never
    CONFIGURED means this deployment chose to run without one, which is a config review, not an
    incident. Collapsing them into one status loses which of those is happening."""
    failed = await NLIFaithfulnessChecker(_unreachable_nli(), fallback=_StubLexical(raises=True)).check(_ANSWER, _SPANS)
    unconfigured = await NLIFaithfulnessChecker(_unreachable_nli()).check(_ANSWER, _SPANS)

    assert failed.method == unconfigured.method == "unavailable"
    assert failed.degraded_reason == "nli_failed_fallback_failed"
    assert unconfigured.degraded_reason == "nli_failed_fallback_unconfigured"


async def test_both_checkers_failing_is_verification_unavailable_not_a_score() -> None:
    """The third outcome, and the one with no safe default. A score of 1.0 here would assert
    that nothing was unsupported, having checked nothing; a score of 0.0 would assert the answer
    is unfaithful on the same evidence. Neither is true, so the verdict must carry no score at
    all and say why -- exactly as the incumbent's aggregate faithfulness is UNKNOWN rather than
    either number."""
    checker = NLIFaithfulnessChecker(_unreachable_nli(), fallback=_StubLexical(raises=True))

    verdict = await checker.check(_ANSWER, _SPANS)

    assert verdict.method == "unavailable"
    assert verdict.degraded is True
    assert verdict.score is None, "a score here would be an assertion about claims nobody checked"
    assert verdict.faithful is None, "and so would a faithful flag"


async def test_no_fallback_configured_is_also_unavailable_rather_than_an_exception() -> None:
    """With no fallback at all the checker raises today, and the caller decides what that means
    -- which is how a verification failure becomes an answer-shaped outcome somewhere far from
    here. The outcome is the same as the case above and should be reported the same way."""
    checker = NLIFaithfulnessChecker(_unreachable_nli())

    verdict = await checker.check(_ANSWER, _SPANS)

    assert verdict.method == "unavailable"
    assert verdict.score is None
