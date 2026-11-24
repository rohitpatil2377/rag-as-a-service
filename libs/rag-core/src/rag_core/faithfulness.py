"""Faithfulness verification (FR-E4) — interfaces + pure logic (no I/O).

After a grounded answer is generated, verify it is actually *supported by* its cited
spans. The verdict feeds a three-way route (pure :func:`route_faithfulness`):

    score >= faithful_threshold          -> GROUNDED   (answer stands)
    flag_threshold <= score < faithful   -> FLAGGED_FOR_REVIEW (to HITL, not final)
    score < flag_threshold               -> ABSTAINED  (faithfulness too low)

Citations remain STRUCTURAL (built from retrieved spans), never scraped from model
text. The concrete checkers live in ``rag_clients`` (lexical + LLM-judge).

Known limitation (ADR-0005): an LLM-as-judge on the small dev model (1.5B/3B) is a
WEAK signal; the judge should be a distinct/stronger model once real vLLM lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag_schemas import RetrievedSpan
from rag_schemas.common import AnswerStatus


@dataclass(frozen=True, slots=True)
class FaithfulnessVerdict:
    """Outcome of a faithfulness check over an answer + its cited spans."""

    # `None` MEANS NOT JUDGED, and is not interchangeable with a number. A score of 1.0 when no
    # check ran would assert that nothing was unsupported having checked nothing; 0.0 would assert
    # the answer is unfaithful on the same non-evidence. Neither is true, so an `unavailable`
    # verdict carries neither -- the same rule the incumbent's aggregate faithfulness follows by
    # being reported as unknown rather than as either number. Every caller that routes on this
    # must treat None as "no judgement was made" and leave the answer's status alone.
    score: float | None  # fraction of the answer supported by the sources, [0,1]; None = not judged
    faithful: bool | None  # convenience flag: no unsupported claims; None = not judged
    unsupported_claims: list[str] = field(default_factory=list)
    # Slice 4: the NOTATION guard (unit/formula/SMILES verbatim-grounding) fired. Distinct from
    # `faithful=False` because the numeric checker also clears `faithful` and its routing must stay
    # unchanged: only a notation hit caps the routed score into the FLAGGED band (see
    # `notation_flag_band_score`), so a notation false-positive goes to human review — not GROUNDED
    # (the pre-cap behaviour, which had no teeth) and not full ABSTAIN.
    notation_flagged: bool = False
    # --- verification PROVENANCE ----------------------------------------------------------------
    #
    # WHY THESE EXIST. A verdict used to carry a score and a claims list and nothing about WHICH
    # checker produced them, so an answer scored by the lexical fallback after an NLI read error
    # was indistinguishable from an NLI-verified one -- the only trace a server log line no
    # reviewer reads. Found in a real run: `nli_scorer_failed_fallback_lexical` fired once across
    # 13 batch verifications, and that batch shipped inside an answer reported as verified.
    #
    # `"unset"`, not `"nli"`. A default that NAMED a method would assert provenance no checker
    # recorded, which is the defect these fields exist to fix arriving as its own default -- so a
    # checker that has not been taught to report provenance reads as `unset`, not as verified.
    # `method` is one of `nli`, `lexical_fallback`, `unavailable` or `lexical` (a deployment whose
    # configured checker IS the lexical one -- not a downgrade, so not degraded). `degraded_reason`
    # is a CODE, for routing and alerting: `nli_failed_fallback_used`,
    # `nli_failed_fallback_failed`, `nli_failed_fallback_unconfigured`. `degraded_detail` carries
    # the original exception text, because only that tells a one-off read error from a scorer that
    # has been down for an hour, and a code alone cannot be triaged.
    method: str = "unset"
    degraded: bool = False
    degraded_reason: str | None = None
    degraded_detail: str | None = None


@runtime_checkable
class FaithfulnessChecker(Protocol):
    """Scores whether an answer is entailed by its cited spans (FR-E4)."""

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        """``query`` is OPTIONAL (default "") and additive: a checker that does not need it
        may ignore it; a checker that does (e.g. the NLI backend's compound-query relevance
        gate) uses it to confirm a cited span actually answers what was asked, not just that
        it is not contradicted by it."""
        ...


def verification_block(verdicts: Sequence[FaithfulnessVerdict]) -> dict[str, Any] | None:
    """The provenance of one answer's verification, from the verdicts that produced it.

    ANSWERS FOUR QUESTIONS WITHOUT A SERVER LOG: which methods were used, how many batches used
    each, whether anything was degraded or unavailable, and which batches were affected.

    ``outcome`` IS A DISCRIMINANT, AND IT IS THE POINT OF THIS SHAPE. A consumer must branch on it
    before reading ``score``:

        scored       exactly one method produced every score, so one number is meaningful and
                     ``score`` is present. May still be ``degraded`` -- a run where every batch
                     fell back to the lexical checker has comparable scores AND a failed intended
                     mechanism, and those are different facts.
        mixed        more than one method. ``score`` is ABSENT, because an NLI score and a lexical
                     score are not the same measurement and a mean over them means nothing while
                     looking authoritative.
        unavailable  at least one batch was never checked. ``score`` is absent.

    OMITTING THE KEY IS NOT A SAFEGUARD, which is why the discriminant exists. In JavaScript an
    absent property is ``undefined``, and ``block.score ?? 0`` yields ``0`` for ``undefined``
    exactly as it does for ``null`` -- so a careless consumer turns "not comparable" into a
    failing grade either way. Omission is the cleaner representation; ``outcome`` is the thing
    that makes the misuse detectable.

    ``methods`` IS AN ORDERED LIST, not a map: it survives a round trip through any JSON tooling
    with its order intact, it reads as a distribution rather than as a lookup, and it is what a
    discriminated union in the console binds to. A UI showing "12 NLI, 1 lexical fallback" must
    not be able to show "NLI" because that was the majority.

    ``degraded`` is about whether the INTENDED mechanism was available, never about grounding. A
    degraded verification does not make an answer ungrounded -- its citations are intact either
    way -- and `lexical` (the configured checker) is NOT degraded while `lexical_fallback` (the
    one that ran because NLI failed) is.
    """
    judged = [v for v in verdicts if v.method != "unset"]
    if not judged:
        return None
    counts: dict[str, int] = {}
    for v in judged:
        counts[v.method] = counts.get(v.method, 0) + 1
    findings = [
        {"method": v.method, "reason_code": v.degraded_reason, "detail": v.degraded_detail}
        for v in judged
        if v.degraded
    ]
    unavailable = any(v.method == "unavailable" for v in judged)
    scores = [v.score for v in judged if v.score is not None]
    if unavailable:
        outcome = "unavailable"
    elif len(counts) > 1:
        outcome = "mixed"
    elif scores:
        outcome = "scored"
    else:
        outcome = "unavailable"
    block: dict[str, Any] = {
        "outcome": outcome,
        "degraded": bool(findings),
        "methods": [{"method": m, "count": n} for m, n in counts.items()],
        "batches": [
            {"method": v.method, "degraded": v.degraded, "score": v.score, "reason_code": v.degraded_reason}
            for v in judged
        ],
    }
    if findings:
        block["findings"] = findings
    if outcome == "scored":
        block["score"] = round(sum(scores) / len(scores), 4)
    return block


def improves_on(candidate: FaithfulnessVerdict, incumbent: FaithfulnessVerdict, *, tolerance: float = 0.0) -> bool:
    """Is ``candidate`` at least as faithful as ``incumbent``, within ``tolerance``?

    FALSE WHEN EITHER MADE NO JUDGEMENT, which is the whole reason this is a function rather than
    an inline comparison. Every re-prompt loop in the generation service -- self-correction, the
    coverage re-prompt, the truncation continuation -- accepts a rewritten answer only if it
    verifies at least as well as the one it replaces. If the checker was unavailable for either
    side there is no "as well as" to evaluate, and the two wrong answers are both available: treat
    an unscored candidate as an improvement and an unverified rewrite silently replaces a verified
    answer; treat an unscored INCUMBENT as unbeatable and a loop that could have improved it never
    runs. Rejecting the candidate keeps the answer that was actually checked, which is the only one
    of the three that cannot mislead a reader."""
    if candidate.score is None or incumbent.score is None:
        return False
    return candidate.score >= incumbent.score - tolerance


def route_unless_unjudged(score: float | None, faithful_threshold: float, flag_threshold: float) -> AnswerStatus | None:
    """:func:`route_faithfulness`, or None when there is no score to route on.

    A status of None is not a fourth answer status -- it means "this candidate was never judged",
    and every caller compares it against ``AnswerStatus.GROUNDED``, so None naturally fails that
    test and the candidate is dropped."""
    return None if score is None else route_faithfulness(score, faithful_threshold, flag_threshold)


def route_faithfulness(score: float, faithful_threshold: float, flag_threshold: float) -> AnswerStatus:
    """Pure three-way routing from a faithfulness score to an answer status."""
    if score >= faithful_threshold:
        return AnswerStatus.GROUNDED
    if score >= flag_threshold:
        return AnswerStatus.FLAGGED_FOR_REVIEW
    return AnswerStatus.ABSTAINED


def notation_flag_band_score(faithful_threshold: float, flag_threshold: float) -> float:
    """A score guaranteed to land in the FLAGGED band ``[flag_threshold, faithful_threshold)`` so a
    notation-ungrounded answer routes to FLAGGED_FOR_REVIEW (Slice 4). Conservative on purpose: a
    unit/formula/SMILES the sources do not contain is held for a human — never shipped as GROUNDED,
    never fully withheld (ABSTAINED). Used by the notation checker to cap its own routed score; the
    numeric-only path never calls this, so its routing is unchanged."""
    if flag_threshold >= faithful_threshold:  # degenerate gate config — the boundary still flags
        return flag_threshold
    return (faithful_threshold + flag_threshold) / 2.0


def build_faithfulness_prompt(answer: str, spans: list[RetrievedSpan]) -> str:
    """Prompt for the LLM-as-judge: return a strict JSON verdict, sources-only."""
    sources = "\n\n".join(f"[{i}] {s.text.strip()}" for i, s in enumerate(spans, start=1))
    return (
        "You are a strict fact-checker. Decide which sentences of the ANSWER are NOT "
        "supported by the SOURCES. Judge ONLY against the SOURCES; do not use outside "
        "knowledge. Respond with ONLY a JSON object of the form "
        '{"score": <fraction 0..1 of the answer supported>, "unsupported_claims": '
        '["<verbatim unsupported sentence>", ...]}.\n\n'
        f"SOURCES:\n{sources}\n\nANSWER:\n{answer.strip()}\n\nJSON:"
    )
