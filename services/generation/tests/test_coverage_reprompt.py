"""Coverage-aware re-prompt wiring in ``GenerationService.answer`` (flag-gated, safe).

Exercises the three behaviours that make the feature safe to ship:
* accept a re-prompt that adds a dropped point WITHOUT regressing faithfulness;
* reject one that regresses the faithfulness gate (keep the original, never downgrade);
* flag OFF ⇒ no extra LLM call and the answer is exactly what it was before.

All fakes — no infra, no real model — so this runs in the default (non-e2e) suite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from svc_generation import GenerationService
from svc_guardrails_eval import AuditService

from rag_clients.memory import InMemoryAuditSink
from rag_core.faithfulness import FaithfulnessVerdict
from rag_schemas import RetrievalQuery, RetrievalResult, RetrievedSpan

_LEAVE = "Annual leave is twenty four working days each year."
_RTO = "Recovery time objective is two hours after an incident."

_A1 = "Annual leave is twenty four working days each year. [1]"  # covers span 1, drops span 2
_A2_GOOD = "Annual leave is twenty four working days each year. [1] Recovery time objective is two hours. [2]"
_A2_BAD = (
    "Annual leave is twenty four working days each year. [1] Recovery time objective is two hours. [2] "
    "Refunds take twelve weeks to process."
)


class _Gateway:
    """Returns queued completions in order (last one repeats); counts calls."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        i = min(self.calls, len(self._answers) - 1)
        self.calls += 1
        return self._answers[i]

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["stub"]


class _Checker:
    """Scores answers from a fixed table (answer text -> (score, unsupported claims))."""

    def __init__(self, table: dict[str, tuple[float, list[str]]]) -> None:
        self._table = table

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        score, unsupported = self._table.get(answer.strip(), (0.9, []))
        return FaithfulnessVerdict(score=score, faithful=not unsupported, unsupported_claims=unsupported)


def _retrieval() -> RetrievalResult:
    spans = [
        RetrievedSpan(chunk_id="a", doc_id="d", text=_LEAVE, score=0.9, scorer="cross_encoder"),
        RetrievedSpan(chunk_id="b", doc_id="d", text=_RTO, score=0.9, scorer="cross_encoder"),
    ]
    q = RetrievalQuery(tenant_id="t", text="what are the leave and recovery rules?", top_k=5)
    return RetrievalResult(query=q, spans=spans, max_score=0.9)


def _svc(gw: _Gateway, checker: _Checker, *, coverage: bool) -> GenerationService:
    return GenerationService(gw, AuditService(InMemoryAuditSink()), checker, coverage_check_enabled=coverage)


async def test_reprompt_accepted_when_it_adds_a_dropped_point_without_regressing() -> None:
    gw = _Gateway(_A1, _A2_GOOD)
    checker = _Checker({_A1: (0.9, []), _A2_GOOD: (0.9, [])})  # both grounded, no regression
    ans = await _svc(gw, checker, coverage=True).answer(_retrieval(), actor="u", trace_id="tr")

    assert gw.calls == 2, "exactly one re-prompt when a drop is detected"
    assert ans.status == "grounded"
    assert "Recovery time objective" in (ans.answer or ""), "the dropped point was folded back in"
    assert any(step.step == "coverage" for step in ans.agent_trace)


async def test_reprompt_rejected_when_it_regresses_faithfulness() -> None:
    gw = _Gateway(_A1, _A2_BAD)
    # The re-prompt covers the point but drops the faithfulness score into the FLAGGED band.
    checker = _Checker({_A1: (0.9, []), _A2_BAD: (0.5, ["Refunds take twelve weeks to process."])})
    ans = await _svc(gw, checker, coverage=True).answer(_retrieval(), actor="u", trace_id="tr")

    assert gw.calls == 2, "the re-prompt was still attempted"
    assert ans.status == "grounded", "never downgrade a grounded answer"
    assert "Refunds" not in (ans.answer or ""), "a faithfulness-regressing re-prompt is discarded"
    assert "Recovery time objective" not in (ans.answer or ""), "the original answer is kept intact"


async def test_flag_off_makes_no_extra_call_and_keeps_the_original() -> None:
    gw = _Gateway(_A1, _A2_GOOD)
    checker = _Checker({_A1: (0.9, []), _A2_GOOD: (0.9, [])})
    ans = await _svc(gw, checker, coverage=False).answer(_retrieval(), actor="u", trace_id="tr")

    assert gw.calls == 1, "flag OFF => no coverage re-prompt, no extra LLM call"
    assert ans.status == "grounded"
    assert "Recovery time objective" not in (ans.answer or "")
    assert not any(step.step == "coverage" for step in ans.agent_trace), "no coverage trace step when OFF"


async def test_no_reprompt_when_the_answer_is_already_complete() -> None:
    gw = _Gateway(_A2_GOOD, _A1)  # first answer already covers both points
    checker = _Checker({_A2_GOOD: (0.9, [])})
    ans = await _svc(gw, checker, coverage=True).answer(_retrieval(), actor="u", trace_id="tr")

    assert gw.calls == 1, "no drop detected => no re-prompt"
    assert ans.status == "grounded"
