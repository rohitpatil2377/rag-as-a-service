"""Service-level tests for the strict, adaptive completeness guard (Slice 1).

Mirrors test_coverage_reprompt.py: all-fake collaborators, no infra, default suite. Covers the two
partial-answer failure modes (truncation continuation + coverage loop), the completeness signal, and
the flag-off byte-identical invariant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from svc_generation import GenerationService
from svc_guardrails_eval import AuditService

from rag_clients.memory import InMemoryAuditSink
from rag_core import FaithfulnessVerdict, GuardBudget
from rag_schemas import RetrievalQuery, RetrievalResult, RetrievedSpan

_LEAVE = "Annual leave is twenty four working days each year."
_RTO = "Recovery time objective is two hours or less."


class _Gateway:
    """Returns queued (text, finish_reason) turns in order (last repeats); counts calls.

    Implements complete_with_usage so the guard sees an authoritative finish_reason, and complete for
    parity — the service prefers complete_with_usage when present (as the real adapter does)."""

    def __init__(self, *turns: tuple[str, str | None]) -> None:
        self._turns = list(turns)
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    def _turn(self) -> tuple[str, str | None]:
        i = min(self.calls, len(self._turns) - 1)
        self.calls += 1
        return self._turns[i]

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        return self._turn()[0]

    async def complete_with_usage(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> tuple[str, dict[str, int | bool | str]]:
        text, fr = self._turn()
        usage: dict[str, int | bool | str] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if fr:
            usage["finish_reason"] = fr
        return text, usage

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["stub"]


class _Checker:
    """Table-driven faithfulness; unknown answers default to a clean 0.9."""

    def __init__(self, table: dict[str, tuple[float, list[str]]] | None = None) -> None:
        self._table = table or {}

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        score, unsupported = self._table.get(answer.strip(), (0.9, []))
        return FaithfulnessVerdict(score=score, faithful=not unsupported, unsupported_claims=unsupported)


def _retrieval() -> RetrievalResult:
    spans = [
        RetrievedSpan(chunk_id="a", doc_id="d", text=_LEAVE, score=0.9, scorer="cross_encoder"),
        RetrievedSpan(chunk_id="b", doc_id="d", text=_RTO, score=0.9, scorer="cross_encoder"),
    ]
    q = RetrievalQuery(tenant_id="t", text="list every leave and recovery rule", top_k=5)
    return RetrievalResult(query=q, spans=spans, max_score=0.9)


def _svc(gw: _Gateway, checker: _Checker, *, completeness: bool, coverage: bool = False) -> GenerationService:
    return GenerationService(
        gw,
        AuditService(InMemoryAuditSink()),
        checker,
        coverage_check_enabled=coverage,
        completeness_guard_enabled=completeness,
    )


# --- Truncation: finish_reason=length triggers a bounded continuation that concatenates + stays grounded ---
async def test_truncation_continuation_concatenates_and_stays_grounded() -> None:
    gw = _Gateway(
        (f"{_LEAVE} [1] Recovery time", "length"),  # initial, cut off mid-sentence
        ("objective is two hours or less. [2]", "stop"),  # continuation completes it
    )
    ans = await _svc(gw, _Checker(), completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert gw.calls == 2, "one continuation for a truncated answer"
    assert ans.status == "grounded"
    assert "two hours" in (ans.answer or ""), "the continuation text was concatenated"
    assert ans.completeness is not None
    assert ans.completeness.complete is True
    assert ans.completeness.truncated is False


async def test_long_list_answer_is_not_cut_off() -> None:
    # A "list every point" answer truncated after two of three points is continued to completion.
    spans = [
        RetrievedSpan(chunk_id="a", doc_id="d", text="Point one is alpha.", score=0.9, scorer="cross_encoder"),
        RetrievedSpan(chunk_id="b", doc_id="d", text="Point two is beta.", score=0.9, scorer="cross_encoder"),
        RetrievedSpan(chunk_id="c", doc_id="d", text="Point three is gamma.", score=0.9, scorer="cross_encoder"),
    ]
    q = RetrievalQuery(tenant_id="t", text="list every point", top_k=5)
    retrieval = RetrievalResult(query=q, spans=spans, max_score=0.9)
    gw = _Gateway(
        ("1. Point one is alpha. [1] 2. Point two is beta. [2] 3. Point three is", "length"),
        ("gamma. [3]", "stop"),
    )
    ans = await _svc(gw, _Checker(), completeness=True).answer(retrieval, actor="u", trace_id="tr")
    assert ans.status == "grounded"
    assert "gamma" in (ans.answer or "")
    assert ans.completeness is not None and ans.completeness.complete is True


# --- Coverage loop reaches completeness ---
async def test_coverage_loop_folds_in_the_omitted_point() -> None:
    complete_answer = f"{_LEAVE} [1] {_RTO} [2]"
    gw = _Gateway(
        (f"{_LEAVE} [1]", "stop"),  # initial: complete sentence, but omits the recovery point
        (complete_answer, "stop"),  # coverage re-prompt folds it in
    )
    ans = await _svc(gw, _Checker(), completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert gw.calls == 2
    assert ans.status == "grounded"
    assert "two hours" in (ans.answer or "")
    assert ans.completeness is not None
    assert ans.completeness.complete is True
    assert (ans.completeness.covered, ans.completeness.expected) == (2, 2)


# --- Coverage loop cannot resolve → mark complete=false (never ship a partial answer as whole) ---
async def test_unresolved_coverage_is_marked_incomplete() -> None:
    stubborn = f"{_LEAVE} [1]"  # every attempt keeps omitting the recovery point
    gw = _Gateway((stubborn, "stop"))
    ans = await _svc(gw, _Checker(), completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert ans.status == "grounded"
    assert ans.completeness is not None
    assert ans.completeness.complete is False
    assert ans.completeness.reason is not None and "not covered" in ans.completeness.reason
    assert (ans.completeness.covered, ans.completeness.expected) == (1, 2)


# --- Flag OFF: byte-identical, no extra call, no completeness field ---
async def test_flag_off_is_byte_identical() -> None:
    gw = _Gateway((f"{_LEAVE} [1] Recovery time", "length"))  # truncated, but guard is OFF
    ans = await _svc(gw, _Checker(), completeness=False).answer(_retrieval(), actor="u", trace_id="tr")
    assert gw.calls == 1, "flag OFF → no continuation, no coverage loop, no extra LLM call"
    assert ans.status == "grounded"
    assert ans.completeness is None
    assert not any(step.step in ("completeness", "coverage") for step in ans.agent_trace)


# --- Never regress: a continuation that lowers faithfulness is rejected, prior answer kept ---
async def test_regressing_continuation_is_rejected() -> None:
    initial = f"{_LEAVE} [1] Recovery time"
    bad_combined = f"{initial} objective is wrong. [2]"
    gw = _Gateway((initial, "length"), ("objective is wrong. [2]", "stop"))
    checker = _Checker({bad_combined.strip(): (0.3, ["objective is wrong. [2]"])})
    ans = await _svc(gw, checker, completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert ans.status == "grounded"
    assert "wrong" not in (ans.answer or ""), "the regressing continuation was not kept"
    assert ans.completeness is not None and ans.completeness.truncated is True  # still cut off → marked


# --- Fix 3: a coverage rewrite after an initial length-truncation is NOT marked truncated ---
async def test_coverage_rewrite_clears_stale_length_truncation() -> None:
    complete_answer = f"{_LEAVE} [1] {_RTO} [2]"
    gw = _Gateway(
        (f"{_LEAVE} [1]", "length"),  # initial: length-truncated AND omits the recovery point
        ("", "stop"),  # continuation yields nothing usable → truncation stays "length"
        (complete_answer, "stop"),  # coverage rewrite completes it (finish_reason=stop)
    )
    ans = await _svc(gw, _Checker(), completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert ans.status == "grounded"
    assert "two hours" in (ans.answer or "")
    assert ans.completeness is not None
    # Without the fix the stale "length" finish_reason would mark this COMPLETE rewrite as truncated.
    assert ans.completeness.truncated is False
    assert ans.completeness.complete is True


# --- Fix 1a: doc-summary mode (completeness_coverage=False) skips the coverage loop entirely ---
async def test_coverage_disabled_skips_the_coverage_loop() -> None:
    gw = _Gateway((f"{_LEAVE} [1]", "stop"))  # omits span b, but not truncated
    ans = await _svc(gw, _Checker(), completeness=True).answer(
        _retrieval(), actor="u", trace_id="tr", completeness_coverage=False
    )
    assert gw.calls == 1, "coverage disabled → no coverage re-prompt even though a point is omitted"
    assert ans.status == "grounded"
    assert ans.completeness is not None
    assert ans.completeness.complete is True  # omission does not count on the intentionally-lossy path
    assert (ans.completeness.covered, ans.completeness.expected) == (0, 0)
    assert not any(step.step == "coverage" for step in ans.agent_trace)


# --- Fix 1b: a shared GuardBudget caps total guard calls across multiple answer() invocations ---
async def test_shared_guard_budget_is_exhausted_across_requests() -> None:
    budget = GuardBudget(1)  # only ONE extra guard call for the whole "request" (e.g. a doc-summary fan-out)

    # Call 1 spends the single budgeted continuation.
    gw1 = _Gateway((f"{_LEAVE} [1] Recovery time", "length"), ("objective is two hours. [2]", "stop"))
    ans1 = await _svc(gw1, _Checker(), completeness=True).answer(
        _retrieval(), actor="u", trace_id="t1", guard_budget=budget
    )
    assert gw1.calls == 2  # initial + one continuation
    assert ans1.completeness is not None and ans1.completeness.complete is True

    # Call 2 shares the now-empty budget → the guard cannot continue, marks incomplete/exhausted.
    gw2 = _Gateway((f"{_LEAVE} [1] Recovery time", "length"))
    ans2 = await _svc(gw2, _Checker(), completeness=True).answer(
        _retrieval(), actor="u", trace_id="t2", guard_budget=budget
    )
    assert gw2.calls == 1, "no budget left → no continuation call"
    assert ans2.completeness is not None
    assert ans2.completeness.complete is False
    assert ans2.completeness.reason is not None and "budget" in ans2.completeness.reason


# --- Fix 4: continuation overlap is de-duplicated (no repeated tail) ---
async def test_continuation_overlap_is_deduped() -> None:
    gw = _Gateway(
        (f"{_LEAVE} [1] Recovery time", "length"),
        ("Recovery time objective is two hours. [2]", "stop"),  # re-emits the "Recovery time" tail
    )
    ans = await _svc(gw, _Checker(), completeness=True).answer(_retrieval(), actor="u", trace_id="tr")
    assert ans.status == "grounded"
    assert (ans.answer or "").count("Recovery time") == 1, "the re-emitted overlap was stripped"
    assert "objective is two hours" in (ans.answer or "")
