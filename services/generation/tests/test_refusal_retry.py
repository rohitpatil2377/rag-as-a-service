"""One re-framed retry when the model refuses despite STRONG retrieval (FR-E2). DEFAULT OFF.

The hypothesis -- that a refusal at high retrieval confidence is about phrasing rather than
evidence -- did not survive measurement (5/5 abstains for both the bare and the imperative
phrasing; the one earlier success was noise). It ships disabled, so these tests set the floor
explicitly rather than relying on the default.

The grounded-or-abstain contract is the thing to protect here: the retry may change how the
question is PHRASED, never what evidence answers it. So the sources are identical on both
attempts, a second refusal still abstains, and a refusal that agrees with weak retrieval is
never second-guessed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from svc_guardrails_eval import AuditService

from rag_clients.memory import InMemoryAuditSink
from rag_schemas import AnswerStatus, GroundedAnswer, RetrievalQuery, RetrievedSpan
from rag_schemas.chunk import Span

_SOURCES = [
    RetrievedSpan(
        chunk_id="c1",
        doc_id="d1",
        text="PBSOP006 is Radioactive Material Handling and Disposal.",
        score=0.95,
        scorer="cross_encoder",
        span=Span(start=0, end=54),
    )
]


class _Gateway:
    """Refuses every attempt until ``answer_on_attempt``, then answers."""

    def __init__(self, answer_on_attempt: int | None) -> None:
        self.attempts = 0
        self.prompts: list[str] = []
        self._answer_on = answer_on_attempt

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.attempts += 1
        self.prompts.append(prompt)
        if self._answer_on is not None and self.attempts >= self._answer_on:
            return "PBSOP006 is Radioactive Material Handling and Disposal [1]."
        return "I don't know."

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError  # this path never streams

    async def list_models(self) -> list[str]:
        return []


async def _answer(gw: _Gateway, *, max_score: float, query: str) -> GroundedAnswer:
    from svc_generation.service import GenerationService

    from rag_schemas import RetrievalResult

    svc = GenerationService(
        gw, AuditService(InMemoryAuditSink()), faithfulness_enabled=False, refusal_retry_min_score=0.5
    )
    return await svc.answer(
        RetrievalResult(query=RetrievalQuery(tenant_id="t", text=query), spans=_SOURCES, max_score=max_score),
        actor="tester",
        trace_id="tr",
    )


@pytest.mark.asyncio
async def test_a_refusal_on_strong_retrieval_is_retried_once_reframed(inmemory: None) -> None:
    gw = _Gateway(answer_on_attempt=2)
    ans = await _answer(gw, max_score=0.92, query="detailed summary analysis for all the 10 SOPs")
    assert gw.attempts == 2, "the re-framed attempt did not happen"
    assert ans.status is AnswerStatus.GROUNDED
    # The retry must carry the SAME evidence -- only the framing may change.
    assert "PBSOP006 is Radioactive" in gw.prompts[1]
    assert "Provide the following from the sources" in gw.prompts[1]


@pytest.mark.asyncio
async def test_a_refusal_on_weak_retrieval_is_never_second_guessed(inmemory: None) -> None:
    """Below the threshold the model is agreeing with the gate. Retrying there is how an
    abstain-by-design turns into a coaxed answer."""
    gw = _Gateway(answer_on_attempt=2)
    ans = await _answer(gw, max_score=0.3, query="detailed summary analysis for all the 10 SOPs")
    assert gw.attempts == 1
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.abstain_reason == "out_of_scope"


@pytest.mark.asyncio
async def test_an_already_explicit_question_is_not_retried(inmemory: None) -> None:
    gw = _Gateway(answer_on_attempt=2)
    ans = await _answer(gw, max_score=0.92, query="What are the ten SOPs?")
    assert gw.attempts == 1
    assert ans.status is AnswerStatus.ABSTAINED


@pytest.mark.asyncio
async def test_a_second_refusal_still_abstains(inmemory: None) -> None:
    """The retry is a re-phrasing, not a way to wear the model down into answering."""
    gw = _Gateway(answer_on_attempt=None)
    ans = await _answer(gw, max_score=0.92, query="detailed summary analysis for all the 10 SOPs")
    assert gw.attempts == 2
    assert ans.status is AnswerStatus.ABSTAINED
    assert ans.citations == []
