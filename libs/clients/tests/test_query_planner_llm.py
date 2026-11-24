"""The LLM query planner: parses a plan from the gateway, and is fail-safe to the heuristic.

The adapter must NEVER raise: an unavailable model, a timeout, or garbage output all degrade to the
deterministic heuristic plan, so a planner failure can never 500 or mis-route a query.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rag_clients.query_planner_llm import LLMQueryPlanner
from rag_core.query_plan import PlanContext

_CTX = PlanContext(doc_count=1, doc_titles=("Employee Handbook",), has_history=False)


class _Gateway:
    """Returns whatever the model is meant to have produced."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        return self._reply

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return []


class _Broken:
    model_name = "broken"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        raise RuntimeError("gateway down")

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_returns_a_parsed_plan_from_the_gateway() -> None:
    reply = (
        '{"strategy":"whole_document","coverage":"exhaustive","target_doc":"Employee Handbook",'
        '"answer_shape":"summary","rationale":"comprehensive coverage of one document","confidence":0.88}'
    )
    plan = await LLMQueryPlanner(_Gateway(reply)).plan("summarise the key obligations in this document", _CTX)
    assert plan.strategy == "whole_document"
    assert plan.coverage == "exhaustive"
    assert plan.target_doc == "Employee Handbook"


@pytest.mark.asyncio
async def test_garbage_output_falls_back_to_heuristic_without_raising() -> None:
    # A pointed factual question → the heuristic fallback plan is "passages".
    plan = await LLMQueryPlanner(_Gateway("the model rambled and produced no json")).plan(
        "what is the notice period?", _CTX
    )
    assert plan.strategy == "passages"


@pytest.mark.asyncio
async def test_gateway_error_falls_back_to_heuristic_and_never_raises() -> None:
    # A catalog question → the heuristic fallback plan is "catalog"; the raising gateway is swallowed.
    plan = await LLMQueryPlanner(_Broken()).plan("what documents do we have?", _CTX)
    assert plan.strategy == "catalog"
