"""Agentic query-planner adapter (implements :class:`~rag_core.query_plan.QueryPlanner`).

The real semantic path: one small planning call to the KB's sovereign gateway, prompted by
:func:`~rag_core.query_plan.build_query_plan_prompt` and parsed by
:func:`~rag_core.query_plan.parse_query_plan`. Fail-safe by construction — an unavailable
model, a timeout, or malformed output all return the HEURISTIC fallback plan, and the call NEVER
raises. Mirrors ``routing.py`` (LLMRouter) and ``intent_classifier.py`` (LLMIntentClassifier).
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.interfaces import LLMGateway
from rag_core.query_plan import (
    HeuristicQueryPlanner,
    PlanContext,
    QueryPlan,
    build_query_plan_prompt,
    parse_query_plan,
)

_log = get_logger(__name__)


class LLMQueryPlanner:
    """LLM-planned execution plan via the sovereign gateway — one small JSON call, fail-safe."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 200) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens
        self._fallback = HeuristicQueryPlanner()

    async def plan(self, query: str, ctx: PlanContext) -> QueryPlan:
        # The deterministic plan is both the fallback AND the parse safety net: any malformed field
        # in the model's reply degrades to it, so a bad completion is exactly today's behaviour.
        fallback = await self._fallback.plan(query, ctx)
        try:
            raw = await self._gw.complete(
                build_query_plan_prompt(query, ctx), max_tokens=self._max_tokens, temperature=0.0
            )
        except Exception as exc:  # fail-safe: a model error must never 500 or mis-route a query
            _log.warning("query_plan_failed_fallback_heuristic", error=str(exc)[:160])
            return fallback
        return parse_query_plan(raw, fallback=fallback)
