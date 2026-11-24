"""Query router adapters (implement :class:`QueryRouter`) — FR-D5.

* :class:`HeuristicRouter` (from rag-core) — deterministic default.
* :class:`LLMRouter` — opt-in LLM classification via the sovereign gateway (one small
  call per query; reserve for vLLM/GPU).
"""

from __future__ import annotations

from rag_common import ConfigError, get_settings
from rag_core import HeuristicRouter, RouteDecision, build_route_prompt
from rag_core.interfaces import LLMGateway
from rag_core.routing import _HEAVY_TOPK_CAP
from rag_schemas import RetrievalQuery


class LLMRouter:
    """LLM-classified routing (standard vs heavy). Sovereign gateway."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gw = gateway

    async def route(self, query: RetrievalQuery) -> RouteDecision:
        raw = (await self._gw.complete(build_route_prompt(query), max_tokens=4, temperature=0.0)).strip().lower()
        if "heavy" in raw:
            return RouteDecision(path="heavy", top_k=min(query.top_k + 3, _HEAVY_TOPK_CAP), reason="llm:heavy")
        return RouteDecision(path="standard", top_k=query.top_k, reason="llm:standard")


def build_query_router(gateway: LLMGateway | None = None) -> HeuristicRouter | LLMRouter:
    """Factory: heuristic (deterministic/default) or llm, per ``RAG_ROUTER_BACKEND``."""
    backend = get_settings().router_backend
    if backend == "heuristic":
        return HeuristicRouter()
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm router backend requires an LLM gateway")
        return LLMRouter(gateway)
    raise ConfigError(f"unknown router backend: {backend}")
