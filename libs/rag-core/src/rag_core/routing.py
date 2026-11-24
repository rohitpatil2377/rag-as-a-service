"""Adaptive query router (FR-D5) — interfaces + deterministic heuristic.

Routes each query to a retrieval *path*: the cheap **standard** hybrid path for simple
questions, or a **heavy** path (deeper retrieval; the seam D6 query-transformation plugs
into) for long / multi-question / comparative / multi-hop queries.

Pure-additive: no schema change. The decision rides the existing QUERY audit payload and
is gated by ``adaptive_router_enabled`` (default OFF → unchanged behaviour). See ADR-0008.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from rag_schemas import RetrievalQuery

Path = Literal["standard", "heavy"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Which path a query takes + the (possibly boosted) top_k + why."""

    path: Path
    top_k: int
    reason: str


@runtime_checkable
class QueryRouter(Protocol):
    async def route(self, query: RetrievalQuery) -> RouteDecision: ...


# Cheap, explainable signals of a "heavy" query.
_COMPLEX_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference",
    "differences",
    "relationship",
    "list all",
    "summari",
    "across",
    "trend",
    "breakdown",
    "and also",
    "as well as",
    "reconcile",
)
_HEAVY_WORD_COUNT = 18
_HEAVY_TOPK_CAP = 12


class HeuristicRouter:
    """Deterministic, no-LLM router. Fast, sovereign, explainable (the default)."""

    async def route(self, query: RetrievalQuery) -> RouteDecision:
        text = query.text.lower()
        signals: list[str] = []
        if len(query.text.split()) >= _HEAVY_WORD_COUNT:
            signals.append("long")
        if query.text.count("?") > 1:
            signals.append("multi-question")
        if any(m in text for m in _COMPLEX_MARKERS):
            signals.append("comparative/multi-hop")
        if signals:
            top_k = min(query.top_k + 3, _HEAVY_TOPK_CAP)
            return RouteDecision(path="heavy", top_k=top_k, reason="; ".join(signals))
        return RouteDecision(path="standard", top_k=query.top_k, reason="simple")


def build_route_prompt(query: RetrievalQuery) -> str:
    """Prompt for the optional LLM router. Returns exactly 'standard' or 'heavy'."""
    return (
        "Classify the retrieval difficulty of the QUESTION. Answer with ONLY one word: "
        "'heavy' if it is multi-part, comparative, multi-hop, or needs broad context; "
        "otherwise 'standard'.\n\n"
        f"QUESTION: {query.text}\n\nANSWER:"
    )
