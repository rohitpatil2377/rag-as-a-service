"""Unit tests for the deterministic heuristic query router (FR-D5)."""

from __future__ import annotations

from rag_core import HeuristicRouter
from rag_schemas import RetrievalQuery


def _q(text: str, top_k: int = 5) -> RetrievalQuery:
    return RetrievalQuery(tenant_id="t", text=text, top_k=top_k)


async def test_simple_query_routes_standard() -> None:
    d = await HeuristicRouter().route(_q("What is the uptime target?"))
    assert d.path == "standard"
    assert d.top_k == 5
    assert d.reason == "simple"


async def test_comparative_query_routes_heavy() -> None:
    d = await HeuristicRouter().route(_q("Compare the ITR and the assessment order and reconcile them"))
    assert d.path == "heavy"
    assert d.top_k > 5
    assert "comparative/multi-hop" in d.reason


async def test_multi_question_routes_heavy() -> None:
    d = await HeuristicRouter().route(_q("What is the target? And what is the peak? And the RPO?"))
    assert d.path == "heavy"
    assert "multi-question" in d.reason


async def test_heavy_top_k_capped() -> None:
    d = await HeuristicRouter().route(_q("compare everything", top_k=50))
    assert d.path == "heavy"
    assert d.top_k <= 12  # _HEAVY_TOPK_CAP
