"""Sovereign Ragas harness core — scores a set of (question, contexts, answer, ground_truth,
faithfulness) tuples end to end with a deterministic fake judge/embedder (no live model)."""

from __future__ import annotations

import pytest
from svc_control_plane.ragas_eval import run_ragas_eval


async def _judge(prompt: str) -> str:
    low = prompt.lower()
    if "write" in low and "questions" in low:  # answer_relevancy question generation
        return "q1\nq2"
    if "CLAIM:" in prompt:  # context_recall
        claim = prompt.split("CLAIM:")[1].split("CONTEXT", maxsplit=1)[0]
        return "YES" if "capital" in claim.lower() else "NO"
    if "useful for answering" in low or "supports or contributes" in low:  # context_precision
        return "YES" if "GOOD" in prompt else "NO"
    return "NO"


async def _embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]  # colinear -> answer_relevancy cosine 1.0


async def test_run_ragas_eval_scores_and_aggregates() -> None:
    items: list[tuple[str, list[str], str, str | None, float | None]] = [
        ("what is the capital?", ["GOOD: Paris is the capital of France."], "Paris", "The capital is Paris.", 0.9),
    ]
    rep = await run_ragas_eval("kb1", items, judge=_judge, embed=_embed)

    assert rep.kb_id == "kb1"
    assert rep.aggregate.n == 1
    s = rep.rows[0].scores
    assert s.answer_relevancy == pytest.approx(1.0)  # colinear embeddings
    assert s.context_precision == pytest.approx(1.0)  # the single context is GOOD -> useful
    assert s.context_recall == pytest.approx(1.0)  # the one claim is about the capital -> supported
    assert s.faithfulness == pytest.approx(0.9)  # carried through
    assert rep.aggregate.context_recall_avg == pytest.approx(1.0)


async def test_run_ragas_eval_recall_none_without_ground_truth() -> None:
    # No ground truth -> context_recall is not computed (None), excluded from the aggregate.
    items: list[tuple[str, list[str], str, str | None, float | None]] = [("q", ["GOOD ctx"], "an answer", None, None)]
    rep = await run_ragas_eval("kb1", items, judge=_judge, embed=_embed)
    assert rep.rows[0].scores.context_recall is None
    assert rep.aggregate.context_recall_avg is None
