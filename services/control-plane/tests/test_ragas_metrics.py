"""Sovereign Ragas-style metric definitions — verified with deterministic fake judge/embedder
(no live model). Proves the maths of answer_relevancy / context_precision / context_recall +
the fail-closed verdict parsing and None-aware aggregation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from svc_control_plane.ragas_metrics import (
    RagasScores,
    aggregate,
    answer_relevancy,
    context_precision,
    context_recall,
)
from svc_control_plane.ragas_metrics import _verdict as verdict


def _judge_yes_if(marker: str) -> Callable[[str], Awaitable[str]]:
    """A fake judge that says YES when the marker text is in the prompt, else NO."""

    async def judge(prompt: str) -> str:
        return "YES" if marker in prompt else "NO"

    return judge


def test_verdict_first_token_and_fails_closed() -> None:
    assert verdict("YES") is True
    assert verdict("Yes.") is True
    assert verdict("YES, clearly supported") is True
    assert verdict("NO") is False
    assert verdict("NO, not attributable") is False
    assert verdict("this is not relevant, despite being useful") is False  # first token "this"
    assert verdict("maybe, unclear") is False  # ambiguous -> fail closed
    assert verdict("") is False


async def test_context_precision_rewards_useful_near_the_top() -> None:
    # Judge marks a context useful iff it contains "GOOD".
    judge = _judge_yes_if("GOOD")
    # Ranking: useful, useless, useful -> AP = (1/1 + 2/3) / 2 useful = 0.8333
    score = await context_precision("q", ["GOOD one", "junk", "GOOD two"], judge=judge)
    assert score == pytest.approx(0.8333, abs=1e-3)

    # All useful, in order -> perfect precision.
    assert await context_precision("q", ["GOOD a", "GOOD b"], judge=judge) == pytest.approx(1.0)
    # None useful -> 0.0 ; empty -> 0.0
    assert await context_precision("q", ["junk", "junk"], judge=judge) == 0.0
    assert await context_precision("q", [], judge=judge) == 0.0


async def test_context_recall_fraction_of_claims_supported() -> None:
    # Ground truth has two claims; the context supports only the first. The fake judge reads
    # the CLAIM half of the prompt (not the whole thing) and says YES iff the claim is about
    # the capital — so claim 1 is supported, claim 2 (population) is not -> 1/2.
    async def judge(prompt: str) -> str:
        claim = prompt.split("CLAIM:")[1].split("CONTEXT", maxsplit=1)[0]
        return "YES" if "capital" in claim else "NO"

    gt = "The capital is Paris. The population is ten million."
    score = await context_recall(gt, ["City facts: Paris is the capital of France."], judge=judge)
    assert score == pytest.approx(0.5)
    # No ground truth or no contexts -> 0.0
    assert await context_recall("", ["ctx"], judge=judge) == 0.0
    assert await context_recall(gt, [], judge=judge) == 0.0


async def test_answer_relevancy_uses_embedding_similarity() -> None:
    async def judge(_prompt: str) -> str:
        return "q1\nq2"  # two reverse-engineered questions

    async def embed_identical(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]  # question and generated qs are colinear -> cos 1

    r = await answer_relevancy("real question", "a real answer", judge=judge, embed=embed_identical)
    assert r == pytest.approx(1.0)

    async def embed_orthogonal(texts: list[str]) -> list[list[float]]:
        # question -> (1,0); generated -> (0,1): orthogonal -> cosine 0
        return [[1.0, 0.0]] + [[0.0, 1.0] for _ in texts[1:]]

    r2 = await answer_relevancy("q", "ans", judge=judge, embed=embed_orthogonal)
    assert r2 == pytest.approx(0.0)

    # Empty answer short-circuits to 0.0 (no judge/embed call needed).
    assert await answer_relevancy("q", "   ", judge=judge, embed=embed_identical) == 0.0


def test_aggregate_excludes_none_rows_from_each_mean() -> None:
    rows = [
        RagasScores(context_recall=1.0, faithfulness=0.9),
        RagasScores(context_recall=0.0, faithfulness=None),  # faithfulness missing here
    ]
    agg = aggregate(rows)
    assert agg.n == 2
    assert agg.context_recall_avg == pytest.approx(0.5)
    assert agg.faithfulness_avg == pytest.approx(0.9)  # averaged over the ONE row that had it
    assert agg.answer_relevancy_avg is None  # no row had it
