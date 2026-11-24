"""Unit tests for query transformation + the clean merge/dedupe (FR-D6)."""

from __future__ import annotations

from rag_core import DecompositionTransformer, merge_retrieved
from rag_schemas import RetrievalQuery, RetrievedSpan


def _q(text: str) -> RetrievalQuery:
    return RetrievalQuery(tenant_id="t", text=text, top_k=5)


def _span(cid: str, score: float) -> RetrievedSpan:
    return RetrievedSpan(chunk_id=cid, doc_id="d", text=cid, score=score, scorer="cross_encoder")


async def test_decomposition_splits_comparative_and_keeps_original() -> None:
    subs = await DecompositionTransformer().transform(_q("Compare the uptime and latency targets"))
    assert "Compare the uptime and latency targets" in subs  # original kept
    assert any("uptime" in s.lower() for s in subs) and any("latency" in s.lower() for s in subs)


async def test_decomposition_multi_question() -> None:
    subs = await DecompositionTransformer().transform(_q("What is the target? And the peak? And the RPO?"))
    assert len(subs) >= 3


async def test_decomposition_simple_query_is_singleton() -> None:
    subs = await DecompositionTransformer().transform(_q("uptime target"))
    assert subs == ["uptime target"]


def test_merge_dedupes_by_chunk_keeping_best_score() -> None:
    # chunk 'a' appears in two sub-query results with different scores -> keep the max, once.
    r1 = [_span("a", 0.4), _span("b", 0.9)]
    r2 = [_span("a", 0.8), _span("c", 0.5)]
    merged, max_score = merge_retrieved([r1, r2], top_k=5)
    ids = [s.chunk_id for s in merged]
    assert ids.count("a") == 1  # de-duplicated
    assert next(s for s in merged if s.chunk_id == "a").score == 0.8  # best score kept
    assert ids[0] == "b" and max_score == 0.9  # sorted desc, max_score = top


def test_merge_respects_top_k() -> None:
    merged, _ = merge_retrieved([[_span(str(i), i / 10) for i in range(10)]], top_k=3)
    assert len(merged) == 3


def test_merge_diversifies_across_subqueries() -> None:
    # Topic A scores high (0.9+), topic B lower (0.4). A plain global top-3 would be all-A;
    # round-robin must surface topic B too (fair share for a "compare A and B" question).
    topic_a = [_span("a1", 0.95), _span("a2", 0.92), _span("a3", 0.90)]
    topic_b = [_span("b1", 0.45), _span("b2", 0.42)]
    merged, max_score = merge_retrieved([topic_a, topic_b], top_k=3)
    ids = [s.chunk_id for s in merged]
    assert ids[0] == "a1" and max_score == 0.95  # strongest evidence still leads
    assert any(i.startswith("b") for i in ids)  # topic B is represented, not crowded out

    # diversify=False keeps the old global-top-k behaviour (all topic A).
    flat, _ = merge_retrieved([topic_a, topic_b], top_k=3, diversify=False)
    assert all(s.chunk_id.startswith("a") for s in flat)


# --- reformulation parsing (P23) ---------------------------------------------------------------


def test_reasoning_is_never_scraped_for_reformulations() -> None:
    """A reasoning model's working is FULL of question-shaped lines it is only considering.
    Feeding those back into retrieval would search on the model's guesses."""
    from rag_core.transform import parse_reformulations

    raw = (
        "<think>\nMaybe the policies talk about last-minute changes?\n"
        "Could it be about seniority?\n</think>\n"
        "Can an unrehearsed speaker go on stage?\n"
        "What is the mandatory rehearsal policy?"
    )
    assert parse_reformulations(raw, 3) == [
        "Can an unrehearsed speaker go on stage?",
        "What is the mandatory rehearsal policy?",
    ]


def test_an_unfinished_reasoning_block_produces_nothing() -> None:
    from rag_core.transform import parse_reformulations

    assert parse_reformulations("<think>\nStill considering whether this is permitted?", 3) == []


def test_numbering_and_bullets_are_stripped() -> None:
    from rag_core.transform import parse_reformulations

    raw = "1. Is rehearsal mandatory?\n- Who signs off?\n* What is the deadline?\nNot a question."
    assert parse_reformulations(raw, 5) == [
        "Is rehearsal mandatory?",
        "Who signs off?",
        "What is the deadline?",
    ]


def test_grounding_passages_reach_the_prompt() -> None:
    """The measured difference between a useless reformulation and a working one.

    Without passages the model invents plausible vocabulary — "unscheduled segments", "event
    programs" — because it cannot see what the corpus calls things, and retrieval stayed at
    0.0084. With them it used "Content Curator", "event sign-off", "No-Change Rule", lifted
    from real text, and the answering chunk came back at 0.9976."""
    from rag_core.transform import build_reformulation_prompt

    q = RetrievalQuery(tenant_id="t", text="Is this permitted?")
    blind = build_reformulation_prompt(q, 3)
    grounded = build_reformulation_prompt(q, 3, ["The Content Curator issues final event sign-off."])

    assert "PASSAGES" not in blind
    assert "Content Curator issues final event sign-off" in grounded
    assert "ONLY vocabulary that appears in them" in grounded


def test_a_grounding_passage_is_truncated_not_dropped() -> None:
    """Several passages plus a reasoning model's own working have to fit one context window,
    so each is capped — but every passage is still shown, because the one that carries the
    right term is not necessarily the highest-scoring one."""
    from rag_core.transform import build_reformulation_prompt

    q = RetrievalQuery(tenant_id="t", text="Is this permitted?")
    prompt = build_reformulation_prompt(q, 3, ["A" * 5000, "rehearsal sign-off"])
    assert "rehearsal sign-off" in prompt
    assert "A" * 5000 not in prompt


def test_the_deterministic_transformer_ignores_context() -> None:
    """Splitting a question needs no knowledge of the corpus, so passing passages must not
    change what it returns — the Protocol widened, the behaviour did not."""
    import asyncio

    from rag_core.transform import DecompositionTransformer

    q = RetrievalQuery(tenant_id="t", text="What is the notice period and the leave entitlement?")
    plain = asyncio.run(DecompositionTransformer().transform(q))
    with_ctx = asyncio.run(DecompositionTransformer().transform(q, ["irrelevant passage"]))
    assert plain == with_ctx
