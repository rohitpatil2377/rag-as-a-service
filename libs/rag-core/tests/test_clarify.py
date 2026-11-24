"""Unit tests for grounded query disambiguation (Layer-2 item 4) — pure logic."""

from __future__ import annotations

from rag_core.clarify import (
    HeuristicClarifyGenerator,
    build_clarify_prompt,
    cluster_spans,
    heuristic_clarify,
    parse_clarify,
)
from rag_schemas import RetrievedSpan


def _span(doc_id: str, text: str, score: float, section: str | None = None) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id=f"{doc_id}-{abs(hash(text)) % 1000}",
        doc_id=doc_id,
        text=text,
        score=score,
        scorer="lexical",
        section=section,
    )


def test_heuristic_clusters_yield_distinct_options() -> None:
    # Two DISTINCT documents (topics) → one grounded option each, highest-scoring cluster first.
    spans = [
        _span("doc-leave", "Annual leave accrues at 1.5 days per month.", 0.7, section="Leave Policy"),
        _span("doc-leave", "Unused leave lapses at year end.", 0.6, section="Leave Policy"),
        _span("doc-sla", "The uptime target is 99.9 percent.", 0.65, section="Service Levels"),
    ]
    out = heuristic_clarify("policy", spans, 3)
    assert len(out) == 2  # two distinct clusters → two options
    assert all(o.startswith("Did you mean:") and o.endswith("?") for o in out)
    assert len(set(out)) == len(out)  # options are distinct
    # Highest-scoring cluster (Leave Policy, 0.7) leads.
    assert "Leave Policy" in out[0]


def test_single_cluster_yields_no_options() -> None:
    # All spans from ONE document → not ambiguous → never fabricate a second topic.
    spans = [
        _span("doc-sla", "The uptime target is 99.9 percent.", 0.8, section="Service Levels"),
        _span("doc-sla", "Credits apply below the target.", 0.7, section="Service Levels"),
    ]
    assert heuristic_clarify("uptime", spans, 3) == []


def test_no_spans_yields_no_options() -> None:
    assert heuristic_clarify("anything", [], 3) == []


def test_cluster_spans_orders_by_best_score() -> None:
    spans = [
        _span("a", "alpha topic text", 0.3),
        _span("b", "beta topic text", 0.9),
        _span("a", "alpha again", 0.5),
    ]
    clusters = cluster_spans(spans)
    assert [c.doc_id for c in clusters] == ["b", "a"]
    assert clusters[1].best_score == 0.5  # best (max) score within the "a" cluster


def test_option_count_is_capped() -> None:
    spans = [_span(f"doc-{i}", f"topic number {i} content here", 0.5, section=f"Section {i}") for i in range(5)]
    assert len(heuristic_clarify("q", spans, 2)) == 2


def test_build_prompt_constrains_to_sources() -> None:
    p = build_clarify_prompt("what is it", [_span("d", "some grounded source text", 0.5)], 3)
    assert "ONLY" in p and "SOURCES:" in p and "Do NOT invent" in p


def test_parse_clarify_cleans_and_dedupes() -> None:
    raw = "1. Did you mean: leave?\n- Did you mean: leave?\n* Did you mean: uptime?\nnot an option\n"
    out = parse_clarify(raw, 3)
    assert out == ["Did you mean: leave?", "Did you mean: uptime?"]


async def test_heuristic_generator_matches_function() -> None:
    gen = HeuristicClarifyGenerator()
    spans = [
        _span("doc-a", "topic alpha content", 0.6, section="Alpha"),
        _span("doc-b", "topic beta content", 0.5, section="Beta"),
    ]
    assert await gen.suggest("q", spans, 3) == heuristic_clarify("q", spans, 3)
