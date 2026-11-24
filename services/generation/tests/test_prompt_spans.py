"""Which passages actually reach the model (FR-E1 + contiguity recall)."""

from __future__ import annotations

from svc_generation.service import _prompt_spans

from rag_schemas import RetrievedSpan


def _span(cid: str, *, scored: bool) -> RetrievedSpan:
    if scored:
        return RetrievedSpan(chunk_id=cid, doc_id="d", text=cid, score=0.8, scorer="cross_encoder")
    return RetrievedSpan(chunk_id=cid, doc_id="d", text=cid)


def test_contiguity_neighbours_do_not_evict_the_passages_they_complete() -> None:
    """``top_k`` answers "how many passages did retrieval judge relevant". A neighbour was never
    judged — it is there because it sits beside one that was. Counting it against ``top_k`` lets
    expansion evict the ranked passage it was fetched to complete. Measured on a real handbook,
    that recovered one missing list item and pushed a different one out — still five of six."""
    spans = [
        _span("r1", scored=True),
        _span("n1", scored=False),
        _span("r2", scored=True),
        _span("n2", scored=False),
        _span("r3", scored=True),
    ]
    assert [s.chunk_id for s in _prompt_spans(spans, top_k=2)] == ["r1", "n1", "r2", "n2"]
    assert "r3" in [s.chunk_id for s in _prompt_spans(spans, top_k=3)]


def test_order_follows_the_document_not_the_ranking() -> None:
    """A split list must reach the model in the order it was written, or it reads item 5 before
    item 3 and reports the list wrongly."""
    spans = [_span("a", scored=False), _span("b", scored=True), _span("c", scored=False)]
    assert [s.chunk_id for s in _prompt_spans(spans, top_k=5)] == ["a", "b", "c"]


def test_without_expansion_the_behaviour_is_exactly_top_k() -> None:
    """The flag is default-off, so the unexpanded path must be byte-identical to what it replaced."""
    spans = [_span(f"r{i}", scored=True) for i in range(6)]
    assert [s.chunk_id for s in _prompt_spans(spans, top_k=3)] == ["r0", "r1", "r2"]
