"""HybridRetriever — what the orchestrator guarantees about the number the gate reads.

`HybridRetriever` had no unit coverage at all, which is how the score semantics drifted:
the confidence the abstain gate consumes is produced here, in one line, from whatever the
reranker returned. These tests pin that line rather than the reranker's own behaviour
(that lives in `libs/clients/tests/test_reranker.py`).
"""

from __future__ import annotations

import asyncio

from rag_core.hybrid import ChunkLookup, HybridRetriever
from rag_core.interfaces import Reranker
from rag_schemas import Chunk, RetrievalQuery, RetrievedSpan
from rag_schemas.chunk import Span

_TENANT = "acme"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id=_TENANT, text=text, span=Span(start=0, end=len(text)))


_CHUNKS = {f"c{i}": _chunk(f"c{i}", f"passage number {i}") for i in range(3)}


class _Embedder:
    dim = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Store:
    """Serves the same candidates on both legs, so RRF has something to fuse."""

    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    async def search(self, tenant_id: str, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return self._hits


async def _lookup(tenant_id: str, ids: list[str]) -> dict[str, Chunk]:
    return {cid: _CHUNKS[cid] for cid in ids if cid in _CHUNKS}


class _UnsortedReranker:
    """Scores honestly but returns the list in its incoming order. The `Reranker` protocol
    (interfaces.py:78) never promised sorted output, and a batching implementation is free
    to return whatever order it likes."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        return [s.model_copy(update={"score": self._scores[s.chunk_id]}) for s in spans][:top_k]


def _retriever(reranker: Reranker) -> HybridRetriever:
    hits = [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)]
    return HybridRetriever(
        _Embedder(),
        _Store(hits),  # type: ignore[arg-type]  # *args search signature, not the protocol's
        _Store(hits),  # type: ignore[arg-type]
        reranker,
        ChunkLookup(_lookup),
    )


def _query() -> RetrievalQuery:
    return RetrievalQuery(tenant_id=_TENANT, text="which passage is relevant?", top_k=3)


class _Recording:
    """Captures what the retriever handed it, then scores normally."""

    def __init__(self) -> None:
        self.seen: list[RetrievedSpan] = []

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        self.seen = list(spans)
        return [s.model_copy(update={"score": 0.7, "scorer": "cross_encoder"}) for s in spans][:top_k]


class _Bypass:
    """A reranker that returns its input untouched — a stand-in for every way the cross-encoder
    can be skipped (an empty-token query, a future adapter, a protocol implementation that
    forgets to score)."""

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        return spans[:top_k]


def test_candidates_reach_the_reranker_unscored_with_fusion_kept_aside() -> None:
    """The fusion score must never occupy `score`, even temporarily. min-max normalisation
    (fusion.py:32) makes the best candidate of EVERY result set exactly 1.0 — and every
    candidate 1.0 when the set is uniform — so a fusion value sitting in `score` is a
    confidence of 1.0 waiting for something to read it."""
    rec = _Recording()
    asyncio.run(_retriever(rec).retrieve(_query()))
    assert rec.seen, "the reranker must actually have been called"
    assert all(s.score == 0.0 and s.scorer == "unscored" for s in rec.seen)
    assert any(s.fusion_score > 0.0 for s in rec.seen), "fusion order must still be carried, just not in `score`"


def test_a_reranker_that_does_not_score_produces_no_confidence() -> None:
    """The regression pin for the whole class: if nothing scores the spans, the gate must see
    0.0 and abstain. Previously a bypass returned spans still carrying the normalised fusion
    score, so the gate read 1.0 and opened on a result set nothing had judged."""
    spans, max_conf = asyncio.run(_retriever(_Bypass()).retrieve(_query()))
    assert spans, "the bypass still returns candidates — it just does not score them"
    assert max_conf == 0.0
    assert all(s.scorer == "unscored" for s in spans)


def test_the_confidence_is_the_best_score_not_the_first_one() -> None:
    """The gate asks "is ANY retrieved span relevant enough to ground an answer?". Reading
    position 0 answers a different question — "is the span the reranker happened to put
    first relevant enough?" — and silently under-reports whenever the list is not sorted."""
    scores = {"c0": 0.20, "c1": 0.95, "c2": 0.40}
    _, max_conf = asyncio.run(_retriever(_UnsortedReranker(scores)).retrieve(_query()))
    assert max_conf == 0.95


# --- P26: the mode recorded is the mode that ran ------------------------------------------------


class _RecordingSparse:
    """A sparse leg that records the positional args it was called with (Slice 2a threading)."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def search(self, tenant_id: str, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        self.calls.append((tenant_id, *args))
        return [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)]


def test_hybrid_threads_the_query_exact_boost_into_the_sparse_leg() -> None:
    """Slice 2a per-KB adaptive: the boost the pipeline stamps on the query must reach the sparse
    index's search, so a per-KB value can weight exact technical-token matches."""
    sparse = _RecordingSparse()
    hits = [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)]
    retriever = HybridRetriever(
        _Embedder(),
        _Store(hits),  # type: ignore[arg-type]
        sparse,  # type: ignore[arg-type]
        _Bypass(),
        ChunkLookup(_lookup),
    )
    query = RetrievalQuery(tenant_id=_TENANT, text="SOP-BULK-001", top_k=3, exact_boost=7.5)
    asyncio.run(retriever.retrieve(query))
    assert sparse.calls, "the sparse leg must have been called"
    # search(tenant_id, query.text, pool, filters, exact_boost) -> exact_boost is the 5th positional.
    assert sparse.calls[0][4] == 7.5


def test_query_exact_boost_defaults_to_none() -> None:
    """Flag-off / unset: the query carries no boost, so the sparse adapter uses its global default."""
    assert RetrievalQuery(tenant_id=_TENANT, text="x", top_k=3).exact_boost is None


def test_the_retriever_declares_the_mode_it_actually_performs() -> None:
    """`RetrievalQuery.mode` was threaded through and echoed back as `mode_used`, and the
    tamper-evident QUERY audit event recorded it — but HybridRetriever never reads it and always
    runs dense + BM25 + RRF. So a request for "dense" produced a hash-chained log entry
    faithfully recording something that did not happen. A retriever must state what it does, the
    way a chunker states its strategy."""
    from rag_core.hybrid import HybridRetriever
    from rag_schemas.common import RetrievalMode

    assert HybridRetriever.mode is RetrievalMode.HYBRID
