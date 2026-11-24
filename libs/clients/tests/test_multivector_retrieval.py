"""Slice-2b retrieval wiring — the HybridRetriever multivector legs over the in-memory twins.

Pins the three safety invariants the design calls decisive:
  * ColBERT MaxSim NEVER becomes a span's ``score`` — the reranker stays the sole score writer and
    the abstain gate reads only that (Decision 1);
  * bge learned-sparse is a SEPARATE fused leg, not a BM25 replacement (Decision 2);
  * grounded-or-abstain is untouched — an unscored pool abstains even with the legs on;
and that flag-OFF takes the plain dense path (no embed_multi call).
"""

from __future__ import annotations

import asyncio

from rag_clients.memory import InMemoryVectorStore, LocalHashEmbedder
from rag_core import MultiVectorOutput
from rag_core.hybrid import ChunkLookup, HybridRetriever
from rag_schemas import Chunk, RetrievalQuery, RetrievedSpan
from rag_schemas.chunk import Span

_TENANT = "acme"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id=_TENANT, text=text, span=Span(start=0, end=len(text)))


_DOCS = {
    "c0": _chunk("c0", "alpha beta gamma delta"),
    "c1": _chunk("c1", "alpha epsilon zeta eta"),
    "c2": _chunk("c2", "theta iota kappa lambda"),
}


async def _lookup(tenant_id: str, ids: list[str]) -> dict[str, Chunk]:
    return {cid: _DOCS[cid] for cid in ids if cid in _DOCS}


class _NoBM25:
    """Isolates the bge learned-sparse leg: the OpenSearch BM25 leg returns nothing here."""

    async def search(self, tenant_id: str, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class _RecordingReranker:
    """The sole score writer: records what it was handed, then stamps one cross-encoder score."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.seen: list[RetrievedSpan] = []

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        self.seen = list(spans)
        return [s.model_copy(update={"score": self._score, "scorer": "cross_encoder"}) for s in spans][:top_k]


class _Bypass:
    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        return spans[:top_k]


class _SpyEmbedder:
    """Wraps the twin embedder to count embed vs embed_multi calls."""

    def __init__(self) -> None:
        self._e = LocalHashEmbedder(dim=64)
        self.embed_calls = 0
        self.embed_multi_calls = 0

    @property
    def dim(self) -> int:
        return self._e.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return await self._e.embed(texts)

    async def embed_multi(self, texts: list[str], **kw: object) -> list[MultiVectorOutput]:
        self.embed_multi_calls += 1
        return await self._e.embed_multi(texts, **kw)  # type: ignore[arg-type]


def _seed() -> tuple[InMemoryVectorStore, LocalHashEmbedder]:
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    outs = asyncio.run(emb.embed_multi([c.text for c in _DOCS.values()]))
    asyncio.run(vs.upsert_multi(_TENANT, list(_DOCS.values()), outs))
    return vs, emb


def _retriever(vs: InMemoryVectorStore, embedder: object, reranker: object, *, mv: bool) -> HybridRetriever:
    return HybridRetriever(
        embedder,  # type: ignore[arg-type]
        vs,
        _NoBM25(),  # type: ignore[arg-type]
        reranker,  # type: ignore[arg-type]
        ChunkLookup(_lookup),
        multivector_enabled=mv,
    )


def test_flag_off_takes_the_plain_dense_path() -> None:
    vs, _ = _seed()
    spy = _SpyEmbedder()
    r = _retriever(vs, spy, _RecordingReranker(0.5), mv=False)
    q = RetrievalQuery(tenant_id=_TENANT, text="alpha beta", top_k=3)
    spans, _ = asyncio.run(r.retrieve(q))
    assert spans
    assert spy.embed_calls == 1 and spy.embed_multi_calls == 0  # dense-only path, byte-identical


def test_colbert_rescores_and_never_writes_score() -> None:
    vs, _ = _seed()
    spy = _SpyEmbedder()
    rr = _RecordingReranker(0.42)
    r = _retriever(vs, spy, rr, mv=True)
    q = RetrievalQuery(
        tenant_id=_TENANT, text="alpha beta gamma", top_k=3, multivector_colbert=True, colbert_prefetch=10
    )
    spans, max_conf = asyncio.run(r.retrieve(q))
    assert spy.embed_multi_calls == 1  # ONE embed_multi on the query
    # Candidates reach the reranker unscored — the ColBERT MaxSim value (unbounded) never leaks in.
    assert rr.seen and all(s.score == 0.0 and s.scorer == "unscored" for s in rr.seen)
    # The gate reads ONLY the reranker's number, not any MaxSim value.
    assert max_conf == 0.42
    assert all(s.scorer == "cross_encoder" for s in spans)


def test_bge_sparse_is_a_separate_leg_from_bm25() -> None:
    vs, _ = _seed()
    emb = LocalHashEmbedder(dim=64)
    r = _retriever(vs, emb, _RecordingReranker(0.5), mv=True)
    # BM25 returns nothing (_NoBM25). A shared rare token must still surface via the bge-sparse leg.
    q = RetrievalQuery(tenant_id=_TENANT, text="kappa", top_k=3, multivector_sparse=True)
    spans, _ = asyncio.run(r.retrieve(q))
    assert any(s.chunk_id == "c2" for s in spans)  # only c2 holds "kappa"


def test_grounded_or_abstain_untouched_with_legs_on() -> None:
    vs, _ = _seed()
    emb = LocalHashEmbedder(dim=64)
    r = _retriever(vs, emb, _Bypass(), mv=True)  # nothing scores the spans
    q = RetrievalQuery(
        tenant_id=_TENANT,
        text="alpha beta gamma",
        top_k=3,
        multivector_sparse=True,
        multivector_colbert=True,
        colbert_prefetch=10,
    )
    spans, max_conf = asyncio.run(r.retrieve(q))
    assert spans  # candidates are returned...
    assert max_conf == 0.0  # ...but an unscored pool abstains, legs or no legs
    assert all(s.scorer == "unscored" for s in spans)
