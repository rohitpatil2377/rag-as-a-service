"""HybridRetriever must hand the document scope to EVERY leg it fans out to.

`RetrievalQuery.doc_ids` is the record-field predicate that replaced scoping-by-metadata-filter
(a `doc_id` filter resolved to `meta.doc_id` on the sparse index and matched nothing, so a
document-scoped query silently ran dense-only). The store adapters now implement the scope, and
`libs/clients/tests/test_document_scope.py` pins that. What those tests cannot see is the
*threading*: deleting a single `doc_ids=query.doc_ids` in `hybrid.py` would widen exactly one leg
back to the whole KB and leave every other test green — the same shape of silent failure.

So these tests assert the plumbing itself, per leg, with recording doubles: BM25, the plain dense
leg, the cross-modal vision leg, and the three multivector legs (named-dense, learned-sparse,
ColBERT). They assert nothing about ranking — that is `test_hybrid.py`'s job.
"""

from __future__ import annotations

import asyncio
from typing import cast

from rag_core.hybrid import ChunkLookup, HybridRetriever
from rag_core.interfaces import Embedder, MultiVectorOutput, Reranker, SparseIndex, VectorStore
from rag_core.vision import VisionEmbedder
from rag_schemas import Chunk, RetrievalQuery, RetrievedSpan
from rag_schemas.chunk import Span
from rag_schemas.common import ChunkLevel

_TENANT = "acme"
_SCOPE = ("wanted.pdf",)


def _chunk(cid: str) -> Chunk:
    text = f"passage {cid}"
    return Chunk(chunk_id=cid, doc_id="wanted.pdf", tenant_id=_TENANT, text=text, span=Span(start=0, end=len(text)))


_CHUNKS = {"c0": _chunk("c0")}
_HITS = [("c0", 0.9)]


async def _lookup(tenant_id: str, ids: list[str]) -> dict[str, Chunk]:
    return {cid: _CHUNKS[cid] for cid in ids if cid in _CHUNKS}


class _Embedder:
    dim = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _MultiEmbedder(_Embedder):
    """Carries `embed_multi`, so `isinstance(..., MultiVectorEmbedder)` holds (runtime_checkable)."""

    async def embed_multi(
        self, texts: list[str], *, dense: bool = True, sparse: bool = False, colbert: bool = False
    ) -> list[MultiVectorOutput]:
        return [
            MultiVectorOutput(
                dense=[0.1, 0.2, 0.3],
                sparse={1: 0.5} if sparse else {},
                colbert=[[0.1, 0.2, 0.3]] if colbert else [],
            )
            for _ in texts
        ]


class _RecordingStore:
    """A VectorStore double that records the `doc_ids` and `levels` each leg was called with.

    Phase 8's grain predicate threads exactly as the document scope does, through the same
    call sites, so it is recorded here rather than in a parallel file — a deleted
    `levels=query.levels` is the same one-leg-widens defect as a deleted `doc_ids=`."""

    def __init__(self) -> None:
        self.calls: dict[str, tuple[str, ...] | None] = {}
        self.levels: dict[str, tuple[ChunkLevel, ...]] = {}

    async def search(
        self,
        tenant_id: str,
        vector: list[float],
        top_k: int,
        filters: object = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        self.calls["search"] = doc_ids
        self.levels["search"] = levels
        return _HITS


class _RecordingMultiStore(_RecordingStore):
    """Adds the three Slice-2b methods, so `isinstance(..., MultiVectorStore)` holds."""

    async def search_dense(
        self,
        tenant_id: str,
        vector: list[float],
        top_k: int,
        filters: object = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        self.calls["search_dense"] = doc_ids
        self.levels["search_dense"] = levels
        return _HITS

    async def search_sparse(
        self,
        tenant_id: str,
        sparse: dict[int, float],
        top_k: int,
        filters: object = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        self.calls["search_sparse"] = doc_ids
        self.levels["search_sparse"] = levels
        return _HITS

    async def search_colbert(
        self,
        tenant_id: str,
        dense: list[float],
        colbert: list[list[float]],
        top_k: int,
        *,
        prefetch: int,
        filters: object = None,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        self.calls["search_colbert"] = doc_ids
        self.levels["search_colbert"] = levels
        return _HITS

    async def ensure_multivector_collection(self, tenant_id: str, dim: int) -> None:  # pragma: no cover - unused
        return None

    async def upsert_multi(self, tenant_id: str, chunks: list[Chunk], outputs: list[object]) -> None:
        return None  # pragma: no cover - unused


class _RecordingSparse:
    """A SparseIndex double that records the `doc_ids` the BM25 leg was called with."""

    def __init__(self) -> None:
        self.doc_ids: tuple[str, ...] | None = None
        self.levels: tuple[ChunkLevel, ...] | None = None

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        filters: object = None,
        exact_boost: float | None = None,
        chem_terms: tuple[str, ...] = (),
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        self.doc_ids = doc_ids
        self.levels = levels
        return _HITS


class _VisionEmbedder:
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Reranker:
    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        return [s.model_copy(update={"score": 0.8}) for s in spans][:top_k]


def _query(**kw: object) -> RetrievalQuery:
    return RetrievalQuery(tenant_id=_TENANT, text="which passage?", top_k=1, **kw)


def _build(
    embedder: object,
    dense: object,
    sparse: object,
    **kw: object,
) -> HybridRetriever:
    """One cast site for the doubles. They implement only what the retriever calls — the
    whole point of a double — so they are not nominal Protocol implementations."""
    vision_embedder = kw.pop("vision_embedder", None)
    vision_store = kw.pop("vision_store", None)
    return HybridRetriever(
        cast(Embedder, embedder),
        cast(VectorStore, dense),
        cast(SparseIndex, sparse),
        cast(Reranker, _Reranker()),
        ChunkLookup(_lookup),
        vision_embedder=cast("VisionEmbedder | None", vision_embedder),
        vision_store=cast("VectorStore | None", vision_store),
        multivector_enabled=bool(kw.pop("multivector_enabled", False)),
    )


# --- the plain dense + BM25 legs ---------------------------------------------------------------


def test_the_scope_reaches_both_the_dense_and_the_bm25_leg() -> None:
    dense, sparse = _RecordingStore(), _RecordingSparse()
    r = _build(_Embedder(), dense, sparse)
    asyncio.run(r.retrieve(_query(doc_ids=_SCOPE)))
    assert dense.calls["search"] == _SCOPE, "the dense leg lost the document scope"
    assert sparse.doc_ids == _SCOPE, "the BM25 leg lost the document scope — the original defect"


def test_the_grain_reaches_both_the_dense_and_the_bm25_leg() -> None:
    """Phase 8's `levels`, threaded through the same call sites as the document scope. A leg
    that lost it would answer a grain-scoped question from every grain — the same one-leg-widens
    failure the scope predicate was built to fix, with the same silence."""
    store, sparse = _RecordingStore(), _RecordingSparse()
    r = _build(_Embedder(), store, sparse)

    grain = (ChunkLevel.CHUNK, ChunkLevel.ROW)
    asyncio.run(r.retrieve(_query(levels=grain)))

    assert store.levels["search"] == grain
    assert sparse.levels == grain


def test_an_unscoped_query_passes_an_empty_grain_to_both_legs() -> None:
    store, sparse = _RecordingStore(), _RecordingSparse()
    r = _build(_Embedder(), store, sparse)

    asyncio.run(r.retrieve(_query()))

    assert store.levels["search"] == ()
    assert sparse.levels == ()


def test_the_grain_reaches_every_multivector_leg() -> None:
    store, sparse = _RecordingMultiStore(), _RecordingSparse()
    r = _build(_MultiEmbedder(), store, sparse, multivector_enabled=True)
    grain = (ChunkLevel.ROW,)

    asyncio.run(r.retrieve(_query(levels=grain, multivector_sparse=True)))
    assert store.levels["search_dense"] == grain
    assert store.levels["search_sparse"] == grain

    asyncio.run(r.retrieve(_query(levels=grain, multivector_colbert=True)))
    assert store.levels["search_colbert"] == grain


def test_an_unscoped_query_passes_an_empty_scope_to_both_legs() -> None:
    """Empty ⇒ every adapter takes its unchanged path, so this pins 'no scope' as well as 'scope'."""
    dense, sparse = _RecordingStore(), _RecordingSparse()
    r = _build(_Embedder(), dense, sparse)
    asyncio.run(r.retrieve(_query()))
    assert dense.calls["search"] == ()
    assert sparse.doc_ids == ()


# --- the cross-modal leg ------------------------------------------------------------------------


def test_the_scope_reaches_the_vision_leg() -> None:
    """The CLIP leg is a real VectorStore over chunks, so an unscoped third leg would reintroduce
    out-of-document spans into a scoped answer."""
    dense, sparse, vision = _RecordingStore(), _RecordingSparse(), _RecordingStore()
    r = _build(_Embedder(), dense, sparse, vision_embedder=_VisionEmbedder(), vision_store=vision)
    asyncio.run(r.retrieve(_query(doc_ids=_SCOPE)))
    assert vision.calls["search"] == _SCOPE, "the vision leg lost the document scope"


# --- the multivector legs -----------------------------------------------------------------------


def test_the_scope_reaches_the_named_dense_and_learned_sparse_legs() -> None:
    store, sparse = _RecordingMultiStore(), _RecordingSparse()
    r = _build(_MultiEmbedder(), store, sparse, multivector_enabled=True)
    asyncio.run(r.retrieve(_query(doc_ids=_SCOPE, multivector_sparse=True)))
    assert store.calls["search_dense"] == _SCOPE, "the named-dense leg lost the document scope"
    assert store.calls["search_sparse"] == _SCOPE, "the learned-sparse leg lost the document scope"


def test_the_scope_reaches_the_colbert_leg() -> None:
    store, sparse = _RecordingMultiStore(), _RecordingSparse()
    r = _build(_MultiEmbedder(), store, sparse, multivector_enabled=True)
    asyncio.run(r.retrieve(_query(doc_ids=_SCOPE, multivector_colbert=True)))
    assert store.calls["search_colbert"] == _SCOPE, "the ColBERT leg lost the document scope"
    assert "search_dense" not in store.calls, "ColBERT replaces the dense leg, it does not add to it"
