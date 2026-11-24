"""HybridRetriever — dense + BM25 + RRF + rerank (FR-D1/D2/D7/D8).

This is the Phase-0 retrieval orchestrator, implemented against the Protocols so it
is store-agnostic. Flow (TDD §4.4):

    dense ANN (top-N) ─┐
                       ├─ RRF fuse ─ hydrate spans ─ rerank ─ top-K
    BM25 sparse (top-N)┘

Tenant + metadata filters are applied inside each store's ``search`` (pre-fusion), so
isolation holds at the store boundary, not just here.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.access import is_cleared, is_entitled
from rag_core.chunking import META_CATEGORY, META_DOC_DATE, META_SECTION
from rag_core.fusion import normalize_scores, reciprocal_rank_fusion
from rag_core.interfaces import (
    Embedder,
    MultiVectorEmbedder,
    MultiVectorStore,
    Reranker,
    SparseIndex,
    VectorStore,
)
from rag_core.parent_child import META_PARENT_ID
from rag_core.vision import VisionEmbedder
from rag_schemas import Chunk, RetrievalQuery, RetrievedSpan
from rag_schemas.common import RetrievalMode

_log = get_logger(__name__)

# Retrieve a wider candidate pool, then rerank down to top_k (TDD: top-50 -> top-K). Injected
# from config by the factory — rag-core stays pure and reads no settings of its own.
DEFAULT_CANDIDATE_POOL = 50
DEFAULT_RERANK_TOP_K = 20


class HybridRetriever:
    """Store-agnostic hybrid retriever. Wired with concrete adapters by the service."""

    # What this retriever ACTUALLY performs, always — dense + BM25 + RRF. Declared the way a
    # chunker declares its strategy, so callers report what ran instead of echoing what was
    # asked for. `RetrievalQuery.mode` is not read here and never has been; recording it as if
    # it were honoured put an untruth on the tamper-evident audit chain (P26).
    mode = RetrievalMode.HYBRID

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        reranker: Reranker,
        chunk_lookup: ChunkLookup,
        *,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        vision_embedder: VisionEmbedder | None = None,
        vision_store: VectorStore | None = None,
        multivector_enabled: bool = False,
    ) -> None:
        self._embedder = embedder
        self._vs = vector_store
        self._sparse = sparse_index
        self._reranker = reranker
        self._lookup = chunk_lookup
        self._pool = candidate_pool
        self._rerank_top_k = rerank_top_k
        # Cross-modal seam (additive, default-OFF): both None -> pure text retrieval, unchanged.
        self._vision_embedder = vision_embedder
        self._vision_store = vision_store
        # Slice-2b master flag, injected from config (rag-core reads no settings). The multivector
        # path additionally requires the embedder + store to carry the capability; OFF or a plain
        # embedder ⇒ the dense+BM25 path below is byte-identical.
        self._multivector_enabled = multivector_enabled

    async def retrieve(self, query: RetrievalQuery) -> tuple[list[RetrievedSpan], float]:
        """Return ``(reranked_spans, max_normalised_confidence)`` for the abstain gate."""
        tenant_id = query.tenant_id
        pool = max(query.top_k, self._pool)

        # Dense (+ optional bge-sparse / ColBERT re-scoring) legs, tenant-scoped + filtered.
        dense, bge_sparse, colbert_used = await self._dense_legs(query, pool)
        # BM25 stays (slice-2a text.exact intact); bge-sparse is a SEPARATE additional leg.
        sparse = await self._sparse.search(
            tenant_id,
            query.text,
            pool,
            query.filters,
            query.exact_boost,
            query.chem_terms,
            doc_ids=query.doc_ids,
            levels=query.levels,
        )

        # Cross-modal (additive/OFF): a THIRD leg over the CLIP image index, so a text query
        # retrieves an image by VISUAL similarity, not only its caption. Wired only when
        # vision_recall is enabled; a vision-backend fault degrades to text-only (logged), never
        # takes retrieval down — the leg is a bonus over the text path, not a dependency of it.
        visual: list[tuple[str, float]] = []
        if self._vision_embedder is not None and self._vision_store is not None:
            try:
                [qimg] = await self._vision_embedder.embed_text([query.text])
                # DELIBERATELY not level-scoped. This leg searches the CLIP image index, whose
                # points are images and carry no RAG v2 level at all, so passing a text grain
                # through would silently delete the whole cross-modal leg rather than narrow it.
                visual = await self._vision_store.search(tenant_id, qimg, pool, query.filters, doc_ids=query.doc_ids)
            except Exception as exc:  # blind by design: additive leg fails open to text-only
                _log.warning("cross_modal_leg_skipped", tenant_id=tenant_id, error=str(exc))

        # The dense leg is EITHER raw dense OR the ColBERT-reordered dense pool (never both — that
        # would double-count the same candidate source). bge-sparse and BM25 are separate legs.
        fused = reciprocal_rank_fusion([dense, bge_sparse, sparse, visual])
        if not fused:
            return [], 0.0
        norm = normalize_scores(fused)

        top_ids = [cid for cid, _ in fused[:pool]]
        chunks = await self._lookup.by_ids(tenant_id, top_ids)
        # Candidates leave here UNSCORED, carrying only their fusion rank. Nothing but a
        # reranker writes `score` — see RetrievedSpan.
        spans = [_to_span(chunks[cid], norm.get(cid, 0.0)) for cid in top_ids if cid in chunks]

        # FR-A6 permission-aware retrieval + banking clearance: drop spans the caller isn't
        # entitled to see (ACL group AND classification level), within the candidate pool
        # (before rerank/trim) so entitled results fill top_k.
        n_before = len(spans)
        spans = [s for s in spans if is_entitled(s.acl, query.acl) and is_cleared(s.classification, query.clearance)]
        acl_filtered = n_before - len(spans)

        keep = min(query.top_k, self._rerank_top_k)  # operator ceiling on prompt size
        reranked = await self._reranker.rerank(query.text, spans, keep)
        # The gate's input is the RERANKER's absolute relevance, never the min-max normalised
        # fusion score — that one makes the best candidate of every result set exactly 1.0.
        # It is the BEST score, not the first: the question the gate asks is "is any retrieved
        # span relevant enough to ground an answer?", and the Reranker protocol never promised
        # to return its list sorted.
        max_conf = max((s.score for s in reranked), default=0.0)
        _log.info(
            "hybrid_retrieve",
            tenant_id=tenant_id,
            dense=len(dense),
            bge_sparse=len(bge_sparse),
            colbert_rescored=colbert_used,
            sparse=len(sparse),
            visual=len(visual),
            fused=len(fused),
            acl_filtered=acl_filtered,
            returned=len(reranked),
        )
        return reranked, max_conf

    async def _dense_legs(
        self, query: RetrievalQuery, pool: int
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], bool]:
        """Build the dense leg (+ optional bge-sparse leg) for the query.

        Legacy / dense-only KBs: one ``embed`` + ``search`` — byte-identical to before. A Slice-2b
        KB (multivector enabled, embedder + store carry the capability): ONE ``embed_multi`` on the
        query, then the dense leg is the named-``dense`` ANN, optionally REORDERED by ColBERT MaxSim
        over the prefetched pool, plus an optional learned-sparse leg. Returns
        ``(dense_leg, bge_sparse_leg, colbert_used)``."""
        tenant_id = query.tenant_id
        embedder = self._embedder
        store = self._vs
        if not (
            self._multivector_enabled
            and isinstance(embedder, MultiVectorEmbedder)
            and isinstance(store, MultiVectorStore)
        ):
            [qvec] = await embedder.embed([query.text])
            return (
                await store.search(tenant_id, qvec, pool, query.filters, doc_ids=query.doc_ids, levels=query.levels),
                [],
                False,
            )

        want_sparse = query.multivector_sparse
        want_colbert = query.multivector_colbert
        [mv] = await embedder.embed_multi([query.text], dense=True, sparse=want_sparse, colbert=want_colbert)
        colbert_used = False
        if want_colbert and mv.colbert:
            prefetch = query.colbert_prefetch or pool
            dense = await store.search_colbert(
                tenant_id,
                mv.dense,
                mv.colbert,
                pool,
                prefetch=prefetch,
                filters=query.filters,
                doc_ids=query.doc_ids,
                levels=query.levels,
            )
            colbert_used = True
        else:
            dense = await store.search_dense(
                tenant_id, mv.dense, pool, query.filters, doc_ids=query.doc_ids, levels=query.levels
            )
        bge_sparse = (
            await store.search_sparse(
                tenant_id, mv.sparse, pool, query.filters, doc_ids=query.doc_ids, levels=query.levels
            )
            if want_sparse
            else []
        )
        return dense, bge_sparse, colbert_used


class ChunkLookup:
    """Resolve chunk_ids -> Chunk records (from the catalog/object store)."""

    def __init__(self, resolver: ChunkResolver) -> None:
        self._resolver = resolver

    async def by_ids(self, tenant_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        return await self._resolver(tenant_id, chunk_ids)


# A resolver is any async callable (tenant_id, ids) -> {id: Chunk}.
from collections.abc import Awaitable, Callable  # noqa: E402

ChunkResolver = Callable[[str, list[str]], Awaitable[dict[str, Chunk]]]


def _to_span(chunk: Chunk, fusion_score: float) -> RetrievedSpan:
    """Hydrate a fused candidate. It gets its fusion rank and NOT a `score`: the fusion value is
    min-max normalised, so seeding `score` with it would hand the gate a confidence of 1.0 for
    the best candidate of every result set, however irrelevant."""
    return RetrievedSpan(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        fusion_score=fusion_score,
        page=chunk.page,
        section=chunk.metadata.get(META_SECTION),
        category=chunk.metadata.get(META_CATEGORY),
        # The document's effective date rides on every chunk (stamped at ingest), so version
        # resolution costs no extra lookup at query time.
        doc_date=chunk.metadata.get(META_DOC_DATE),
        span=chunk.span,
        source_uri=chunk.source_uri,
        acl=list(chunk.acl),
        classification=chunk.classification,
        modality=chunk.modality,
        # Parent/child (FR-C2): a child stamped with its parent's id lets _parent_expand fetch the
        # context bank at query time. None for legacy/non-parent-child chunks (feature no-ops).
        parent_id=chunk.metadata.get(META_PARENT_ID),
    )
