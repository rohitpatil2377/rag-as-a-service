"""Hand-rolled RAG interfaces (Protocols) — the self-owned contracts.

These Protocols are the seams that keep RAG as a Service free of RAG-framework lock-in
(ADR-0002) and make every backend swappable (Qdrant->Milvus, TEI->vLLM,
Postgres->OpenMetadata) as a pure adapter change. Services depend ONLY on these
Protocols; concrete implementations live in ``rag_clients``.

All methods are async and tenant-aware: implementations derive their namespace from
the passed ``tenant_id`` via :mod:`rag_common.tenancy` and never trust ambient
values for isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import IO, Protocol, runtime_checkable

from rag_core.graph import GraphCommunity, GraphNode, GraphStats
from rag_schemas import AuditEvent, Chunk, DocumentEnvelope, RetrievalQuery, RetrievedSpan
from rag_schemas.common import ChunkLevel, ChunkStrategy


@runtime_checkable
class Embedder(Protocol):
    """Self-hosted embedding service (TEI/vLLM). Sovereign: local endpoints only."""

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class MultiVectorOutput:
    """The three representations bge-m3 emits for one text from a single ``/embed`` call.

    ``dense`` is the SAME 1024-d vector the plain ``embed()`` returns (so retrieval/index paths and
    the abstain calibration are unchanged). ``sparse`` maps a token id → learned weight (bge-m3
    lexical, IDF-modified in Qdrant). ``colbert`` is the late-interaction multivector (N_tokens x
    dim). ``sparse``/``colbert`` are empty when the caller did not request them."""

    dense: list[float]
    sparse: dict[int, float] = field(default_factory=dict)
    colbert: list[list[float]] = field(default_factory=list)


@runtime_checkable
class MultiVectorEmbedder(Protocol):
    """Additive capability on top of :class:`Embedder` (Slice 2b) — a bge-m3 ``/embed`` adapter.

    Duck-typed exactly like ``token_budget``/``token_offsets``: a plain :class:`Embedder` (TEI,
    LocalHash) simply lacks ``embed_multi`` and the multivector path no-ops. The base
    :class:`Embedder` Protocol stays byte-identical, so nothing else in the type surface moves."""

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_multi(
        self, texts: list[str], *, dense: bool = True, sparse: bool = True, colbert: bool = True
    ) -> list[MultiVectorOutput]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector index (Qdrant Phase-0; Milvus later). Namespaced per tenant."""

    async def ensure_collection(self, tenant_id: str, dim: int) -> None: ...

    async def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def search(
        self,
        tenant_id: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, score), ...]`` for the tenant's collection only.

        ``doc_ids`` (document scope): when non-empty, only chunks whose RECORD ``doc_id`` is in
        the set are returned (an OR over the set), in ADDITION to ``filters``. Empty ⇒ unchanged.

        ``levels`` (retrieval grain, RAG v2 Phase 8): when non-empty, only chunks whose RECORD
        ``level`` is in the set. Empty is NO FILTER, not "every level" — a v1 chunk has no level
        and must stay visible to every caller that does not ask for a grain.
        """
        ...

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """How many units this store holds for the tenant, per RAG v2 level (the enum's value).

        The Phase 6 acceptance gate: v2 chunks are indexed into the tenant's OWN namespace rather
        than a parallel ``_v2`` one (ADR-0024), so "did every level land in all three stores?" is
        answered by comparing this count across the vector store, the sparse index and the catalog
        — which is only meaningful if all three answer the same question the same way.

        ``None`` keys a chunk with no level, i.e. a v1 chunk. Returns an EMPTY mapping for a
        namespace that does not exist yet (nothing indexed is not an error), and never raises on a
        missing collection/index/schema.
        """
        ...

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Remove specific chunks (document delete). No-op for unknown ids."""
        ...

    async def delete_tenant(self, tenant_id: str) -> None: ...


@runtime_checkable
class MultiVectorStore(Protocol):
    """Additive Slice-2b capability on a :class:`VectorStore` — a NAMED collection carrying a
    dense vector, a learned-sparse vector and a ColBERT multivector per point.

    Duck-typed (checked via ``isinstance``/``hasattr``), like the optional cross-modal leg: a store
    that lacks these methods (or a legacy single-unnamed-vector collection) makes the multivector
    legs no-op, so the base :class:`VectorStore` contract and every existing call path are
    byte-identical. ``search_colbert`` re-scores the dense candidate POOL by MaxSim and its output
    is fed to RRF as one ranked list — it NEVER writes ``RetrievedSpan.score`` (the cross-encoder
    reranker stays the sole score writer and the abstain gate is unchanged)."""

    async def ensure_multivector_collection(self, tenant_id: str, dim: int) -> None:
        """Create the named dense+colbert+sparse collection (no-op if it already exists)."""
        ...

    async def upsert_multi(self, tenant_id: str, chunks: list[Chunk], outputs: list[MultiVectorOutput]) -> None:
        """Upsert children with their dense + (optional) sparse + (optional) colbert vectors."""
        ...

    async def search_dense(
        self,
        tenant_id: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        """Dense ANN against the NAMED ``dense`` vector (the 2b equivalent of ``search``).

        ``doc_ids`` (document scope): when non-empty, only chunks whose RECORD ``doc_id`` is in
        the set are returned (an OR over the set), in ADDITION to ``filters``. Empty ⇒ unchanged.
        """
        ...

    async def search_sparse(
        self,
        tenant_id: str,
        sparse: dict[int, float],
        top_k: int,
        filters: dict[str, str] | None = None,
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        """Learned-sparse (dot-product) search against the named ``sparse`` vector.

        ``doc_ids`` (document scope): when non-empty, only chunks whose RECORD ``doc_id`` is in
        the set are returned (an OR over the set), in ADDITION to ``filters``. Empty ⇒ unchanged.
        """
        ...

    async def search_colbert(
        self,
        tenant_id: str,
        dense: list[float],
        colbert: list[list[float]],
        top_k: int,
        *,
        prefetch: int,
        filters: dict[str, str] | None = None,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        """MaxSim-rescore the top-``prefetch`` dense candidates; return the reordered ranked list.

        The returned scores are MaxSim values (unbounded, uncalibrated) used ONLY for ranking into
        RRF — they must never become a span's ``score`` or reach the abstain gate.

        ``doc_ids`` (document scope): when non-empty, only chunks whose RECORD ``doc_id`` is in
        the set are returned (an OR over the set), in ADDITION to ``filters``. Empty ⇒ unchanged.
        """
        ...


@runtime_checkable
class SparseIndex(Protocol):
    """BM25 keyword index (OpenSearch Phase-0). Namespaced per tenant."""

    async def ensure_index(self, tenant_id: str) -> None: ...

    async def index(self, tenant_id: str, chunks: list[Chunk]) -> None: ...

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
        exact_boost: float | None = None,
        chem_terms: tuple[str, ...] = (),
        *,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, score), ...]`` for the tenant's index only.

        ``levels`` (retrieval grain, RAG v2 Phase 8): when non-empty, only chunks whose RECORD
        ``level`` is in the set. Empty is NO FILTER, not "every level" — a v1 chunk has no level
        and must stay visible to every caller that does not ask for a grain.

        ``exact_boost`` (Slice 2a): a per-KB weight for the exact technical-token match path
        (whole-token, case-preserving — chemical formulas/SMILES/identifiers). ``None`` ⇒ the
        implementation falls back to the global ``exact_term_boost`` setting. The exact path is
        gated by ``exact_term_match_enabled``; when that flag is off the parameter is ignored and
        retrieval is byte-identical to a plain ``match`` on ``text``.

        ``chem_terms`` (Slice 6, Phase C): canonical SMILES/InChIKey terms for the molecule(s) named
        in the query, resolved server-side for a chemistry KB. When non-empty, the adapter adds a
        boosted match on the whitespace ``chem`` field so an equivalent-but-differently-drawn molecule
        matches. Empty ⇒ no chem clause ⇒ byte-identical (gated on the terms, not on a global flag).

        ``doc_ids`` (document scope): when non-empty, only chunks whose RECORD ``doc_id`` is in
        the set are returned (an OR over the set), in ADDITION to ``filters``. Empty ⇒ unchanged.
        """
        ...

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """How many units this store holds for the tenant, per RAG v2 level (the enum's value).

        The Phase 6 acceptance gate: v2 chunks are indexed into the tenant's OWN namespace rather
        than a parallel ``_v2`` one (ADR-0024), so "did every level land in all three stores?" is
        answered by comparing this count across the vector store, the sparse index and the catalog
        — which is only meaningful if all three answer the same question the same way.

        ``None`` keys a chunk with no level, i.e. a v1 chunk. Returns an EMPTY mapping for a
        namespace that does not exist yet (nothing indexed is not an error), and never raises on a
        missing collection/index/schema.
        """
        ...

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Remove specific chunks (document delete). No-op for unknown ids."""
        ...

    async def delete_tenant(self, tenant_id: str) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker (BGE). Re-scores candidates against the query (FR-D2)."""

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]: ...


@runtime_checkable
class LLMGateway(Protocol):
    """Model-agnostic gateway to a sovereign LLM endpoint (self-hosted or managed). FR-E1."""

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str: ...

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        """Stream the answer token-by-token (same content as :meth:`complete`). ``max_tokens``
        None falls back to the gateway's configured default."""
        ...

    @property
    def model_name(self) -> str:
        """The model/deployment name this gateway targets (shown with the answer)."""
        ...

    async def list_models(self) -> list[str]:
        """Deployed models the configured endpoint exposes (empty if not discoverable)."""
        ...


@runtime_checkable
class IntentClassifier(Protocol):
    """Classifies a query's semantic INTENT before retrieval (catalog | summarize | rag).

    Deterministic heuristic (default) or LLM adapter (opt-in) behind one seam; both live in
    ``rag_clients``. ``has_history`` gates the ``summarize`` label — it is only meaningful with a
    conversation. Returns a plain ``str`` label (see :data:`rag_core.intent.Intent`)."""

    async def classify(self, query: str, *, has_history: bool) -> str: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits extracted document text into citable chunks (FR-C1)."""

    strategy: ChunkStrategy

    def chunk(self, doc: DocumentEnvelope, text: str, *, page_map: list[int] | None = None) -> list[Chunk]: ...


@runtime_checkable
class ObjectStore(Protocol):
    """S3-compatible blob store (MinIO Phase-0) for raw docs + audit log (WORM later)."""

    async def put(self, tenant_id: str, key: str, data: bytes, content_type: str) -> str: ...

    async def put_stream(self, tenant_id: str, key: str, stream: IO[bytes], length: int, content_type: str) -> str:
        """Stream ``length`` bytes from a file-like ``stream`` to the key WITHOUT buffering the
        whole object in memory — for large uploads (e.g. multi-GB files). See :meth:`put` for the
        buffered, bytes-in-hand variant."""
        ...

    async def get(self, tenant_id: str, key: str) -> bytes:
        """Raise ``KeyError`` if the key (or the tenant's namespace) does not exist."""
        ...

    async def delete(self, tenant_id: str, key: str) -> None:
        """Remove an object (document delete). No-op if it doesn't exist."""
        ...


@runtime_checkable
class ExternalObjectFetcher(Protocol):
    """Pull an object from an EXTERNAL, tenant-provided S3-compatible bucket (FR-A2b).

    Distinct from :class:`ObjectStore` (the platform's own store): the endpoint and
    credentials are supplied per call, not from platform config. Sovereignty is enforced
    by the *caller* via ``assert_sovereign`` BEFORE this is invoked, so it is provable
    regardless of which fetcher implementation runs. Must raise ``KeyError`` if the object
    does not exist (matching :meth:`ObjectStore.get`)."""

    async def fetch(
        self, *, endpoint: str, bucket: str, key: str, access_key: str, secret_key: str, secure: bool
    ) -> bytes: ...

    async def list_keys(
        self,
        *,
        endpoint: str,
        bucket: str,
        prefix: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        limit: int = 200,
        start_after: str = "",
    ) -> list[str]:
        """List object keys under ``prefix`` (folder sync, FR-A2b batch), at most ``limit``.

        ``start_after`` CONTINUES a previous listing: only keys strictly greater than it are
        returned, in the lexicographic order every backend already lists in. Without it a
        listing always restarts at the beginning, so nothing past ``limit`` is reachable however
        many times a sync runs -- measured on a real bucket, two syncs at limit=5 returned the
        same five keys. Implementations MUST keep that ordering stable or paging skips objects.
        """
        ...


@runtime_checkable
class CatalogAdapter(Protocol):
    """Metadata/lineage catalog (Postgres Phase-0; OpenMetadata later, FR-G7)."""

    async def register_document(self, doc: DocumentEnvelope) -> None: ...

    async def list_documents(self, tenant_id: str) -> list[DocumentEnvelope]:
        """All documents registered for a tenant, newest first (for the console)."""
        ...

    async def record_chunks(self, tenant_id: str, chunks: list[Chunk]) -> None: ...

    async def delete_document(self, tenant_id: str, doc_id: str) -> list[str]:
        """Delete a document + its chunk rows; return the deleted chunk_ids (to purge the
        vector/sparse indexes). Empty list if the document is unknown."""
        ...

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Remove specific chunks, leaving the document row alone.

        What a write-then-swap re-index needs: index the new version first, then drop only the
        ids it no longer contains. ``delete_document`` cannot do that — it removes the row too,
        so using it first is what leaves a document with zero chunks when the re-index then
        fails."""
        ...

    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Resolve chunk_ids -> Chunk for the tenant (retrieval hydration)."""
        ...

    async def document_chunks(self, tenant_id: str, doc_id: str) -> list[Chunk]:
        """All chunks of one document, in reading order (page, then span) — the source
        text for the in-document citation viewer (FR-I3)."""
        ...

    async def chunk_hashes(self, tenant_id: str, doc_id: str) -> dict[str, str]:
        """``chunk_id -> chunk_hash`` for one document's already-indexed chunks.

        What "make ingestion idempotent by ``chunk_hash``" (plan Phase 6) buys that content-derived
        ids do not. The ids already make the WRITES idempotent — unchanged text keeps its id and is
        simply re-upserted — but nothing stops the EMBEDDER being called again for it, and that is
        the expensive half of an index. A caller compares these hashes against the chunks it is
        about to write and embeds only the difference.

        Empty for an unknown document, and for rows written before ``chunk_hash`` was populated.
        Both mean "nothing may be reused", which is the safe direction: the cost of a false empty
        is a re-embed, the cost of a false match is a stale vector.
        """
        ...

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """How many chunk records the catalog holds for the tenant, per RAG v2 level.

        The catalog half of the Phase 6 reconciliation (see :meth:`VectorStore.count_by_level`).
        PARENT context records (FR-C2) are EXCLUDED, because they are deliberately never embedded
        or BM25-indexed: counting them would make a reconciliation FAIL on a parent/child KB whose
        three stores are in fact exactly right. Parents can only ever fall in the ``None`` bucket —
        no v2 strategy produces one — so excluding them cannot hide a v2 discrepancy.

        ``None`` keys a chunk with no level. Empty mapping for an unknown tenant; never raises.
        """
        ...

    async def get_tenant_config(self, tenant_id: str) -> dict[str, str]: ...

    async def set_tenant_config(self, tenant_id: str, key: str, value: str) -> None:
        """Upsert one per-tenant config value (threshold overrides, golden set, …)."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Per-KB knowledge graph for GraphRAG (FR-C6/D4). Postgres adjacency tables in
    Phase-2; a dedicated graph DB (Neo4j) sits behind this same seam at scale."""

    async def ensure(self, tenant_id: str) -> None: ...

    async def add_chunk(self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str]]) -> None:
        """Record a chunk's entities (as nodes mentioning it) + co-occurrence edges."""
        ...

    async def add_diagram_edges(
        self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str, str]]
    ) -> None:
        """Record a diagram's real arrow structure: each ``(source, target, relation)`` is ONE
        precise directed, labeled edge — NOT the pairwise co-occurrence fan-out ``add_chunk``
        does. Separate write path so a diagram's boxes never become a near-complete clique."""
        ...

    async def remove_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Forget these chunks — drop them from every node and edge, and drop anything left
        with no evidence at all.

        ``add_chunk`` only ever ADDS, so a re-chunked document left its old ids in the graph
        forever. Content-derived ids (P44) mean those ids are genuinely gone, and community
        evidence rebuilt from such a graph still cites them (P48)."""
        ...

    async def linked_chunks(self, tenant_id: str, entities: list[str], hops: int = 1) -> list[str]:
        """Chunk ids reachable from ``entities`` within ``hops`` (local graph search)."""
        ...

    async def stats(self, tenant_id: str, top: int = 10) -> GraphStats:
        """The KB's knowledge-graph size + most-connected entities."""
        ...

    async def clear(self, tenant_id: str) -> None:
        """Drop the whole KB graph (on KB delete / full reindex)."""
        ...

    async def graph_data(self, tenant_id: str) -> tuple[list[GraphNode], list[tuple[str, str]]]:
        """The whole graph (nodes + edges) for community detection (slice 2)."""
        ...

    async def save_communities(self, tenant_id: str, communities: list[GraphCommunity]) -> None:
        """Replace the KB's detected communities (thematic clusters + summaries)."""
        ...

    async def list_communities(self, tenant_id: str) -> list[GraphCommunity]:
        """The KB's communities, largest theme first (empty until built)."""
        ...


@runtime_checkable
class AuditSink(Protocol):
    """Append-only, tamper-evident audit log (FR-G4)."""

    async def append(self, event: object) -> str:
        """Persist a sealed AuditEvent; return its event_hash."""
        ...

    async def list_events(self, tenant_id: str, trace_id: str | None, limit: int) -> list[AuditEvent]:
        """Read audit events for a tenant (optionally one trace), newest-first."""
        ...

    async def count_actions(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        """Count audit events per action for a tenant since ``since`` (all-time if None).

        Aggregation happens in the store (SQL GROUP BY / in-memory tally), not by pulling
        every row — this backs usage metering / FinOps showback (FR-H4)."""
        ...

    async def count_actions_kb(self, kb_id: str, since: datetime | None) -> dict[str, int]:
        """KB-scoped per-action counts: the KB's own schema UNION any events that landed in the
        owning ORG schema but self-attribute to this KB (``payload.kb_id == kb_id``).

        Query-time audit events are recorded under the KB id going forward, so they live in the
        KB schema; this union additionally recovers any that reached the org schema (a genuine
        multi-KB ask, or history) yet name this KB in their payload — so KB-scoped dashboards are
        not blind to query-time activity. When ``kb_id`` IS the org (single-KB org, ``kb_id``
        without ``__``), the KB schema and the org schema are the same and this is exactly
        :meth:`count_actions` (a no-op union). Fail-safe: a missing org schema/table contributes
        nothing, never raises."""
        ...

    async def list_events_kb(
        self, kb_id: str, since: datetime | None, actions: list[str] | None, limit: int
    ) -> list[AuditEvent]:
        """KB-scoped event read (newest-first), unioning the KB schema with org-schema events that
        self-attribute to this KB (``payload.kb_id == kb_id``), deduped by ``event_id``. Optional
        ``actions`` restricts to those action values. Same union/no-op/fail-safe semantics as
        :meth:`count_actions_kb`; backs the abstain/feedback analytics surface."""
        ...

    async def token_totals(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        """Sum prompt/completion/total tokens over GENERATE events since ``since``
        (token metering, FR-H4). Zeroes when nothing was generated."""
        ...

    async def quality_stats(self, tenant_id: str, since: datetime | None) -> dict[str, object]:
        """Aggregate answer-quality signals since ``since`` (FR-G5 dashboards):
        faithfulness count/avg + low/mid/high buckets, outcome counts, abstain reasons,
        and guardrail stage counts. Aggregated in-store, never by pulling rows."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Top-level retrieval orchestration (hybrid + rerank) (TDD §4.4)."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedSpan]: ...
