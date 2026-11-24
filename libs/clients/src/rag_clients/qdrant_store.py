"""Qdrant vector store adapter (implements :class:`VectorStore`) — FR-C4/D1/D8.

Isolation: one Qdrant **collection per tenant**, named via
:func:`rag_common.tenancy.vector_collection`. A caller can only ever touch the
collection derived from the explicit ``tenant_id`` argument, so cross-tenant reads
are structurally impossible (see ``tests/integration/test_tenant_isolation.py``).

Milvus is the deferred at-scale alternative (ADR-0003); swapping is a new adapter
implementing the same Protocol — no service changes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_common import ConfigError, Settings, get_logger, get_settings, vector_collection
from rag_core import MultiVectorOutput
from rag_schemas import Chunk
from rag_schemas.common import ChunkLevel

_log = get_logger(__name__)

# Deterministic point id from chunk_id so re-indexing upserts (not duplicates).
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000e5d5")
_HTTP_NOT_FOUND = 404
# A 4xx from a named-vector query on a not-yet-recreated collection degrades an additive 2b leg.
_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500

# Collections whose Slice-2b named legs have FAILED OPEN at least once this process (a query hit a
# collection not yet recreated to the multivector shape and degraded to BM25/dense instead of
# erroring). Durable, observable state — surfaced on /system/model-health so a silently-degraded KB
# is visible to an operator, not just a per-query log line. Keyed by collection name; value = count.
_DEGRADED_MULTIVECTOR: dict[str, int] = {}


def multivector_degraded_collections() -> dict[str, int]:
    """Collections currently in Slice-2b degraded mode (name → fail-open count this process)."""
    return dict(_DEGRADED_MULTIVECTOR)


# When quantization is on, rescore the top candidates against full-precision vectors so
# recall holds (critical for binary). Oversample the candidate pool before rescoring.
_RESCORE_OVERSAMPLING = 2.0


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _quantization_config(mode: str) -> models.QuantizationConfig | None:
    """Map the ``qdrant_quantization`` setting to a Qdrant quantization config (or None)."""
    if mode == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        )
    if mode == "binary":
        return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
    return None


def _collection_kwargs(dim: int, settings: Settings) -> dict[str, Any]:
    """Pure builder for create_collection kwargs from settings (unit-testable, no client).

    Defaults (quantization=none, on_disk=False, shards=1, replication=1) reproduce the
    original single-node, full-precision collection exactly.
    """
    return {
        "vectors_config": models.VectorParams(
            size=dim, distance=models.Distance.COSINE, on_disk=settings.qdrant_on_disk
        ),
        "quantization_config": _quantization_config(settings.qdrant_quantization),
        "shard_number": settings.qdrant_shards,
        "replication_factor": settings.qdrant_replication,
    }


def _multivector_collection_kwargs(dim: int, settings: Settings) -> dict[str, Any]:
    """create_collection kwargs for a Slice-2b NAMED collection (unit-testable, no client).

    Three vectors per point: a named ``dense`` (unchanged 1024-d space), a named ``colbert``
    MULTIVECTOR scored by MaxSim and stored rescore-only (``datatype=FLOAT16`` + ``on_disk`` +
    HNSW ``m=0`` = no independent index, so MaxSim runs only over a prefetched pool — the storage
    bound), and a named learned-sparse ``sparse`` vector with the IDF modifier bge-m3 expects.

    NOTE FLOAT16 (not int8): the ColBERT components are cosine-space floats that include negatives,
    which an unsigned int8 store cannot represent; FLOAT16 gives the 2x storage cut without that
    correctness hazard, and qdrant-client exposes no signed int8 datatype."""
    return {
        "vectors_config": {
            "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=settings.qdrant_on_disk),
            "colbert": models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                datatype=models.Datatype.FLOAT16,
                on_disk=True,
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        "sparse_vectors_config": {
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
        "quantization_config": _quantization_config(settings.qdrant_quantization),
        "shard_number": settings.qdrant_shards,
        "replication_factor": settings.qdrant_replication,
    }


def _search_params(settings: Settings) -> models.SearchParams | None:
    """Rescore against full vectors when quantization is on; else no override (None)."""
    if settings.qdrant_quantization == "none":
        return None
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(rescore=True, oversampling=_RESCORE_OVERSAMPLING)
    )


class QdrantVectorStore:
    """Dense vector index, namespaced per tenant by collection."""

    def __init__(self, url: str | None = None, namespace: Callable[[str], str] = vector_collection) -> None:
        self._settings = get_settings()
        self._url = url or self._settings.qdrant_url
        self._client = AsyncQdrantClient(url=self._url)
        # Per-tenant collection name. Defaults to the text collection; the cross-modal image
        # index injects ``vision_collection`` so image vectors live in their own collection.
        self._ns = namespace

    async def ensure_collection(self, tenant_id: str, dim: int) -> None:
        name = self._ns(tenant_id)
        if not await self._client.collection_exists(name):
            await self._client.create_collection(collection_name=name, **_collection_kwargs(dim, self._settings))
            _log.info(
                "qdrant_collection_created",
                collection=name,
                dim=dim,
                quantization=self._settings.qdrant_quantization,
                on_disk=self._settings.qdrant_on_disk,
                shards=self._settings.qdrant_shards,
                replication=self._settings.qdrant_replication,
            )

    async def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        name = self._ns(tenant_id)
        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector=vec,
                payload=_payload(c),
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        for batch in _batched(points):
            await self._client.upsert(collection_name=name, points=batch)

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
        name = self._ns(tenant_id)
        query_filter = _build_filter(filters, doc_ids, levels)
        try:
            res = await self._client.query_points(
                collection_name=name,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                search_params=_search_params(self._settings),
            )
        except UnexpectedResponse as exc:
            # A tenant that has never ingested anything has no collection yet — that is
            # "no results", not an error. Every other status still propagates.
            if exc.status_code == _HTTP_NOT_FOUND:
                return []
            raise
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points if p.payload]

    # --- Slice 2b: named dense+sparse+colbert collection (additive; MultiVectorStore) ----------

    def multivector_degraded(self, tenant_id: str) -> bool:
        """True when this tenant's collection has failed a Slice-2b named leg open this process
        (answering on BM25/dense only, pending a reindex). Read by /system/model-health."""
        return self._ns(tenant_id) in _DEGRADED_MULTIVECTOR

    async def ensure_multivector_collection(self, tenant_id: str, dim: int) -> None:
        """Create the NAMED dense+colbert+sparse collection.

        Three cases, all data-loss-averse:
        * The collection does NOT exist → create it named (as today).
        * It exists and is ALREADY the named schema → no-op (as today).
        * It exists but is an UNNAMED/legacy (single-VectorParams, dense-only) collection AND is
          EMPTY (``points_count == 0``) → recreate it named. Safe: there is no indexed data to lose.
          This is the fresh-KB path — attaching ``bge_m3_multi`` as the FIRST embedder makes it
          active directly (no staged switch to promote), so nothing else recreates the unnamed
          collection ``ensure_namespaces`` made at KB creation, and the first ingest would otherwise
          fail the ``upsert_multi`` mismatch guard.

        A POPULATED unnamed collection is NEVER auto-dropped: migrating indexed vectors is the
        deliberate ``embedder_switch.apply`` (``delete_tenant`` first, then recreate + re-embed),
        and a flag-on-but-unmigrated populated KB must still raise the clear reindex-required
        ConfigError at ``upsert_multi`` rather than silently discarding its vectors."""
        name = self._ns(tenant_id)
        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                collection_name=name, **_multivector_collection_kwargs(dim, self._settings)
            )
            _log.info("qdrant_multivector_collection_created", collection=name, dim=dim)
            return
        info = await self._client.get_collection(name)
        params = info.config.params.vectors
        # Same shape detection as ``collection_dims``: a named 2b collection is a dict carrying a
        # ``dense`` entry; a legacy collection is a single unnamed VectorParams.
        already_named = isinstance(params, dict) and params.get("dense") is not None
        if already_named:
            return
        points = info.points_count or 0
        if points == 0:
            # Unnamed AND empty → recreate named (no data to lose).
            await self._client.delete_collection(name)
            await self._client.create_collection(
                collection_name=name, **_multivector_collection_kwargs(dim, self._settings)
            )
            _log.info("qdrant_multivector_collection_recreated_from_empty_legacy", collection=name, dim=dim)
        else:
            # Unnamed AND populated → leave it. ``upsert_multi`` will raise the clear reindex-required
            # ConfigError; the deliberate migration path (``embedder_switch.apply``) drops it FIRST.
            _log.warning(
                "qdrant_multivector_collection_populated_legacy_left_intact",
                collection=name,
                points=points,
                detail="unnamed collection has indexed vectors — a reindex (embedder apply) is "
                "required to migrate; not auto-dropping to avoid data loss",
            )

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """Counted in Qdrant, not pulled: one ``count`` per level plus one for the whole collection,
        with no vectors or payloads transferred at all. ``exact=True`` because a reconciliation that
        reports an estimate is not a reconciliation. Like every other payload predicate in this
        adapter (the D8 filters, the ``doc_ids`` scope), it is an unindexed scan — this collection
        has no payload indexes — so it is an operator-run check, not a per-query path.

        The level-less (v1) bucket is the REMAINDER — total minus the levelled counts — rather than
        a filter of its own, because "has no level" is the absence of a payload key and a
        ``must_not`` over three values would also have to hold for a point whose key is present but
        unknown to this build. A remainder stays correct if a later phase adds a fourth level before
        this code learns about it: the unknown level lands in the v1 bucket and the TOTAL still
        reconciles, which is the property the gate actually rests on.
        """
        name = self._ns(tenant_id)
        try:
            total = (await self._client.count(collection_name=name, exact=True)).count
            counts: dict[str | None, int] = {}
            for level in ChunkLevel:
                hits = (
                    await self._client.count(
                        collection_name=name,
                        count_filter=models.Filter(
                            must=[models.FieldCondition(key="level", match=models.MatchValue(value=level.value))]
                        ),
                        exact=True,
                    )
                ).count
                if hits:
                    counts[level.value] = hits
        except UnexpectedResponse as exc:
            # A tenant that has never ingested has no collection — "nothing indexed", not an error,
            # exactly as `search` treats the same 404.
            if exc.status_code == _HTTP_NOT_FOUND:
                return {}
            raise
        remainder = total - sum(counts.values())
        if remainder:
            counts[None] = remainder
        return counts

    async def upsert_multi(self, tenant_id: str, chunks: list[Chunk], outputs: list[MultiVectorOutput]) -> None:
        if len(chunks) != len(outputs):
            raise ValueError("chunks/outputs length mismatch")
        name = self._ns(tenant_id)
        points = []
        for c, out in zip(chunks, outputs, strict=True):
            # Named vectors, all optional per point: a KB with the colbert/sparse leg off simply
            # omits those vectors (Qdrant allows optional named vectors) and stores dense alone.
            vector: dict[str, Any] = {"dense": out.dense}
            if out.colbert:
                vector["colbert"] = out.colbert
            if out.sparse:
                vector["sparse"] = models.SparseVector(indices=list(out.sparse), values=list(out.sparse.values()))
            points.append(
                models.PointStruct(
                    id=_point_id(c.chunk_id),
                    vector=vector,
                    payload=_payload(c),
                )
            )
        try:
            for batch in _batched(points):
                await self._client.upsert(collection_name=name, points=batch)
        except UnexpectedResponse as exc:
            # A named-vector upsert against a legacy UNNAMED (dense-only) collection is a 4xx.
            # Turn Qdrant's raw "not existing vector name" 400 into a clear, actionable error:
            # activation REQUIRES the attach + apply reindex (or delete_tenant + reindex), which
            # RECREATES the collection in the named shape — there is no silent auto-recreate that
            # could drop data.
            if _HTTP_CLIENT_ERROR_MIN <= (exc.status_code or 0) < _HTTP_SERVER_ERROR_MIN:
                raise ConfigError(
                    f"collection '{name}' is not migrated to the multivector schema — a reindex is "
                    "required (attach the bge_m3_multi embedder and apply the re-embed, which "
                    "recreates the collection with named dense+sparse+colbert vectors).",
                    details={"collection": name, "status": exc.status_code},
                ) from exc
            raise

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
        return await self._named_query(tenant_id, vector, "dense", top_k, filters, doc_ids, levels)

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
        if not sparse:
            return []
        query = models.SparseVector(indices=list(sparse), values=list(sparse.values()))
        return await self._named_query(tenant_id, query, "sparse", top_k, filters, doc_ids, levels)

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
        """MaxSim-rescore the top-``prefetch`` dense candidates (Prefetch on ``dense``, MaxSim on
        ``colbert``). The scores are MaxSim values used ONLY for ranking into RRF — never a span
        ``score`` and never the abstain gate."""
        if not colbert:
            return []
        name = self._ns(tenant_id)
        query_filter = _build_filter(filters, doc_ids, levels)
        try:
            res = await self._client.query_points(
                collection_name=name,
                prefetch=models.Prefetch(query=dense, using="dense", limit=prefetch, filter=query_filter),
                query=colbert,
                using="colbert",
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            return self._named_leg_fail_open("colbert", name, exc)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points if p.payload]

    async def _named_query(
        self,
        tenant_id: str,
        query: Any,
        using: str,
        top_k: int,
        filters: dict[str, str] | None,
        doc_ids: tuple[str, ...] = (),
        levels: tuple[ChunkLevel, ...] = (),
    ) -> list[tuple[str, float]]:
        name = self._ns(tenant_id)
        try:
            res = await self._client.query_points(
                collection_name=name,
                query=query,
                using=using,
                limit=top_k,
                query_filter=_build_filter(filters, doc_ids, levels),
                with_payload=True,
                search_params=_search_params(self._settings) if using == "dense" else None,
            )
        except UnexpectedResponse as exc:
            return self._named_leg_fail_open(using, name, exc)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points if p.payload]

    @staticmethod
    def _named_leg_fail_open(using: str, name: str, exc: UnexpectedResponse) -> list[tuple[str, float]]:
        """A Slice-2b named leg is ADDITIVE: it must fail open, never take a query down. A missing
        collection (404) is "no results"; a 4xx such as querying the named ``dense``/``sparse``/
        ``colbert`` vector on a collection NOT yet recreated to the 2b shape degrades this leg to
        empty (RRF then leans on BM25/dense), matching the embedder-switch reindex-window behaviour
        — the un-reindexed KB keeps answering rather than erroring. Any other status still propagates.
        """
        status = exc.status_code or 0
        if status == _HTTP_NOT_FOUND:
            return []
        if _HTTP_CLIENT_ERROR_MIN <= status < _HTTP_SERVER_ERROR_MIN:
            # Fail open (don't take the query down) BUT record it durably so the degradation is
            # observable on /system/model-health, not just in a per-query log line. Loud once,
            # counted thereafter.
            first = name not in _DEGRADED_MULTIVECTOR
            _DEGRADED_MULTIVECTOR[name] = _DEGRADED_MULTIVECTOR.get(name, 0) + 1
            _log.warning(
                "multivector_degraded_mode",
                leg=using,
                collection=name,
                status=status,
                first=first,
                count=_DEGRADED_MULTIVECTOR[name],
                detail="named 2b leg unavailable — this KB is answering on BM25/dense only; a reindex "
                "to the multivector schema is required",
            )
            return []
        raise exc

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        name = self._ns(tenant_id)
        await self._client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=[_point_id(c) for c in chunk_ids]),
        )

    async def delete_tenant(self, tenant_id: str) -> None:
        name = self._ns(tenant_id)
        if await self._client.collection_exists(name):
            await self._client.delete_collection(name)

    async def collection_dims(self) -> dict[str, int]:
        """Every collection's configured vector width, for the startup preflight.

        A width that disagrees with RAG_EMBED_DIM makes every query against that collection
        fail — and only at query time, as a raw Qdrant 400 in front of a user. Cheap to read
        once at boot; introspection only, never used on a request path."""
        out: dict[str, int] = {}
        collections = await self._client.get_collections()
        for c in collections.collections:
            try:
                info = await self._client.get_collection(c.name)
                params = info.config.params.vectors
                # A Slice-2b collection uses NAMED vectors (a dict), so ``params`` has no ``size``;
                # the dense width lives on the ``dense`` entry. A legacy collection is a single
                # unnamed VectorParams. Reading the wrong shape here makes the whole boot preflight
                # miss every 2b collection.
                if isinstance(params, dict):
                    size = getattr(params.get("dense"), "size", None)
                else:
                    size = getattr(params, "size", None)
                if isinstance(size, int):
                    out[c.name] = size
            except Exception:  # noqa: S112 - a collection we cannot read must not break the probe
                continue
        return out

    async def aclose(self) -> None:
        await self._client.close()


def _batched(points: list[Any]) -> Iterator[list[Any]]:
    """Split an upsert into requests Qdrant will accept.

    Qdrant caps one request at ``max_request_size_mb`` (32 by default) and rejects an oversized
    one with a 400 WHILE THE CLIENT IS STILL UPLOADING, so httpx raises a bare ``ReadError`` and
    the ingest 500s with nothing written and no hint that size was the problem. A document can
    pass that cap on its own: measured on an 80-page tender, RAG v2 ingest produced 1801 points
    (1425 table rows) totalling 36.9 MB, and the ONLY upsert call was never going to succeed.

    Batching by COUNT rather than by measured bytes, deliberately: the serialised size depends on
    the embedding dimension and on how much metadata the caller stamped on each chunk, so a byte
    budget here would be a second place that has to know those things. A conservative count is
    checkable by hand (see ``qdrant_upsert_batch``) and cannot silently drift.
    """
    size = get_settings().qdrant_upsert_batch
    for start in range(0, len(points), size):
        yield points[start : start + size]


def _payload(c: Chunk) -> dict[str, object]:
    """The point payload: the chunk's tags FIRST, then the record fields, so the RECORD WINS.

    The payload is deliberately FLAT (tags filterable at the top level alongside the record, which
    is what every stored D8 filter already relies on), but flat means a tag called ``doc_id`` and
    the record's own ``doc_id`` are the same key. Spread last, a tag silently REDEFINED the record
    it was only supposed to describe — the same fault already fixed on the sparse side, where tags
    are namespaced under ``meta.*`` (see ``opensearch_index._source``).

    The producer of a colliding key is a CALLER: ``DocumentEnvelope.metadata`` supplied at ingest.
    B7 auto-metadata cannot do it — ``tagging.DERIVED_KEYS`` is
    ``{doc_type, doc_date, year, entities, category, categories}``, none of which is a record
    field. So no in-repo path emits one today, and this is a guard against a caller, not a fix for
    a live fault. Ordering fixes it without moving any field: tags stay queryable where they
    always were, and none of them can overwrite the record. Note it heals only on (re)index — an
    already-stored point whose tag won keeps its bad payload until the document is re-indexed.
    """
    payload: dict[str, object] = {
        **c.metadata,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "tenant_id": c.tenant_id,
        "page": c.page,
        "acl": c.acl,
    }
    # RAG v2 retrieval grain (section|chunk|row), the enum's VALUE so the payload stays JSON. Added
    # ONLY when the chunk has one: a v1 chunk's payload is byte-identical to before, and an
    # unconditional `"level": None` would rewrite every point in every live collection for a field
    # no v1 chunk has. It sits with the record fields, AFTER the tags, so — like every other record
    # field here — a tenant tag called `level` cannot redefine it.
    #
    # Written so the index itself can answer "how many chunks of each level do you hold?", which is
    # Phase 6's acceptance gate now that v2 shares the tenant's collection rather than a parallel
    # `_v2` one (ADR-0024). `count` with a payload filter answers it without pulling any vectors.
    if c.level is not None:
        payload["level"] = c.level.value
    return payload


def _build_filter(
    filters: dict[str, str] | None,
    doc_ids: tuple[str, ...] = (),
    levels: tuple[ChunkLevel, ...] = (),
) -> models.Filter | None:
    """Metadata filters applied pre-fusion (FR-D8), plus the optional document and grain scopes.

    Tenant isolation is via collection. ``doc_ids`` is a predicate on the TOP-LEVEL ``doc_id``
    payload key (the record, never a tag) and is ANDed with ``filters`` as one more ``must``
    condition — ``MatchAny`` makes the set itself an OR. Empty ⇒ the filter is byte-identical.

    ``levels`` (RAG v2 Phase 8) is the same shape on the ``level`` key, which Phase 6 wrote for
    exactly this ("later a level-scoped query leg"). It is present on a point only when the chunk
    HAS a level, so a ``MatchAny`` over the v2 levels excludes every v1 point — which is the
    intended meaning of asking for a grain, and the reason an empty tuple must stay no filter at
    all rather than widening to "every level".
    """
    if not filters and not doc_ids and not levels:
        return None
    must: list[models.Condition] = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in (filters or {}).items()
    ]
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
    if levels:
        must.append(models.FieldCondition(key="level", match=models.MatchAny(any=[lv.value for lv in levels])))
    return models.Filter(must=must)
