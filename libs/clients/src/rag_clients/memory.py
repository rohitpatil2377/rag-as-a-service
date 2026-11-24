"""In-memory adapters + deterministic local embedder (dev/test/CI, no infra).

Every class implements the SAME rag-core Protocol as its real counterpart and enforces
the SAME per-tenant namespacing (via :mod:`rag_common.tenancy`), so:

* the walking skeleton and CI run with zero external stores/models, and
* tenant-isolation guarantees are provable without standing up the stack
  (``tests/integration/test_tenant_isolation.py``).

These are NOT for production — the factory only returns them when ``RAG_INMEMORY`` is
set, and never in prod (guarded).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import IO, Any

from rag_clients.layout_rows import BlockRow, SectionRow, flatten_layout_tree
from rag_common import (
    ConfigError,
    assert_sovereign,
    get_settings,
    object_prefix,
    org_of,
    sparse_index,
    vector_collection,
)
from rag_core import MultiVectorOutput
from rag_core.chem import CHEM_TERMS_META
from rag_core.chunking import text_hash
from rag_core.doc_profile import DocProfile
from rag_core.facet_vocab import CanonicalFacet, FacetVocabulary
from rag_core.graph import DeclaredRelation
from rag_core.layout import LayoutTree
from rag_core.mapreduce import MapCacheEntry
from rag_core.parent_child import is_parent
from rag_core.router import DocCard
from rag_core.section_enrich import EnrichmentManifest, SectionEnrichment
from rag_core.strategies import SectionCard
from rag_schemas import AuditEvent, Chunk, DocumentEnvelope
from rag_schemas.common import ChunkLevel

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Faithfulness distribution buckets (mirror the routing thresholds' defaults).
_FAITH_LOW = 0.4
_FAITH_HIGH = 0.7


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _tally(levels: Iterable[str | None]) -> dict[str | None, int]:
    """Count per RAG v2 level for ``count_by_level``. ONE implementation for all three twins, so
    they cannot disagree with each other about a reconciliation whose whole purpose is agreement."""
    return dict(Counter(levels))


# --- Deterministic local embedder -------------------------------------------
class LocalHashEmbedder:
    """Hashing bag-of-words embedder: deterministic, offline, dependency-free.

    Good enough for cosine similarity in tests/demos (semantically-overlapping text
    lands nearby). NOT a semantic model — the real path is TEI (``bge-small``).
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or min(get_settings().embed_dim, 256)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_multi(
        self, texts: list[str], *, dense: bool = True, sparse: bool = True, colbert: bool = True
    ) -> list[MultiVectorOutput]:
        """Deterministic twin of :class:`MultiVectorEmbedder` (Slice 2b), so the sparse + MaxSim
        round-trip is provable in CI with no live endpoint. Sparse keys reuse the SAME hash bucket
        as the dense dim, so an identical token maps to the same index in query and document (a
        real dot-product match); each ColBERT per-token vector is the single-token bag embedding,
        so MaxSim rewards shared tokens. Base ``embed`` stays byte-identical."""
        out: list[MultiVectorOutput] = []
        for t in texts:
            toks = _tokens(t)
            dv = self._embed_one(t) if dense else []
            sv: dict[int, float] = {}
            if sparse:
                counts: dict[int, float] = defaultdict(float)
                for tok in toks:
                    h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324 - non-crypto hashing
                    counts[h % self._dim] += 1.0
                sv = dict(counts)
            cb = [self._embed_one(tok) for tok in toks] if colbert else []
            out.append(MultiVectorOutput(dense=dv, sparse=sv, colbert=cb))
        return out

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324 - non-crypto hashing
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# --- In-memory stores (tenant-namespaced) -----------------------------------
class InMemoryVectorStore:
    def __init__(self, namespace: Callable[[str], str] = vector_collection) -> None:
        self._c: dict[str, dict[str, tuple[list[float], dict[str, str]]]] = defaultdict(dict)
        # Slice-2b learned-sparse + ColBERT multivector, keyed alongside the dense vector so the
        # named-collection round-trip is provable in CI. Absent for legacy dense-only upserts.
        self._mv: dict[str, dict[str, tuple[dict[int, float], list[list[float]]]]] = defaultdict(dict)
        # Per-collection schema: "unnamed" (legacy dense-only) vs "named" (Slice-2b
        # dense+sparse+colbert). Mirrors the real Qdrant collection config so the twin can model
        # the fresh-KB recreate vs the populated-KB reindex-required guard. Absent ⇒ not created.
        self._schema: dict[str, str] = {}
        # The per-tenant collection name. Defaults to the text collection; the cross-modal
        # image index injects ``vision_collection`` for a separate namespace on the same store.
        self._ns = namespace

    async def ensure_collection(self, tenant_id: str, dim: int) -> None:
        name = self._ns(tenant_id)
        self._c[name]  # touch namespace
        # A first ``ensure_collection`` creates an UNNAMED (legacy dense-only) collection, exactly
        # like ``ensure_namespaces`` does at KB creation with the default openai_compat embedder.
        self._schema.setdefault(name, "unnamed")

    async def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        col = self._c[self._ns(tenant_id)]
        for c, v in zip(chunks, vectors, strict=True):
            # Tags first, record LAST — mirrors QdrantVectorStore._payload: the payload is flat,
            # so a tag called `doc_id` must not redefine the record it describes.
            payload = {**c.metadata, "doc_id": c.doc_id}
            # The RAG v2 level, mirroring `QdrantVectorStore._payload` — present only when the
            # chunk has one, so a v1 point's payload is byte-identical. On the payload rather than
            # in a side map so a `filters={"level": ...}` predicate behaves the same in the twin
            # as in Qdrant, and so `count_by_level` reads the same place in both.
            if c.level is not None:
                payload["level"] = c.level.value
            col[c.chunk_id] = (v, payload)

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        col = self._c.get(self._ns(tenant_id), {})
        return _tally(payload.get("level") for _vec, payload in col.values())

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        col = self._c[self._ns(tenant_id)]
        mv = self._mv[self._ns(tenant_id)]
        for cid in chunk_ids:
            col.pop(cid, None)
            mv.pop(cid, None)

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
        col = self._c[self._ns(tenant_id)]
        scored = []
        wanted_levels = {lv.value for lv in levels}
        for cid, (vec, payload) in col.items():
            if filters and any(payload.get(k) != v for k, v in filters.items()):
                continue
            # Document scope: the payload's `doc_id` is the RECORD's (written last, so no tag can
            # have overwritten it) — mirrors the Qdrant MatchAny condition on the top-level key.
            if doc_ids and payload.get("doc_id") not in doc_ids:
                continue
            # Retrieval grain (Phase 8) — mirrors the Qdrant MatchAny on `level`. A v1 point has
            # no `level` key at all and is therefore excluded by any non-empty request, which is
            # what asking for a grain means.
            if wanted_levels and payload.get("level") not in wanted_levels:
                continue
            scored.append((cid, _cosine(vector, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete_tenant(self, tenant_id: str) -> None:
        name = self._ns(tenant_id)
        self._c.pop(name, None)
        self._mv.pop(name, None)
        self._schema.pop(name, None)

    # --- Slice 2b: named dense+sparse+colbert twin (MultiVectorStore) --------------------------

    async def ensure_multivector_collection(self, tenant_id: str, dim: int) -> None:
        name = self._ns(tenant_id)
        self._c[name]  # touch namespace
        self._mv[name]
        schema = self._schema.get(name)
        if schema is None:
            self._schema[name] = "named"  # does not exist → create named
        elif schema == "named":
            return  # already named → no-op
        elif len(self._c[name]) == 0:
            self._schema[name] = "named"  # unnamed AND empty → recreate named (no data to lose)
        # else: unnamed AND populated → leave as-is; ``upsert_multi`` raises reindex-required.

    async def upsert_multi(self, tenant_id: str, chunks: list[Chunk], outputs: list[MultiVectorOutput]) -> None:
        name = self._ns(tenant_id)
        if self._schema.get(name) == "unnamed":
            # Mirror the real store's mismatch guard: a named-vector upsert against a legacy
            # UNNAMED collection is not migrated and needs a reindex — never a silent auto-recreate
            # that could drop indexed data.
            raise ConfigError(
                f"collection '{name}' is not migrated to the multivector schema — a reindex is "
                "required (attach the bge_m3_multi embedder and apply the re-embed, which "
                "recreates the collection with named dense+sparse+colbert vectors).",
                details={"collection": name},
            )
        self._schema.setdefault(name, "named")
        col = self._c[name]
        mv = self._mv[name]
        for c, out in zip(chunks, outputs, strict=True):
            col[c.chunk_id] = (out.dense, {**c.metadata, "doc_id": c.doc_id})  # record last, see `upsert`
            mv[c.chunk_id] = (dict(out.sparse), [list(tok) for tok in out.colbert])

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
        return await self.search(tenant_id, vector, top_k, filters, doc_ids=doc_ids, levels=levels)

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
        col = self._c[self._ns(tenant_id)]
        mv = self._mv[self._ns(tenant_id)]
        scored: list[tuple[str, float]] = []
        wanted_levels = {lv.value for lv in levels}
        for cid, (doc_sparse, _colbert) in mv.items():
            payload = col.get(cid, ([], {}))[1]
            if filters and any(payload.get(k) != v for k, v in filters.items()):
                continue
            if doc_ids and payload.get("doc_id") not in doc_ids:
                continue
            if wanted_levels and payload.get("level") not in wanted_levels:
                continue
            dot = sum(w * doc_sparse[i] for i, w in sparse.items() if i in doc_sparse)
            if dot > 0.0:
                scored.append((cid, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

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
        """Emulate Qdrant Prefetch+MaxSim: take the dense top-``prefetch`` pool, then reorder it by
        MaxSim = sum over query tokens of max over doc tokens cosine(q_tok, d_tok)."""
        if not colbert:
            return []
        pool = await self.search(tenant_id, dense, prefetch, filters, doc_ids=doc_ids, levels=levels)
        mv = self._mv[self._ns(tenant_id)]
        rescored: list[tuple[str, float]] = []
        for cid, _dense_score in pool:
            doc_colbert = mv.get(cid, ({}, []))[1]
            if not doc_colbert:
                continue
            maxsim = sum(max(_cosine(q, d) for d in doc_colbert) for q in colbert)
            rescored.append((cid, maxsim))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]


class InMemorySparseIndex:
    def __init__(self) -> None:
        # index -> {chunk_id: (text, filterable_payload, record_doc_id)}
        #
        # The record's `doc_id` is carried OUT OF BAND, in its own tuple slot, NOT merged into the
        # filterable payload — exactly as the OpenSearch adapter keeps `doc_id` a top-level field
        # while every tag lives under `meta.*`. That separation is the point: a `doc_id` TAG must
        # keep meaning the tag (see test_sparse_metadata), while the `doc_ids` document scope is a
        # predicate on the record. Putting the record id in the payload would collapse the two.
        # The RAG v2 level rides in its own slot for the same reason: the OpenSearch adapter writes
        # it as a TOP-LEVEL `_source` field, not a `meta.*` tag, so it is a fact about the record.
        self._i: dict[str, dict[str, tuple[str, dict[str, str], str, str | None]]] = defaultdict(dict)

    async def ensure_index(self, tenant_id: str) -> None:
        self._i[sparse_index(tenant_id)]

    async def index(self, tenant_id: str, chunks: list[Chunk]) -> None:
        idx = self._i[sparse_index(tenant_id)]
        for c in chunks:
            # Tags kept SEPARATE from the record, exactly as the OpenSearch adapter namespaces
            # them: a tag called "doc_id" describes the chunk, it does not redefine it.
            idx[c.chunk_id] = (c.text, dict(c.metadata), c.doc_id, c.level.value if c.level is not None else None)

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        return _tally(level for *_rest, level in self._i.get(sparse_index(tenant_id), {}).values())

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        idx = self._i[sparse_index(tenant_id)]
        for cid in chunk_ids:
            idx.pop(cid, None)

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
        idx = self._i[sparse_index(tenant_id)]
        q = set(_tokens(query))
        settings = get_settings()
        exact_on = settings.exact_term_match_enabled
        # Slice 2a exact-match boost: mirror the OpenSearch `text.exact` whitespace sub-field with
        # WHOLE, case-preserving tokens (split on whitespace only) so a chemical formula / SMILES /
        # identifier the `_tokens` fragmenter shreds is still matched verbatim. None ⇒ the global
        # setting. Flag OFF ⇒ no exact terms, no boost ⇒ byte-identical to the fragment-only score.
        exact_query_terms = set(query.split()) if exact_on else set()
        # Slice 6, Phase C chemical normalization: mirror the OpenSearch `chem` field. Gated on the
        # server-resolved `chem_terms` being non-empty (NOT the master flag), so a non-chemistry /
        # flag-off query carries none ⇒ no contribution ⇒ byte-identical even when the flag is on for
        # other KBs. A chunk's canonical chemical terms (stamped at ingest under CHEM_TERMS_META) are a
        # SEPARATE match set from its text — never mixed into the text overlap.
        chem_active = bool(chem_terms)
        chem_query_terms = set(chem_terms)
        boost = (
            (exact_boost if exact_boost is not None else settings.exact_term_boost)
            if (exact_on or chem_active)
            else 0.0
        )
        scored = []
        wanted_levels = {lv.value for lv in levels}
        for cid, (text, payload, record_doc_id, level) in idx.items():
            if filters and any(payload.get(k) != v for k, v in filters.items()):
                continue  # D8 metadata filter (mirrors the OpenSearch term filter)
            if doc_ids and record_doc_id not in doc_ids:
                continue  # document scope (mirrors the OpenSearch `terms` clause on top-level doc_id)
            if wanted_levels and level not in wanted_levels:
                continue  # retrieval grain (mirrors the OpenSearch `terms` clause on top-level level)
            toks = _tokens(text)
            base = overlap = sum(1 for t in toks if t in q)
            score = (base / math.sqrt(len(toks))) if (toks and overlap) else 0.0  # tf-ish
            if exact_query_terms:
                # A verbatim whole-token hit scores the chunk even when the fragmented overlap is 0
                # (the "shredded to a miss" case), and lifts it above a fragment-only competitor.
                text_terms = set(text.split())
                score += boost * sum(1 for t in exact_query_terms if t in text_terms)
            if chem_active:
                # Canonical chemical tokens are matched against the chunk's OWN chem terms only (the
                # separate `chem` field), never its text — so a non-chemistry chunk scores 0 here.
                doc_chem = set(payload.get(CHEM_TERMS_META, "").split())
                if doc_chem:
                    score += boost * sum(1 for t in chem_query_terms if t in doc_chem)
            if score > 0.0:
                scored.append((cid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete_tenant(self, tenant_id: str) -> None:
        self._i.pop(sparse_index(tenant_id), None)


class InMemoryObjectStore:
    def __init__(self) -> None:
        self._o: dict[str, bytes] = {}

    async def put(self, tenant_id: str, key: str, data: bytes, content_type: str) -> str:
        full = f"{object_prefix(tenant_id)}{key}"
        self._o[full] = data
        return f"mem://{full}"

    async def put_stream(self, tenant_id: str, key: str, stream: IO[bytes], length: int, content_type: str) -> str:
        full = f"{object_prefix(tenant_id)}{key}"
        self._o[full] = stream.read()  # twin: fine to buffer in tests
        return f"mem://{full}"

    async def get(self, tenant_id: str, key: str) -> bytes:
        return self._o[f"{object_prefix(tenant_id)}{key}"]

    async def delete(self, tenant_id: str, key: str) -> None:
        self._o.pop(f"{object_prefix(tenant_id)}{key}", None)


class InMemoryExternalFetcher:
    """Twin for :class:`ExternalObjectFetcher` (FR-A2b). Seeded per (endpoint, bucket, key).

    Still runs ``assert_sovereign`` so the sovereignty guarantee is provable without a
    live external bucket — a non-allowlisted endpoint is refused here exactly as in prod.
    """

    def __init__(self) -> None:
        self._o: dict[tuple[str, str, str], bytes] = {}

    def seed(self, endpoint: str, bucket: str, key: str, data: bytes) -> None:
        self._o[(endpoint, bucket, key)] = data

    async def fetch(
        self, *, endpoint: str, bucket: str, key: str, access_key: str, secret_key: str, secure: bool
    ) -> bytes:
        assert_sovereign(f"{'https' if secure else 'http'}://{endpoint}")
        try:
            return self._o[(endpoint, bucket, key)]
        except KeyError as exc:
            raise KeyError(f"{bucket}/{key}") from exc

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
        assert_sovereign(f"{'https' if secure else 'http'}://{endpoint}")
        keys = sorted(k for (ep, b, k) in self._o if ep == endpoint and b == bucket and k.startswith(prefix))
        return [k for k in keys if k > start_after][:limit] if start_after else keys[:limit]


# The ordering key `PostgresCatalog.document_chunks` sorts on, expressed once here so the twin and
# the real adapter cannot drift. `ORDER BY page NULLS FIRST, span.start NULLS FIRST, sequence NULLS
# FIRST, chunk_id` — NULL sorts first, hence the -1 sentinel. The twin previously sorted stably on
# `span.start or 0` while Postgres tied every v2 chunk on NULL, so the in-memory profile was
# deterministic, the real one was not, and no test could tell.
def _document_order(c: Chunk) -> tuple[int, int, int, str]:
    return (
        c.page if c.page is not None else -1,
        c.span.start if c.span is not None else -1,
        c.sequence if c.sequence is not None else -1,
        c.chunk_id,
    )


class InMemoryCatalog:
    def __init__(self) -> None:
        self._docs: dict[str, dict[str, DocumentEnvelope]] = defaultdict(dict)
        self._chunks: dict[str, dict[str, Chunk]] = defaultdict(dict)
        self._config: dict[str, dict[str, str]] = defaultdict(dict)
        # Mirrors the RAG v2 `documents.minio_layout_key` column. Kept OUT of the
        # DocumentEnvelope: it is a per-document catalog fact the platform records, not part of
        # the contract a source submits, and Postgres holds it as its own column for the same
        # reason. A separate map also means a re-register cannot blank it, matching the Postgres
        # adapter's ON CONFLICT.
        self._layout_keys: dict[str, dict[str, str]] = defaultdict(dict)
        # Mirrors the RAG v2 `sections` / `blocks` tables, keyed tenant -> doc_id -> rows. Keyed by
        # DOCUMENT rather than by row id because every operation on them is document-scoped
        # (replace-the-document's-tree, purge-the-document), which is exactly what the Postgres
        # adapter's `WHERE doc_id = :d` does.
        self._sections: dict[str, dict[str, list[SectionRow]]] = defaultdict(lambda: defaultdict(list))
        self._blocks: dict[str, dict[str, list[BlockRow]]] = defaultdict(lambda: defaultdict(list))
        # Mirrors `doc_relations` + `sections.amends`, keyed tenant -> doc_id -> relations.
        # `replace_document_profile` writes the `doc_relations` half (Phase 3); a test fills it
        # directly for the `sections.amends` half, which Phase 4 produces.
        self._relations: dict[str, dict[str, list[DeclaredRelation]]] = defaultdict(lambda: defaultdict(list))
        # Mirrors the RAG v2 profile-card columns on `documents` plus the `facets` table. Held
        # beside the envelope rather than on it for the reason `_layout_keys` is: the card is a
        # platform-derived fact about the document, not part of the contract a source submits.
        self._profiles: dict[str, dict[str, DocProfile]] = defaultdict(dict)
        # Mirrors the Phase-4 enrichment columns on `sections`, keyed tenant -> doc_id ->
        # section_id. Held per document for the same reason `_sections` is: every operation on it
        # is document-scoped, which is what the Postgres adapter's `WHERE doc_id = :d` does.
        self._enrichment: dict[str, dict[str, dict[str, SectionEnrichment]]] = defaultdict(lambda: defaultdict(dict))
        self._manifests: dict[str, dict[str, EnrichmentManifest]] = defaultdict(dict)
        # Mirrors the `facets.revision` column: the document revision a stored facet vocabulary
        # describes. Kept beside `_profiles` rather than on the card, because Postgres keeps it on
        # the `facets` ROWS and a twin that hung it on the profile object would let a test pass on
        # a shape production does not have.
        self._facet_revisions: dict[str, dict[str, str]] = defaultdict(dict)
        # Mirrors the RAG v2 `query_log` table (Phase 12), tenant -> rows in insert order. This
        # twin exists where the others in this class do not have to: `query_log` had no writer
        # until Phase 12, and `factory.purge_query_log_citing` could honestly answer 0 for the
        # in-memory profile because nothing wrote it. Now something does, and a twin that stored
        # nothing would make the unit and e2e gates blind to both the write and the erasure.
        self._query_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._query_log_seq = 0
        # Mirrors the RAG v2 `map_cache` table (Phase 9), tenant -> cache_key -> entry. Keyed by
        # the cache key because that is how a READ addresses it, with `section_id` carried on the
        # entry because that is how a PURGE addresses it — the same two access paths the Postgres
        # table has an index and a DELETE for.
        self._map_cache: dict[str, dict[str, MapCacheEntry]] = defaultdict(dict)

    async def register_document(self, doc: DocumentEnvelope) -> None:
        self._docs[doc.tenant_id][doc.doc_id] = doc

    async def list_documents(self, tenant_id: str) -> list[DocumentEnvelope]:
        return sorted(self._docs[tenant_id].values(), key=lambda d: d.ingested_at, reverse=True)

    async def record_chunks(self, tenant_id: str, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[tenant_id][c.chunk_id] = c

    async def document_relations(self, tenant_id: str, doc_id: str) -> list[DeclaredRelation]:
        """Twin of ``PostgresCatalog.document_relations``.

        BOTH HALVES ARE LIVE: :meth:`replace_document_profile` writes the `doc_relations` half
        (Phase 3) and :meth:`replace_section_enrichment` the `sections.amends` half (Phase 4), so
        the graph seeding that Phase 6 shipped inert has a producer at both grains under the
        in-memory profile too. The section half is derived on read rather than stored twice, which
        is what the Postgres adapter does — it reads `sections` in the same query.
        """
        declared = list(self._relations.get(tenant_id, {}).get(doc_id, ()))
        return declared + self._section_relations(tenant_id, doc_id)

    def _section_relations(self, tenant_id: str, doc_id: str) -> list[DeclaredRelation]:
        """The `sections.amends` half, in the exact shape the Postgres query returns it: the
        section's BREADCRUMB as the source node's name, the target as the document wrote it, and
        the relation type implied by the column."""
        crumbs = {row.section_id: row.breadcrumb for row in self._sections.get(tenant_id, {}).get(doc_id, ())}
        return [
            DeclaredRelation(
                rel_type="amends",
                source_ref=crumbs.get(section_id, ""),
                target_ref=e.amends.target_ref,
                evidence_quote=e.amends.quote,
            )
            for section_id, e in self._enrichment.get(tenant_id, {}).get(doc_id, {}).items()
            if e.amends is not None
        ]

    async def replace_document_profile(
        self, tenant_id: str, doc_id: str, profile: DocProfile, *, revision: str = ""
    ) -> bool:
        """Twin of ``PostgresCatalog.replace_document_profile``: REPLACE, and False with no row.

        The relations are converted to :class:`DeclaredRelation` exactly as the Postgres query
        does: `source_ref` is the document's title (its `doc_id` where the card has none), and
        `target_ref` is what the document wrote, unresolved. A twin that stored them in the
        profile's own shape would make the seeding tests pass on a shape production never sees.
        """
        if doc_id not in self._docs.get(tenant_id, {}):
            return False
        self._profiles[tenant_id][doc_id] = profile
        self._facet_revisions[tenant_id][doc_id] = revision
        # REPLACE: the document's own declared relations are dropped before the new card's are
        # added, and a Phase-4 `sections.amends` entry a test injected is NOT one of them --
        # Postgres keeps those in `sections`, which this method does not touch.
        kept = [
            r for r in self._relations[tenant_id][doc_id] if r.rel_type == "amends" and r.source_ref != profile.title
        ]
        self._relations[tenant_id][doc_id] = kept + [
            DeclaredRelation(
                rel_type=r.rel_type,
                source_ref=profile.title or doc_id,
                target_ref=r.target_ref,
                evidence_quote=r.evidence_quote,
            )
            for r in profile.relations
        ]
        return True

    async def replace_enrichment_manifest(self, tenant_id: str, doc_id: str, manifest: EnrichmentManifest) -> bool:
        """Twin of ``PostgresCatalog.replace_enrichment_manifest``.

        REFUSES A DOCUMENT IT DOES NOT KNOW, because the Postgres statement is an UPDATE on
        ``documents`` and hits nothing for an unregistered doc_id. A twin that stored the manifest
        anyway would let a test pass on a shape production cannot produce -- the same trap
        :meth:`replace_section_enrichment` names about section rows."""
        if doc_id not in self._docs.get(tenant_id, {}):
            return False
        self._manifests[tenant_id][doc_id] = manifest
        return True

    async def facet_vocabulary(self, tenant_id: str, doc_id: str) -> FacetVocabulary | None:
        """Twin of ``PostgresCatalog.facet_vocabulary``: None when there is nothing to reuse.

        None for an unprofiled document, for a card with no facets, and for one stored without a
        revision — the same three cases the Postgres query returns nothing for, so a reuse test
        behaves the same against either backend."""
        profile = self._profiles.get(tenant_id, {}).get(doc_id)
        revision = self._facet_revisions.get(tenant_id, {}).get(doc_id, "")
        if profile is None or not profile.facets or not revision:
            return None
        return FacetVocabulary(
            revision=revision,
            facets=tuple(
                CanonicalFacet(facet_id=f.facet_id, name=f.name, description=f.description, aliases=tuple(f.aliases))
                for f in profile.facets
            ),
        )

    async def enrichment_manifests(
        self, tenant_id: str, doc_ids: Sequence[str]
    ) -> dict[str, tuple[EnrichmentManifest, str]]:
        """Twin of ``PostgresCatalog.enrichment_manifests``: manifest + the row's CURRENT hash.

        An absent document means "no readable manifest", which a caller must not read as
        "nothing failed". The current hash comes from the registered envelope, exactly as the
        Postgres query reads it from the ``documents`` row, so a staleness test behaves the same
        against either backend."""
        stored = self._manifests.get(tenant_id, {})
        docs = self._docs.get(tenant_id, {})
        return {
            doc_id: (stored[doc_id], getattr(docs.get(doc_id), "content_hash", "") or "")
            for doc_id in doc_ids
            if doc_id in stored
        }

    async def replace_section_enrichment(
        self, tenant_id: str, doc_id: str, enrichments: Mapping[str, SectionEnrichment]
    ) -> int:
        """Twin of ``PostgresCatalog.replace_section_enrichment``: UPDATE-only, returns rows hit.

        UPDATE-ONLY is the part worth mirroring exactly. The Postgres statement matches on
        ``section_id AND doc_id``, so an enrichment naming a section this document has no row for
        updates nothing; a twin that simply stored every key would make a test pass on a shape
        production cannot produce, and the shortfall count is precisely the signal the Postgres
        adapter warns on."""
        if not enrichments:
            return 0
        known = {row.section_id for row in self._sections.get(tenant_id, {}).get(doc_id, ())}
        hit = {sid: e for sid, e in enrichments.items() if sid in known}
        self._enrichment[tenant_id][doc_id].update(hit)
        return len(hit)

    async def section_enrichment(self, tenant_id: str, doc_id: str) -> dict[str, SectionEnrichment]:
        """The stored enrichment, for tests and for the in-memory profile's own reads. Postgres
        has no counterpart, for the reason :meth:`document_profile` gives: nothing in this build
        reads an enrichment back out of the catalog, and a query with no caller is a query nobody
        has checked."""
        return dict(self._enrichment.get(tenant_id, {}).get(doc_id, {}))

    async def stale_profiles(self, tenant_id: str, *, below: int) -> list[str]:
        """Twin of ``PostgresCatalog.stale_profiles``: cardless documents count as stale.

        Cardless is the commonest stale case — every document ingested before the v2 flag — so
        a twin that returned only the profiled-but-old ones would make the CLI's tests pass on
        the rare half of its input."""
        profiles = self._profiles.get(tenant_id, {})
        return sorted(
            doc_id
            for doc_id in self._docs.get(tenant_id, {})
            if doc_id not in profiles or profiles[doc_id].profile_version < below
        )

    async def read_map_cache(self, tenant_id: str, keys: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Twin of ``PostgresCatalog.read_map_cache``.

        A twin that stored nothing and always missed would be simpler and would make the cache
        tests vacuous: every assertion about a HIT would pass against a function that never
        returns one. So this stores."""
        if not keys:
            return {}
        held = self._map_cache.get(tenant_id, {})
        return {k: list(held[k].items) for k in keys if k in held}

    async def write_map_cache(self, tenant_id: str, entries: Sequence[MapCacheEntry]) -> int:
        """Twin of ``PostgresCatalog.write_map_cache``: UPSERT on the key, empty lists kept."""
        for entry in entries:
            self._map_cache[tenant_id][entry.cache_key] = entry
        return len(entries)

    def _purge_map_cache(self, tenant_id: str, section_ids: set[str]) -> int:
        """Drop every cached entry for these sections.

        The twin of the ``DELETE ... WHERE section_id IN (...)`` that ``replace_document_structure``
        and ``delete_document`` both run in Postgres, and the reason ``section_id`` is stored on
        the entry rather than only hashed into the key: a cache nobody can purge by section
        outlives the text it describes."""
        held = self._map_cache.get(tenant_id)
        if not held:
            return 0
        doomed = [k for k, e in held.items() if e.section_id in section_ids]
        for key in doomed:
            held.pop(key, None)
        return len(doomed)

    async def document_sections(self, tenant_id: str, doc_ids: Sequence[str]) -> list[SectionCard]:
        """Twin of ``PostgresCatalog.document_sections``: every section of ``doc_ids``, in
        document order.

        THE JOIN IS THE PART WORTH MIRRORING. Postgres holds the Phase-2 identity columns and the
        Phase-4 derived ones on ONE row, so a section whose enrichment never ran still comes back
        — with an empty summary, no facets and ``amends`` false — and is still selectable by the
        reranker on its breadcrumb alone. This twin keeps the two halves in separate maps
        (``_sections`` and ``_enrichment``, mirroring who writes each), so it joins them here and
        must join them OUTER: a twin that returned only enriched sections would hide the ordinary
        case where enrichment is partial, which is the case the selector is built to survive.

        An empty ``doc_ids`` returns nothing, exactly as the Postgres adapter does and for the
        same reason: "nothing in scope" must not widen to the whole tenant."""
        if not doc_ids:
            return []
        wanted = set(doc_ids)
        out: list[SectionCard] = []
        for doc_id, rows in self._sections.get(tenant_id, {}).items():
            if doc_id not in wanted:
                continue
            enriched = self._enrichment.get(tenant_id, {}).get(doc_id, {})
            for row in rows:
                e = enriched.get(row.section_id)
                out.append(
                    SectionCard(
                        section_id=row.section_id,
                        doc_id=row.doc_id,
                        breadcrumb=row.breadcrumb,
                        heading=row.heading or "",
                        summary=e.summary if e else "",
                        facet_ids=e.facet_ids if e else (),
                        amends=bool(e and e.amends is not None),
                        ordinal=row.ordinal,
                        depth=row.depth,
                    )
                )
        out.sort(key=lambda c: (c.doc_id, c.ordinal, c.section_id))
        return out

    async def document_cards(
        self, tenant_id: str, *, doc_ids: Sequence[str] | None = None, limit: int = 20
    ) -> list[DocCard]:
        """Twin of ``PostgresCatalog.document_cards``: PROFILED documents only, newest first.

        The profiled-only filter is the part worth mirroring exactly. A twin that returned every
        registered document would hand the router a scope of cardless entries and make its tests
        pass on a shape production cannot produce — the router's whole input is the card."""
        profiles = self._profiles.get(tenant_id, {})
        wanted = set(doc_ids) if doc_ids else None
        docs = [
            doc
            for doc_id, doc in self._docs.get(tenant_id, {}).items()
            if doc_id in profiles and (wanted is None or doc_id in wanted)
        ]
        docs.sort(key=lambda d: (d.ingested_at, d.doc_id), reverse=True)
        cards: list[DocCard] = []
        for doc in docs[: max(1, limit)]:
            profile = profiles[doc.doc_id]
            cards.append(
                DocCard(
                    doc_id=doc.doc_id,
                    title=profile.title or doc.doc_id,
                    doc_type=profile.doc_type,
                    purpose=profile.purpose,
                    facet_ids=tuple(f.facet_id for f in profile.facets),
                )
            )
        return cards

    async def record_query_log(
        self,
        tenant_id: str,
        *,
        query: str,
        answer: str,
        cited_doc_ids: Sequence[str],
        latency_ms: Mapping[str, int],
        router_json: str | None = None,
        selected_sections: Sequence[str] | None = None,
        verification: str | None = None,
    ) -> int:
        """Twin of ``PostgresCatalog.record_query_log``: APPEND, and a monotonic row id.

        The id counter is process-wide rather than per tenant, mirroring a ``BIGSERIAL`` closely
        enough for the only thing any caller does with it (log it), and deliberately NOT restarting
        per tenant — a test that asserted on a per-tenant id would be asserting on something
        Postgres does not promise."""
        self._query_log_seq += 1
        self._query_log[tenant_id].append(
            {
                "id": self._query_log_seq,
                "ts": datetime.now(UTC),
                "query": query,
                "answer": answer,
                "cited_doc_ids": list(cited_doc_ids),
                "latency_ms": dict(latency_ms),
                "router_json": router_json,
                # `is not None`, NOT a truthiness test: Phase 8 made the empty case meaningful.
                # An EMPTY array says the section pass RAN and selected nothing; NULL says no pass
                # ran at all. Collapsing the two would make a dashboard read "nothing selects
                # sections yet" for a strategy that ran and found nothing — the exact confusion the
                # TODO(Phase-8) note here was left to avoid.
                "selected_sections": list(selected_sections) if selected_sections is not None else None,
                "verification": verification,
            }
        )
        return self._query_log_seq

    async def delete_query_log_citing(self, tenant_id: str, doc_ids: set[str]) -> int:
        """Twin of ``PostgresCatalog.delete_query_log_citing``: array OVERLAP, not containment.

        A row goes if it cited ANY of ``doc_ids`` — a partially-erased answer still quotes the
        erased document. A row that cited nothing is never collaterally purged, which is what
        Postgres's ``&&`` does with a NULL or empty array."""
        if not doc_ids:
            return 0
        rows = self._query_log.get(tenant_id, [])
        kept = [r for r in rows if not (set(r["cited_doc_ids"]) & doc_ids)]
        removed = len(rows) - len(kept)
        self._query_log[tenant_id] = kept
        return removed

    async def query_log(self, tenant_id: str) -> list[dict[str, Any]]:
        """The stored rows, for tests and for the in-memory profile's own reads. Postgres has no
        counterpart, for the reason :meth:`document_profile` gives: nothing in this build reads a
        log row back out, and a query with no caller is a query nobody has checked."""
        return [dict(r) for r in self._query_log.get(tenant_id, [])]

    async def document_profile(self, tenant_id: str, doc_id: str) -> DocProfile | None:
        """The stored card, for tests and for the in-memory profile's own reads. Postgres has no
        counterpart yet: nothing in this build READS a card back out of the catalog — Phase 4 is
        its first reader — and a query with no caller is a query nobody has checked."""
        return self._profiles.get(tenant_id, {}).get(doc_id)

    async def chunk_hashes(self, tenant_id: str, doc_id: str) -> dict[str, str]:
        """Derived with the SAME `text_hash` the Postgres adapter writes into `chunks.chunk_hash`,
        so a re-embed skip cannot behave one way in CI and another on a real catalog."""
        return {c.chunk_id: text_hash(c.text) for c in self._chunks.get(tenant_id, {}).values() if c.doc_id == doc_id}

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """Parents excluded, as the Protocol requires — they are never indexed, so counting them
        would make an otherwise-correct parent/child KB fail reconciliation."""
        return _tally(
            c.level.value if c.level is not None else None
            for c in self._chunks.get(tenant_id, {}).values()
            if not is_parent(c)
        )

    async def delete_document(self, tenant_id: str, doc_id: str) -> list[str]:
        chunk_ids = [cid for cid, c in self._chunks[tenant_id].items() if c.doc_id == doc_id]
        for cid in chunk_ids:
            self._chunks[tenant_id].pop(cid, None)
        self._docs[tenant_id].pop(doc_id, None)
        # The row is gone, so the pointer to its layout overlay goes with it — the Postgres
        # adapter drops the whole row and cannot leave one behind either.
        self._layout_keys[tenant_id].pop(doc_id, None)
        # ...and so does its structure tree. `blocks.text` holds verbatim document text, so this is
        # the twin's half of the erasure guarantee the Postgres adapter's `delete_document` states.
        # ...the map cache first, keyed by section_id and so unreachable once the sections go.
        self._purge_map_cache(tenant_id, {row.section_id for row in self._sections[tenant_id].get(doc_id, ())})
        self._sections[tenant_id].pop(doc_id, None)
        self._blocks[tenant_id].pop(doc_id, None)
        # ...and its profile card with its `facets`/`doc_relations` rows, which the Postgres
        # adapter's `delete_document` deletes explicitly for the same erasure reason.
        self._profiles[tenant_id].pop(doc_id, None)
        self._facet_revisions[tenant_id].pop(doc_id, None)
        self._relations[tenant_id].pop(doc_id, None)
        # ...and the Phase-4 enrichment hanging off its sections. `summary` and the quotes inside
        # `key_values`/`amends` are derived from verbatim document text, so this is the same
        # erasure guarantee: the Postgres adapter drops it with the `sections` rows.
        self._enrichment[tenant_id].pop(doc_id, None)
        # ...and every query-log row that quoted it. `query_log.answer` holds text lifted out of
        # the corpus, so this is the same erasure guarantee, and the Postgres adapter deletes
        # these rows inside the same transaction as the rest of the document.
        await self.delete_query_log_citing(tenant_id, {doc_id})
        return chunk_ids

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Remove chunks without touching the document row (write-then-swap re-index)."""
        for cid in chunk_ids:
            self._chunks[tenant_id].pop(cid, None)

    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        store = self._chunks[tenant_id]
        return {cid: store[cid] for cid in chunk_ids if cid in store}

    async def document_chunks(self, tenant_id: str, doc_id: str) -> list[Chunk]:
        chunks = [c for c in self._chunks[tenant_id].values() if c.doc_id == doc_id]
        return sorted(chunks, key=_document_order)

    async def get_tenant_config(self, tenant_id: str) -> dict[str, str]:
        return dict(self._config[tenant_id])

    async def set_tenant_config(self, tenant_id: str, key: str, value: str) -> None:
        self._config[tenant_id][key] = value

    async def set_layout_key(self, tenant_id: str, doc_id: str, key: str) -> bool:
        """Twin of ``PostgresCatalog.set_layout_key``: False when the document is unknown."""
        if doc_id not in self._docs[tenant_id]:
            return False
        self._layout_keys[tenant_id][doc_id] = key
        return True

    def layout_key(self, tenant_id: str, doc_id: str) -> str | None:
        """The recorded layout-overlay key, or None — the twin's read of what Postgres keeps in
        ``documents.minio_layout_key``. Sync, like the twin's other test-facing readers."""
        return self._layout_keys[tenant_id].get(doc_id)

    async def replace_document_structure(self, tenant_id: str, tree: LayoutTree) -> tuple[int, int]:
        """Twin of ``PostgresCatalog.replace_document_structure``: REPLACE this document's stored
        section/block rows with ``tree``'s, returning ``(sections, blocks)``.

        Replace, not append — the assignment below IS the contract, and it is what makes a second
        run (and a reindex) idempotent rather than doubling the tree. There is no transaction and no
        lock here because the twin is single-process and in-memory: the two assignments cannot be
        observed half-done, which is precisely what the Postgres adapter needs a transaction and an
        advisory lock to achieve."""
        sections, blocks = flatten_layout_tree(tree)
        # The cache goes FIRST, for the reason the Postgres adapter gives: its rows are keyed by
        # `section_id`, so they must not outlive the sections they point at. A section surviving
        # the replace with the same id has still had its text replaced, so a cached extraction
        # over it is stale either way.
        self._purge_map_cache(tenant_id, {row.section_id for row in self._sections[tenant_id][tree.doc_id]})
        self._sections[tenant_id][tree.doc_id] = sections
        self._blocks[tenant_id][tree.doc_id] = blocks
        return len(sections), len(blocks)

    def document_structure(self, tenant_id: str, doc_id: str) -> tuple[list[SectionRow], list[BlockRow]]:
        """The document's stored structure rows — the twin's read of the ``sections``/``blocks``
        tables. Sync, like ``layout_key``."""
        return list(self._sections[tenant_id][doc_id]), list(self._blocks[tenant_id][doc_id])


class InMemoryAuditSink:
    def __init__(self) -> None:
        self._log: dict[str, list[AuditEvent]] = defaultdict(list)

    async def append(self, event: object) -> str:
        if not isinstance(event, AuditEvent):
            raise TypeError("AuditSink.append expects an AuditEvent")
        chain = self._log[event.tenant_id]
        prev = chain[-1].event_hash if chain else None
        sealed = event.model_copy(update={"prev_hash": prev}).sealed()
        chain.append(sealed)
        assert sealed.event_hash is not None
        return sealed.event_hash

    def events(self, tenant_id: str) -> list[AuditEvent]:
        return list(self._log[tenant_id])

    async def list_events(self, tenant_id: str, trace_id: str | None, limit: int) -> list[AuditEvent]:
        evs = [e for e in self._log[tenant_id] if trace_id is None or e.trace_id == trace_id]
        return list(reversed(evs))[:limit]

    async def count_actions(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self._log[tenant_id]:
            if since is None or e.at >= since:
                counts[e.action.value] += 1
        return dict(counts)

    async def count_actions_kb(self, kb_id: str, since: datetime | None) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self._kb_scoped(kb_id, since, None):
            counts[e.action.value] += 1
        return dict(counts)

    async def list_events_kb(
        self, kb_id: str, since: datetime | None, actions: list[str] | None, limit: int
    ) -> list[AuditEvent]:
        events = self._kb_scoped(kb_id, since, actions)
        events.sort(key=lambda e: e.at, reverse=True)
        return events[:limit]

    def _kb_scoped(self, kb_id: str, since: datetime | None, actions: list[str] | None) -> list[AuditEvent]:
        """The KB's own events UNION org-schema events that self-attribute to it
        (``payload.kb_id == kb_id``), deduped by ``event_id``. When ``kb_id`` IS the org the two
        chains are the same list, so the dedup collapses them (a true no-op union)."""
        org = org_of(kb_id)
        candidates = list(self._log[kb_id])
        if org != kb_id:
            candidates += [e for e in self._log[org] if e.payload.get("kb_id") == kb_id]
        want = set(actions) if actions else None
        seen: set[str] = set()
        out: list[AuditEvent] = []
        for e in candidates:
            if e.event_id in seen:
                continue
            if since is not None and e.at < since:
                continue
            if want is not None and e.action.value not in want:
                continue
            seen.add(e.event_id)
            out.append(e)
        return out

    async def token_totals(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for e in self._log[tenant_id]:
            if e.action.value == "generate" and (since is None or e.at >= since):
                for k in totals:
                    v = e.payload.get(k)
                    if isinstance(v, int):
                        totals[k] += v
        return totals

    async def quality_stats(self, tenant_id: str, since: datetime | None) -> dict[str, object]:
        scores: list[float] = []
        outcomes: dict[str, int] = defaultdict(int)
        abstain_reasons: dict[str, int] = defaultdict(int)
        guardrail_stages: dict[str, int] = defaultdict(int)
        feedback: dict[str, int] = defaultdict(int)
        for e in self._log[tenant_id]:
            if since is not None and e.at < since:
                continue
            if e.action.value == "faithfulness":
                v = e.payload.get("score")
                if isinstance(v, int | float):
                    scores.append(float(v))
                out = e.payload.get("outcome")
                if isinstance(out, str):
                    outcomes[out] += 1
            elif e.action.value == "abstain":
                reason = e.payload.get("reason")
                if isinstance(reason, str):
                    abstain_reasons[reason] += 1
            elif e.action.value == "guardrail":
                stage = e.payload.get("stage")
                if isinstance(stage, str):
                    guardrail_stages[stage] += 1
            elif e.action.value == "feedback":
                rating = e.payload.get("rating")
                if isinstance(rating, str):
                    feedback[rating] += 1
        low = sum(1 for x in scores if x < _FAITH_LOW)
        high = sum(1 for x in scores if x >= _FAITH_HIGH)
        return {
            "faithfulness_n": len(scores),
            "faithfulness_avg": round(sum(scores) / len(scores), 4) if scores else None,
            "faithfulness_buckets": {"low": low, "mid": len(scores) - low - high, "high": high},
            "outcomes": dict(outcomes),
            "abstain_reasons": dict(abstain_reasons),
            "guardrail_stages": dict(guardrail_stages),
            "feedback": dict(feedback),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
