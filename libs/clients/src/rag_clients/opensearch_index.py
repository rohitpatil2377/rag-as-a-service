"""OpenSearch BM25 sparse index adapter (implements :class:`SparseIndex`) — FR-C5/D1.

Isolation: one OpenSearch **index per tenant**, named via
:func:`rag_common.tenancy.sparse_index`. Searches target only the tenant's index.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import AsyncOpenSearch, helpers
from opensearchpy.exceptions import NotFoundError, RequestError, TransportError

from rag_common import get_logger, get_settings, sparse_index
from rag_common.errors import ServiceUnavailableError
from rag_core.chem import CHEM_TERMS_META
from rag_schemas import Chunk
from rag_schemas.common import ChunkLevel

_log = get_logger(__name__)

# Document/auto tags live under ONE object field rather than at the top level. Spreading them
# alongside `text`/`doc_id` meant a chunk carrying a tag called "text" replaced the indexed body
# with its tag value — a collision that cannot happen once they are namespaced.
META_FIELD = "meta"

# The aggregation bucket key for "this document has no RAG v2 level" (a v1 chunk). A sentinel is
# needed because a bucket key is a string: `count_by_level` maps it back to None. Chosen so it
# cannot collide with a real `ChunkLevel` value.
_NO_LEVEL = "__no_level__"


def _safe_key(key: str) -> str:
    """Flatten dots in a metadata key to underscores. OpenSearch reads a dot in a field name as a
    nested-object PATH, so a tag like ``field.date`` makes it try to nest under ``meta.field`` —
    which collides with the ``meta.*`` keyword mapping and fails the whole bulk with a
    ``KeywordFieldMapper cannot be cast to ObjectMapper`` class-cast (observed live on
    B7 auto-metadata: ``field.date``/``field.amount``/``field.amounts``). Flattening keeps every
    tag a single scalar keyword. Applied on BOTH the write and query paths (via ``meta_path``) so
    a D8 filter still resolves to the same field."""
    return key.replace(".", "_")


def _is_too_many_clauses(exc: RequestError) -> bool:
    """Whether an OpenSearch RequestError is the boolean maxClauseCount overflow.

    The top-level error is a generic ``search_phase_execution_exception``; the real cause
    (``too_many_clauses`` / ``maxClauseCount``) sits in the nested ``info`` root_cause, so both
    the info payload and the string form are checked."""
    blob = f"{getattr(exc, 'info', '')}{exc}".lower()
    return "too_many_clauses" in blob or "maxclausecount" in blob


def meta_path(key: str) -> str:
    """Where a D8/B7 filter key lives in the document. One function so the write path and the
    query path cannot disagree about it."""
    return f"{META_FIELD}.{_safe_key(key)}"


_MAPPING: dict[str, Any] = {
    "mappings": {
        # Tags are matched with `term`, which needs an EXACT keyword. Left to dynamic mapping
        # OpenSearch types them as analysed text, so a filter matched only values that survive
        # analysis unchanged. Measured on the running server before this: term lang="en" hit
        # (single lowercase token, by luck) while term category="Business Travel Policy"
        # returned NOTHING — and B7 auto-metadata produces exactly those multi-word values.
        # A filter that under-matches reads as "no results", never as an error.
        "dynamic_templates": [{"tags_are_keywords": {"path_match": f"{META_FIELD}.*", "mapping": {"type": "keyword"}}}],
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            # `standard` fragments chemical formulas / SMILES / identifiers; the `whitespace`
            # sub-field (Slice 2a) keeps each whole (splits ONLY on whitespace) and is
            # CASE-PRESERVING — correct for SMILES (aliphatic `C` != aromatic `c`). A built-in
            # analyzer needs no `settings.analysis` block, so this multi-field can be added to an
            # EXISTING index live via put_mapping (see _ensure_exact_subfield) with no drop/close.
            "text": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"exact": {"type": "text", "analyzer": "whitespace"}},
            },
            "page": {"type": "integer"},
            "acl": {"type": "keyword"},
            # Canonical chemical terms (Slice 6, Phase C) — a SEPARATE whitespace-analyzed field so a
            # SMILES's RDKit canonical form/InChIKey are whole, case-preserving searchable tokens the
            # augmented query matches, WITHOUT touching `text` (citations show the original text). A
            # built-in analyzer, so — like `text.exact` — it can be added to an existing index live via
            # put_mapping (see _ensure_chem_field). Empty for non-chemistry docs ⇒ matches nothing ⇒
            # additive. Only added to a fresh index when the chem-normalization flag is on.
            "chem": {"type": "text", "analyzer": "whitespace"},
            # RAG v2 retrieval grain (section|chunk|row). A KEYWORD, because what it is for is
            # counting and filtering an EXACT value — the per-level reconciliation that is Phase 6's
            # acceptance gate (ADR-0024), and later a level-scoped query leg.
            #
            # UNCONDITIONAL, unlike `chem` above, and for the reason the RAG v2 Postgres DDL is also
            # unconditional (see `postgres_catalog._v2_table_ddl`): a mapping that depends on which
            # env var was set when an index happened to be created is the schema-drift class of bug.
            # Here it is worse than drift, because OpenSearch CANNOT RETYPE A FIELD. A v2 chunk
            # reaching an index with no `level` mapping gets dynamically typed as analysed text, and
            # then every aggregation over it fails outright ("Text fields are not optimised for
            # operations that require per-document field data", observed on this deployment) and
            # every `term` filter under-matches — for the life of that index, recoverable only by
            # dropping and rebuilding it. The field costs nothing while no document carries one.
            "level": {"type": "keyword"},
            META_FIELD: {"type": "object"},
        },
    }
}

# The `chem` field spec, injected into a NEW index's mapping only when chem normalization is enabled
# (so a fresh index is byte-identical to today when the flag is off), and added live to an existing
# index by _ensure_chem_field.
_CHEM_FIELD_SPEC = {"chem": {"type": "text", "analyzer": "whitespace"}}

# The `level` field spec, for adding it live to an index that predates it (_ensure_level_field). A
# fresh index always gets it from _MAPPING — unlike `chem`, it is not flag-gated; see the note there.
_LEVEL_FIELD_SPEC = {"level": {"type": "keyword"}}


def _index_mapping() -> dict[str, Any]:
    """The create-time mapping. The `chem` field is present ONLY when the chem-normalization flag is
    on — with it off a fresh index's mapping is byte-identical to before that slice. (`level` is
    unconditional; see the note on it in `_MAPPING`.)"""
    if get_settings().chem_normalization_enabled:
        return _MAPPING
    body = {"mappings": {**_MAPPING["mappings"]}}
    props = {k: v for k, v in _MAPPING["mappings"]["properties"].items() if k != "chem"}
    body["mappings"]["properties"] = props
    return body


# OpenSearch markers for "the cluster/index is blocking writes/creates" — almost always a disk
# flood-stage watermark (data volume full) that set a cluster create-index block or flipped
# indices read-only. These are operational/capacity conditions, not a bad request, so they map
# to a 503 the caller can act on rather than an opaque 500.
_STORE_BLOCK_MARKERS = (
    "index_create_block",
    "create-index blocked",
    "cluster_block",
    "read_only",
    "read-only",
    "flood",
    "disk usage exceeded",
)


def _as_store_unavailable(exc: TransportError, index: str) -> Exception:
    """Translate an OpenSearch create/write block (disk-full / read-only) into a 503
    :class:`ServiceUnavailableError`; return the original exception otherwise (so it re-raises)."""
    marker = f"{getattr(exc, 'error', '')} {exc}".lower()
    if any(m in marker for m in _STORE_BLOCK_MARKERS):
        return ServiceUnavailableError(
            "the search index store is read-only or out of disk, so new knowledge-base indexes "
            "cannot be created right now — free disk space on the OpenSearch node and retry",
            details={"index": index, "cause": str(exc)},
        )
    return exc


class OpenSearchSparseIndex:
    """BM25 keyword index, namespaced per tenant by index name."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or get_settings().opensearch_url
        self._client = AsyncOpenSearch(hosts=[self._url])

    async def ensure_index(self, tenant_id: str) -> None:
        name = sparse_index(tenant_id)
        if not await self._client.indices.exists(index=name):
            try:
                await self._client.indices.create(index=name, body=_index_mapping())
            except TransportError as exc:
                raise _as_store_unavailable(exc, name) from exc
            _log.info("opensearch_index_created", index=name)
            return
        await self._warn_if_stale(name)
        await self._ensure_exact_subfield(name)
        await self._ensure_chem_field(name)
        await self._ensure_level_field(name)

    async def _warn_if_stale(self, name: str) -> None:
        """An index created before tags were namespaced keeps its old mapping forever.

        OpenSearch will not retype an existing field, so re-indexing documents into it is NOT
        enough — the index has to be dropped and recreated. Until then D8/B7 filters go on
        silently under-matching, which is the failure this whole change is about, so it is
        stated rather than left to be discovered."""
        try:
            mapping = (await self._client.indices.get_mapping(index=name))[name]["mappings"]
        except Exception:  # a mapping we cannot read is not a reason to fail an ingest
            return
        if any(
            META_FIELD in tpl.get(next(iter(tpl)), {}).get("path_match", "")
            for tpl in mapping.get("dynamic_templates", [])
        ):
            return
        _log.warning(
            "opensearch_index_mapping_stale",
            index=name,
            detail="this index predates namespaced, keyword-mapped tags, so metadata filters "
            "(D8/B7 and extraction provenance) will silently return fewer results. OpenSearch "
            "cannot retype an existing field: the index must be DROPPED and rebuilt, not just "
            "re-indexed into.",
        )

    async def _ensure_exact_subfield(self, name: str) -> None:
        """Add the Slice 2a ``text.exact`` whitespace sub-field to an index that predates it.

        A NEW multi-field on an existing field is one of the few mapping changes OpenSearch allows
        live — unlike retyping ``text`` itself — so this is a put_mapping, NOT a drop/rebuild.
        But a multi-field is only POPULATED when a document is (re)indexed: existing documents keep
        an empty ``text.exact`` until reindexed, so the exact-match path silently under-matches old
        content until then. That is stated (like _warn_if_stale), not left to be discovered. New
        ingests populate it automatically."""
        try:
            props = (await self._client.indices.get_mapping(index=name))[name]["mappings"].get("properties", {})
        except Exception:  # a mapping we cannot read is not a reason to fail an ingest
            return
        text_field = props.get("text", {})
        if "exact" in text_field.get("fields", {}):
            return
        try:
            # Same `text` field spec a fresh index gets: standard analyzer + the whitespace
            # `exact` sub-field (the multi-field being added live).
            text_spec = {
                "type": "text",
                "analyzer": "standard",
                "fields": {"exact": {"type": "text", "analyzer": "whitespace"}},
            }
            await self._client.indices.put_mapping(index=name, body={"properties": {"text": text_spec}})
        except Exception as exc:  # never fail an ingest on a best-effort mapping upgrade
            _log.warning("opensearch_exact_subfield_add_failed", index=name, error=str(exc)[:160])
            return
        _log.warning(
            "opensearch_exact_subfield_added",
            index=name,
            detail="added the whitespace 'text.exact' sub-field (Slice 2a exact-term matching) to an "
            "index that predated it. A new multi-field is only populated on (re)index, so EXISTING "
            "documents must be REINDEXED before an exact formula/SMILES/identifier query can match "
            "them via text.exact; new ingests populate it automatically.",
        )

    async def _ensure_chem_field(self, name: str) -> None:
        """Add the Slice-6-C ``chem`` whitespace field to an index that predates it — the same live
        put_mapping upgrade as :meth:`_ensure_exact_subfield` (a NEW field is one of the few mapping
        changes OpenSearch allows without a drop/rebuild). Gated on the flag so a chem-off deployment
        never touches an existing index's mapping. Only POPULATED on (re)index, so existing documents
        carry an empty ``chem`` until reindexed — new ingests populate it automatically."""
        if not get_settings().chem_normalization_enabled:
            return
        try:
            props = (await self._client.indices.get_mapping(index=name))[name]["mappings"].get("properties", {})
        except Exception:  # a mapping we cannot read is not a reason to fail an ingest
            return
        if "chem" in props:
            return
        try:
            await self._client.indices.put_mapping(index=name, body={"properties": _CHEM_FIELD_SPEC})
        except Exception as exc:  # never fail an ingest on a best-effort mapping upgrade
            _log.warning("opensearch_chem_field_add_failed", index=name, error=str(exc)[:160])
            return
        _log.warning(
            "opensearch_chem_field_added",
            index=name,
            detail="added the whitespace 'chem' field (Slice 6 chemical normalization) to an index "
            "that predated it. A new field is only populated on (re)index, so EXISTING chemistry "
            "documents must be REINDEXED before a canonical-SMILES query can match them; new ingests "
            "populate it automatically.",
        )

    async def _ensure_level_field(self, name: str) -> None:
        """Add the RAG v2 ``level`` keyword field to an index that predates it — the same live
        put_mapping upgrade as :meth:`_ensure_chem_field`, but NOT gated on a flag.

        It must be in place BEFORE the first v2 chunk is written, not when some env var happens to
        be set, because OpenSearch cannot retype a field: a `level` that arrives unmapped is typed
        as analysed text and stays that way, breaking every aggregation and filter over it until the
        index is dropped and rebuilt. Running it unconditionally also means the flag can be turned
        on anywhere — env, a reindex on one KB — without a mapping having to be remembered first.

        Only POPULATED on (re)index, which is exactly how a corpus adopts v2 anyway (reindex with
        the flag on), so no existing document should have had a level and lacks one."""
        try:
            props = (await self._client.indices.get_mapping(index=name))[name]["mappings"].get("properties", {})
        except Exception:  # a mapping we cannot read is not a reason to fail an ingest
            return
        if "level" in props:
            # Present, but is it the right TYPE? A `level` that was once written to an index with no
            # mapping for it is analysed text now and forever, and every count/filter over it is
            # broken — the same unrecoverable state _warn_if_stale exists to name, so it is named
            # here too rather than surfacing later as an opaque aggregation 400.
            if props["level"].get("type") != "keyword":
                _log.warning(
                    "opensearch_level_field_mistyped",
                    index=name,
                    found=props["level"].get("type"),
                    detail="`level` is not mapped as a keyword — it was written to this index "
                    "before the mapping existed. OpenSearch cannot retype a field, so per-level "
                    "counts and filters will fail until the index is DROPPED and rebuilt.",
                )
            return
        try:
            await self._client.indices.put_mapping(index=name, body={"properties": _LEVEL_FIELD_SPEC})
        except Exception as exc:  # never fail an ingest on a best-effort mapping upgrade
            _log.warning("opensearch_level_field_add_failed", index=name, error=str(exc)[:160])
            return
        _log.info(
            "opensearch_level_field_added",
            index=name,
            detail="added the RAG v2 'level' keyword field to an index that predated it; it is "
            "populated as documents are (re)indexed with the v2 ingest flag on.",
        )

    async def index(self, tenant_id: str, chunks: list[Chunk]) -> None:
        name = sparse_index(tenant_id)
        actions = [{"_index": name, "_id": c.chunk_id, "_source": self._source(c)} for c in chunks]
        await helpers.async_bulk(self._client, actions, refresh=True)

    @staticmethod
    def _source(c: Chunk) -> dict[str, object]:
        source: dict[str, object] = {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "tenant_id": c.tenant_id,
            "text": c.text,
            "page": c.page,
            "acl": c.acl,
            # Document/auto tags (FR-A7/B7), namespaced so they can never overwrite a
            # field of the record they describe. Keys are dot-flattened (see _safe_key):
            # a dotted tag would make OpenSearch nest under meta.<prefix> and fail the bulk.
            META_FIELD: {_safe_key(k): v for k, v in c.metadata.items()},
        }
        # Canonical chemical terms (Slice 6, Phase C) — lifted from chunk metadata into the dedicated
        # whitespace `chem` field so they are tokenized/searchable. Written ONLY when the chunk
        # carries them (chem normalization was on + the KB is chemistry at ingest); absent otherwise,
        # so the written `_source` is byte-identical to before this slice for every non-chem chunk.
        chem = c.metadata.get(CHEM_TERMS_META)
        if chem:
            source["chem"] = chem
        # RAG v2 retrieval grain. Written ONLY when the chunk has one, so a v1 chunk's `_source` is
        # byte-identical to before — and so "how many chunks of each level does this index hold?",
        # the Phase 6 acceptance question (ADR-0024), is answerable from the index itself rather
        # than inferred from the catalog.
        if c.level is not None:
            source["level"] = c.level.value
        return source

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
        name = sparse_index(tenant_id)
        settings = get_settings()
        exact_on = settings.exact_term_match_enabled
        # Chem is gated on the RESOLVED terms, NOT on the master flag: a non-chemistry / flag-off
        # query carries no chem_terms ⇒ no chem clause ⇒ no clause-cap change ⇒ byte-identical, even
        # when the master flag is on for other (chemistry) KBs in the same deployment.
        chem_active = bool(chem_terms)
        # Defense in depth (belt-and-braces alongside the API-level QueryRequest.max_length):
        # a `match` query becomes one boolean clause PER TOKEN under the hood, and OpenSearch's
        # default maxClauseCount is 1024 -- an oversized query raises an uncaught RequestError.
        # Any caller of this client (not only the HTTP API) gets the same floor. When the exact
        # sub-field / chem field are queried too, each contributes a clause per token, so the
        # per-field cap is halved to keep the COMBINED count within maxClauseCount.
        _MAX_QUERY_TERMS = 256 if (exact_on or chem_active) else 512  # noqa: N806
        terms = query.split()
        if len(terms) > _MAX_QUERY_TERMS:
            query = " ".join(terms[:_MAX_QUERY_TERMS])
        filter_clauses: list[dict[str, object]] = [{"term": {meta_path(k): v}} for k, v in (filters or {}).items()]
        if doc_ids:
            # Document scope (RetrievalQuery.doc_ids) — a predicate on the TOP-LEVEL `doc_id`
            # keyword field, i.e. the chunk RECORD, NOT a `meta.*` tag. It is deliberately NOT
            # routed through `filters`: every D8 filter key is namespaced to `meta.<key>` by
            # `meta_path`, so a `doc_id` FILTER means the tag and matched nothing here (measured:
            # 181 hits on `doc_id`, 0 on `meta.doc_id`) — document-scoped queries silently
            # returned no BM25 at all. One `terms` clause expresses the whole OR, and it counts as
            # ONE boolean clause regardless of how many values it carries, so the `_MAX_QUERY_TERMS`
            # budget above is unaffected.
            filter_clauses.append({"terms": {"doc_id": list(doc_ids)}})
        if levels:
            # Retrieval grain (RetrievalQuery.levels, RAG v2 Phase 8) — the top-level `level`
            # keyword field this adapter already maps, for the reason its mapping note gave
            # ("later a level-scoped query leg"). A `meta.level` filter would have been the same
            # silent miss `doc_id` was, which is why `level` is now a reserved filter key. One
            # `terms` clause, so the `_MAX_QUERY_TERMS` budget above is unaffected.
            filter_clauses.append({"terms": {"level": [lv.value for lv in levels]}})
        if exact_on or chem_active:
            # Match the base `text` plus whichever whole-token fields are enabled; any one qualifies a
            # hit (minimum_should_match: 1). `text.exact` keeps verbatim technical tokens (Slice 2a);
            # `chem` matches a molecule's canonical SMILES/InChIKey (Slice 6, Phase C) — from the
            # server-resolved `chem_terms`, NOT the raw query text — so an equivalent-but-differently-
            # drawn molecule matches. `chem` is empty for non-chemistry docs ⇒ matches nothing ⇒
            # additive. When only `exact_on`, the body is byte-identical to the Slice-2a body.
            boost = exact_boost if exact_boost is not None else settings.exact_term_boost
            should: list[dict[str, object]] = [{"match": {"text": query}}]
            if exact_on:
                should.append({"match": {"text.exact": {"query": query, "boost": boost}}})
            if chem_active:
                should.append({"match": {"chem": {"query": " ".join(chem_terms), "boost": boost}}})
            bool_query: dict[str, object] = {"should": should, "minimum_should_match": 1}
            if filter_clauses:
                bool_query["must"] = filter_clauses
        else:
            # Flag OFF: byte-identical to the original single-field query body.
            must: list[dict[str, object]] = [{"match": {"text": query}}]
            must.extend(filter_clauses)
            bool_query = {"must": must}
        body = {"size": top_k, "query": {"bool": bool_query}}
        try:
            res = await self._client.search(index=name, body=body)
        except NotFoundError:
            # A tenant that has never ingested anything has no index yet — that is
            # "no results", not an error. Matches the VectorStore adapter's behaviour.
            return []
        except RequestError as exc:
            # The exact sub-field ~doubles boolean clauses AND the `standard` leg shreds each
            # whitespace token into several, so a pathological formula/SMILES-dense query can still
            # blow maxClauseCount even after the halved per-field cap. Degrade THAT query to the
            # text-only body (identical to the flag-off path) — the correct, grounded fallback for
            # an oversized query — rather than 500 the whole answer. Only when we added the exact
            # clause; anything else re-raises unchanged.
            if not ((exact_on or chem_active) and _is_too_many_clauses(exc)):
                raise
            _log.warning("opensearch_exact_query_degraded_text_only", index=name, terms=len(terms))
            # The degraded body drops the SCORING clauses only. Every filter clause — D8 tags, the
            # `doc_ids` document scope AND the `levels` grain — rides along: a scoped query that
            # quietly widened to the whole KB would be a worse failure than the one being
            # degraded around.
            fallback = {"size": top_k, "query": {"bool": {"must": [{"match": {"text": query}}, *filter_clauses]}}}
            res = await self._client.search(index=name, body=fallback)
        return [(h["_source"]["chunk_id"], float(h["_score"])) for h in res["hits"]["hits"]]

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """One ``terms`` aggregation with ``size: 0`` — counted in OpenSearch, no documents fetched.

        ``missing`` buckets the documents that have no ``level`` at all, which is what makes this
        correct on an index predating the field: verified against this deployment's live
        ``t_public__company_wide_bm25`` (181 documents, ``level`` unmapped) — all 181 land in the
        missing bucket rather than the aggregation failing.
        """
        name = sparse_index(tenant_id)
        body = {
            "size": 0,
            # Generously above the three levels that exist, so a bucket can never be dropped into
            # `sum_other_doc_count` and silently undercount the thing being reconciled.
            "aggs": {"levels": {"terms": {"field": "level", "missing": _NO_LEVEL, "size": 50}}},
        }
        try:
            res = await self._client.search(index=name, body=body)
        except NotFoundError:
            return {}  # never ingested ⇒ nothing indexed, not an error
        agg = res["aggregations"]["levels"]
        if agg.get("sum_other_doc_count"):
            _log.warning("opensearch_level_buckets_truncated", index=name, other=agg["sum_other_doc_count"])
        return {
            (None if b["key"] == _NO_LEVEL else str(b["key"])): int(b["doc_count"])
            for b in agg["buckets"]
            if b["doc_count"]
        }

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        name = sparse_index(tenant_id)
        actions = [{"_op_type": "delete", "_index": name, "_id": cid} for cid in chunk_ids]
        await helpers.async_bulk(self._client, actions, refresh=True, raise_on_error=False)

    async def delete_tenant(self, tenant_id: str) -> None:
        name = sparse_index(tenant_id)
        if await self._client.indices.exists(index=name):
            await self._client.indices.delete(index=name)

    async def aclose(self) -> None:
        await self._client.close()
