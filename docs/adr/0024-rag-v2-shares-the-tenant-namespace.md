# ADR-0024 — RAG v2 indexes into the tenant's own namespace, not a parallel `_v2` one

**Status:** accepted · 2026-10-21
**Decides** the deviation flagged while scoping RAG v2 Phase 6 (upgrade plan §"Phase 6 — Indexing
fan-out (v2) and GraphRAG seeding"). **Restates that phase's acceptance criterion.**
**Builds on** ADR-0003 (Qdrant first, collection per tenant), ADR-0016 (per-KB embedder, staged
apply + write-then-swap reindex).

## Context

The upgrade plan's Phase 6 says: *"upsert to Qdrant `_v2` with full payload … index `text` +
payload in OpenSearch `_v2`"*, and its acceptance is *"fixture corpus fully indexed into `_v2`;
counts per level reconcile across Qdrant, OpenSearch, Postgres."*

There is no `_v2` namespace in this build, and there is no partial one either. `rag_common.tenancy`
is the single source of truth for how a `tenant_id` becomes a store namespace
(`vector_collection` / `sparse_index` / `object_prefix` / `pg_schema`), and the v2 work so far —
Phases 1, 2a/2b/2c and 5 — has written v2 structure into the tenant's existing schema and indexes
throughout. Phase 6 is where that either gets ratified or reversed, so it is decided here rather
than reinterpreted a second time in passing.

What a parallel namespace would be FOR, in the plan's own terms, is building and validating the v2
index beside the live v1 one and then cutting over. Three facts about this build bear on whether a
second namespace is the way to get that:

1. **The platform already has a build-then-cut-over mechanism, and it is not a second namespace.**
   The write-then-swap reindex (`svc_control_plane.pipeline`) indexes the new version of a document
   first and purges only the ids the new version no longer contains; the per-KB embedder switch
   (ADR-0016) is staged behind an explicit `POST /kb/{id}/embedder/apply` and re-embeds through
   that same path. "Rebuild without breaking reads" is therefore an existing, exercised property of
   the namespace a KB already has.
2. **A second collection is reserved for a second VECTOR SPACE.** The one parallel collection that
   does exist — `vision_collection`, the CLIP image vectors — exists because its embedder has a
   different dimensionality and a different space, so its vectors *cannot* share the text
   collection. v2 chunks are embedded by the same embedder at the same dimension as v1 chunks, so a
   separate collection would hold the same kind of vector in a second place.
3. **The cost lands on all 25 live tenant namespaces.** A `_v2` variant of each namespace helper
   doubles the namespace surface every adapter, purge path, DPDP erasure, delete, backup and
   isolation test has to cover. The isolation gate (`tests/integration/test_tenant_isolation.py`)
   proves tenant B cannot read tenant A across four stores; each new namespace per tenant is
   another dimension that proof has to hold in, for the lifetime of the platform, in exchange for a
   migration window.

## Decisions

1. **v2 chunks are indexed into the tenant's existing collection, index and schema.** No `_v2`
   namespace is introduced, now or by Phase 6. `rag_common.tenancy` keeps one namespace per tenant
   per store.

2. **`level` is what separates v1 from v2 inside that namespace.** It is a first-class `Chunk`
   field, a `chunks` column with an index on `(section_id, level)`, a Qdrant payload key and an
   OpenSearch `keyword` field. Each of the three stores can therefore answer "how many units of
   each level do you hold?" about itself.

3. **Phase 6's acceptance criterion is restated as:** *the fixture corpus is fully indexed, and
   counts per level reconcile across Qdrant, OpenSearch and Postgres **within the tenant's own
   namespace**.* The substance — every level landed in all three stores, and in equal numbers — is
   unchanged. `scripts/reconcile_v2_index.py` is that check, executable per KB and exiting non-zero
   on disagreement, so the criterion is run rather than asserted.

4. **Parent context records are excluded from the catalog side of that count.** FR-C2 parents are
   deliberately never embedded or BM25-indexed, so counting them would make a parent/child KB fail
   a reconciliation that is in fact exactly right. Parents can only ever fall in the level-less
   bucket — no v2 strategy produces one — so the exclusion cannot hide a v2 discrepancy.

5. **Rollout stays a per-document reindex behind the existing flag**, not a namespace swap: ingest
   with `RAG_V2_INGEST` on, or reindex an already-ingested document with it on.
   `pipeline_fingerprint` gains a v2 component when the flag is on, so an unconverted corpus reads
   as stale rather than as converted.

## Consequences

- A KB mid-conversion holds v1 and v2 chunks in one collection and index. That is intended and is
  what `level` is for; it is also what a `_v2` namespace would have hidden behind a cutover, at the
  cost of every page above.
- **A level-mixed KB is only as good as the retrieval path's handling of levels.** Nothing filters
  or weights by level yet (Phase 8). Until it does, a converted document's `chunk` and `row` units
  compete in the same candidate pool, which is a retrieval-quality question to measure, not an
  isolation one.
- **Reverting the decision is a real project, not a flag.** It would mean namespace helpers,
  adapters, every purge/erasure path and the isolation proof. What would justify it: needing to
  serve v1 and v2 to different callers simultaneously from the same KB, or a v2 embedder at a
  different dimension — the second being the case ADR-0016's staged apply already exists for.
- An operator cannot tell "my corpus is on v2" from a collection name. They read it from the
  per-level counts (`scripts/reconcile_v2_index.py`) and from document staleness against
  `pipeline_fingerprint`.
