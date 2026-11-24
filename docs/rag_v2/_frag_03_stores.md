## Stores: Qdrant, OpenSearch, Postgres, MinIO, Graph

Read-only discovery. Paths relative to the repository root. "live dev stack" blocks are read-only queries run against the running containers (postgres, qdrant, opensearch, minio).

Tenant model used throughout: a "tenant" in store code is a KB id (`{org}` or `{org}__{slug}`; `libs/common/src/rag_common/tenancy.py:88-97`). Every store namespace is derived from it by `_slug` (lowercase, `-` -> `_`; `tenancy.py:32-35`).

### 1. Qdrant

**Collection naming (per tenant/KB).**
- Text: `t_<slug>_chunks` (`tenancy.py:58-60`). Cross-modal CLIP images: `t_<slug>_vchunks` (`tenancy.py:63-70`). Prefix `_NS_PREFIX = "t_"` (`tenancy.py:29`).
- One collection per KB, no shared collection (`qdrant_store.py:3-6`). `QdrantVectorStore.__init__` takes an injectable `namespace` callable defaulting to `vector_collection` (`qdrant_store.py:126-132`).
- Point id = `uuid5(NAMESPACE, chunk_id)`, so re-upserts overwrite (`qdrant_store.py:27-28, 51-52`).

**Vector config. Two shapes exist.**
- Legacy unnamed, dense only: `VectorParams(size=dim, distance=COSINE, on_disk=settings.qdrant_on_disk)` plus optional scalar/binary quantization, shards, replication (`qdrant_store.py:66-79`). This is the default and what almost every live collection uses.
- "Slice 2b" named multivector, used only when `multivector_enabled` and the embedder is `bge_m3_multi` (`qdrant_store.py:82-111`; `services/indexing/.../service.py:133-141`):
  - named `dense`: size=dim, COSINE.
  - named `colbert`: size=dim, COSINE, `MultiVectorConfig(MAX_SIM)`, FLOAT16, `on_disk=True`, `HnswConfigDiff(m=0)` (rescore-only, no independent HNSW).
  - sparse vector `sparse`: `SparseVectorParams(modifier=IDF)`.
- dim comes from the embedder (`RAG_EMBED_DIM`). Live: 1024.
- Migration from unnamed to named is delete-and-recreate. A populated unnamed collection is never auto-dropped (`qdrant_store.py:198-246`).

**Payload fields written (complete list).**
- Base, from `upsert` (`qdrant_store.py:156-163`) and `upsert_multi` (`qdrant_store.py:265-272`): `chunk_id`, `doc_id`, `tenant_id`, `page`, `acl`, plus `**c.metadata`. Metadata is `dict[str,str]` (`libs/schemas/src/rag_schemas/chunk.py:49`).
- No chunk `text` in the payload (comment at `indexing/service.py:283`: "payload has no text"). Text lives only in Postgres `chunks.record` and the OpenSearch `text` field. `section` is inside `metadata`, so it is in the payload.
- Metadata keys seen live (see block below): `lang`, `doc_type`, `entities`, `field.amount(s)`, `extraction_*` (extractor, capabilities, degraded, degradation, tables_found, table_pages, table_runs, table_char_share, boilerplate_lines), `ocr_*`, `sanitization_*`, `hidden_chars_removed`, `section`. Postgres metadata also carries `index_chunk_strategy` and `index_pipeline_version`.
- `classification`, `modality`, `span`, `source_uri`, `version`, `lineage_id` exist on `Chunk` (`chunk.py:29-49`) but are NOT in the Qdrant payload unless echoed into `metadata`.

**Payload indexes created: NONE.**
- `grep create_payload_index|payload_schema` over all `*.py` outside tests returns zero hits.
- Filters (`_build_filter`, `qdrant_store.py:428-434`) are `FieldCondition(MatchValue)` on whatever metadata keys the caller passes, so they run as unindexed payload scans.
- Filters are `dict[str,str]` exact-match only (no range, no `MatchAny`).

**Live (collections list)**

```text
# live dev stack: curl -s localhost:6333/collections
21 collections, all named t_<slug>_chunks, e.g.
t_public__company_wide_chunks, t_devdiag__qa_sop_diagnostic_chunks, t_acme2026__just_chunks,
t_acme__zz_multivector_test_chunks, ... (no *_vchunks exist; no *_v2)
```

```text
# live dev stack: per-collection shape + points + payload_schema (loop over all 21)
t_public__company_wide_chunks            181  unnamed []         {}
t_devdiag__qa_sop_diagnostic_chunks        7  unnamed []         {}
t_acme2026__just_chunks                    1  unnamed []         {}
t_acme__zz_multivector_test_chunks         0  named   ['sparse'] {}
t_acme__zz_mv_freshfix_test_chunks         0  named   ['sparse'] {}
(all other collections: 0 points, unnamed, {})
```

```json
# live dev stack: curl -s localhost:6333/collections/t_public__company_wide_chunks (trimmed)
"payload_schema": {}                       <-- no payload indexes
"params": {"vectors": {"size": 1024, "distance": "Cosine", "on_disk": false},
           "shard_number": 1, "replication_factor": 1, "on_disk_payload": true}
```

```json
# live dev stack: points/scroll limit=1, with_vector=false (payload keys, values abbreviated)
{"chunk_id":"SOPs - Leslie Dan ... .pdf::535383a19329","doc_id":"SOPs - Leslie Dan ... .pdf",
 "tenant_id":"public__company-wide","page":25,"acl":[],"field.amount":",","field.amounts":",, 7, 6, 8",
 "lang":"en","doc_type":"agreement","entities":"Standard Operating Procedure ...",
 "extraction_extractor":"pdfplumber","extraction_capabilities":"tables,text","extraction_degraded":"true",
 "extraction_degradation":"ocr_failed","extraction_tables_found":"true","extraction_table_pages":"37",
 "extraction_table_runs":"10","extraction_table_char_share":"0.1367","extraction_boilerplate_lines":"0",
 "ocr_applied":"true","ocr_reasons":"embedded_images","ocr_pages":"5","ocr_images":"5",
 "sanitization_risk":"clean","sanitization_hits":"0","hidden_chars_removed":"0","section":"7 Revision History"}
```

Live finding: the only populated shape is unnamed dense-only, and no live collection is multivector with data. Dotted keys like `field.amount` sit in the Qdrant payload as-is. OpenSearch flattens them to `field_amount` (see 2).

### 2. OpenSearch

**Index naming.** `t_<slug>_bm25` (`tenancy.py:73-75`), one index per KB (`opensearch_index.py:1-5`). Live: 21 `t_*_bm25` indexes, 1 primary / 1 replica (yellow on single node).

**Create-time mapping** (`opensearch_index.py:54-89`). There is NO `settings.analysis` block and NO custom analyzer. Only built-in `standard` and `whitespace`.

| Field | Type / analyzer | Cite |
|---|---|---|
| `chunk_id`, `doc_id`, `tenant_id` | keyword | :64-66 |
| `text` | text, analyzer `standard`; sub-field `text.exact` = text, analyzer `whitespace` (Slice 2a) | :72-76 |
| `page` | integer | :77 |
| `acl` | keyword | :78 |
| `chem` | text, `whitespace`; created only if `chem_normalization_enabled` | :85, :97-105 |
| `meta` | object, dynamic template `meta.*` -> keyword | :62, :86 |

- Document `_source` (`opensearch_index.py:252-272`): `chunk_id, doc_id, tenant_id, text, page, acl, meta{...}`, plus `chem` when present. Metadata keys are dot-flattened (`field.date` -> `field_date`) by `_safe_key` (`:27-35`).
- Indexed text is the contextualized text when contextual retrieval is on. The catalog keeps the original (`indexing/service.py:246-258, 284`).
- Live-upgrade helpers `_ensure_exact_subfield` (`:181-216`) and `_ensure_chem_field` (`:218-244`) `put_mapping` new fields into existing indexes. Newly added subfields stay empty until documents are re-indexed.

**Fields queries search** (`opensearch_index.py:274-341`):
- Default (`exact_term_match_enabled=False`, `config.py:988`): `bool.must[ match(text) , term(meta.<k>) filters ]` (`:317-322`).
- When `exact_term_match_enabled` (and/or chem terms): `bool.should[ match(text), match(text.exact){boost=exact_term_boost default 2.0, config.py:989}, match(chem) ]`, `minimum_should_match=1`, filters in `must` (`:301-316`).
- Query term cap is 256 (or 512 when the exact path is off) to stay under `maxClauseCount`. A `too_many_clauses` error falls back to text-only (`:296-299, 329-340`).
- No BM25 parameter tuning, no per-field multi_match, no phrase or `match_phrase` leg.

**Live (indices)**

```text
# live dev stack: curl -s 'localhost:9200/_cat/indices?v'  (abridged)
yellow open t_public__company_wide_bm25        1 1 181 docs
yellow open t_devdiag__qa_sop_diagnostic_bm25  1 1   7 docs
yellow open t_acme2026__just_bm25              1 1   1 doc
... 18 more t_*_bm25 (0 docs); no *_v2; plus .opensearch-observability, .plugins-ml-config
```

```json
# live dev stack: GET t_public__company_wide_bm25/_mapping (properties, meta elided)
{"acl":{"type":"keyword"},"chem":{"type":"text","analyzer":"whitespace"},
 "chunk_id":{"type":"keyword"},"doc_id":{"type":"keyword"},"page":{"type":"integer"},
 "tenant_id":{"type":"keyword"},
 "text":{"type":"text","fields":{"exact":{"type":"text","analyzer":"whitespace"}},"analyzer":"standard"}}
dynamic_templates: tags_are_keywords meta.* -> keyword
meta keys: doc_date doc_type entities extraction_* extractor field_amount field_amounts field_date
  hidden_chars_removed index_chunk_strategy index_pipeline_version lang ocr_* parent_id
  sanitization_* section vision_caption year        (all keyword)
# GET .../_settings?filter_path=*.settings.index.analysis  ->  {}   (no custom analysis)
```

**CRITICAL: does the analyzer preserve `61000-4-2` / `802.1x` / `IEC 62443-4-2`?**

Verdict: `text` (standard) SPLITS hyphenated standards numbers. Only `text.exact` (whitespace) keeps them whole, and `text.exact` is only queried when `exact_term_match_enabled` is true (default False, `config.py:988`).

```text
# live dev stack: POST t_public__company_wide_bm25/_analyze {"field":"<f>","text":"<q>"}
== field=text        q=61000-4-2        ['61000', '4', '2']
== field=text        q=802.1x           ['802.1x']
== field=text        q=IEC 62443-4-2    ['iec', '62443', '4', '2']
== field=text        q=EN 61000-4-2:2008 ['en', '61000', '4', '2', '2008']
== field=text.exact  q=61000-4-2        ['61000-4-2']
== field=text.exact  q=802.1x          ['802.1x']
== field=text.exact  q=IEC 62443-4-2   ['IEC', '62443-4-2']
== field=text.exact  q=EN 61000-4-2:2008 ['EN', '61000-4-2:2008']
```

Caveats on `text.exact`:
- Whitespace-only split and case-preserving. A token followed by punctuation (`62443-4-2.`, `(61000-4-2)`, `61000-4-2,`) will not match the bare form.
- There is no lowercase filter, so `iec` does not match `IEC`.
- Existing documents carry an empty `text.exact` until re-indexed (`opensearch_index.py:181-216`).
- Under `standard`, `61000-4-2` becomes `61000`,`4`,`2`. The digits `4` and `2` are near-ubiquitous, so BM25 on `text` has almost no discriminating power for this identifier. This is the real accuracy bug the plan targets. The plan's code-preserving analyzer is a net-new custom analysis block and requires a new index (see 7).

### 3. Postgres

**Layout.**
- One database `rag`, user `rag`.
- Shared: schema `public` (`tenants`, `alembic_version`) and schema `rag_identity` (global control-plane tables).
- One schema per KB: `t_<slug>` (`tenancy.py:83-85`). Live: 25 `t_*` schemas.
- Per-tenant tables are created at RUNTIME by store adapters (`CREATE ... IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`), NOT by Alembic (`libs/clients/migrations/env.py:3-6`: "Per-tenant schemas ... created idempotently by the store adapters at runtime ... a documented, deliberate split").

**Alembic revision head: `0007_tenants`** (live `public.alembic_version` = `0007_tenants`; chain below).

| Rev | File | Adds |
|---|---|---|
| 0001_baseline | versions/0001_baseline.py:20 | `public.tenants` shared tenant registry |
| 0002_rag_identity | :27-28 | `rag_identity` schema (orgs, users, api_keys, auth_tokens, approval_requests, egress_requests, eval_schedules, ingest_jobs, ingest_job_items, platform_flags, sync_schedules ...) |
| 0003_federation_grants | :24-25 | `rag_identity.federation_grants` |
| 0004_grant_delegation | :20-21 | `federation_grants.parent_grant_id` + index `ix_grants_parent` |
| 0005_eval_federate_from | :19-20 | `eval_schedules.federate_from` |
| 0006_org_units | :19-20 | `rag_identity.org_units` |
| 0007_tenants | :18-19 | `rag_identity.tenants` (managed-cloud registry) |

No Alembic migration touches any `t_<slug>` schema, `documents` or `chunks`.

**Per-tenant catalog DDL as it exists** (`libs/clients/src/rag_clients/postgres_catalog.py:27-56`, executed by `ensure_schema` `:65-71`):

```sql
CREATE SCHEMA IF NOT EXISTS "t_<slug>";
CREATE TABLE IF NOT EXISTS "t_<slug>".documents (
    doc_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    mime TEXT NOT NULL,
    uri TEXT,
    lineage_id TEXT,
    source_acl JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}',
    content_hash TEXT,
    ingested_at TIMESTAMPTZ NOT NULL
);
ALTER TABLE "t_<slug>".documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
CREATE INDEX IF NOT EXISTS documents_by_hash ON "t_<slug>".documents (content_hash);
CREATE TABLE IF NOT EXISTS "t_<slug>".chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    record JSONB NOT NULL            -- full Chunk model_dump_json (text, span, metadata, acl, classification, ...)
);
CREATE TABLE IF NOT EXISTS "t_<slug>".tenant_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Graph tables, same schema (`libs/clients/src/rag_clients/graph_store.py:148-181`):

```sql
CREATE TABLE IF NOT EXISTS "{schema}".graph_nodes (
    name_key TEXT PRIMARY KEY, display TEXT NOT NULL, chunk_ids JSONB NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS "{schema}".graph_edges (
    source TEXT NOT NULL, target TEXT NOT NULL, chunk_ids JSONB NOT NULL DEFAULT '[]',
    relation TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target));
CREATE TABLE IF NOT EXISTS "{schema}".graph_communities (
    community_id INTEGER PRIMARY KEY, label TEXT NOT NULL, size INTEGER NOT NULL,
    top_entities JSONB NOT NULL DEFAULT '[]', chunk_ids JSONB NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '', members JSONB NOT NULL DEFAULT '[]');
```

Other per-tenant tables created by control-plane modules (live columns below): `audit_log` (`postgres_audit.py:28-31`), `conversations`, `conversation_messages`, `hitl_reviews`, `knowledge_bases`, `model_connections`, `sql_connections`.

```text
# live dev stack: schemas
rag_identity, public, t_demo, t_devdiag, t_devdiag__qa_sop_diagnostic, t_acme, t_acme2026,
t_acme2026__cool, t_acme2026__dl, t_acme2026__just, t_acme__sanity_check, t_acme__zz_* (x10),
t_acme_platform_operations, t_public, t_public__company_wide, t_public__company_wide__zz_ocr_v3, t_public__ragas_cert
# public + rag_identity tables
rag_identity.{api_keys,approval_requests,auth_tokens,egress_requests,eval_schedules,federation_grants,
  ingest_job_items,ingest_jobs,org_units,orgs,platform_flags,sync_schedules,tenants,users}
public.{alembic_version,tenants};   alembic_version = 0007_tenants
```

```text
# live dev stack: columns of t_public__company_wide (information_schema)
documents  : doc_id text, source_id text, source_kind text, mime text, uri text, lineage_id text,
             source_acl jsonb, metadata jsonb, content_hash text, ingested_at timestamptz
chunks     : chunk_id text, doc_id text, page integer, version integer, record jsonb
tenant_config: key text, value text
audit_log  : seq bigint, event_id text, trace_id text, action text, actor text, at timestamptz, payload jsonb, prev_hash text, event_hash text
graph_nodes: name_key text, display text, chunk_ids jsonb
graph_edges: source text, target text, chunk_ids jsonb, relation text
graph_communities: community_id int, label text, size int, top_entities jsonb, chunk_ids jsonb, summary text, members jsonb
conversations / conversation_messages / hitl_reviews / knowledge_bases / model_connections / sql_connections (control-plane)
# indexes: only PKs + documents_by_hash(content_hash) + conv_msg_by_conv
# rows: chunks=181, documents=1; graph_nodes=382, graph_edges=3925 (relation='' for all)
```

Live finding on `chunks.record`: a full Chunk JSON (`text`, `span{start,end}`, `page`, `metadata{...}`, `acl`, `version`, `chunk_id`, `doc_id`).

**Important live finding.** There is no column or table for:
- sections / blocks / layout
- doc relations
- facets (the closest are the free-form `documents.metadata` JSONB keys `doc_type`, `lang`, `year`, `doc_date`)
- map cache or query log. `audit_log` is a hash-chained audit trail, not a query log (`postgres_audit.py:31`).

Sections are NOT persisted as rows. `detect_sections(text)` (`libs/rag-core/src/rag_core/sections.py`) runs at chunk time (`indexing/service.py:217`) and only the resulting per-chunk `section` string lands in `metadata`.

### 4. MinIO

- Single bucket `rag` (`minio_store.py:24`, default arg `:30`). Key = `object_prefix(tenant) + key` = `t_<slug>/<key>` (`tenancy.py:78-80`; `minio_store.py:50, 63`). Phase-1 note at `minio_store.py:4-5` says bucket-per-tenant + WORM is deferred.
- Interface: `put`, `put_stream`, `get`, `delete` only (`minio_store.py:48-106`). There is no list, stat, or presign.
- Objects stored today:
  - `raw/<doc_id>` = original uploaded bytes, written once by ingestion (`services/ingestion/src/svc_ingestion/service.py:51`). The key is built by `raw_object_key` (`tenancy.py:104-115`). Non-object-safe ids (URLs) become `raw/<slug>-<sha1[:12]>`.
  - Bulk-ingest staging keys (`_staging_key(job_id, doc_id)`) from `ingest_jobs.py:415, 436`.
  - Image bytes read back for multimodal (`multimodal.py:134`) and crawl re-use (`crawl_jobs.py:122`).
- Raw bytes are the replay source: `reindex_document` reads `get_object_store().get(tenant, raw_object_key(doc_id))` (`pipeline.py:548`) and re-runs extraction. Raw is deleted with the document (`pipeline.py:429, 662`).
- NOT stored: extracted text, Chandra/Datalab layout JSON/HTML, page images, blocks, bbox. grep of `doc-intelligence` shows layout/bbox handling only in `ocr.py`/`tables.py`/`classify.py` in-process; none of it is uploaded. Extracted text is not stored in MinIO or in a document-level Postgres column; only per-chunk text exists in `chunks.record`.
- Equivalent of the plan's `minio_layout_key`: NONE. No column on `documents` and no object convention for layout.

```text
# live dev stack: MinIO data dir, bucket rag (no mc in container; listed via filesystem)
/data/rag/t_public__company_wide/raw/SOPs_-_Leslie_Dan_Faculty_of_Pharmacy_-_Single_Document.pdf-dcff4f0a229f/
/data/rag/t_devdiag__qa_sop_diagnostic/raw/{SOP-QA-001,SOP-QA-002,SOP-QA-003,SOP-QA-004}
/data/rag/t_acme2026__just/raw/WhatsApp_Image_2026-09-11_at_3.48.39_PM.jpeg-ca04490f2b34
# only a raw/ prefix exists under each tenant prefix; nothing else.
```

### 5. GraphRAG

**What is extracted** (`libs/clients/src/rag_clients/graph_extract.py`):
- `LLMGraphExtractor` prompts the KB model for `{"entities":[...],"edges":[[A,B],...]}` (`graph_extract.py:29-55`; prompt `libs/rag-core/src/rag_core/graph.py:238-247`, text truncated to 2000 chars, max 16 entities per chunk `:26`). Output is UNTYPED entity names plus UNDIRECTED, UNLABELLED co-occurrence pairs. Falls back to `HeuristicGraphExtractor` (capitalised phrases) on bad JSON.
- `LLMDomainRelationExtractor` (`:58-105`): typed, directed `(subject, relation, object)` edges constrained to a KB domain-profile vocabulary. Only active with `domain_profile_enabled` and the per-KB `domain_relations_enabled` (requires `graphrag_enabled`, `pipeline.py:812-822, 946-948`).
- Diagram edges: Mermaid/vision arrows from image docs become directed labelled edges (`graph.py` `GraphTriple.diagram_edges`; `pipeline.py:991-998`, needs `diagram_graph_enabled`).
- `graph_summary.py` (`LLMCommunitySummarizer`) writes a one-line theme per community.
- Backends: `graph_extract_backend` = heuristic (default) | llm | translate (`config.py:1164`).

**Schema.**
- Nodes: `(name_key, display, chunk_ids[])`. NO entity type column.
- Edges: `(source, target, chunk_ids[], relation)`. `relation` is '' for co-occurrence, set only for diagram or domain-typed edges.
- Communities: `(community_id, label, size, top_entities, chunk_ids, summary, members)` (`graph_store.py:148-181`).
- Store is `PostgresGraphStore` in the per-KB schema; an in-memory twin is used for CI (`graph_store.py:1-12`). No Neo4j. Node key = `normalize_entity(name).lower()` (`:25`), shared across docs.
- Both live graphs are co-occurrence only: `t_public__company_wide` has 382 nodes / 3925 edges, all `relation=''`. `t_devdiag__qa_sop_diagnostic` has 40 nodes / 220 edges, all `relation=''`.

**When populated.**
- At ingest and reindex, per document, after indexing: `pipeline._index_graph(kb, doc_id)` (`pipeline.py:487, 641, 800`).
- NO-OP unless `graphrag_enabled` (default False, `config.py:1160`; `pipeline.py:812`). Parent chunks are excluded. Fail-open, never breaks ingest.
- Also seeded from connected SQL/NoSQL/REST/SharePoint source schemas at connection time (`services/control-plane/.../graph_sync.py:70-87`), and from live source answers at query time (`graph_bridge.py:128-147`, `graph_bridge_enabled`, `pipeline.py:1989`).
- Communities are built on demand and refreshed after reindex (`pipeline.py:1372-1396`, `_refresh_communities` `:641`).

**When queried.**
- `_graph_augment` (`pipeline.py:1061-1091`): when `graphrag_enabled`, extract entities from the query, call `GraphStore.linked_chunks(entities, hops=graph_retrieval_hops)`. It walks undirected one or more hops, returns the chunk ids of the entity and its neighbours, fetches them from the catalog, and merges them as graph spans, then re-scores (`_score_graph_spans`).
- Global-search path (`:1470`) when `graph_global_search` and `is_global_query(text)`, using community summaries.
- Graph is a chunk-recall booster only. It is never consulted for doc-to-doc navigation.

**Doc-to-doc or section-to-section relation edges today: NONE.**
- Nodes are entity strings, not doc or section ids. The only doc linkage is indirect: two docs mentioning the same entity name share a node via `chunk_ids`.
- Edge `relation` is only for diagram arrows and domain-profile vocab relations between entities, not documents.
- There is no table or API for supersedes / amends / references / depends_on. Plan Phase 1 `doc_relations` is entirely new.

### 6. Re-index tooling and idempotency

**What exists.**
- `scripts/reindex_corpora.py:1-69`: CLI over HTTP (`localhost:8000`, dev token). For each KB, lists documents and POSTs `/v1/kb/{kb}/documents/{doc}/reindex` with `{"strategy": clause}`, one doc at a time, prints per-doc chunk count and timing. NO dry-run flag, NO resume, NO stale filter.
- Control plane `services/control-plane/src/svc_control_plane/reindex.py`:
  - `stale_report(kb_id)` (`:72-77`) is a read-only report. A document is stale when `metadata.index_pipeline_version != pipeline_fingerprint(settings)`. This is the closest thing to a dry run. It is exposed via the API near `api.py:3180` (returns total / stale / stale_doc_ids / `auto_reindex`).
  - `reindex_kb(kb_id, stale_only=True, limit=50, heartbeat=...)` (`:80-123`): sequential, per-doc failures reported not fatal.
  - `worker_loop` / `_run_leader_pass` (`:126-210`): opt-in auto-reindex for KBs named in `RAG_AUTO_REINDEX_KBS`, Redis-lease leader-gated so exactly one worker runs.
  - API: `POST /kb/{kb_id}/reindex` (`api.py:3204`), `POST /kb/{kb_id}/documents/{doc_id}/reindex` (`api.py:3140`).
- `pipeline.reindex_document` (`pipeline.py:528-650`): reads raw bytes from MinIO, re-extracts, re-chunks, re-embeds, re-indexes, and runs a WRITE-THEN-SWAP: old chunk ids not in the new set are deleted from catalog, Qdrant, OpenSearch, image vectors and graph afterwards (`:622-633`). Failure leaves the old version serving. It then rebuilds the doc graph and refreshes communities.
- `pipeline_fingerprint` (`indexing/service.py:51-85`) hashes chunk strategy/size/overlap/max_chars, embed model/dim, contextual retrieval, graphrag + extract backend, ocr/image/vision backends, multimodal, and the multivector marker. It does NOT include any v2 schema version.

**Idempotency.**
- Ingest delta by content hash: `sha256(raw)` is stored in `documents.content_hash` (indexed `documents_by_hash`). Re-ingesting identical bytes for the same `doc_id` skips extract/embed/index and returns `outcome="unchanged"` (`pipeline.py:271-296`). Identical bytes under a different `doc_id` is flagged near-duplicate (`:299-301`).
- Chunk ids are content-derived (P44 per `reindex_corpora.py:3-5`; `chunk_id = <doc_id>::<12-hex>` live), and Qdrant point id = `uuid5(chunk_id)`. Qdrant upsert, OpenSearch bulk with `_id=chunk_id`, and Postgres `ON CONFLICT (chunk_id) DO UPDATE` are all idempotent upserts.
- `register_document` is `ON CONFLICT (doc_id) DO UPDATE` (`postgres_catalog.py:84-85`).
- No global dry-run for ingest, and no dry-run flag on reindex itself. The plan's blue/green `_v2` reindex has no existing driver.

### 7. How a `_v2` parallel index must be named (design constraint the plan missed)

All naming is centralised in `libs/common/src/rag_common/tenancy.py` and derived per KB, so a v2 index cannot be one global name like `ragaas_chunks_v2`. It must be one v2 object per KB.

Concrete names for a KB whose slug is `S` (e.g. KB `public__company-wide` has `S = public__company_wide`):

| Store | Today | v2 (coexisting) |
|---|---|---|
| Qdrant text | `t_S_chunks` | `t_S_chunks_v2` |
| Qdrant vision (leave) | `t_S_vchunks` | unchanged |
| OpenSearch | `t_S_bm25` | `t_S_bm25_v2` |
| Postgres schema | `t_S` (shared with v1) | same schema `t_S`; new tables suffixed e.g. `t_S.documents_v2`, `chunks_v2`, `sections`, `blocks`, ... |
| MinIO | `t_S/raw/<doc>` | same bucket and prefix `t_S/`; add `t_S/layout/<doc>.json` etc. |

Constraints and traps:
1. Add `vector_collection_v2(tenant)`, `sparse_index_v2(tenant)` in `tenancy.py` and thread them through `QdrantVectorStore(namespace=...)` (already injectable, `qdrant_store.py:126-132`). OpenSearch has NO namespace injection (`sparse_index` is hard-imported and used at `opensearch_index.py:144, 247, 283, 346, 351`), so it needs a new constructor parameter or a subclass.
2. Collision check: a `_v2` suffix cannot collide with another KB's v1 name. v1 Qdrant names always end in `_chunks` / `_vchunks` and OpenSearch in `_bm25`; `_slug` permits `[A-Za-z0-9_-]` only. KB `foo` v2 = `t_foo_chunks_v2`, while another KB `foo_chunks_v2` would map to `t_foo_chunks_v2_chunks`. Distinct, so safe. `assert_registrable` (`tenancy.py:38-55`) guards only slug collapse between tenant ids, not v1/v2.
3. Postgres table-name clash: `documents` and `chunks` already exist in every `t_S` schema with different shapes (see 3). The plan's new `documents`/`chunks` tables therefore need a different name or an `ALTER`. The plan's "Alembic migrations" also do not apply to tenant schemas: Alembic only manages `public` and `rag_identity` (`env.py:3-6`). Tenant tables are runtime `CREATE IF NOT EXISTS` in `PostgresCatalog._ddl` (`postgres_catalog.py:27-56`), run per tenant on first touch across 25+ schemas. A v2 schema change must be added there (or a per-tenant migration runner), otherwise it silently applies only to tenants that get touched.
4. Cleanup and delete paths enumerate the v1 names only, and must learn v2 or v2 will orphan:
   - `delete_tenant` in `QdrantVectorStore` (`qdrant_store.py:393-396`) and `OpenSearchSparseIndex` (`opensearch_index.py:350-353`).
   - KB delete / `tenant_control` provisioning (non-atomic, per `CLAUDE.md`).
   - `embedder_switch.apply` (delete_tenant then recreate).
   - Reindex swap deletes (`pipeline.py:622-633`) and DPDP erasure cascade (`pipeline.py` ~660-670).
5. `collection_dims()` preflight iterates ALL Qdrant collections (`qdrant_store.py:398-422`), so `_v2` collections get checked against `RAG_EMBED_DIM` automatically (good, but they must match it).
6. Both-read cutover: queries and indexing resolve names from the bound tenant. A flag (`RAG_V2_QUERY`) must pick the v1 or v2 name at the single `tenancy.py` seam. Dual-write during migration needs `IndexingService` to write both sets (today one fan-out at `indexing/service.py:281-285`).
7. OpenSearch analyzer change (the code-preserving analyzer) cannot be applied to the existing `text` field, since OpenSearch cannot retype or re-analyze it, and a new analyzer needs `settings.analysis` at index creation. Only a new index (`_bm25_v2`) works. Cheaper alternative: a new sub-field on the existing index via `put_mapping`, but a custom analyzer cannot be added without closing the index, and the repo's live-upgrade pattern is limited to built-in analyzers (`opensearch_index.py:67-71`).
8. Qdrant payload indexes can be added live to existing collections via `create_payload_index`, so `_v2` is not needed purely for indexes. It is needed only for new payload semantics or a changed vector config.

### Gaps vs plan (Phases 1 & 6)

- Qdrant per-tenant collection `t_<slug>_chunks`, cosine, 1024-d, deterministic uuid5 point ids, upsert idempotency: ALREADY EXISTS, reuse. Named dense+colbert+sparse variant (`qdrant_store.py:82-111`): ALREADY EXISTS, reuse as the base for `_v2`. `collection_exists`/create/`delete_chunks`/`collection_dims`: ALREADY EXISTS, reuse.
- Qdrant payload indexes: NONE exist (`payload_schema: {}` live). Gap.
- Qdrant payload lacks plan fields. Present: `chunk_id, doc_id, tenant_id, page, acl, section` (via metadata), `doc_type`, `lang`. Missing as first-class: doc family / standard id, version, `is_current`, normative flag, block type, section path/breadcrumb, `classification`, `doc_ver`. `doc_type` today is a coarse free-form tag (live value `agreement` for an SOP PDF).
- OpenSearch per-tenant index `t_<slug>_bm25`, `text` + `text.exact` whitespace sub-field, `meta.*` keyword dynamic template, `chem` field, term-cap and fallback handling: ALREADY EXISTS, reuse the exact-match scaffolding and the `exact_term_match_enabled` / `exact_term_boost` flags (`config.py:988-989`). Do not rebuild.
- OpenSearch code-preserving analyzer: ABSENT. No `settings.analysis`; `text` = standard splits `61000-4-2` into `['61000','4','2']`. `text.exact` already keeps it whole but is case-sensitive, punctuation-sensitive, off by default, and empty for pre-existing docs. Needs a new `_bm25_v2` index with a custom analyzer (word-delimiter with `preserve_original` plus lowercase, or a pattern tokenizer), not a mapping tweak.
- Postgres per-tenant `documents` (with `content_hash`, `metadata` JSONB) and `chunks` (JSONB record): ALREADY EXISTS, reuse. Name clash with the plan's tables (see 7.3). Not present: `doc_relations`, `facets`, `sections`, `blocks`, `map_cache`, `query_log`, and per-doc `minio_layout_key` / `layout`. All net-new. Tenant tables are runtime-DDL, not Alembic; the plan's migration mechanism does not match the repo (Alembic head `0007_tenants`, shared schemas only).
- MinIO: bucket `rag`, prefix `t_<slug>/`, `raw/<doc_id>` raw bytes (+ `raw_object_key` safe-key encoding): ALREADY EXISTS, reuse. No layout/extracted-text/page-image artifacts and no `minio_layout_key`; Chandra/Datalab output is not persisted. Re-extraction on every reindex is the cost. No list/stat/presign in `MinioObjectStore`.
- GraphRAG store (nodes/edges/communities), LLM/heuristic extractors, local + global retrieval, `graph_extract_backend`, `graphrag_enabled`, on-ingest population, write-then-swap graph cleanup: ALREADY EXISTS, reuse. Typed directed edges also exist for diagrams and domain vocab, limited to entity-to-entity. Doc-to-doc and section-to-section edges (supersedes / references / amends) do NOT exist. Phase 6 "graph seeding" from `doc_relations` is net-new, but the `graph_edges.relation` column is a ready home if doc nodes are encoded as node keys. The nodes have no type column, so doc/section nodes would need a `doc::`/`sec::` key convention or a new `node_type` column (runtime ALTER pattern at `graph_store.py:161-168`).
- Re-index tooling: write-then-swap `reindex_document`, `reindex_kb(stale_only)`, `stale_report`, leader-gated auto-reindex, `pipeline_fingerprint` stale detection, content-hash delta skip: ALREADY EXISTS, reuse as the blue/green driver; add a v2 marker to `pipeline_fingerprint` so the staleness report drives the migration. `scripts/reindex_corpora.py` has no dry-run and no `--stale-only`; `stale_report` is the de-facto dry run. No dual-write or cutover.
- Section detection `detect_sections` (`rag_core/sections.py`) and the per-chunk `section` metadata: ALREADY EXISTS, reuse. Section is not persisted as a row, and sections are only recognised for multi-level numbered headings by design (narrow regex). Parent/child chunking (`parent_id`, `is_parent`) and contextual retrieval (FR-C2, `contextualize_text`): ALREADY EXISTS, but covered by other slices.
- Top 3 design constraints: (1) `_v2` names must be per-KB (`t_<slug>_chunks_v2`, `t_<slug>_bm25_v2`) via new `tenancy.py` helpers, with OpenSearch lacking namespace injection; (2) tenant-schema DDL is runtime, not Alembic, and `documents`/`chunks` already exist with different shapes; (3) every delete/erase/reindex-swap path must be taught about v2 or it orphans data.
