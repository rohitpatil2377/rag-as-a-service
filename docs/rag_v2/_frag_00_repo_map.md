## Repo map & orchestration (Phase 0 task 1)

### Shape

Modular monolith (ADR-0001): seven service packages mounted as routers in one FastAPI
process, `apps/api` (`rag_api`). `apps/api/src/rag_api/app.py:43` lists `_SERVICE_ROUTERS`.

| Layer | Package | Notes |
|---|---|---|
| Ingestion | `services/ingestion` (`svc_ingestion`) | `router.py` is **health-only** (`services/ingestion/src/svc_ingestion/router.py:8`); invoked in-process by control-plane |
| Doc Intelligence | `services/doc-intelligence` | `service.py` 600 lines |
| Indexing | `services/indexing` | `service.py` 304 lines — the fan-out |
| Retrieval | `services/retrieval` | `service.py` only 28 lines; the real logic is `libs/rag-core` |
| Generation | `services/generation` | `service.py` 818 lines |
| Guardrails/Eval | `services/guardrails-eval` | |
| Control Plane | `services/control-plane` | the `/v1` surface + **the orchestrator** |

**The orchestrator is `services/control-plane/src/svc_control_plane/pipeline.py` — 5051 lines.**
Every ingest and every answer flows through it. `api.py` (the HTTP surface) is a thin
validation layer over it. Any Phase 1-12 work lands mostly here, in `libs/rag-core`, and in
`libs/clients`.

### Entry points that matter

**Ingestion:** `POST /v1/kb/{kb_id}/ingest` → `api.py:883` → `pipeline.ingest_document()`
(`pipeline.py:253`). Related: `pipeline.reindex_document()` (`pipeline.py:528`),
`pipeline.delete_document()` (`pipeline.py:647`), DAG view `/kb/{kb_id}/documents/{doc_id}/dag`,
bulk jobs under `/kb/{kb_id}/ingest/jobs/*`, `svc_control_plane/ingest_dag.py` (458 lines),
`ingest_jobs.py`, `ingest_progress.py`.

**Query:** `POST /v1/kb/{kb_id}/query` → `api.py:2400` → builds a `RetrievalQuery`
(tenant_id = kb_id, text, top_k, mode, filters, **acl**, **clearance**, after/before/freshness)
→ `pipeline.answer_query()` (`pipeline.py:4913`) → `_answer_query_impl()` (`pipeline.py:3559`).
Streaming variant `POST /kb/{kb_id}/query/stream` → `_stream_answer_impl()` (`pipeline.py:3896`).
Conversational variants: `/kb/{kb_id}/conversations/{id}/ask` (+`/stream`) →
`answer_turn()`/`stream_turn()` (`pipeline.py:4939`, `:4989`). Retrieval-only:
`POST /kb/{kb_id}/retrieve`. Federated: `answer_across()` (`:4396`), `answer_federated()` (`:4520`).

**Other relevant surfaces:** `/admin/features` + `/admin/features/{name}` (feature flags),
`/kb/{kb_id}/profile/detect` + `/profile/apply` (domain profile — **note: a profile concept
already exists**, see `svc_control_plane/domain_profile.py`, 301 lines, and
`libs/rag-core/domain_profile.py`), `/kb/{kb_id}/golden-set` + `/golden-set/generate`,
`/kb/{kb_id}/eval` + `/eval/run-now` + `/eval/schedule`, `/kb/{kb_id}/graph` +
`/graph/communities` + `/graph/subgraph`, `/kb/{kb_id}/reindex`, `/kb/{kb_id}/quality`.

### The existing answer-strategy dispatcher — ALREADY EXISTS, read before Phases 7-9

`pipeline._execute_query_plan()` (`pipeline.py:3497`) maps `plan.strategy` → an answer
mechanism. The strategy taxonomy that exists **today**:

| `plan.strategy` | Mechanism | Function |
|---|---|---|
| `whole_document` (or `passages` + `answer_shape=summary` + `coverage=exhaustive`) | 1 doc → single whole-document read; 2..cap docs → summarise each and combine; > cap → normal RAG + "did you mean…" titles; 0 → normal RAG | `_answer_document_full_summary` (`:2999`), `_answer_documents_full_summary` (`:3299`) |
| `catalog` | answer from the document catalog | `_answer_catalog` (`:2099`) |
| `count` | counting questions w/ breakdown | `_answer_count` (`:2316`) |
| `filter` | criteria-filtered doc list | `_answer_filter` (`:2391`) |
| `compare` | multi-entity comparison | `_answer_compare` (`:2450`) |
| `whole_kb`, `passages`, unresolved | the normal RAG path | `_answer_query_impl` |

The plan assumes it must build a 6-intent router (`lookup|aggregate|compare|summarize|
procedural|relational`) and per-intent strategies from scratch. It must not — there is a
planner with `strategy` + `answer_shape` + `coverage` dimensions, a `QueryPlan` contract, and
a fail-safe contract worth preserving: **an unresolved strategy returns normal RAG, so a
planner decision can never turn an answerable question into an error** (`pipeline.py:3519`).
Phase 7 is therefore an *extension* of `QueryPlan`, not a new router.

### Map→reduce ALREADY EXISTS (relevant to Phase 9)

`_answer_document_full_summary` (`pipeline.py:2999`) is documented as "a linear map→reduce with
guarded stages" and already does:

- **Fetch** every chunk of one doc in document order via `CatalogAdapter.document_chunks`,
  ACL- and clearance-filtered (`is_entitled` / `is_cleared`), capped by
  `whole_doc_summary_max_chunks` with honest `truncated` marking.
- **Single-full-read optimisation**: if the whole doc fits `min(whole_doc_full_read_token_budget,
  llm_context_window - reserve)` × safety, it does ONE generation call instead of map-reduce
  (`read_mode = "full_read" | "map_reduce"`, `pipeline.py:3060`).
- **MAP** — batches of `whole_doc_summary_batch_chunks` through the *unchanged* grounded
  generation path, so each batch extract is faithfulness-verified and cites real chunk ids;
  per-batch coverage re-prompting deliberately disabled. `_run_whole_doc_batches` (`:2806`).
- **REDUCE** — assembles batch extracts into one answer, renumbering `[n]` citation markers
  into a combined union list; grounded-or-abstain preserved end to end.
- **Coverage accounting** — `expected` = chunks read, `covered` = chunks in grounded batches;
  `_verify_whole_doc_coverage` (`:2874`).

Difference from the plan's Phase 9: today's map unit is a **batch of N chunks in document
order**, driven by token budget. The plan's map unit is a **semantically selected section**,
driven by relevance to the query and its sub-queries, and its reduce groups by **facet**. The
missing ingredients are the section tree (Phase 2) and the facet taxonomy (Phase 3) — not the
map-reduce machinery.

### Retrieval-path augmentation stages that already exist (relevant to Phase 8)

All in `pipeline.py`: `_graph_augment` (`:1061`), `_neighbour_augment` (`:1102`),
`_mark_superseded` (`:1166` — **supersession handling already exists in some form**),
`_parent_expand` (`:1213`), `_global_augment` (`:1464` — GraphRAG global search),
`_coverage_augment` (`:1578`), `_apply_doc_scope` (`:1531`) / `_doc_scope_for` (`:1519`),
`_apply_recency` (`:1639`), `_drop_navigation_spans` (`:1625`), `_with_exact_boost` (`:1868`),
`_with_glossary_expansion` (`:1927`), `_with_multivector_flags` (`:1903`),
`_cross_lingual_retrieve` (`:2014`), `_fuse_structured_evidence` (`:1971`),
`_semantic_cache_lookup` (`:1750`), `_agentic_retry` (`:1788`), `kb_threshold_overrides`
(`:1835` — **per-KB threshold overrides already exist**).

### Dev-stack reality check (live, `docker ps`)

Qdrant v1.18.0, OpenSearch 2.17.1, Postgres 16-alpine, MinIO, three TEI containers
(embedder `tei`, `reranker`, `nli-checker`), a **llama.cpp server (fallback only)**, plus
otel-collector / Prometheus / Grafana and four `ragaas-sovereign-node` federation nodes.

Deviation from the plan's "Current system" table: the plan names **Gemma 31B via vLLM** and
**Chandra** OCR. CORRECTED after `_frag_05`: the dev `api` does NOT use the local llama.cpp
container by default — it points at `models-dev.example.internal`, model `qwendeploy` (Qwen-32B); the
local llama.cpp container (Qwen2.5-3B Q4_K_M, `-c 8192`) is only the fallback, and ADR-0006
describes that older design. The deploy/prod config DOES use `gemma4-31b` at
`models-uat.example.internal`, matching the plan's "Gemma 31B". Helm defaults to Sarvam-30B.
Model identity per environment must be read from config, never assumed — see `_frag_05`.
