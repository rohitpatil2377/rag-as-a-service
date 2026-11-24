## Query Path & Retrieval

Paths are relative to the repository root. `pipeline.py` = `services/control-plane/src/svc_control_plane/pipeline.py`; `config.py` = `libs/common/src/rag_common/config.py`. Scope: incoming query to final ranked spans (generation excluded). Flags quoted are code defaults unless stated; deployment overrides are listed in Q7.

Important framing: `services/retrieval` is a thin wrapper (`service.py:21-28`, `router.py:1-14` is health only). The real orchestration of the "query path" lives in `pipeline.py::_answer_query_impl` (3559-3895), with a near-duplicate in `_stream_answer_impl` (3896-4350, same steps at 3906-4088). The hybrid retrieval primitive is `HybridRetriever.retrieve` (`libs/rag-core/src/rag_core/hybrid.py:79-143`).

### 1. End-to-end retrieval flow (ordered)

HTTP entry: `POST /kb/{kb_id}/query` (`api.py:2400-2419`) builds `RetrievalQuery` with `tenant_id=kb_id`, `acl`, `clearance`, `top_k` default `settings.query_top_k`=16 (`api.py:405-407`, `config.py:322`), after `tenant_guard` (`api.py:162-183`). Then `pipeline.answer_query` -> `_answer_query_impl`:

1. `_with_exact_boost` (pipeline.py:3573; def 1868) - stamps per-KB BM25 exact-token boost. Flag `exact_term_match_enabled` (config.py:988) default off.
2. `_with_chem_normalization` (3576; def 1886) - adds `chem_terms`; off unless chemistry KB.
3. `_with_multivector_flags` (3579; def 1903) - per-KB bge-sparse/ColBERT leg flags; needs `multivector_enabled` (config.py:166, default False).
4. `_with_glossary_expansion` (3582; def 1927) - appends canonical terms to `query.text` via `expand_glossary` (`glossary.py:167`); flag `glossary_expansion_enabled` (config.py:1012) default off. Original kept in `original_text`.
5. Input guard `wiring.input_guard().check` (3587) - `guardrails_enabled` default True (config.py:1089).
6. Process-intent shortcut `process_flow.is_process_query` (3611) - `process_intent_enabled` default off (config.py:717).
7. `_semantic_cache_lookup` (3616; def 1750) - `semantic_cache_enabled` default False (config.py:1122), threshold 0.97 (1123).
8. Agentic planner (3641-3667), only if `query_planner_enabled` (config.py:1046, default False): `resolve_tenant_query_planner(...).plan` (3649) then `_execute_query_plan` (3651; def 3497). Can short-circuit to whole-document / catalog / count / filter / compare answers (see Q2, Q6).
9. Intent classifier (3672-3693), only if `intent_routing_enabled` (config.py:1029, default False) and planner not on: `resolve_tenant_intent_classifier(...).classify` (3674); may short-circuit to `_answer_catalog` / `_answer_doc_summary` / `_answer_count` / `_answer_filter` / `_answer_compare` (3680-3688).
10. Adaptive router (3695-3706), only if `adaptive_router_enabled` (config.py:978, default False): `wiring.query_router().route(query)` (3698). `heavy` path sets `top_k = settings.heavy_top_k` (3700, =10).
11. Recency over-fetch (3710-3712): `top_k = min(50, want*recency_overfetch)` (config.py:902, 4) when `after/before/freshness` set.
12. `_doc_scope_for` (3714; def 1519) - if question names a document (`scope_for_query`, `docscope.py:124`), over-fetch `top_k=min(50, max(top_k, want*5))` (3715-3716). `doc_scope_enabled` default **True** (config.py:911).
13. Retrieval, one of:
    - (a) Decomposition (3721-3743) if `query_transform_enabled and heavy` (config.py:1078 default False; if router off, `heavy` is True, 3692): `query_transformer_for(...).transform` (3722) -> one `retrieval_service_for(...).retrieve` per sub-query (3723-3727) -> `merge_retrieved` (3735; `transform.py:180`) with budget `min(top_k*n_sub, transform_span_budget=40)` (3734; config.py:481).
    - (b) else `_cross_lingual_retrieve` (3744; def 2014): single `svc.retrieve(query)` unless `cross_lingual_enabled` (config.py:877, default False) and non-Latin script.
14. Inside every `svc.retrieve` -> `RetrievalService.retrieve` (`service.py:21`) -> `HybridRetriever.retrieve` (`hybrid.py:79`):
    1. `pool = max(query.top_k, candidate_pool)` (hybrid.py:82); `candidate_pool = settings.rerank_candidates`=50 (`factory.py:376`, config.py:218).
    2. Dense leg `_dense_legs` (hybrid.py:85; def 145-179): `embedder.embed([text])` + `vector_store.search(tenant_id, qvec, pool, filters)` (164), or multivector path (166-178) if enabled.
    3. BM25 leg `sparse_index.search(tenant_id, text, pool, filters, exact_boost, chem_terms)` (hybrid.py:87-89; `opensearch_index.py:275-340`).
    4. Optional CLIP image leg (hybrid.py:96-101), `vision_recall_enabled` default off (config.py:797).
    5. `reciprocal_rank_fusion([dense, bge_sparse, sparse, visual])` (hybrid.py:105; `fusion.py:17`, k=60 at `fusion.py:14`).
    6. `normalize_scores` min-max (hybrid.py:108; `fusion.py:32`) -> goes only to `fusion_score`.
    7. Top `pool` ids hydrated from Postgres `catalog.get_chunks` (hybrid.py:110-114); spans created UNSCORED (`_to_span`, hybrid.py:198-221).
    8. ACL + clearance post-filter `is_entitled` / `is_cleared` (hybrid.py:120; `access.py:7-35`).
    9. `reranker.rerank(query.text, spans, keep)` with `keep = min(query.top_k, rerank_top_k=20)` (hybrid.py:123-124; config.py:260). Only the reranker writes `score`/`scorer` (`reranker.py:193`).
    10. `max_conf = max(score)` (hybrid.py:130) -> `RetrievalResult.max_score` (`service.py:27`).
15. `_agentic_retry` (3769; def 1788) - only if `agentic_retry_enabled` (config.py:1151, default False) and `max_score < abstain_gate`; re-transforms with failed-passage grounding, re-retrieves, `merge_retrieved`.
16. `_graph_augment` (3782; def 1061) - GraphRAG local; `graphrag_enabled` default False (config.py:1160). Extra rerank of graph-linked chunks (1058).
17. `_global_augment` (3786; def 1464) - GraphRAG global communities; `graph_global_search` default True (config.py:1205) but needs graphrag.
18. `_neighbour_augment` (3790; def 1102) - +/- `neighbour_window`=2 adjacent chunks of top `neighbour_expand_top`=3 docs' anchors; `neighbour_expansion_enabled` default False (config.py:248). Neighbours arrive UNSCORED.
19. `_apply_recency` (3796; def 1639) - time window + freshness boost (`recency.py`), trims to `want`.
20. `_coverage_augment` (3797; def 1578) - if question names >=3 identifiers (`enumeration.py:405`, `MIN_ITEMS=3`), one extra `svc.retrieve` per item (`coverage_per_item`=3, `coverage_max_items`=12; config.py:486-487), merged cap `min(80, ...)`; then widens prompt budget `_prompt_budget_for_items` (3799-3806; def 1571; 2 spans/item, max 40). Always on (no flag).
21. `_apply_doc_scope` (3808; def 1531) - post-filter to the named doc, `kept[:want]`; falls back to unscoped if empty.
22. `_drop_navigation_spans` (3816; def 1625; `navigation.py`) - removes table-of-contents spans when other evidence exists.
23. `_fuse_structured_evidence` (3819; def 1971) - SQL/NoSQL/REST sources, `unified_kb_enabled` default off (config.py:679).
24. `_mark_superseded` (3820; def 1166) - `version_conflict_enabled` default off (config.py:229).
25. `_parent_expand` (3821; def 1213) - attaches non-citable `parent_text`; `parent_child_enabled` default False (config.py:348). See Q5.
26. Hand-off to `generation_service().answer` (3825) - out of scope.

Conversation turns (`answer_turn` 4939 / `stream_turn` 4989) first condense the follow-up (`_condense_turn`, 4754) and classify intent with `has_history=True` (4967, 5023), then reuse the same impl.

### 2. Query routing

There are FOUR independent routing mechanisms, all flag-gated, all default OFF in code:

| Mechanism | Backend(s) | Output label set | File:line | Flag |
|---|---|---|---|---|
| Adaptive router (FR-D5, ADR-0008) | `HeuristicRouter` (default, keyword/length) or `LLMRouter` (1 call, `max_tokens=4`) | `Path = Literal["standard","heavy"]` | `routing.py:18`, `routing.py:57-71` (markers 34-51, `_HEAVY_WORD_COUNT=18` at 53); `libs/clients/.../routing.py:17-39` | `adaptive_router_enabled` (config.py:978), `router_backend` (979) |
| Intent classifier | `HeuristicIntentClassifier` or `LLMIntentClassifier` (1 call, `max_tokens=8`, fail-safe to `rag`) | `Intent = Literal["catalog","doc_summary","summarize","count","filter","compare","rag"]` | `intent.py:36`; `intent.py:38-46` (`_ALLOWED`); `intent_classifier.py:60-82`, `_guard_meta_intent` 112-125 | `intent_routing_enabled` (config.py:1029), `intent_backend` (1030) |
| Agentic query planner | `HeuristicQueryPlanner` (wraps intent heuristic) or `LLMQueryPlanner` (1 JSON call, `max_tokens=200`) | `Strategy = Literal["whole_document","passages","whole_kb","catalog","count","filter","compare"]`, `Coverage = Literal["exhaustive","targeted"]`, `AnswerShape = Literal["summary","direct","list","comparison"]` + `target_doc`, `rationale`, `confidence` | `query_plan.py:34-36`, `QueryPlan` 44-58; `query_planner_llm.py:162-181`; `_INTENT_TO_PLAN` `query_plan.py:84-92` | `query_planner_enabled` (config.py:1046) |
| Doc-scope matcher | deterministic token-coverage (`MIN_COVERAGE=0.6`, `MIN_MARGIN=0.2`) | `DocScope(doc_id, coverage, margin)` or None | `docscope.py:71,74,115-141` | `doc_scope_enabled` default **True** (config.py:911) |

**Comparison with the plan's 6 intents** `{lookup, aggregate, compare, summarize, procedural, relational}`:
- `rag` ~ `lookup` (default path). `compare` exists (`intent.py:36`, planner `compare`), but only for exactly TWO named documents; otherwise falls back to RAG (`pipeline.py:2466-2469`).
- `summarize` exists but means "summarize THIS CONVERSATION" (`intent.py` prompt 392-401; handler `_answer_summary` 2201), not document overview. `doc_summary` (summarize each doc) and planner `whole_document`+`exhaustive` / `whole_kb` are the nearest to plan's `summarize`/`aggregate`.
- No `procedural` and no `relational` label anywhere. GraphRAG augment (`_graph_augment`, 1061) is entity-triggered, not intent-routed.
- Existing labels the plan lacks: `catalog`, `count`, `filter` (KB-inventory meta-intents answered from the catalog, not retrieval), and planner `whole_kb`.
- No `aggregate` label per se; planner `whole_document`+`exhaustive` is the functional equivalent for one document (Q6).
- No `facets`, `doc_ids` list, `sub_queries`, `query_lang`/`answer_lang`, `needs_amendment_view` in any router output. Planner emits a single `target_doc` title (`query_plan.py:53-58`).

**Rewriting / sub-queries / multi-query today:**
- `QueryTransformer` (`transform.py:42-50`; clients `transform.py`): `DecompositionTransformer` (deterministic split on `?` and ` vs `/` and `, max 4 sub-queries, `transform.py:20,64-87`), `HyDETransformer`, `RewriteTransformer`, `ReformulationTransformer` (LLM, up to 3 rule-shaped questions, grounded on prior-retrieval excerpts, `transform.py:27,99-147`; clients `transform.py:51-81`), `CompositeQueryTransformer` (default `auto` = decomposition + reformulation, clients `transform.py:84-111,114-135`). Backend setting `query_transform_backend` default `auto` (config.py:1084).
- Wired in two places: heavy-path fan-out (pipeline.py:3721-3743) and agentic retry (1788). **`query_transform_enabled` default False** (config.py:1078); `agentic_retry_enabled` default False (config.py:1151). Deployment files that turn router+transform ON: `infra/helm/values.yaml:142-144` and `infra/compose/docker-compose.dev.yml:367-369`; `infra/compose/docker-compose.deploy.example.yml` does NOT set them (it sets only `RAG_INTENT_ROUTING_ENABLED=true`/`RAG_INTENT_BACKEND=llm`, lines 122-123).
- Cross-lingual translation variant (`_cross_lingual_retrieve`, 2014-2046), off by default.
- Per-enumerated-item sub-retrieval (`_coverage_augment`, 1578) is always on, shape-triggered, not LLM-driven.
- Glossary query expansion (`glossary.py:167`) off by default.
- Sub-queries are NOT facet-specific, are not tied to document structure, and are capped at 4 (+3 reformulations).

ADR-0008 records that the heavy path only "boosts top_k, capped" and is a seam for D6. CLAUDE.md:1120 records the router A/B as a measured NULL result (~+0.0%).
Quirk worth flagging: `heavy_top_k` default 10 (config.py:315) is LOWER than `query_top_k` default 16 (config.py:322), so the heavy branch at pipeline.py:3700 can reduce top_k relative to the standard path unless the deployer raises it.

### 3. Fusion

- Algorithm: Reciprocal Rank Fusion, `RRF(d)=sum 1/(k+rank)` with `RRF_K = 60` (`fusion.py:14`, function `fusion.py:17-30`). Rank-based; input scores only order each list. `k` is a module constant; not an env var or Settings field.
- Lists fused (`hybrid.py:105`): `[dense, bge_sparse, bm25, visual]`. `bge_sparse` and ColBERT only when `multivector_enabled` and per-KB flags (hybrid.py:166-178); ColBERT REORDERS the dense pool rather than adding a list (hybrid.py:103-104, 171-175). No per-list weights; all lists equally weighted.
- Top-n per stage:
  - dense ANN: `pool = max(query.top_k, rerank_candidates)` = 50 by default (hybrid.py:82; config.py:218, `RAG_RERANK_CANDIDATES`, `.env.example:206`).
  - BM25: same `pool` (hybrid.py:87-89).
  - RRF output truncated to `fused[:pool]` (hybrid.py:110); all ids hydrated (111).
  - ACL/clearance filter, then reranker scores all remaining (<=50), keeps `min(query.top_k, rerank_top_k=20)` (hybrid.py:123; config.py:260).
- Dense legs use Qdrant `query_points(limit=top_k)` (`qdrant_store.py:175-182`); BM25 is OpenSearch `match` on `text` (+ optional `text.exact`/`chem`) with `size=top_k` (`opensearch_index.py:322`). "Both analyzers" in the plan: today the index has `standard` on `text` and a `whitespace` sub-field `text.exact` (`opensearch_index.py:74-75`), the latter only queried when `exact_term_match_enabled` (config.py:988; boost `exact_term_boost`=2.0 at 989).
- Min-max normalised fused score goes ONLY to `RetrievedSpan.fusion_score` (diagnostic, ADR-0015 decision 1).

### 4. Reranker

- Wiring: `build_reranker()` (`reranker.py:371-403`) selects from `rerank_backend` in `{cross_encoder, lexical, openai_compat}` (config.py:204, default `cross_encoder`); optional `FallbackReranker(primary, secondary)` when `rerank_fallback_backend` set (`reranker.py:327-345`; config.py:270). Per-KB override via `resolve_tenant_reranker` (`wiring.py:708-732`: KB attach -> org default -> platform).
- Real cross-encoder? YES in every non-lexical configuration: `CrossEncoderReranker` posts `{query, texts[]}` to a TEI `/rerank` (`reranker.py:176-177`), `OpenAICompatReranker` to a cohere/TEI-shaped `/rerank` (`reranker.py:248-256`). `LexicalReranker` (term-overlap, `reranker.py:90-104`) exists only if explicitly selected, and is the in-memory test default (ADR-0015 Consequences).
- Model name: **the plan's "BGE reranker" is only partly accurate; it differs by config source.**
  - Code default: `cross-encoder/ms-marco-MiniLM-L-6-v2` (config.py:214; `.env.example:202`; dev compose `docker-compose.dev.yml` reranker service, 22M, English-only).
  - `infra/compose/docker-compose.deploy.example.yml:48-62`: `RAG_RERANK_BACKEND=openai_compat`, URL `https://models.example.internal/openai/v1`, `RAG_RERANK_MODEL=bge-reranker` (variant not stated in config; `docs/calibration.md:154` titles the calibration "bge-reranker-v2-m3"); fallback = same endpoint/model (a retry, not redundancy).
  - `infra/helm/values.yaml:100-113`: `jina-reranker-v3.5` via `api.jina.ai`, shape `cohere+sigmoid`, abstain 0.50. `docs/calibration.md:157-160` says Jina was REPLACED 2026-09-11 for sovereignty, but Helm still carries it (config drift).
  - `docker-compose.dev.yml:197-203` documents an opt-in GPU `reranker-gpu` = `BAAI/bge-reranker-v2-m3`.
  - Model identity is verified only for TEI (`_verify_model` GETs `/info`, `reranker.py:130-163`); not for `openai_compat`.
- Max input length: **NOT set or enforced client-side anywhere** (no truncate/max_length in `reranker.py`). Candidate text is sent whole (`reranker.py:177,248`); truncation is whatever the server does (TEI model limit; MiniLM-L6 is 512 tokens). Chunk size 900 chars default (config.py:335; deploy compose overrides to `RAG_CHUNK_SIZE=250`, deploy.yml:70). Plan's "truncate candidate text, not query" is NOT implemented. Not verified for the remote bge/jina endpoints.
- Top-k and batching: scores up to `rerank_candidates`=50 candidates in batches of `rerank_batch_size`=32 (config.py:262; `reranker.py:173`), sorted globally, returns `scored[:top_k]` with `top_k=min(query.top_k, rerank_top_k)` (hybrid.py:123).
- **Threshold semantics: the reranker applies NO score threshold. Candidates below any threshold are NOT dropped by the reranker; they simply sort lower and survive if inside top_k.** The only threshold is the abstain gate on `max_score` (best span), applied downstream (`grounding.py:75` `should_abstain`; `gate_for` 86-107) with `abstain_threshold` = 0.15 code default (config.py:380), 0.23 in deploy.yml:67, 0.50 in Helm (values.yaml:113), plus per-KB override via `kb_threshold_overrides` (`pipeline.py:1835`, `_THRESHOLD_KEYS` 737). `abstain_threshold_lexical`=0.60 for the lexical scale (config.py:413). The plan's `rerank_threshold` + `max_keep` per-candidate filter does not exist.
- **ADR-0015 contract (summary):** (1) `RetrievedSpan.score` = ABSOLUTE cross-encoder relevance in [0,1], comparable across queries, the only gate input; `fusion_score` is set-relative and diagnostic, never shares the field. (2) Only a reranker writes `score` and must write `scorer` in the same op; candidates are hydrated unscored (score 0.0, `scorer="unscored"`) so a span that skips the reranker fails safe and abstains (schema validator `retrieval.py:139-146`). (3) Graph-contributed candidates are reranked by the same reranker (no fabricated constants; `_score_graph_spans`, pipeline.py:1045). (4) `max_score = max(s.score for s in spans)`, never `spans[0]`. (5) `RAG_RERANK_MODEL` is load-bearing; mismatched server model is refused (TEI path). `_probability` (`reranker.py:72-87`) raises on out-of-[0,1] scores; `+sigmoid` api_shape converts logits (`reranker.py:235-238`). Neighbour spans are unscored and cannot open the gate (pipeline.py:1116-1119).

### 5. Parent expansion

- Yes, but narrow, flag-gated, and non-citable. `_parent_expand` (`pipeline.py:1213-1290`, called at 3821) attaches `parent_text` to child spans; ranked/cited spans remain the children; `max_score` is unchanged.
- Parent unit: a **section-bounded group of consecutive child chunks**, not a page and not a whole section. `build_parents` / `_group_children` (`parent_child.py:220-279`): consecutive same-`META_SECTION` children up to `parent_child_parent_max_chars`=1800 (config.py:354); a section change always starts a new parent; a lone child gets no parent (`parent_child.py:239-241`). A short section = one parent = the section. The parent text is children joined by "\n" (parent_child.py:242). Children are ~350 chars (`parent_child_child_max_chars`, config.py:351).
- Link: child carries `metadata["parent_id"]` (`META_PARENT_ID`), copied to `RetrievedSpan.parent_id` by `_to_span` (hybrid.py:220).
- Query-time bounds: at most `parent_child_max_parents`=5 distinct parents (config.py:357), aggregate char budget `parent_child_context_max_chars`=5000 (config.py:362), best-child-score ordered, per-KB overridable (pipeline.py:1241-1242). Same ACL/clearance/filter guard (1244-1247).
- Gates: `parent_child_enabled` default False (config.py:348) AND the KB must have been (re)indexed with `chunk_strategy=parent_child` (default chunk strategy is `clause`, config.py:332). A legacy index has no `parent_id` so the step self-skips (1235-1237). Enabled in dev compose (`docker-compose.dev.yml:385`).
- Section identity today is only a string heading path in chunk metadata (`chunking.py:42,78`, `sections.py`); there is NO section table or section-level unit/summary in the catalog (Postgres access is `documents` + chunks: `postgres_catalog.py:101,152,169`).
- Related, different expansion: `_neighbour_augment` (+/-2 chunks, flag off) adds contiguous chunks by reading order, unscored (pipeline.py:1102-1163).
- Dedup helper `dedup.py` (near-duplicate chunk pairs) is ingest-side/diagnostic, not on the retrieve path.

### 6. Aggregate / exhaustive retrieval

Honest answer: **there is no top-k-free, section-level aggregate strategy in the retrieval layer, but there ARE exhaustive-by-enumeration paths one level up, all flag-gated and none query-relevance aware.**

- `enumeration.py` is NOT a document-unit enumerator. It extracts identifier tokens (e.g. `PBSOP001`) from the question (`enumeration.py:405-417`, `MIN_ITEMS=3`) so `_coverage_augment` can run one top-3 retrieval per identifier (pipeline.py:1578-1622). It is top-k per item.
- `docscope.py` is a document-NAME matcher (`docscope.py:124`). It does scope a query to one document ("this document" only when the title is named, or when the KB has exactly one visible doc: `_match_plan_targets` pipeline.py:3490-3494). It post-filters an over-fetched pool (`top_k=min(50, want*5)`, pipeline.py:3715) and keeps `kept[:want]` (1531-1552); it does NOT enumerate. The docstring explains why it is a post-filter: the BM25 index only filters on `metadata.*` (pipeline.py:1537-1541; `opensearch_index.py:48-51,300`), so a `doc_id` store filter would empty the sparse leg. Note `_answer_compare` (pipeline.py:2474-2476) and `_answer_doc_summary` nonetheless pass `filters={"doc_id": ...}` - worth verifying whether the BM25 leg is silently empty there (OpenSearch `_source` has top-level `doc_id`, `opensearch_index.py:254`, but filters target `metadata.doc_id`).
- `coverage.py` (`uncovered_spans`, lines 1-54) and `completeness.py` are POST-generation checks (answer omitted a high-score span), not retrieval.
- **Whole-document read** (planner `whole_document`+`exhaustive`, or `passages`+`summary`+`exhaustive`): `_execute_query_plan` (pipeline.py:3497-3558) -> `_answer_document_full_summary` (def 2999): `catalog.document_chunks(tenant, doc)` loads ALL chunks of the doc in order, ACL/clearance filtered (3040-3043), capped at `whole_doc_summary_max_chunks`=400 (config.py:1068), then either a single full-context read if est. tokens (chars/4) <= 0.9 * `min(whole_doc_full_read_token_budget=96000, llm_context_window - reserve)` (3056-3064), else map-reduce in batches of `whole_doc_summary_batch_chunks`=12 (config.py:1050) with concurrency `whole_doc_map_concurrency`=4 (config.py:1073). 2..`whole_doc_multi_target_cap`=3 matched docs -> per-doc summary combined (`_answer_documents_full_summary`, def 3299); more -> RAG + clarify options.
  - Gate: `query_planner_enabled` default **False** (config.py:1046); ON only in `docker-compose.dev.yml:412` (not in `docker-compose.deploy.example.yml`, not in Helm values). Unit is the **chunk** (all of them), no relevance filtering, no section summaries, no facets, no rerank-vs-query selection.
- `doc_summary` intent (`_answer_doc_summary`, pipeline.py:2590-): per-document top-k sampling, `doc_summary_chunks_per_doc`=6, `doc_summary_max_docs`=8 (config.py:1034-1035). Not exhaustive.
- `whole_kb` planner strategy routes to normal RAG (+ GraphRAG global augment); not exhaustive (pipeline.py:3556-3558).
- Document-scoped ("this document") mode: partial. Implicit via `doc_scope_for` (title named), via sole-doc KB, and via planner `target_doc`. There is no explicit request parameter to scope a query to a document (`QueryRequest` fields: query, top_k, mode, filters, model, after/before/recency_days/freshness; `api.py:398-415`) other than generic `filters` (which hit the `doc_id` payload key in Qdrant). `RetrievalQuery.mode` is declared but NOT read by the retriever (`hybrid.py:44-48`).

### 7. Configurable vs hard-coded (query path)

| parameter | current value | where set |
|---|---|---|
| RRF k | 60 | literal const `fusion.py:14` (`RRF_K`); not env |
| Dense candidates / BM25 candidates (pool) | `max(query.top_k, 50)` | `rerank_candidates` default 50, env `RAG_RERANK_CANDIDATES` (config.py:218; `.env.example:206`); fallback literal `DEFAULT_CANDIDATE_POOL=50` (hybrid.py:37) |
| Post-rerank cap | `min(query.top_k, 20)` | `rerank_top_k` default 20, `RAG_RERANK_TOP_K` (config.py:260); literal `DEFAULT_RERANK_TOP_K=20` (hybrid.py:38) |
| API query top_k | 16 (max 50) | `query_top_k` (config.py:322); `QueryRequest.top_k` (api.py:405-407); `RetrievalQuery.top_k` default 5, le=100 (`retrieval.py:30`) |
| Heavy-path top_k | 10 | `heavy_top_k` (config.py:315) used at pipeline.py:3700 |
| Heuristic router thresholds | >=18 words; `top_k+3` capped at 12; `?`>1; marker list | literals `routing.py:34-54,63-69` |
| Sub-query cap | 4 (decompose); 3 (reformulate) | literals `transform.py:20` (`_MAX_SUBQUERIES`), `transform.py:27` (`MAX_REFORMULATIONS`) |
| Sub-query merged budget | `min(top_k*n_sub, 40)` | `transform_span_budget` (config.py:481) |
| Reformulation LLM params | max_tokens 900, 400-char excerpts, 6 excerpts, temp 0 | literals `transform.py:31,34,37`; `clients/transform.py:69-72` |
| Query-transform backend | `auto` | `query_transform_backend` (config.py:1084) |
| Reranker backend / URL | `cross_encoder` / `http://reranker:80` | `rerank_backend` (config.py:204), `rerank_url` (205); deploy overrides above |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (code); `bge-reranker` (deploy.yml:51); `jina-reranker-v3.5` (helm values.yaml:102) | `rerank_model` (config.py:214) / env `RAG_RERANK_MODEL`; per-KB attach via `resolve_tenant_reranker` (wiring.py:708) |
| Reranker api shape | `cohere` (code); `cohere+sigmoid` (helm) | `rerank_api_shape` (config.py:209) |
| Reranker fallback | off (code) | `rerank_fallback_*` (config.py:270-275) |
| Reranker batch / timeout | 32 / 30 s | `rerank_batch_size` (config.py:262), `rerank_timeout_seconds` (263) |
| Reranker max input length | none set client-side | not configured (server default) |
| Per-candidate rerank threshold / max_keep | none | not present |
| Abstain threshold (gate on max_score) | 0.15 code; 0.23 deploy.yml:67; 0.50 helm values.yaml:113; per-KB override | `abstain_threshold` (config.py:380), `RAG_ABSTAIN_THRESHOLD`, `tenant_config` key (pipeline.py:738, 1845) |
| Lexical-scale abstain | 0.60 | `abstain_threshold_lexical` (config.py:413) |
| Embedding model | `BAAI/bge-m3`, dim 1024 | `embed_model` (config.py:147), per-KB attach `resolve_tenant_embedder` (wiring.py:668) |
| ColBERT prefetch pool | 100 | `colbert_prefetch_pool` (config.py:179) |
| BM25 exact boost | 2.0 (flag off) | `exact_term_boost` (config.py:989) |
| BM25 query-term cap | 512 (256 with exact/chem) | literal `opensearch_index.py:292` |
| Qdrant quantization | none | `qdrant_quantization` (config.py:124) |
| Neighbour expansion | window 2, top 3 docs, off | `neighbour_*` (config.py:248-256) |
| Parent/child | child 350, parent 1800, max 5 parents, 5000 char ctx; off | config.py:348-362 |
| Chunk size / overlap | 900 / 150 (deploy: 250 / 50) | config.py:335-336; deploy.yml:70-71 |
| Recency | overfetch 4x, half-life 180 d | config.py:902, 912 |
| Doc-scope | enabled; coverage>=0.6, margin>=0.2; over-fetch `want*5`, cap 50 | `doc_scope_enabled` (config.py:911); literals `docscope.py:71,74`, pipeline.py:3716 |
| Enumeration coverage | >=3 ids; 12 items; 3 spans/item; cap 80; prompt 2 spans/item max 40 | `MIN_ITEMS` (`enumeration.py:397`), config.py:486-487, literals pipeline.py:1604, 1569-1570 |
| Compare handler | exactly 2 docs; 4 spans/doc | literals pipeline.py:2281-2282 |
| GraphRAG | hops 1, candidates 30, max chunks 10; off | config.py:1190-1198, `graphrag_enabled` 1160 |
| Whole-doc read | budget 96000 tok, ctx window 128000 (dev compose: 32768), batch 12 (dev: 40), cap 400 chunks, concurrency 4, safety 0.9 | config.py:1050-1073, 1060; literals pipeline.py:2774, 2777 |
| Semantic cache | off; cosine>=0.97; 256 entries | config.py:1122-1124 |
| Agentic retry | off | `agentic_retry_enabled` (config.py:1151) |
| Clarify | off; min clusters 2; ceiling 0.75 | config.py:933-937 |
| Router / intent / planner / transform enables | all False in code | config.py:978, 1029, 1046, 1078 |
| Deployed enables | Helm: router ON, transform ON, contextual ON; deploy.yml: intent ON (llm) only | Helm `values.yaml:142-144`; `docker-compose.deploy.example.yml:122-123` (intent only); planner ON only in `docker-compose.dev.yml:412`, neighbour+parent-child ON in dev compose (:385, :438). Live VM values must be verified in-container (see memory note: always verify env changes in-container). |

### 8. Multi-tenancy and access filters

- **Tenant = KB id.** `tenant_guard` (`api.py:162-195`) requires `kb_id == principal.tenant_id`, or `kb_id.startswith(f"{tid}__")`, or `kb_id == SHARED_KB_ID`; plus per-user `kb_access` list and live cross-KB access grants. The endpoint builds `RetrievalQuery(tenant_id=kb_id, ...)` (`api.py:2405-2417`), so `tenant_id` is server-derived from the path after the guard, never client-supplied in the body. `RetrievalQuery` is frozen, `extra="forbid"` (`retrieval.py:22`).
- **Isolation at the store boundary, by namespace, not by payload filter**: Qdrant collection per tenant `vector_collection(tenant_id)` (`tenancy.py:58`; `qdrant_store.py:3-6,172`), OpenSearch index per tenant `sparse_index(tenant_id)` (`tenancy.py:73`; `opensearch_index.py:283`), Postgres schema per tenant `pg_schema(tenant_id)` (`postgres_catalog.py:103,155`), CLIP index per tenant `vision_collection` (`tenancy.py:63`). `service.py:6-8` states `query.tenant_id` is the sole authority. Tenant id is also a payload field (`qdrant_store.py:159`, `opensearch_index.py:255`) but is not used as a filter.
- **Metadata filters (D8)**: `query.filters` -> Qdrant `Filter(must=[FieldCondition(key=k, match=value)])` (`qdrant_store.py:428-434`), OpenSearch `term` on `metadata.<k>` (`opensearch_index.py:300`). Applied pre-fusion in each store (hybrid.py:9-11). Post-hoc augment steps re-apply them with `_chunk_matches_filters` (pipeline.py:1513).
- **ACL (FR-A6)**: `is_entitled(span.acl, query.acl)` - empty span ACL = tenant-public, otherwise group intersection (`access.py:7-23`). Applied as a post-filter in `HybridRetriever` on the hydrated candidate pool BEFORE rerank (hybrid.py:116-121), not as a store filter, so the pool is 50 pre-ACL and a heavily-restricted caller may see fewer than top_k.
- **Classification clearance**: `is_cleared(span.classification, query.clearance)` is `classification <= clearance` (`access.py:26-35`), applied in the same line (hybrid.py:120). `clearance` 0-3 from `kb_roles.effective_kb_clearance` (api.py:2414); default 0 = Public only (`retrieval.py:38-44`).
- **Every augment path re-checks ACL+clearance+filters**: `_graph_augment` (pipeline.py:1083-1087), `_neighbour_augment` (1141-1153), `_parent_expand` (1244-1247), `_global_augment` (1488-1491), whole-doc read (3040-3043), `_visible_docs` (2285-2288).
- Note: the `ACL` and `clearance` of the semantic cache are part of the cache key (`semantic_cache.put(... list(query.acl) ... query.clearance, query.top_k)`, pipeline.py:3874-3876), and cache is skipped under recency/filters (1765).
- `/kb/{id}/retrieve` (`api.py:2442-2458`) uses `retrieval_service_for(kb_id).retrieve(rq)` directly, so none of steps 1-13 (glossary, router, transform, doc-scope, coverage, parent expand) apply to agent retrieval-only calls.

### Gaps vs plan (Phases 7-8)

- **ALREADY EXISTS - reuse:** hybrid primitive with Qdrant dense + OpenSearch BM25 + RRF k=60 and pool 50 (`hybrid.py:79-143`, `fusion.py:14-30`). Plan Phase 8 "hybrid(q, filters, levels, n=50)" matches except there is no `levels` concept (chunk/row/proposition); all units are one flat chunk collection.
- **ALREADY EXISTS - reuse:** real cross-encoder rerank with calibrated [0,1] absolute scores, `scorer` discriminator, ADR-0015 gate contract, per-KB reranker attach and fallback (`reranker.py:107-271,327-403`; `retrieval.py:89-146`). A `section_threshold`/`rerank_threshold` can be added as a new post-rerank filter without touching the gate. Gap: no per-candidate threshold/`max_keep`, no truncation of candidate text to reranker max length, and the calibrated abstain threshold is model-specific (0.15 / 0.23 / 0.50 by deployment) so any new low section threshold (~0.15) must be calibrated per model, not assumed.
- **ALREADY EXISTS - reuse:** LLM planner scaffold with JSON-constrained output, robust parse and heuristic fallback (`query_plan.py:117-227`, `query_planner_llm.py:162-181`) and an intent-prompt/parse/guard pattern (`intent.py:386-461`, `intent_classifier.py:67-125`). Phase 7's router can extend `QueryPlan`/`Strategy`/`PlanContext` instead of starting fresh; the "fallback to lookup on invalid output" rule matches the existing fail-safe-to-passages behaviour.
- **ALREADY EXISTS - reuse:** query transformation/sub-query fan-out and merge (`transform.py:64-230`, `clients/transform.py:51-135`, `pipeline.py:3721-3743`, `merge_retrieved` round-robin with best-score dedupe). Phase 8 `sub_queries` can feed this merge; only the sub-query SOURCE (router, facet-aware) changes.
- **ALREADY EXISTS - reuse:** whole-document exhaustive read with full-read vs map-reduce selection, ACL-safe chunk loading, completeness accounting (`pipeline.py:2999`, `3497-3558`) and `catalog.document_chunks` (`postgres_catalog.py:169`) as the enumeration primitive. This is the closest thing to the plan's aggregate path but is chunk-level, query-blind, single-doc/<=3-doc, flag-off in deploy.
- **ALREADY EXISTS - reuse:** document scoping by name (`docscope.py`, `pipeline.py:1519-1552`), per-item coverage retrieval (`enumeration.py`, `pipeline.py:1578`), neighbour expansion (`pipeline.py:1102`), parent-context bank with ACL guard and char budget (`parent_child.py:220-279`, `pipeline.py:1213`), compare-with-two-docs handler (`pipeline.py:2450`), merge/diversify utilities, `_chunk_matches_filters`/ACL checks to reuse in any new strategy.
- **GAP 1 (biggest): no section-level unit.** There is no `sections`/section-summary store, no section IDs, breadcrumbs, `facet_ids` or `amends` fields; sections are only a heading-path string in chunk metadata, and parents are char-bounded groups of children. Phase 8 aggregate step 1 ("load ALL section units from Postgres") has nothing to load. Needs the Phase 1-5 data model first.
- **GAP 2: no aggregate retrieval strategy in the retrieval layer.** No "enumerate all units, rerank against query + every sub-query, score = max, low threshold, union facet-matched, cap max_sections" exists. The reranker has no multi-query scoring entrypoint (`Reranker.rerank(query, spans, top_k)` takes one query). Whole-document read is the nearest and is off by default.
- **GAP 3: router taxonomy and outputs.** Labels differ (`rag/catalog/doc_summary/summarize(conversation)/count/filter/compare` plus planner strategies); missing `aggregate`, `procedural`, `relational`, facet mapping, `doc_ids[]`, `sub_queries`, `query_lang`/`answer_lang`, `needs_amendment_view`. No profile cards exist to feed the router. Router also receives only document TITLES (`PlanContext.doc_titles`, bounded sample `_PLAN_TITLE_SAMPLE`), not profiles. Everything routing-related is default OFF in code and only partially enabled in deployments, so Phase 7's 90% accuracy test has no baseline harness for the existing intents.
- **GAP 4: no relational/graph intent routing.** GraphRAG augment is entity-regex triggered and flag-off (`config.py:1160`), not driven by a "relational" intent; Phase 8 relational strategy needs doc/entity linking that is not modelled.
- **GAP 5: no "levels" in the index** (chunk / row / proposition / table-as-unit); tables are only atomic chunks inside parent/child (`parent_child.py:153-164`). Phase 8 lookup "expand to parent (section text or full table)" requires section/table IDs.
- **GAP 6: no pre-rerank truncation or explicit reranker max length;** the plan's "respect reranker max input length" is unimplemented and the actual remote reranker variant (bge-reranker vs bge-reranker-v2-m3 vs Jina) differs across `config.py`, `docker-compose.deploy.example.yml`, Helm; confirm on the live VM before Phase 0 baseline.
- **GAP 7 (config drift to resolve in Phase 0):** the plan says Gemma + BGE; code defaults say MiniLM; Helm says Jina; deploy compose says `bge-reranker`. `heavy_top_k` (10) < `query_top_k` (16). Possible silent empty BM25 leg for `filters={"doc_id": ...}` in compare/doc_summary (see Q6) - verify with a test before relying on doc-id filters for the aggregate path.
- Latency note: reranker cost is linear in candidates (CPU ~100 ms/candidate MiniLM, ~500 ms bge-base; `config.py:215-217`). Reranking every section of a long document against 2-8 sub-queries multiplies calls, so the aggregate strategy's latency target must be set against the remote reranker's measured throughput.
