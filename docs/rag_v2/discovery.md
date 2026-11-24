# RAG v2 — Phase 0 Discovery

**Date:** 2026-10-16 · **Branch:** `phase-3` (ahead of `origin/phase-3`, unpushed)
**Plan under execution:** `RAG_UPGRADE_PLAN.md` (owner: Alex Mercer)
**Status:** Phase 0 discovery complete. Baseline evaluation **NOT** run — blocked, see §7.

Detail lives in seven fragments in this directory; this file is the synthesis, the
defect list, the gap matrix and the decisions needed before Phase 1.

This is a DATED measurement record, not a living design doc. Where a later phase proved a
finding wrong, the row is corrected in place and marked **Corrected <date> (<phase>)** rather
than rewritten — the original reading is part of the evidence trail. Corrections so far:
Phase 2a, on Chandra layout and layout persistence (§3 and the Phase 2/6 rows in §5); Phase 5, on
chunking and contextualize (the Phase 5 row in §5); Phase 3, on the document profile card (the
Phase 3 row in §5); Phase 4, on section enrichment (the Phase 4 row in §5); the Phase 12 ops
slice, on query-path latency and the query log (§8.2 and the Phase 12 row in §5); Phase 10, on
answer verification (the Phase 10 row in §5); Phase 11, on the eval harness (the Phase 11 row in
§5); Phase 7, on the query router (the Phase 7 row in §5).

| Fragment | Slice |
|---|---|
| `_frag_00_repo_map.md` | Repo map, orchestration, existing strategy dispatcher |
| `_frag_01_ingest_docintel.md` | Ingestion, Doc Intelligence, OCR/layout, tables, quarantine |
| `_frag_02_chunk_context_embed.md` | Chunking, Contextualize, Embedding |
| `_frag_03_stores.md` | Qdrant, OpenSearch, Postgres, MinIO, Graph |
| `_frag_04_query_retrieval.md` | Query path, routers, fusion, reranker |
| `_frag_05_generation_verification.md` | Generation, grounding, citations, faithfulness |
| `_frag_06_config_flags_eval.md` | Config, feature flags, eval harness, metrics |

---

## 1. Headline

**The plan's diagnosis is partly wrong, and that is good news.** It treats the aggregate-query
failure as missing architecture and proposes twelve phases of new building. Discovery found
that most of the proposed architecture already exists in some form — and that at least three
concrete *defects* in the existing implementation independently produce the exact symptoms
the plan opens with ("retrieved 4 chunks matching the literal word 'compliance'", "missed the
most important sections entirely", "truncated lists").

Those defects are cheap to fix and must be fixed and re-measured **before** Phase 1 is
scoped, or the programme will be sized against a baseline that is artificially bad.

---

## 2. Confirmed defects (verified in this session, not inferred)

### D1 — A `doc_id` filter silently empties the BM25 leg · **confirmed live** · severity: high

`meta_path()` (`libs/clients/src/rag_clients/opensearch_index.py:47-50`) maps **every** filter
key to `meta.<key>`, with no special case for the top-level fields. Filter clauses are built
with it unconditionally (`opensearch_index.py:300`). But the write path stores `doc_id` as a
**top-level** keyword (`opensearch_index.py:65`, `:263`). So a filter `{"doc_id": X}` becomes
`term: {"meta.doc_id": X}` — a field that does not exist in any indexed document.

Measured on the live dev cluster, index `t_public__company_wide_bm25`:

```
live dev stack
top-level keys  : ['acl','chunk_id','doc_id','meta','page','tenant_id','text']
meta keys       : ['doc_type','entities','extraction_*','field_*','lang','ocr_*','sanitization_*']
top-level doc_id: 'SOPs - Leslie Dan Faculty of Pharmacy - Single Document.pdf'
meta.doc_id     : None

term on doc_id      -> count=181
term on meta.doc_id -> count=0
```

**CORRECTED ATTRIBUTION (this is not an OpenSearch mapping mistake).** The sparse adapter is
honouring its documented contract. `RetrievalQuery.filters` is "Metadata filters (FR-D8)"
(`retrieval.py:32`), and that filters mean *tags* is deliberate and tested:
`libs/clients/tests/test_sparse_metadata.py::test_a_filter_on_a_reserved_name_matches_the_tag_not_the_field`
asserts that a `doc_id` filter key means the metadata TAG, not the record field, and the
in-memory twin states the same (`memory.py:244-246`: *"a tag called `doc_id` describes the
chunk, it does not redefine it"*). Changing `meta_path` would reverse a deliberate decision —
exactly what `CLAUDE.md` warns against.

The real defect is a **contract divergence between the legs**, plus callers using the wrong
channel:

| Leg | `filters` semantics | Record fields filterable? |
|---|---|---|
| Sparse (OpenSearch + twin) | metadata tags under `meta.*` (`opensearch_index.py:47-50`, `:300`) | **No** — by design |
| Dense (Qdrant + twin) | flat payload, key passed through (`qdrant_store.py:432-433`) | **Yes** — record and tags share one namespace |

Confirmed live on the dense side too: a populated payload is flat —
`['acl','chunk_id','doc_id','doc_type','entities','extraction_*','field.*','lang','ocr_*','page','sanitization_*','section','tenant_id']`
— so `doc_id` is filterable there and `section`/`field.*` sit top-level in Qdrant but under
`meta.*` in OpenSearch. The failure is asymmetric and silent: a filter that under-matches reads
as "no results", never as an error (a hazard the file's own comment at
`opensearch_index.py:56-61` documents for *tags*, without noticing it applies to `doc_id`).

**Affected callers** — both pass `filters={..., "doc_id": ...}`:
- `pipeline.py:2475` — `_answer_compare`
- `pipeline.py:2666` — `_answer_doc_summary`

**Consequence:** every document-scoped compare and per-document summary runs **dense-only**.
For a tender corpus whose distinguishing tokens are exact standards codes (`IEC 62443-4-2`,
`61000-4-2`, `802.1x`), losing the lexical leg is close to the worst possible degradation —
and it matches the reported symptom precisely. `_answer_document_full_summary`
(the `whole_document` path) is **not** affected: it reads via `catalog.document_chunks`.

**Fix (chosen: option A, first-class scoping).** Add an explicit keyword-only
`doc_ids: tuple[str, ...] = ()` to `RetrievalQuery` and to every store search method that takes
`filters` — `VectorStore.search`, `SparseIndex.search`, and `MultiVectorStore.search_dense` /
`search_sparse` / `search_colbert` — implemented identically on both legs against the
**top-level** `doc_id`, in addition to (never instead of) `filters`/ACL/clearance. `hybrid.py`
threads it to every leg; the two callers switch from the metadata channel to it. This keeps the
FR-D8 tag semantics and the existing test intact, and matches the repo's own philosophy that
`tenant_id` is always an explicit parameter rather than inferred. Rejected alternatives:
changing `meta_path` (reverses a tested decision) and caller-side-only (leaves the divergence
for the next caller to trip on).

**Regression test that must fail on the old code:** a `doc_ids`-scoped **sparse** search over a
KB with ≥2 documents returns a non-empty result for the scoped document.

### D2 — `llm_context_window` is set in dev only; production falls back to 128000 · severity: high (verify on VM)

`llm_context_window` defaults to **128000** (`libs/common/src/rag_common/config.py:1060`).
The real `gemma4-31b` window was measured at **32768** and set — but commit `933ccef`
*"config(dev): set real gemma context window so whole-doc read map-reduces"* changed **only**
`infra/compose/docker-compose.dev.yml` (8 lines, one file). `RAG_LLM_CONTEXT_WINDOW` appears
in **no** other environment source: not `.env.prod-vm.example`, not
`docker-compose.deploy.example.yml`, not the Helm values.

`_answer_document_full_summary` chooses its read mode from it (`pipeline.py:3055-3062`):

```python
est_tokens = sum(len(c.text) for c in chunks) // 4
reserve = settings.heavy_max_tokens + _WHOLE_DOC_PROMPT_RESERVE_TOKENS
effective_budget = min(settings.whole_doc_full_read_token_budget,      # 96000
                       max(0, settings.llm_context_window - reserve))  # 128000 - ~4-6k
full_read = est_tokens <= int(effective_budget * _WHOLE_DOC_FULL_READ_SAFETY)
```

With the default, the budget lands near 96k tokens, so a document far larger than the model's
real 32k window takes the **single full-read** path instead of map-reduce. The commit message
states the purpose of setting the true window plainly: *so the whole-doc read map-reduces*.
The result is a silently truncated read — the tail of the document never reaches the model.

Both containers on this box are correctly set to 32768 (`ragaas-dev-api-1`, `gloss-api`).
**Unverified:** the UAT VM. `.env.prod` is filled in on the target host and is not in the repo,
and that VM runs `ragaas-api:redis-scale-p2e`, which may be wired differently.

**Check on the VM:** `docker exec <api-container> env | grep RAG_LLM_CONTEXT_WINDOW`
**Fix — DONE (config side).** `RAG_LLM_CONTEXT_WINDOW=32768` added to
`.env.prod-vm.example:119` with the reasoning inline. Helm had the same hole but serves a local
vLLM mount (`RAG_LLM_MODEL: "/model"`, Sarvam-30B), so the correct value must be read off that
server; a documented commented line was added at `infra/helm/values.yaml:101` instead of a guess.
**Still open:** a preflight guard that fails startup when the configured window exceeds what the
served model reports (`llm_context_window` has exactly one consumer, `pipeline.py:3062`), and
confirmation of the UAT VM's own `.env.prod`.

### D3 — `heavy_top_k` (10) is below `query_top_k` (16) · severity: medium (needs confirmation)

Flagged during discovery: the "heavy"/thorough retrieval path retrieves **fewer** passages
than the standard path. If confirmed as reached in production, the adaptive router escalating
a hard question to `heavy` would *reduce* evidence. Not yet verified end to end — listed here
so it is not lost. See `_frag_04_query_retrieval.md` §7.

### D4 — in Qdrant, a metadata tag can overwrite the record field it describes · severity: low (latent)

`qdrant_store.py:156-163` builds the payload as
`{"chunk_id": …, "doc_id": …, "tenant_id": …, "page": …, "acl": …, **c.metadata}` — metadata
spread **last**, so a chunk whose metadata carries `doc_id`/`chunk_id`/`tenant_id`/`page`/`acl`
redefines the record field. The in-memory vector twin has it too (`memory.py:122`, `:180`:
`{"doc_id": c.doc_id, **c.metadata}`). This is the same class of bug already fixed on the sparse
side, where it has a test (`test_metadata_cannot_overwrite_the_indexed_body`).

No live payload currently shadows a record field (verified over sampled points in
`t_public__company_wide_chunks`), but B7 auto-metadata can produce such keys.

**Fix:** spread metadata **first** and let the record fields win. This keeps today's behaviour
that tags are filterable at the top level — so no existing filter changes meaning — while making
it impossible for a tag to redefine the record. Fixed alongside D1.

---

## 3. Plan premises that are false

The plan's "prior hardening work in this codebase (reuse, don't redo)" list is partly wrong.
Correcting it changes phase sizing.

| Plan premise | Reality |
|---|---|
| "cross-page table merging" exists | **False.** Only an upper-bound *metric* exists (`table_page_runs`, `ocr.py:324-336`); its own docstring calls it an upper bound. A table spanning a page break emerges as two tables. Phase 2 table work is larger than budgeted. |
| "quote-based citation resolution" exists | **False.** Citations are index-based `[n]` markers. `Citation.quote` is chosen *post hoc* as the span sentence with the most query overlap (`_span_quote`, `grounding.py:569-590`). The plan's Phases 9-10 depend on quote-anchored items; that resolution must be **built**. |
| NLI is `mDeBERTa-v3-base-xnli-multilingual` | **True in every deployed environment** (`.env.prod-vm.example:122`, `docker-compose.dev.yml:133`). Only the code default differs (`config.py:488` = `cross-encoder/nli-deberta-v3-base`). No model swap needed. |
| NLI gate is "entailment ≥ 0.70" | **False.** A claim passes when `1 − P(contradiction) ≥ 0.5` plus a 0.25 lexical anchor (`nli_faithfulness.py:553-589`, `config.py:489`). **Neutral currently passes** — exactly the class Phase 10 wants caught. Long premises get one best 300-char window inside a 1000-char budget, not max-over-windows. |
| "an eval harness" to extend | **True, and richer than implied** — but it measures none of the plan's new metrics. See §6. |
| Chandra emits layout we can consume | ~~**Not as called.** Chandra runs in chat-completion mode, the prompt asks for Markdown, it answers HTML, and `normalize_ocr_markup` (`ocr_check.py:312`) flattens to text; the HTML is discarded (`ocr.py:841`). No block types, no bboxes. Only the **raw upload** is persisted to MinIO (`ingestion/service.py:48`) — there is no extracted-text or layout artifact at all.~~ **Corrected 2026-10-19 (Phase 2a) — half right, and the useful half was free.** The flattening and the discarding were real, but the conclusion was wrong: block types, headings, table structure and reading order were ALREADY arriving as HTML (`normalize_ocr_markup`'s own docstring says Chandra answers HTML however it is asked), so neither the prompt nor the call ever needed changing. Phase 2a keeps that markup as a per-page overlay, writes it to the object store at `layout/<safe-doc-id>` under the tenant's prefix, and records that key in `documents.minio_layout_key` — the object IS the artifact, the column only points at it. **Still gated:** the write is behind `RAG_V2_INGEST`, which defaults false and appears nowhere in `infra/compose/**`, so no deployed environment persists anything today — the capability exists, the data does not. Both Chandra engines emit markup but in different languages (self-hosted chat → HTML, hosted `chandra_api` /convert → Markdown), declared per page. Unchanged and still true: a chat-shaped VLM gives no **coordinates**, so `blocks.bbox` stays NULL and the plan's font-size/weight heading cue is unavailable; plain-text engines yield no markup at all. |

---

## 4. What already exists (do not rebuild)

The plan's Phases 7-9 are **extensions**, not new builds.

**An agentic query planner with a strategy dispatcher** — `_execute_query_plan`
(`pipeline.py:3497`) maps `plan.strategy` ∈ {`whole_document`, `whole_kb`, `catalog`, `count`,
`filter`, `compare`, `passages`} plus `answer_shape` and `coverage` (`exhaustive`) to a
mechanism. Preserve its fail-safe contract: an unresolved strategy falls through to normal
RAG, so a planner decision can never turn an answerable question into an error
(`pipeline.py:3519`).

**A map→reduce over a whole document** — `_answer_document_full_summary` (`pipeline.py:2999`):
fetches every chunk in document order (ACL/clearance filtered), a full-read fast path when it
fits the budget, otherwise concurrent batched MAP through the *unchanged* grounded generation
path (so each batch is faithfulness-checked and cites real chunk ids), then a REDUCE that
renumbers `[n]` markers into a union citation list, with honest `truncated`/coverage
accounting (`_run_whole_doc_batches` `:2806`, `_verify_whole_doc_coverage` `:2874`).
Its REDUCE is concatenation + line dedupe, **not** an LLM synthesis — no JSON items, quotes,
facets or old→new. That is the real Phase 9 delta.

**Retrieval augmentation stages**, all in `pipeline.py`: `_graph_augment` (`:1061`),
`_neighbour_augment` (`:1102`), `_mark_superseded` (`:1166`), `_parent_expand` (`:1213`),
`_global_augment` (`:1464`), `_coverage_augment` (`:1578`), `_apply_doc_scope` (`:1531`),
`_with_exact_boost` (`:1868`), `_with_glossary_expansion` (`:1927`),
`_semantic_cache_lookup` (`:1750`), `_agentic_retry` (`:1788`),
`kb_threshold_overrides` (`:1835`).

**Also reusable:** content-hash chunk IDs, stable across re-chunk and test-asserted
(`chunking.py:96-112`); the `Contextualizer` seam, which keeps original text for citations;
parent-child + atomic-unit chunking; RRF fusion (k=60) and the reranker contract of ADR-0015
(`score` is an absolute [0,1] cross-encoder probability and the only gate input,
`fusion_score` is diagnostic); `text.exact` whitespace subfield; `documents.content_hash`
delta-skip idempotency; write-then-swap reindex keyed on `pipeline_fingerprint` with a
stale-document report; boilerplate stripping (`ocr.py:1008`, Tesseract/Paddle/native-PDF
only — not Chandra); numeracy and notation checkers; three-way faithfulness routing;
completeness/coverage re-prompt loops with call budgets; `eval_gate.py` with baseline
`--compare`; the RAGAS metrics and floors.

---

## 5. Gap matrix

| Phase | Verdict | Notes |
|---|---|---|
| 1 Data model | **Partly blocked — redesign needed** | Alembic manages only `public` + `rag_identity` (head `0007_tenants`); the 25 live per-KB schemas get tables from runtime `CREATE TABLE IF NOT EXISTS` (`postgres_catalog.py:27-56`). New per-tenant tables belong there, not in a migration. The plan's `documents` and `chunks` **collide with existing tables of those names in every tenant schema** — rename or `ALTER`. Qdrant has **no payload indexes at all** (live `payload_schema` is `{}`). `_v2` names must be per-KB (`t_<slug>_chunks_v2`, `t_<slug>_bm25_v2`) and need new `tenancy.py` helpers; OpenSearch hard-imports `sparse_index`, so it needs a constructor change; delete/erase/reindex-swap must learn the v2 names or orphan data. |
| 2 Structure tree | **Mostly net-new, larger than planned** | `sections.py` is a numbering-regex detector producing a **flat** list, heading scope capped at 2000 chars, no IDs, no tree; unnumbered headings undetected. Prerequisite work the plan omits: ~~change the Chandra call to request layout, and persist a layout artifact (none exists today)~~. **Corrected 2026-10-19 (Phase 2a):** the call did NOT need changing — it already returned HTML — and the layout artifact now exists: captured, persisted and purged with the raw bytes, though only when `RAG_V2_INGEST` is on, which no deployment sets, so no corpus has an overlay yet. The remaining Phase-2 work is the parser (tree + breadcrumbs) and cross-page table merging, both of which must still be built. **Corrected 2026-10-19 (Phase 2c):** born-digital PDFs DO now carry heading styles and fonts — pdfplumber's `page.chars` give real bbox + font size + weight, which is more signal than the Chandra markup path has, and the `layout-json` overlay records it. One consequence needed its own decision: a page title set in heading type on every page produced one root section per page (it is a real heading, so the furniture rule rightly refuses it), now folded into one section by `_merge_repeated_page_titles` under the same >=3-page vote the furniture rule uses, with the title KEPT rather than stripped. Born-digital/DOCX paths capture no heading styles or fonts. |
| 3 Doc profile | **Built 2026-10-21 — acceptance MEASURED 2026-10-23: 3 of 4 assertions pass.** doc_type, the `amends` relations (2, both on target) and the verbatim evidence-quote rule all pass. FACET COVERAGE still fails — **0.50-0.60, mean 0.54 over 5 runs, against a 0.80 bar** — but it was **0.38**: `_profile_head` took the plan's "first ~2 pages" literally, 4,119 of 193,432 chars (2.1%), so the card was built from one fiftieth of the document and mirrored the heading outline. Spending the whole `doc_profile_head_chars` budget contiguously fixed that half. SAMPLING the rest of the document was tried and measured WORSE (0.44) — fragments are poorer evidence than one coherent passage — so that route is closed. What remains splits: `training` (char 61,842) and the standards clauses (33,817) are past any front-loaded window, while supply-chain, OEM credentials and lifecycle sit INSIDE it and are still not faceted. | A crude semantic `doc_type` exists (`tagging.py:25-33`, keyword substring, first-match-wins, labels agreement/invoice/policy/report/manual/resume/minutes, gated by `auto_metadata_enabled=False`). No purpose, issuer, relations or facets. Supersession is a query-time filename-Jaccard guess (`versioning.py`, threshold 0.6), never persisted. A separate `domain_profile` concept exists with `/profile/detect` — reconcile with it rather than adding a second profile notion. **Corrected 2026-10-21 (Phase 3):** the card is built and persisted. One gateway call per document (`rag_clients.doc_profiler.LLMDocProfiler`) over the plan's four inputs — title, the opening pages, the full breadcrumb list from the Phase-2 tree, and platform metadata — prompted by a reviewable FILE (`rag_core/prompts/doc_profile.md`, loaded through a new `prompts` subpackage; discovery §8.5's prompt loader, placed INSIDE the package because the API ships as a wheel and a repo-root `prompts/` would not be in the image). The reply is strictly validated and never partially stored: `doc_type` and `purpose` are the card, so a reply missing either yields no card at all, and a declared relation whose `evidence_quote` is not in the text the model was SHOWN is dropped and reported — matched with `graph.quote_key`, promoted to public so Phase 3's write-time check and Phase 6's seed-time anchoring cannot use two different normalisations. One retry on an unusable reply, carrying the reason (a greedy decode re-reading an identical prompt writes identical bytes), and fail-safe throughout: a model outage costs a document its card and nothing else. Persisted by `replace_document_profile` — the `documents` card columns, `facets` and `doc_relations`, one transaction under the existing advisory lock, REPLACE rather than upsert because a per-document taxonomy's ids differ between two runs and a union is not a taxonomy any profile call described. **This is the producer the Phase-6 relation seeding was waiting for (§8.12): that path is live from here.** Two consequences in the chunker: Phase 5's `TODO(Phase-3)` is closed — the card now prefixes every unit's `embed_text` (a row gets the card but still not the breadcrumb, which it already carries), bounded at 200 characters and subtracted from the token-derived character cap, since it spends the same embedder budget — and plan §3's `doc_type → chunk profile` map exists as `layout_chunk_max_chars_by_doc_type`, **shipped EMPTY** because no document class has been measured. `domain_profile` is NOT merged (§8.13). **The plan's acceptance is not met and is not claimed:** every clause of it names the AEGCL corrigendum, which is still not in this repository (§7, §9.1) — see §8.14. |
| 4 Section enrichment | **Built 2026-10-21 — acceptance MET and MEASURED 2026-10-23: 3 of 3.** On the real corrigendum the MAAT section yields `amends {old: Rs. 35 Crores, new: Rs. 19.44 Crores}`; the router section's propositions carry both HBOM and SBOM; and quote validation is **398/398 = 1.000** against the plan's >= 0.95 bar — every quote the enricher attached is verbatim in its section. 38 of 39 enrichable sections enriched, 1 failed. | Was: net-new, depends entirely on Phase 2. **Corrected 2026-10-21 (Phase 4):** built. One gateway call per enrichable SECTION (`rag_clients.section_enricher.LLMSectionEnricher`), prompted by a second reviewable FILE (`rag_core/prompts/section_enrich.md`) over the plan's three inputs plus the Phase-3 facet taxonomy spelled out as an explicit list — a deviation, because "from the provided facet_taxonomy only" is a hard validation rule and burying the permitted values inside the profile JSON makes the model hunt for the one list it may not depart from. The reply is strictly validated by `rag_core.section_enrich` and never partially stored: the SUMMARY is the verdict (a reply without one yields no enrichment, the same all-or-nothing rule Phase 3 applies to `doc_type`/`purpose`), and every `key_value`, `amends` and `proposition` whose quote is not in the text the model was SHOWN is dropped and reported — the plan's own rule, through the same `graph.quote_key` Phase 3 and Phase 6 use. One retry per window, carrying the reason. Fail-safe PER SECTION and per window: a model outage costs those sections their summaries and nothing else. Long sections are read in windows and merged (`section_windows`/`merge_enrichments`); each window is its own evidence window. Persisted by `replace_section_enrichment` — an UPDATE of `summary`/`facet_ids`/`key_values`/`amends` on the rows `replace_document_structure` wrote, one transaction under the existing advisory lock, run AFTER the structure tree because that call DELETEs and re-INSERTs the very rows this updates (a real constraint, so both writes live behind one `_persist_v2_derived` rather than being repeated at two call sites). **This is the producer the other half of the Phase-6 relation seeding was waiting for (§8.12): `sections.amends` now seeds section-level edges.** Two chunk LEVELS follow from it: `section` returns, holding the SUMMARY — the one text in a document no other unit carries, which is what closes Phase 5's duplication objection — and `proposition` is new, one unit per validated standalone statement. `doc` stays deferred (§8.17). Enrichment has its own switch and four bounds (§8.16) because one call per section is where this plan's cost actually lands; it enriches from MASKED text and does not send tables as HTML (§8.15). **The plan's acceptance is not met and is not claimed:** every clause of it names the AEGCL corrigendum (§7, §9.1) — see §8.18. |
| 5 Chunking/Contextualize | **Extend** | Three chunkers: FIXED (900/150 overlap), CLAUSE (700, default), PARENT_CHILD (350 child/1800 parent) — all **character**-based. Only PARENT_CHILD protects tables/code/formulas. A chunk **can** cross a section (FIXED, or overlap carry-over) and no test forbids it. Contextualize output today is `[context] Document: <title>, section <path>, page N, k=v; …` + blank line + text (`contextual.py:44-55`) — no doc_type, no purpose; the `k=v` tail is unbounded. Levels present: child/chunk, parent (catalog-only, joined text not summary), atomic-unit (a guard, not a level). No doc/section/row/proposition levels, and `Chunk` has no `level`/`section_id`/`parent_id`/`embed_text` fields — structure lives in a string-only `metadata` dict. **Corrected 2026-10-19 (Phase 5):** the reading above was accurate and is now partly history. A fourth chunker exists (`rag_core.layout_chunking.LayoutAwareChunker`, `ChunkStrategy.LAYOUT_AWARE`): it chunks the Phase-2 structure tree, emits the `chunk` and `row` levels with resolving parent links, and cannot cross a section boundary by construction (tested) — `doc` and `proposition` remain deferred to Phases 3/4, and `section` joined them in review: it was built and measured, but `LayoutSection.text` is the concatenation of the section's own blocks, so every unit it produced duplicated a `chunk` or every `row` beside it verbatim, which is the double-counting the module forbids itself elsewhere. It returns with the Phase-4 summary. **Corrected 2026-10-21 (Phase 4): `section` and `proposition` are emitted, from the enrichment, per section that has one; `doc` is still not reserved (§8.17).** `Chunk` gained `level`/`section_id`/`parent_id`/`embed_text`/`sequence` as defaulted-None additive fields (`sequence` is document order: a v2 chunk has no `span`, which is the v1 ordering key, so without it `document_chunks` returned a document in whatever order Postgres chose — the in-memory twin sorted stably, which is why the unit gate could not see it), and `embed_text` carries the plan's breadcrumb prefix (the profile half of it still waits on Phase 3). FIXED/CLAUSE/PARENT_CHILD are untouched, and with `RAG_V2_INGEST` off chunking is byte-identical (tested). Two things the row did not anticipate: `atomic_units` is effectively INERT on the markup path, because Phase 2's HTML parser collapses whitespace and every signal in that module is line-based — a table is protected there by structure (one `table_row` block = one whole unit) instead; and the size target is still CHARACTERS (1600, calibrated to ~310-370 tokens), because the markup path has no token offsets — the ones the indexing service fetches describe the plain-text layer. The Phase-1 `chunks` COLUMNS (`level`/`section_id`/…/`token_count`) are still NULL: the values now exist on the contract and ride in the `record` JSONB, but promoting them onto the columns is the Phase-6 catalog write, and `token_count` still has no honest source. The TOKEN budget is no longer discarded: `max_tokens` is converted to a character cap using the document's OWN measured chars-per-token ratio (`len(text) / len(token_offsets)`), so a Devanagari document lowers its own cap where 1600 chars would have been ~1000 tokens against a 512-token embedder that truncates silently; a `row` is exempt because it is never split. A container heading's own text is no longer a chunk (it is carried by every descendant's breadcrumb), which is this path's answer to v1's `_MIN_CHUNK_CHARS` merge — that merge joins across heading boundaries and so cannot be reused where no unit may span two sections. Capture is bounded by `RAG_LAYOUT_CAPTURE_MAX_PAGES` (3000) and the stored object by `RAG_LAYOUT_OVERLAY_MAX_BYTES` (32 MiB); measured cost of the capture is +0-20% extraction time and 6.7 KiB/page at 29 blocks/page, so the earlier "roughly doubles" estimate was wrong. |
| 6 Indexing fan-out | **Extend** | Idempotency and write-then-swap reindex already solid. ~~No layout persistence.~~ **Corrected 2026-10-19 (Phase 2a):** layout persistence now exists — the overlay is an object at `layout/<safe-doc-id>` in the tenant's prefix and `documents.minio_layout_key` records its key, refreshed on reindex so an existing corpus can be backfilled. Behind `RAG_V2_INGEST` (default false, set in no deployment), so nothing is persisted in any environment yet. Graph has only untyped co-occurrence edges, no doc→doc or section→section, and is off by default. `reindex_corpora.py` has no `--dry-run` (`stale_report` is the nearest equivalent). **Corrected 2026-10-21 (pre-Phase-6 decisions):** the plan's `_v2` namespace will NOT be built — **ADR-0024** — so this phase indexes into the tenant's own collection/index/schema and its acceptance is restated as *counts per level reconcile within the tenant namespace*, run by `scripts/reconcile_v2_index.py`. The three stores can now each answer that about themselves (`count_by_level` on `VectorStore`/`SparseIndex`/`CatalogAdapter`, with parents excluded on the catalog side because they are never indexed), which needed the level to BE in each index: it is now a Qdrant payload key and an OpenSearch `keyword` field, written only on a v2 chunk. The OpenSearch mapping for it is deliberately UNCONDITIONAL, unlike `chem` — the first live run of the reconciliation found that a `level` arriving at an index without a mapping is dynamically typed as analysed text, which OpenSearch cannot retype, after which every aggregation over it fails outright and every `term` filter under-matches for the life of that index. `token_count` now has its honest source too (§8.10). **Phase 6 proper landed 2026-10-21:** the nine per-chunk columns are WRITTEN (`level`/`section_id`/`parent_id`/`embed_text`/`sequence`/`page_start`/`page_end`/`token_count`/`chunk_hash`), with `sequence` added to the DDL (the plan did not foresee it, because it assumed v2 chunks would keep a span) and a `chunks_by_doc` index added for the per-document reads that had none; `chunk_hash` is derived at the one place a chunk is written so it cannot disagree with the text beside it, and `document_chunks` orders on the promoted column with a COALESCE back to `record` so no corpus needs a flag day. `chunk_hash` idempotency SKIPS re-embedding text that has not changed — ingest-only, see §8.11. Declared relations (`doc_relations`, `sections.amends`) are seeded into the graph as directed, labeled edges anchored to the chunk holding each relation's evidence quote; both tables are empty until Phases 3-4, so that path is complete but INERT today (§8.12). The reindex CLI gained `--docs`/`--target`/`--dry-run`. Still open: nothing from the plan's Phase 6 list. |
| 7 Router | **Built 2026-10-21 — acceptance STILL UNVERIFIED: it needs the owner's 40-query intent set, which does not exist** | Was: extend, do not rebuild. | Four routers exist (adaptive `standard|heavy`; intent classifier `catalog/doc_summary/summarize/count/filter/compare/rag`; planner strategies; doc-scope), all default **off** in code except doc-scope. Missing from the plan's contract: `aggregate`, `procedural`, `relational`, plus `facets`, `doc_ids`, `sub_queries`, `needs_amendment_view`. The planner sees only document **titles**, not profile cards. Query rewriting (decompose ≤4, LLM reformulate, HyDE, composite `auto`) exists but is flag-gated and wired to paths that are off. **Corrected 2026-10-21 (Phase 7):** built, behind `RAG_V2_ROUTER` (default off — a decision scopes retrieval and drives decomposition). `rag_core.router` holds the plan's §7 contract: the six intents (now the CANONICAL definition, imported by the Phase-11 gold loader so a fixture cannot label against a vocabulary the router no longer emits), `doc_ids`, per-document `facets`, `sub_queries`, the two language fields and `needs_amendment_view`. It EXTENDS rather than replaces, per §8.4: `plan_for_intent` maps the six onto the existing `QueryPlan`, so a v2 decision selects an EXISTING answer mechanism and the fall-through-to-RAG fail-safe is preserved. Two backends behind one seam, heuristic DEFAULT — the deterministic path is the fallback and the only one whose intent accuracy can be asserted in a test, which matters because the plan's acceptance IS an accuracy figure. Validation is against the SCOPE: a `doc_id` not in scope and a `facet_id` the card does not define are dropped and reported, because a decision scoping retrieval to an unseeable document returns a confident empty answer and an invented facet joins to no `facets` row. **This is also the first read of a Phase-3 card out of Postgres** (`document_cards`) — the card had a writer since Phase 3 and no reader, and the router is the consumer it was written for. Applied today: `doc_ids` through the first-class predicate D1's fix introduced, and `sub_queries` through the EXISTING decomposition machinery. Recorded to `query_log.router_json`, the Phase-1 column Phase 12 gave a writer and left NULL with a `TODO(Phase-7)`. **Acceptance UNVERIFIED** — §8.24. |
| 8 Strategies | **Built 2026-10-22 — acceptance 13/14, MEASURED** | Nothing enumerates section units and reranks them against sub-queries at a low threshold — step 1 has nothing to load. The reranker applies **no per-candidate threshold**: it keeps `min(top_k, 20)` of 50 candidates; the only threshold is the abstain gate on `max_score` (0.15 code / 0.23 deploy / 0.50 Helm, with per-KB overrides). No client-side candidate truncation to the reranker's max length. Reranker model disagrees across config sources: code `ms-marco-MiniLM-L-6-v2`, deploy `bge-reranker`, Helm `jina-reranker-v3.5` (and `docs/calibration.md` says Jina was replaced on 2026-09-11). |
| 9 Map→reduce | **Built 2026-10-22 — acceptance 82-85% over 3 runs, BELOW the 85% bar** | Scaffolding exists (§4). Deltas: structured per-section extraction with validated verbatim quotes, facet grouping, old→new, coverage line in the answer text, LLM reduce instead of concatenation. Quote-anchored citation resolution must be built (D3 in §3). |
| 10 Verification | **Built 2026-10-21 — acceptance MET and measured** | Was: extend, semantics differ. NLI model is right; the gate is not (§3). Numeracy is wired on for every backend but only checks that each answer number appears in *some* retrieved span (`numeracy.py:94-109`) and a hit **does not change routing** (`faithfulness.py:251-267`). `notation.py` does cap into FLAGGED but is chemistry-oriented — no standard codes, crore, kV — and is off by default. `coverage.py` is span-level term overlap and **cannot** catch an "HBOM and SBOM" → "SBOM" drop. No per-claim regenerate, no "could not be verified" footnote, no `query_log.verification` (no query-log table at all). Claims are split by sentence/bullet, not as atomic list items. **Streaming verifies after the tokens have already been sent.** **Corrected 2026-10-21 (Phase 10):** built, behind `RAG_V2_VERIFY` (default off, because it EDITS THE ANSWER). `rag_core.verification` is the new pure layer: an atomic claim split that is abbreviation-aware, a VALUE check (figures, quantities carrying their unit, standards designations, dates as format-agnostic component sets) and the ENUMERATION check the plan's own worked example needs. The premise is THE SPAN THE CLAIM CITES (`Citation.quote`), not the union of everything retrieved — which is the one change that makes a mis-attribution visible, and which `numeracy.unsupported_numbers` structurally cannot see. A failing claim is footnoted (default) or dropped (`RAG_V2_VERIFY_ACTION`), per the plan; the status is deliberately NOT changed (§8.22). The report is persisted to `query_log.verification` — the plan's §10 item 6, a column that has existed since Phase 1 and whose writer landed in the Phase-12 slice, so this is the first thing that fills it. **THE ACCEPTANCE IS MET AND IS THE FIRST ONE IN THIS RUN THAT COULD BE:** it is stated against a seeded error set, not against the AEGCL PDF, so injected faults are legitimate evidence — measured 24/24 (100%) against the plan's 90% floor, with no fault kind blind, plus the companion measurement the plan does not ask for and that decides whether a guard survives contact with a corpus: 0 false positives across 38 faithful restatements. Items 2, 4 and 5 of the plan's Phase-10 list are NOT built — see §8.21-8.22. |
| 11 Eval | **Harness extended 2026-10-21; the corpus is still one document class** | Was: gold today is `eval/golden_sop.json` = `{"questions":[{question, expected?}]}`, 24 questions (20 expected, 4 abstain), all on one SOP document; plus `sample-data/golden/golden.json` (4) and `corpus_golden.json` (45 across 3 KBs). No `intent`, `checklist`, tiers or `supersessions`. `eval run --suite rag_v2` does not exist. **Corrected 2026-10-21 (Phase 11):** the format and the metrics exist. `svc_guardrails_eval.suite` carries the plan's §11 schema (`intent`; `expected_answer` XOR `checklist` of `{id, facet, must_contain, tier}`; `supersessions`), a STRICT loader (a malformed gold file raises rather than being skipped — a dropped gold query makes a suite smaller and its score better, the one failure mode a regression gate must not have), and the metrics: lookup accuracy, item recall reported PER TIER, value accuracy (reusing Phase 10's `extract_values`, so a figure written differently is the same figure), supersession correctness, context precision, faithfulness carried through, intent counts, and per-stage latency p95. `scripts/eval_gate.py` gained `--suite rag_v2` (the plan's `eval run --suite rag_v2`, every file in `eval/gold/`) and `--floor metric=value`, keeping its existing `--repeat` discipline — the report is a RATE, never a single verdict. NO FLOOR IS SET BY DEFAULT (§8.23). `eval/gold/sop.yaml` is the one suite this repository can honestly carry: its lookup items are `golden_sop.json`'s verified facts and its aggregate items are DERIVED from those same facts, so the aggregate metrics have something real to run against; it has no supersession items because the document is not an amendment, and that is reported as null rather than as a perfect score. **Corpus diversity is 1 of the plan's 4 document classes** — the other three are §9.5. Shadow mode and router intent accuracy are NOT built (§8.23). |
| 12 Ops | **Complete 2026-10-22 — the slice, then budgets, caches, the re-enrichment CLI and the streamed reduce** | Was: no per-stage latency on the query path, no query-log table, OTel spans only for federation (ingest does have `stage_ms`). The plan's §0 criteria cannot be measured without this. **Corrected 2026-10-21 (Phase 12 slice):** the query path now records per-stage DURATIONS (`rag_common.timing.StageTimeline`) and emits them as a `query_stage_timing` log line unconditionally, plus a `query_log` row behind `RAG_QUERY_LOG_ENABLED` (default off). `query_log` had a table, a GIN index and a DPDP erasure scrub since Phase 1 and **no writer at all** — `PostgresCatalog.record_query_log` is it, and writing it closed the gap `delete_document` had flagged in its own docstring since Phase 1 ("needs a decision when the table starts being written: either scope rows by the doc ids they cited, or keep the log free of answer text" — the first was taken). The in-memory twin gained a query log too, which forced a correction: `factory.purge_query_log_citing` used to answer 0 for that profile truthfully, and now purges for real. `router_json`/`selected_sections`/`verification` stay NULL with `TODO(Phase-7/8/10)` rather than being filled with today's nearest equivalent. **NOT built, and why, in §8.20:** the map/router/section caches, batched map calls and the reduce-streaming items are all about machinery Phases 7-9 have not built; the per-claim NLI stage is Phase 10's; the re-enrichment CLI needs an input this build does not persist. |

---

## 6. Metrics: the §0 criteria are mostly unmeasurable today

| Plan metric | Measured today? | Where / what |
|---|---|---|
| Aggregate item recall (gold checklist) | **No** | no checklist concept |
| Value accuracy (numbers/units/codes) | **Partly** | runtime `numeracy` check only, not an eval metric |
| Claim faithfulness | **Partly** | per-answer faithfulness + RAGAS floor 0.70; not claim-level |
| Supersession correctness | **No** | — |
| Lookup no-regression | **Yes** | `eval_gate.py` baseline + `--compare`; `make eval` |
| Router intent accuracy | **No** | — |
| Section coverage | **No** | — |
| p95 latency per stage | **No** | whole-query external timing only (`benchmark.summarize`) |

Existing metrics: citation coverage, abstain correctness, context hit rate, recall@k, MRR,
pass_rate, RAGAS answer_relevancy / context_precision / context_recall (floors
0.80/0.70/0.70/0.70, min sample 8).

---

## 7. Baseline: not run, and why

Plan task 4 asks for a baseline of the existing harness **and** of the Appendix A gold sets.
The second is impossible today and the first would be misleading:

1. **The AEGCL corrigendum is not in the repo.** No file matches `*aegcl*` or `*corrigend*`.
   Appendix A's 40 checklist items and the "known v1 failures to assert fixed" need the source
   PDF, and ideally the KB it is already ingested into.
2. **Most fixtures are not in the repo either.** Only `sample-data/platform_brief.md`
   (948 B) is committed; the HR policy PDFs, the content-governance SOP and the CAG tender EOI
   exist only as live-stack KBs referenced by the gold files. Document types available as gold
   today: SOP, HR policy, tender. The plan wants ≥4 including a contract and a technical paper.
3. **D1 and D2 would contaminate it.** Baselining before fixing a silently dead BM25 leg and a
   possibly truncated whole-doc read measures the bugs, not the architecture.

---

## 8. Proposed deviations from the plan

1. **Phase 0.5 — fix and re-measure. APPROVED by the owner on 2026-10-16; in progress.**
   D1 (leg contract divergence, fixed via option A: first-class `doc_ids`), D2 (context-window
   drift, config side done) and D4 (Qdrant payload hardening), each with a regression test, then
   run the baseline. Re-scope Phases 1-12 against *that* baseline. All are small and
   independently correct regardless of whether v2 proceeds.
2. **Pull a thin slice of Phase 12 forward**: per-stage timings + the `query_log` table, so §0
   has something to measure against. Phase 11's new metrics follow it.
   **BUILT 2026-10-21, out of §13 order and on purpose** — this was argued from Phase 0 and is
   the reason the Phase-0 baseline was never run. `rag_common.timing.StageTimeline` records
   DURATIONS (not the offsets the ingest DAG uses: a query is tuned, not drawn), marked across
   `guard` / `cache` / `route` / `retrieve` / `generate` / `post` on the non-streaming answer
   path, and emitted two ways — a `query_stage_timing` structured log line, ALWAYS, carrying no
   question text; and `query_log.latency_ms`, behind `RAG_QUERY_LOG_ENABLED` (default off,
   because the row holds the question and the answer). See §8.19 for what the slice deliberately
   does not do and §8.20 for what is still missing from the phase.
3. **Phase 1 per-tenant tables go in the runtime-DDL path**, not Alembic, and `sections`/
   `blocks`/`facets`/`doc_relations` must not reuse the names `documents`/`chunks`.
4. **Phase 7 extends `QueryPlan`** rather than adding a parallel router; map the plan's six
   intents onto the existing strategy taxonomy and keep the fall-through-to-RAG fail-safe.
5. **Add a prompt-file loader** as explicit scope. Prompts are inline Python constants today
   (`grounding.py:266-403`, `llm_gateway.py:59`); the plan assumes `prompts/*.md`.
   **Done 2026-10-21 (Phase 3):** `rag_core.prompts` — `load_prompt` / `render_prompt` over
   templates packaged INSIDE the library (not the repo-root `prompts/` the plan's path implies: the
   API ships as a wheel built from `src/rag_core`, so a repo-root directory would be absent from
   the image and only the first deployment would find out). Substitution is `string.Template`, because
   these templates contain a literal JSON schema and `str.format` raises on every brace in it.
   `prompts/doc_profile.md` is the first template; the existing inline prompts are NOT migrated — that
   is a change to live generation behaviour and belongs with its own measurement, not inside Phase 3.
6. **`RAG_V2_QUERY` cannot use the existing flag store.** It is global-scope and BOOLEAN-only
   (16 flags in `TOGGLEABLE`, `feature_flags.py:36`, persisted in
   `rag_identity.platform_flags`); `effective()` would read `"shadow"` as ON. Make the
   tri-state env/`LOCKED`-only, or widen the column and the admin PUT body. The booleans
   `RAG_V2_INGEST`/`RAG_V2_VERIFY` fit as-is. Note the env prefix makes them `RAG_V2_*`
   unless an `AliasChoices` is added.
7. **Tighten the NLI gate as its own change** (neutral should not pass), measured separately
   from the Phase 10 build, since it will move every existing faithfulness number.
8. **Evaluate flipping `exact_term_match_enabled` early.** `text.exact` already keeps standards
   codes whole — live `_analyze` gives `['61000-4-2']` on `text.exact` versus `['61000','4','2']`
   on `text`. Caveats: case-sensitive, breaks on trailing punctuation, and **empty for
   documents indexed before the subfield existed**, so it needs a reindex to be meaningful.
9. **Phase 6 indexes into the tenant's OWN namespace — there is no `_v2` collection or index.**
   DECIDED 2026-10-21, **ADR-0024**. The plan's `_v2` namespace is not built: v2 chunks share the
   tenant's existing Qdrant collection, OpenSearch index and Postgres schema, and `level`
   separates them. Phase 6's acceptance is restated as *counts per level reconcile across Qdrant,
   OpenSearch and Postgres **within the tenant namespace***, and `scripts/reconcile_v2_index.py`
   runs it (non-zero exit on disagreement) rather than it being asserted. Parents are excluded
   from the catalog side — never indexed by design, and they can only fall in the level-less
   bucket. See the ADR for why a second namespace is reserved for a second vector space.
10. **`chunks.token_count` is the embedder's own count of the chunk's stored text, or NULL.**
   The document-level `token_offsets` cannot supply it: a v2 chunk is assembled from markup blocks
   and has no span into the plain-text layer to slice. So `Embedder.count_tokens` (optional and
   duck-typed, like `token_budget`/`token_offsets`) asks the running server, batched, once per
   document; an embedder with no tokenizer endpoint leaves it unset and says so in the log. It is
   never estimated — measured through the adapter against this box's running bge-small, 1 000
   characters of ToC dot leaders tokenize to **1 000 tokens**, twice the 510-token budget, where
   characters/4 would have recorded 250 and called the chunk comfortably inside a limit it is
   double (Devanagari ~1.6 chars/token, prose ~5.1, so the error has no safe direction), and
   nothing downstream could distinguish the guess from a measurement. **Live limit:** only `TeiEmbedder` implements
   it, because only TEI exposes `/tokenize`; this deployment's production embedder is bge-m3 behind
   the AIOps `/embed` facade, so `token_count` is NULL there until that endpoint exposes a
   tokenizer (its ColBERT output is one vector per token, which is a candidate but unmeasured).
   Written only on v2 chunks: it is a plan §1.1 v2 column, and adding a key to a v1 chunk's
   serialised record would make every record this build writes unreadable by the previous image,
   whose `Chunk` is `extra="forbid"`.
11. **The re-embed skip is INGEST-ONLY — reindex always rebuilds.** `chunk_hash` idempotency
   (plan Phase 6, "make ingestion idempotent by `chunk_hash`") skips the embedder for text that is
   byte-identical to what is already indexed, gated on the document's recorded
   `pipeline_fingerprint` still matching the running one. It is NOT applied on reindex, which is a
   narrowing with a reason: the fingerprint answers "could a different pipeline have produced these
   vectors", not "are they still there", and a collection recreated or wiped out of band keeps the
   fingerprint while losing the vectors. On ingest that gap cannot open — the write-then-swap has
   purged nothing yet and the catalog row proves the vector and BM25 writes completed. On reindex it
   is the whole point of the command: reindex is the repair path the auto-reindexer and every
   runbook reach for, and a repair that quietly declined to rebuild would be the one failure nobody
   could see. The e2e invariant test for a failed reindex (`test_delta_sync`) is therefore unchanged
   and still passing. Reusing on reindex needs the stores asked whether they still hold each id —
   a bigger change, with its own measurement.
12. **The graph seeding from declared relations is COMPLETE but INERT.** Nothing writes
   `doc_relations` (Phase 3) or `sections.amends` (Phase 4), and §13 puts both after this phase, so
   every document declares nothing and the seeding writes nothing. It is a full read path rather
   than a stub: `relation_edges` resolves each relation to the chunk whose text CONTAINS its
   evidence quote — the plan's own Phase-4 validation rule, whitespace-normalised and case-folded —
   and a relation nothing proves is dropped and counted, never anchored to an arbitrary chunk,
   because an amendment edge pointing at the wrong evidence would make an answer cite text that
   does not say it. Edges go out through the existing `GraphStore.add_diagram_edges` writer, which
   Slice 6-B already established as the directed-labeled-edge path rather than a diagram-specific
   one. Tested as a pure function, through the pipeline wiring on the twins, and against live
   Postgres with rows inserted by hand (the only way to exercise a consumer before its producer).
   **Corrected 2026-10-21 (Phase 3): no longer inert.** `doc_relations` has a producer — the profile
   card's validated relations — so this path now carries real edges on the ordinary ingest path. The
   `sections.amends` half is still Phase 4's and is still empty. The two sides share one quote
   normalisation by construction (`graph.quote_key`, promoted from private for exactly this), so a
   relation cannot pass Phase 3's write-time validation and then silently fail to anchor here.
   **Corrected 2026-10-21 (Phase 4): both halves are live.** `sections.amends` now has its producer
   too (`replace_section_enrichment`), so a SECTION-level amendment seeds an edge whose source node
   is the clause's breadcrumb rather than the document's title. Nothing in the seeding changed to
   make that work — the query already read the column; what it lacked was a writer.
13. **`domain_profile` and the Phase-3 profile card stay TWO concepts.** DECIDED 2026-10-21,
   **ADR-0025** (§9.8 asked the question). They answer different questions at different scopes and neither can
   stand in for the other: `domain_profile` is KB-SCOPED — it samples a corpus, classifies it
   (chemistry / legal / financial / code), stores three JSON values in `tenant_config` and drives
   RETRIEVAL KNOBS under a detect-and-suggest rule that refuses to move a reindex-class knob
   without an operator. The profile card is PER-DOCUMENT — it is a model's reading of one
   document, stored in that document's own columns and rows, and it drives chunk sizing, the
   contextualize prefix, the relation graph and (from Phase 4) facet grouping. Merging them would
   mean either a corpus-level classifier that cannot distinguish a corrigendum from the tender it
   amends, or a per-document card that moves a KB's retrieval knobs — and the second is exactly
   the auto-apply the domain-profile module exists to refuse. They are documented as neighbours
   instead; nothing reads across.

14. **Phase 3's acceptance is UNVERIFIED and is not claimed met.** The plan's acceptance is
   "AEGCL fixture yields doc_type≈corrigendum, a relation 'amends' pointing to the original
   bidding document / Corrigendum-II, and facets covering eligibility, financial qualification,
   documentation/undertakings, supply-chain security, certifications/standards, OEM credentials,
   deployment/sovereignty, sizing/availability, lifecycle/support, training". That document is not
   in this repository (§7, §9.1) and NO FIXTURE STANDS IN FOR IT HERE — deliberately, and unlike
   Phases 2 and 5, which did use stand-ins.

   The difference is the kind of claim each acceptance makes. Phase 2's and Phase 5's are
   STRUCTURAL INVARIANTS ("no chunk crosses a section boundary", "the page header no longer
   appears inside block text"); any document of the right shape exercises those honestly, which is
   what makes `test_layout.py`'s stand-in defensible. Phase 3's is an EXTRACTION RESULT — a
   judgement a model makes about one specific document. A fixture written here would be a reply
   its author composed, so asserting the parser keeps it would prove only that the author can
   write a passing fixture. It would say nothing about whether a model reads the real corrigendum
   as a corrigendum, and a later reader would reasonably mistake it for evidence that it does.

   **What IS verified** (and is the whole of what the tests claim): the card is all-or-nothing;
   a relation whose quote is not in the window the model was shown is dropped and reported; that
   quote check is byte-for-byte the one Phase 6 anchors with; facet ids are normalised and
   collisions resolved first-wins; `issue_date` is accepted in one unambiguous form only; the card
   prefix is bounded and reaches `embed_text` and never `text`; breadcrumbs are masked before they
   reach the prompt (they come off the PRE-masking tree); the Postgres round trip, REPLACE
   semantics, the `document_relations` join and erasure, against a live server in a throwaway
   tenant; the flag-off path makes no model call at all; and a model outage leaves the document
   indexed without a card.

   **What is NOT verified, and cannot be without the PDF:** every clause of the plan's acceptance;
   the whole of Appendix A (40 checklist items, 3 supersessions, 7 lookup queries, and the four
   named "known v1 failures to assert fixed"); and whether a facet taxonomy is STABLE enough
   across re-profiling to be the join key Phases 4 and 8 depend on, which is content-dependent and
   so unanswerable on a composed fixture. Phase 4's acceptance (MAAT ₹35 Cr → ₹19.44 Cr, HBOM
   *and* SBOM in the router section's propositions, ≥95% of quotes validating) inherits the same
   block.

   **One real-model run, recorded because it is evidence and because n=1 is all it is.** The model
   the plan names is reachable — `gemma4-31b` answers on this box via the api container's
   `RAG_VLLM_URL` — so the prompt was run against it once, on the only real multi-section document
   in the dev corpus (a 68-page pharmacy-faculty SOP in `t_public__company_wide`; of 25 tenant
   schemas on this box the others hold a JPEG, four synthetic SOP text files, and nothing).
   Result: **valid JSON on the first attempt, no retry**; 2 974 prompt tokens, 686 completion
   tokens, 13.9 s; `doc_type` "sop"; the issuer read correctly off the cover page; **8 facets,
   inside the prompt's 6-15 band, with no duplicate ids**; 2 `references` relations, **both quotes
   validating — nothing dropped**. That exercises the JSON-reliability risk the plan flags in §14
   and it came out clean, on one document.

   Two things that run surfaced, which no composed fixture would have:

   * **`lang` came back `"english"`, not `"en"`.** The platform answers this deterministically in
     ISO-639-1 (`detect_language`) and writes codes into `documents.lang` from the tagging path, so
     a model's prose answer in the same column would be a second vocabulary and a filter joining
     them would match nothing. Fixed before the phase closed: the CALLER's detection wins and the
     model's value is used only when there is none. The field stays in the prompt — it is in the
     plan's schema and costs nothing.
   * **`issue_date` came back 2012-03-22, which is the first row of the document's own SOP master
     listing table, not the document's date** (its cover says "Last Updated: 05-Dec-17"). NOT
     fixed, because there is no evidence yet for what the right rule is — "prefer the latest date
     in a revision table" is a guess, and `tagging.py`'s `_RELEASE_CUES` already encodes a
     different one for a different purpose. Recorded as a known weakness of the field, and a
     reason not to build supersession ordering on `issue_date` until it has been measured on more
     than one document.

   The unblock for the acceptance itself is still one file, not a KB.

15. **Phase 4 enriches from MASKED section text, and does not send tables as HTML.** The plan's
   prompt says "Section text (tables as HTML)". Two reasons it does not, and both are contracts
   rather than preferences. **Masking:** `blocks.text` and the text behind `sections.text_hash` are
   UNMASKED — the structure tree is captured pre-masking — while `Chunk.text` is masked (FR-G2),
   and a proposition BECOMES a chunk's text while a summary becomes a `section` unit's. An
   unmasked enrichment pass would route document PII into the index through a path that looks like
   derived metadata, and into `sections.summary` beside it. So the caller masks the breadcrumb and
   the body before building the prompt, and every quote validates against the masked text.
   **Tables:** `LayoutBlock.html` is the extractor's own markup, kept for `table` blocks only, and
   it is ALWAYS None on the `layout-json` path — which is every born-digital PDF, i.e. the
   majority of documents. "Tables as HTML" would therefore have been available for a minority and
   unmasked for all of them. What the prompt gets instead is `section_prompt_text`: the section's
   blocks with the `table` block's own pipe-delimited lines kept and its `table_row` blocks
   dropped, because Phase 2 emits BOTH and `LayoutSection.text` consequently carries every table
   twice.

16. **Phase 4 has its own switch, and section enrichment is where the cost of this plan actually
   lands.** `RAG_SECTION_ENRICH_ENABLED` (default true, read only when `v2_ingest` is on).
   Phase 3 costs a document ONE model call; this costs one per enrichable section, so a
   200-section tender is a different order of spend and latency and an operator has to be able to
   decline it without giving up v2 chunking, the structure tree and the card. Bounded on four
   axes, none of them invented loosely: a 200-character floor on a section's own text (the same
   floor `_LAYOUT_CHUNK_MIN_CHARS` sets for a chunk — in a deep outline most nodes are container
   headings whose summary would restate the breadcrumb every descendant already carries), a
   400-section ceiling per document, 12 000-character windows capped at 4 per section (and the
   window is also the evidence window — a quote is validated against what the model SAW, never
   the whole section), and a concurrency of 4, because "parallel, batched" fired unbounded at a
   tenant's own endpoint is a self-inflicted denial of service. Fail-safe PER SECTION: a failure
   costs that section its summary and propositions, never the document its ingest.

17. **Phase 4 restores the `section` level and adds `proposition`; `doc` stays deferred.** Phase 5
   built the `section` level, measured it and withdrew it because its only possible text was the
   section's own concatenated blocks, which duplicated the `chunk`/`row` units beside it in every
   reachable case. The summary is what closes that argument — it is the one text in a document
   that no other unit carries — so the level returns, emitted only for a section that has an
   enrichment. `proposition` is new and is one unit per validated standalone statement; the
   statement is indexed and its QUOTE is not, because the quote is by definition already in the
   section's own units verbatim. `doc` is still not reserved in `ChunkLevel`: Phase 3 built the
   card, but the card already reaches the index as the `embed_text` prefix on every unit, and a
   separate `doc` unit would index it a second time — the same objection, unanswered.

18. **Phase 4's acceptance is UNVERIFIED and is not claimed met,** for the same reason Phase 3's
   is (§8.14) and from the same single blocker. Every clause of the plan's Phase-4 acceptance
   names the AEGCL corrigendum: "the MAAT section yields `amends {old: Rs. 35 Crores, new: Rs.
   19.44 Crores}`", "the router section's propositions include both HBOM and SBOM", ">= 95% of
   quotes validate". All three are claims about what a MODEL EXTRACTS from a real document. A
   fixture reply asserting them would only show that a reply written here parses into the fields
   it was written to parse into — self-fulfilling, and no amount of it is evidence.

   **What IS verified, and holds whatever any model answers:** a quote not found in the text the
   model was shown loses its item, measured through the whole pipeline rather than only in the
   parser (the pipeline test answers two sections with one section's quotes and only the section
   whose own text contains them keeps the quoted items); a facet outside the document's Phase-3
   taxonomy is dropped rather than created; a reply with no summary yields no enrichment at all; a
   model failure is scoped to one section and one window; the windows of a long section merge
   without losing or duplicating anything; the four `sections` columns are written as real
   `TEXT[]`/`JSONB` onto the rows the structure tree wrote, after that tree is written and never
   before it; and a section-level `amends` becomes a declared relation whose source node is the
   clause's breadcrumb. 57 new tests across the five layers plus 10 against live Postgres.

   **What is NOT verified and cannot be without the file:** the quote-validation RATE on real
   model output (the plan's own 95% threshold — the code logs a WARNING below it and enforces
   nothing, because the items that did validate are still correct and still worth storing), the
   usefulness of a real summary as a retrievable unit, whether a real model finds a real
   amendment, and whether the 200-character floor and the 400-section cap are the right numbers
   for a real tender. Unlike Phase 3, NO real-model run was made here: one call per section on a
   68-page SOP is ~40 calls at ~14 s each against the shared box, and the only document available
   to spend them on is not the document the acceptance is about. The unblock is still one file.

19. **The Phase-12 timing slice adds no audit event, and that is a decision, not an omission.**
   The obvious place to hang a latency number is the QUERY audit payload. It is the wrong place
   twice over. `AuditSink.count_actions` METERS those events for FinOps showback (FR-H4), so one
   extra event per question would silently inflate every tenant's bill — a billing defect
   introduced by an observability feature. And the existing QUERY event is recorded MID-FLIGHT,
   before generation, so a total could not honestly go on it. So the timings go to a structured
   log line and to `query_log.latency_ms`, and the audit trail is byte-identical with the feature
   on or off (pinned by a test, because this is the kind of thing a later edit re-breaks).

   Two more limits of the slice, both stated rather than papered over:

   * **The STREAMING path gets two numbers, not six.** `_stream_answer_impl` is a generator with
     half a dozen `yield done; return` exits, so stage marks inside it would be six call sites
     that each have to stay correct; and the number that actually matters for a stream is TIME TO
     FIRST TOKEN, which only an observer of the event sequence can see. It is therefore timed at
     the public `stream_answer` wrapper, which records `first_token` and the total and does not
     pretend to the breakdown.
   * **The sub-intent routes are not timed.** `catalog` / `doc_summary` / `count` / `filter` /
     `compare` return into their own functions before the main path's marks begin. They carry no
     timeline and write no row, so a KB with intent routing on will see fewer rows than questions.
     Each is a separate entry point that would need its own instrumentation.

20. **What is still missing from Phase 12, and why none of it is buildable yet.** The phase's own
   bullet list is mostly about machinery later phases have not built, which is the honest reason
   the slice is a slice:

   * **Batched map calls, the `map_cache`, the router-output cache and the per-(query, doc)
     section reranker cache** — there is no map step, no router output and no section reranking.
     Those are Phases 9, 7 and 8. `map_cache` has had a table since Phase 1 and, like `query_log`
     until today, no writer.
   * **Streaming the reduce, and progressive verification of completed bullets** — Phases 9 and 10.
   * **Context budgeting before each call** — partially addressed already, but not as the plan
     frames it: the Phase-3 and Phase-4 prompts are bounded in CHARACTERS, and a real token budget
     needs a tokenizer the deployed embedder/LLM does not expose (§8.10 records the same gap for
     `token_count`). Doing it properly is its own measurement, not a line of config.
   * **The re-enrichment CLI** ("bump `profile_version`/`model_version` and provide a CLI to
     re-run Phases 3-4 only") — the capability already exists in the wrong shape and the gap is
     not a script. `reindex_document` re-runs both phases, so a corpus CAN be re-enriched today;
     what is missing is (a) a way to re-run them WITHOUT re-extraction, which needs the masked
     text layer and `page_map` persisted — they are not, and nothing reads the stored layout
     overlay back, so `_profile_head`'s input would have to come from somewhere new; and (b)
     SELECTION by contract version, which needs `documents.profile_version` exposed on the
     documents API before a thin HTTP client like `reindex_corpora.py` can filter on it. Both are
     real decisions about what gets stored and what the API promises, so neither was slipped in
     under an ops slice.

21. **Phase 10 does NOT rewrite the NLI checker, and there are now two claim splitters.** The
   plan's Phase-10 item 2 is per-claim entailment with windowing, an entailment floor, and a
   neutral-retry against wider premises. That is `rag_clients.nli_faithfulness`, which already
   exists, already splits claims, already windows long premises (`_focus`, `_table_focus`) and
   already carries a measured compound-query relevance gate — all of it tuned against live
   corpora, with the measurements written into the module. Rewriting it to the plan's letter would
   change every faithfulness score on the platform for every tenant, which is a re-calibration
   (ADR-0005's thresholds are already documented as uncalibrated), not a phase.

   THE COST IS A DUPLICATED SPLITTER, and it is named here rather than hidden.
   `verification.split_claims` and the NLI checker's `_to_claims` both segment an answer, and they
   do not agree: the new one is abbreviation-aware (it had to be — "Rs. 19.44 Crores" was being
   torn in half, which silently emptied every monetary claim of its figure), strips citation
   markers, and drops sub-claim fragments. The NLI one is not changed because its segmentation IS
   its calibration. This is the same hazard `graph.quote_key` was promoted to public to avoid, and
   it is accepted here only because unifying them moves a live gate. The unification is a
   measurement, and it belongs with the threshold recalibration ADR-0005 already asks for.

   The plan's claim-split wording ("split conjunction lists into items when they are
   enumerations") is also deliberately NOT implemented as a split. Splitting "A, B and C" into
   three claims multiplies one assertion into three, each separately scored against the same quote
   and each separately counted in a rate the plan states per claim — inflating the denominator
   with the same sentence three times. The enumeration is compared IN PLACE instead
   (`enumeration_items`), which is the same check with an honest count.

22. **What Phase 10 does NOT do, and why each is a decision rather than a gap.**

   * **It never changes an answer's STATUS.** The plan's remedy for a failing claim is editorial —
     footnote it or drop it — and both are stronger consequences than a score nudge, which is
     exactly what §5 criticised the pre-existing numeric guard for lacking. Routing a
     DETERMINISTIC check into ABSTAIN would let one false positive withhold a correct answer; the
     notation guard already shows the care that needs, capping into the FLAGGED band rather than
     failing outright. A test pins the status unchanged with the flag on or off, and pins that no
     audit event is added (`count_actions` meters QUERY events for FinOps showback, so one extra
     per question would inflate every tenant's bill).
   * **REGENERATE-THE-FAILING-BULLET is not implemented** (plan item 4's first half). The answer
     text carries positional `[n]` markers into `GroundedAnswer.citations`, so regenerating one
     bullet means a model may return different markers, no markers, or markers pointing at the
     wrong entry — and re-anchoring them needs the citation assembler to re-run over the edited
     text, which is Phase 9's reduce. Shipping it without that would risk editing an answer into
     one whose citations no longer resolve, which is worse than the unverified claim it replaces.
     The configured fallback (footnote, or drop-and-footnote) is implemented and is what the plan
     itself falls back to.
   * **The COMPLETENESS LOOP is not implemented** (plan item 5). It is explicitly aggregate-only
     and re-runs the map step for a high-scoring section that produced zero items. There is no
     aggregate strategy and no map step — Phases 8 and 9.
   * **STREAMING verifies after the tokens have been sent.** §5 flagged this before Phase 10 and
     it is still true: verification runs at the `stream_answer` wrapper, so the footnote reaches
     the final `done` answer but not the token stream. A console that renders `done` shows it; one
     that only concatenates tokens does not. The plan's own remedy is "verify before display
     (config)", which is a UI contract rather than a pipeline change.
   * **The value vocabulary is deliberately SHORT.** The unit list is explicit (crore/lakh/kV/
     GB/TB/Gbps/EPS/nits/°C/%/…), standards bodies are matched case-SENSITIVELY so "when 50"
     cannot read as "EN 50", and an article-prefixed non-acronym list is not read as a list at
     all. Every one of those bounds trades a miss for a false positive on purpose: `numeracy`'s
     own docstring states the rule this phase is governed by — "a guard that fires on correct
     answers gets turned off" — and the measured false-positive rate is what the fixture set
     exists to protect.

23. **What Phase 11 does NOT do, and the defect it found on the way.**

   * **NO DEFAULT FLOOR.** `SuiteFloors` defaults every threshold to None = ungated, and an
     operator passes the floors they have agreed. That is the difference between a gate and a
     guess: the plan's §0 criteria are owner-confirmable numbers — §9.2 still lists the latency
     targets as open — and a floor invented here would either be so low it gates nothing or so
     high it fails every build until somebody lowers it, at which point the gate has taught the
     team to edit the threshold instead of the code. Relatedly, a metric a suite does not ASSERT
     can never fail a floor: `sop.yaml` has no supersession items, so its supersession
     correctness is null, not 1.0.
   * **ROUTER INTENT ACCURACY has no producer.** The gold format carries `intent` and the report
     counts them, but nothing measures whether a router AGREES: there is no Phase-7 router, and
     the existing intent classifier labels a different taxonomy
     (`catalog/doc_summary/summarize/count/filter/compare/rag` against the plan's
     `lookup/aggregate/compare/summarize/procedural/relational`). Scoring one against the other
     would report a number about a mismatch of vocabularies. The labels are a fixture waiting for
     its consumer, which is why the format was worth building first.
   * **SECTION COVERAGE is not computed.** It needs the sections an aggregate query selected —
     `query_log.selected_sections`, Phase 8's column, still NULL.
   * **PER-STAGE LATENCY reaches the metrics layer but not the CLI.** `score_query` takes the
     Phase-12 timeline and `score_suite` reports a per-stage p95, both tested. The CLI can only
     supply END-TO-END, because the query response does not carry `latency_ms` — the stage
     breakdown lives in the `query_stage_timing` log line and in `query_log.latency_ms`, and
     surfacing it on the response is an API-contract change that an eval harness should not make
     on its own. An operator reads the breakdown from `query_log` today.
   * **SHADOW MODE is not built.** It needs `RAG_V2_QUERY=shadow`, a tri-state the boolean-only
     platform flag store cannot hold (§8.6), AND a v2 query path to shadow — Phases 7-9. There is
     nothing to run beside v1 yet.

   **A DEFECT THIS PHASE FOUND, in the harness it was extending.** `scripts/eval_gate.py` has
   claimed since it was written that its folding makes "NO₃⁻" match "NO3-". It did not: NFKC maps
   the SUPERSCRIPT MINUS (U+207B) to the MINUS SIGN (U+2212), not to the ASCII hyphen, so a
   chemistry gold answer typed with a hyphen scored as a MISS against an extraction that preserved
   the real character — a silent false negative in the release gate, on the corpus class the
   notation work exists for. Caught by a test written to assert the documented behaviour rather
   than the implemented one. `fold_text` now folds the whole dash family and is the single
   definition both the v1 gate and the v2 suite use. The change can only make a match more
   generous, so nothing that passed before fails now.

24. **Phase 7's acceptance is UNVERIFIED, and what was measured instead.** The plan's acceptance
   is "the router labels a 40-query mixed test set (owner + Appendix A) with >= 90% intent
   accuracy". Neither half of that set exists: the owner's queries have not been supplied and
   Appendix A is the AEGCL corrigendum (§7, §9.1). So the same measurement was made at the same
   size against a set composed in `libs/rag-core/tests/test_router.py` — 40 queries across the
   six intents, over-sampling the collisions the cue ordering has to resolve.

   **It scored 38/40 on first run and 40/40 after two fixes, and the second number is the weaker
   one.** Both misses were real defects the set exposed — "amend" matched as a SUBSTRING, turning
   "tl;dr of the amendment" (a summarize question whose topic is an amendment) into a relational
   one, and "refers to" missed "which clauses does this corrigendum refer to" because the verb
   was inflected differently — and the fixes are principled (word boundaries, an inflection
   group). But a score measured on the set that shaped the code is not independent, and 40/40
   should be read as "no known ordering or matching defect", not as an accuracy estimate. The
   owner's set is the real test.

   **One genuinely independent disagreement was found and is worth more than the 40.** The
   Phase-11 gold set labelled "Which designations are responsible for preparing, approving and
   authorising an SOP?" as `aggregate` — written before this classifier existed — and the
   classifier said `lookup`, because the question carries no all/list/every cue. The gold label
   was right: a PLURAL SUBJECT with a plural verb expects more than one answer however the
   question is phrased. That rule is now in the classifier, and the Phase-11 suite's labels are
   asserted against the router in `services/guardrails-eval/tests/test_suite.py` — which closes
   the loop §8.23 left open, where the intent metric was "a fixture waiting for its consumer".

   **What the router does NOT do yet, because the consumers do not exist.** `facets` and
   `needs_amendment_view` are decided, validated and persisted, and nothing reads them: grouping
   an aggregate answer by facet and unioning the amending sections are both Phase 8. The
   `aggregate` intent maps to the existing `whole_document`/exhaustive mechanism, which is the
   NEAREST available answer and not the plan's aggregate strategy — that one enumerates
   section-level units and reranks them against the sub-queries, and has no implementation. The
   router also runs on the non-streaming path only, for the reason the Phase-12 timing slice and
   the Phase-10 verifier do: the streaming impl is a generator with half a dozen exits, and
   wiring a decision through it is a second change of its own. And the plan's cross-lingual step
   ("if query_lang != document lang, translate the query for the OpenSearch leg") is NOT wired —
   `answer_lang` is captured and stored, but the existing `_cross_lingual_retrieve` path decides
   translation on its own signals, and routing that decision through the router is a behaviour
   change to a live path that needs its own measurement.

---

## 9. Decisions needed before Phase 1 — [ASK]

1. **AEGCL source PDF** + the KB it is ingested into (blocks Appendix A, §7). **Still open as of 2026-10-21, and now blocking the acceptance of THREE phases that are otherwise built — §8.14 (Phase 3), §8.18 (Phase 4) and §8.24 (Phase 7, which also needs the owner's 40-query intent set).** The KB is not required; the file alone is enough to run both acceptances against the live `gemma4-31b`.
2. **Latency targets** — confirm ≤8 s lookup / ≤60 s aggregate p95, or defer until §8.2 makes
   them measurable. Note the dev stack is CPU-bound while the real target is the UAT VM.
3. ~~Approve Phase 0.5~~ — **approved 2026-10-16** (option A + the Qdrant hardening). Baseline still pending items 1 and 2 below.
4. **Run the D2 check on the UAT VM** — `docker exec <api> env | grep RAG_LLM_CONTEXT_WINDOW`.
5. **Two more structure fixtures** for Phase 2 acceptance, and — now that the Phase-11 harness
   exists and has one suite to run — the **HR policy handbook, contract and technical-paper gold
   fixtures** that the plan's "corpus diversity: at least 4 document types" needs. `eval/gold/`
   holds `sop.yaml`; the format is in `eval/README.md`. The AEGCL corrigendum (item 1) is the
   fourth and would also unblock the aggregate and supersession halves, which `sop.yaml` can only
   partly exercise.
6. **Multi-document scope rules** — current upload/session, or whole tenant corpus? Today a
   document-scoped mode exists only implicitly (title match, single-doc KB, planner
   `target_doc`); there is no explicit request parameter.
7. **UI changes** for the coverage line, observations section and unverified-claims footnote.
8. ~~**Reconcile `domain_profile` with the Phase 3 doc profile** — one profile concept or two?~~ — **answered 2026-10-21: two** (§8.13).
