## Chunking, Contextualize & Embedding

Scope: `libs/rag-core/src/rag_core/{chunking,contextual,atomic_units,parent_child,sections}.py`, `libs/clients/src/rag_clients/*embedder*.py`, `services/indexing/src/svc_indexing/service.py`. Read-only discovery; all line numbers verified against the working tree. Paths below are relative to the repository root.

### 1. Chunk

**Splitters / strategies.** Three are wired; two enum members are unimplemented stubs.
- Enum: `FIXED, CLAUSE, LAYOUT_AWARE, PARENT_CHILD, CONTEXTUAL` (`libs/schemas/src/rag_schemas/common.py:31-38`). `get_chunker` builds only FIXED / CLAUSE / PARENT_CHILD (`libs/rag-core/src/rag_core/chunking.py:489-513`); anything else raises `NotImplementedError` (`chunking.py:514-515`).
- `FixedChunker` — fixed character windows + overlap (`chunking.py:115-155`).
- `ClauseChunker` — splits on `(?<=[.;:\n])\s+` (`chunking.py:158`), accumulates clauses to `max_chars`, carries a word-aligned tail as overlap (`chunking.py:207-222`, `266-369`), then folds fragments forward (`_merge_fragments`, `chunking.py:372-421`).
- `ParentChildChunker` — clause engine at child size + per-section parent records (`libs/rag-core/src/rag_core/parent_child.py:80-133`, `220-254`).

**Default and config** (`libs/common/src/rag_common/config.py`):
- `chunk_strategy` default `"clause"`, pattern `fixed|clause|parent_child` (`config.py:332`). `.env.example:98` also says `clause | fixed`.
- `chunk_size=900` and `chunk_overlap=150` are FIXED only (`config.py:335-336`).
- `chunk_max_chars=700` is CLAUSE only (`config.py:340`). The comment calls it "currently a FLOOR, not a cap" (`config.py:337-339`), but the code now flushes before appending (`chunking.py:321-351`).
- CLAUSE overlap: `_DEFAULT_CLAUSE_OVERLAP=100` (`chunking.py:67`); the service passes `chunk_overlap` (150) into `get_chunker` (`service.py:185-189`), so CLAUSE gets 150 clamped to `max_chars//2` (`chunking.py:276`).
- Parent-child: child 350 chars and parent 1800 chars (`config.py:351,354`). Both are overridable per KB (`service.py:193-196`, `services/control-plane/src/svc_control_plane/pipeline.py:763-770`). `parent_child_enabled` defaults False (`config.py:348`), but the chunker is selected by `chunk_strategy`, not that flag.
- Units are CHARACTERS, not tokens. A token cap is layered on when the embedder can report one: `max_tokens`/`token_offsets` come from TEI `/info` + `/tokenize` (`service.py:143-154,207`; `libs/clients/src/rag_clients/tei_embedder.py:70-127`). Only the TEI adapter implements `token_budget`/`token_offsets`. `openai_embedder.py` and `bge_m3_embedder.py` do not, so the deployed config (see 4) has the char cap only.
- No sentence/token target of 300-500 exists. At 700 chars the target is about 140 prose tokens (`chunking.py:296-298`).

**Respects headings?** Partly.
- `detect_sections(text)` runs once per document (`service.py:217`). It is numbering-regex based: multi-level `6.1.1 Title` (`sections.py:31`), a trailing heading merged onto a paragraph (`sections.py:35`), and single-level `1. Title` only if the numbers ascend (`sections.py:83-88,155-171`). Markdown `#` headings and unnumbered headings are NOT detected (explicitly rejected, `sections.py:66-76`).
- ClauseChunker flushes at a STRUCTURAL heading (`opens_section`, `chunking.py:346-348`). Single-level headings are `structural=False` and only label (`sections.py:113,171`).
- FixedChunker does NOT respect headings (no `opens_section` logic, `chunking.py:141-155`). It only records a section label.
- Heading offsets are bounded by `_MAX_SECTION_SPAN=2000` (`sections.py:57,222`). Beyond that the path is empty.

**Respects tables?**
- ClauseChunker: NO. It has no atomic-unit logic, so a table can be split mid-row by the clause regex or hard wrap.
- ParentChildChunker: YES, via `atomic_spans` (`parent_child.py:127-133`, `atomic_units.py:435-442`). A table is a run of >=2 consecutive lines containing `" | "` (`atomic_units.py:308,360-375`). Code fences are included (`378-384`), and so are short formula lines (`387-414`). An atomic unit is emitted as ONE child even if oversized (`parent_child.py:158-165`).
- Table detection is a text heuristic on `" | "`. There is no layout/HTML table model, no header-name propagation and no row-level units.

**Can a chunk cross a section boundary today?**
- Only for heading-less or unreliably detected structure.
- CLAUSE and PARENT_CHILD hard-split at structural headings (`chunking.py:346-351`, `parent_child.py:170-175`). The code comment measured 28.9% of SOP chunks straddling one before this change (`chunking.py:334-338`).
- Carry-over overlap can pull the previous chunk's tail across a heading. The span covers the carry but "context may cross a page" (`chunking.py:325-330`, `_tail` at `207-222`).
- `_merge_fragments` folds a small chunk only within the same section label (`chunking.py:407`, `439-440`).
- A non-structural (single-level) heading never forces a boundary (`chunking.py:339-345`), and if a chunk range contains a heading, `section_path_for` returns `""` (`sections.py:244-245`).
- FIXED can cross sections freely.
- Pages are also a hard boundary (`chunking.py:331-333`).
- No test asserts "no chunk crosses a section". The Phase 5 acceptance test does not exist yet.

### 2. Contextualize (most important)

**Gate.**
- `contextual_retrieval_enabled: bool = False` (`config.py:974`). `.env.example:124` is `false`.
- `contextual_retrieval_backend: str = "metadata"` (`config.py:975`), with values `metadata | llm`.
- The flag is ON in `infra/compose/docker-compose.dev.yml:367`, `infra/compose/docker-compose.mesh.example.yml:125` and `infra/helm/values.yaml:143`. It is absent from `infra/compose/docker-compose.deploy.example.yml` — but **CORRECTED: that is not the production path.** Production runs the single-file `infra/compose/docker-compose.prod.yml` with `--env-file .env.prod`, and `infra/compose/.env.prod-vm.example:88` sets `RAG_CONTEXTUAL_RETRIEVAL_ENABLED=true`. `deploy.yml` is one overlay in the dev-only five-file merge (`Makefile:15-19`). Both live containers on this box have it true. NOT a production gap.
- Applied in `service.py:129-131` (flag) and `service.py:247-258` (use): `contextual = self._contextual_enabled and self._contextualizer is not None`.
- The contextualizer is injected by `get_contextualizer()` (`libs/clients/src/rag_clients/factory.py:276-280`). It is wired at `services/control-plane/src/svc_control_plane/wiring.py:235,762`.

**Implementations** (Protocol `Contextualizer.context_for(doc, chunk, full_text) -> str`, `libs/rag-core/src/rag_core/contextual.py:21-24`):
1. `MetadataContextualizer` — deterministic, no LLM, DEFAULT (`contextual.py:27-55`). Selected at `libs/clients/src/rag_clients/contextual.py:31-32`.
2. `LLMContextualizer` — one LLM call per chunk, `max_tokens=64, temperature=0.0` (`libs/clients/src/rag_clients/contextual.py:17-25`). It needs the gateway or raises `ConfigError` (`:33-36`). It is opt-in only.

**Exact output per chunk (default / metadata backend).** Template code is at `contextual.py:44-55`:
```
parts = [f"Document: {title}"]                       # contextual.py:46   title = doc.metadata["title"] or doc.doc_id (:45)
if section: parts.append(f"section {section}")       # :47-49  section = chunk.metadata["section"] (heading path, " > " joined, sections.py:42)
if chunk.page is not None: parts.append(f"page {chunk.page}")   # :50-51
tags = "; ".join(f"{k}={v}" for k,v in doc.metadata.items() if k not in {"title","section"})  # :52
if tags: parts.append(tags)                          # :53-54
return "[context] " + ", ".join(parts)               # :55
```
Composition: `contextualize_text(context, text) = f"{context}\n\n{text}"` (`contextual.py:69-71`). It is applied at `service.py:249-252`.

**LLM backend prompt** (`contextual.py:58-66`). It sends the first 2000 chars of the document (`:60`), not the full document:
```
<document>\n{full_text[:2000]}\n</document>\n\nHere is a chunk from that document:\n<chunk>\n{chunk.text.strip()}\n</chunk>\n\nWrite ONE short sentence that situates this chunk within the document to improve search retrieval (what section/topic it belongs to). Answer with ONLY the sentence.
```
LLM output is used as the context string verbatim (stripped). It is NOT prefixed with `[context]`.

**What is NOT in the output:** no doc_type, no purpose, no issuer/date unless present as free-form `doc.metadata` tags, and no table/column context. The breadcrumb is only the numbered-heading path; it is empty for unnumbered documents. The "title" is `doc.metadata["title"]`; where this metadata is populated at ingest was not traced. The fallback is `doc_id`, i.e. the filename (example 3). `pipeline.py:2068` has a separate `title or filename or doc_id` helper that the contextualizer does NOT use.

**Three real examples.** All three were produced by executing `ClauseChunker(max_chars=90, overlap=0)` + `MetadataContextualizer` + `contextualize_text` in the repo venv (`.venv/bin/python`) on the fixture text from `libs/rag-core/tests/test_contextual.py:48-55` (`_sop_text`). The doc `metadata` was `{"title": "Acme Content Governance Master SOP", "category": "policy", "doc_date": "2026-03-01"}`, with `page_map=[len(text)]` so every chunk is page 1. Source type: text from a test fixture, metadata constructed, output executed (not a stored snapshot). The repo has no stored snapshot of contextualized text. The nearest assertions are `test_contextual.py:29-34` (`"Platform Brief"`, `"page 1"`, `"dept=cloud"`, `startswith("[context]")`) and `:77-97` (section differs per heading).

Example A — executed from the `test_contextual.py` fixture; section 6.1.1:
```
[context] Document: Acme Content Governance Master SOP, section 6.1.1 Mandatory Rehearsal Policy, page 1, category=policy; doc_date=2026-03-01

6.1.1 Mandatory Rehearsal Policy
Written sign-off is required before publication.
```

Example B — executed from the same fixture; identical body, different heading (the case `test_contextual.py:77-97` asserts):
```
[context] Document: Acme Content Governance Master SOP, section 6.4.1 Central Content Repository, page 1, category=policy; doc_date=2026-03-01

6.4.1 Central Content Repository
Written sign-off is required before publication.
```

Example C — constructed from the template, same fixture chunk (the title block), with EMPTY doc metadata (no `title`, no tags), showing the fallback to `doc_id` and no section. The title-only chunk has no governing heading, so the section is omitted (`test_contextual.py:100-114`):
```
[context] Document: sop, page 1

Acme Content Governance Master SOP
```

(The "Mandatory Rehearsal Policy" section label has no ancestors because the fixture contains no `6` / `6.1` headings. Per `sections.py:206-233` ancestors appear only if the document contains them.)

**Where the context goes.**
- The contextualized string is used for embedding (`service.py:279,282`) and BM25 (`index_chunks[].text`, `service.py:253-255,284`).
- The catalog stores the ORIGINAL child text (`service.py:285`), so answers/citations never show the prefix (`contextual.py` docstring lines 1-9).
- The Qdrant payload carries `chunk_id, doc_id, tenant_id, page, acl, **metadata` and NO text (`libs/clients/src/rag_clients/qdrant_store.py:152-166`).

### 3. Embed

**Which text is embedded.**
- Contextualized when the gate is on, raw child text otherwise. The selection is `service.py:247-258` (`embed_texts = [contextualize_text(...)...]` at `:249-252`, else `embed_texts = [c.text for c in children]` at `:257`). The embed call is `await self._embedder.embed(embed_texts)` at `service.py:282`, or `embed_multi(embed_texts)` at `:279`.
- Parents are never embedded or BM25-indexed (`service.py:228-235,285`; test `services/indexing/tests/test_parent_child_partition.py:53-87`).
- The LLM at answer time sees the original text, plus a non-citable parent context bank at query time (`_parent_expand`, `pipeline.py:1213`).

**Batch size.** 32 in every adapter: `tei_embedder.py:20,133`, `openai_embedder.py:38,78`, `bge_m3_embedder.py:43,113`.

**Truncation / max length.**
- TEI: sends `"truncate": true` (`tei_embedder.py:137`), so over-length input is silently clipped. The adapter and chunker try to bound this by reading `max_input_length` from `/info` (`tei_embedder.py:84-87`).
- `openai_compat` (`openai_embedder.py:81`) and `bge_m3_multi` (`bge_m3_embedder.py:83-97`) send NO truncate flag and no max-length. Server-side truncation or rejection applies.

**Normalised?** There is NO client-side L2 normalisation anywhere in the three adapters (grep for normali/l2 found none). Vectors are stored as returned and Qdrant uses `Distance.COSINE` (`qdrant_store.py:74,95,98`), which is scale-invariant. Server-side normalisation by TEI / the AIOps endpoint is assumed but not verified in the repo.

**Dimension check.**
- `openai_embedder.py:34-40` and `bge_m3_embedder.py:99-106` raise `ConfigError` on width mismatch.
- TEI only logs `embed_dim_mismatch` (`tei_embedder.py:140-141`).

### 4. Default embedder, dimension, and BGE-M3

- Code default: `embed_backend="tei"`, `embed_model="BAAI/bge-m3"`, `embed_dim=1024`, `tei_url="http://tei:80"` (`config.py:145-148`).
- Factory dispatch `get_embedder()`: `tei` -> `TeiEmbedder`, `openai_compat` -> `OpenAICompatEmbedder`, `bge_m3_multi` -> `BgeM3MultiEmbedder`, in-memory -> `LocalHashEmbedder` (`factory.py:86-116`).
- Dev and CI compose run TEI with `BAAI/bge-small-en-v1.5` (`infra/compose/docker-compose.dev.yml:99`, `docker-compose.ci.yml:51`). Per the `TeiEmbedder` docstring that is 384-dim (`tei_embedder.py:7`), so dev does not use BGE-M3 unless `RAG_EMBED_*` overrides it (`docker-compose.dev.yml:229` comments a GPU/bge-m3 swap).
- Deploy compose (the UAT/POC-bundle shape): `RAG_EMBED_BACKEND=openai_compat`, `RAG_EMBED_MODEL=bge-m3`, `RAG_EMBED_DIM=1024`, endpoint `https://models.example.internal/openai/v1` (`infra/compose/docker-compose.deploy.example.yml:43-48`). So BGE-M3 IS in use in the deploy config, via the OpenAI-compatible `/embeddings` facade, DENSE ONLY (`openai_embedder.py:81`).
- Sparse and ColBERT heads exist as code but are OFF by default:
  - `multivector_enabled: bool = False` (`config.py:166`). It requires `embed_backend=bge_m3_multi` and `embed_api_auth=bearer` (`config.py:145,182-190`).
  - `service.py:133-141` `_use_multivector()` requires the flag AND `MultiVectorEmbedder` AND `MultiVectorStore`.
  - `BgeM3MultiEmbedder.embed_multi` returns dense + learned-sparse + ColBERT (`bge_m3_embedder.py:121-142`), written via `upsert_multi` (`service.py:278-280`, `qdrant_store.py:248-270`).
  - Per-KB legs: `multivector_sparse_enabled`, `multivector_colbert_enabled`, `colbert_prefetch_pool` (`pipeline.py:751-762`).
  - Not found enabled in `docker-compose.deploy.example.yml` (grep MULTIVECTOR = no match).
- `pipeline_fingerprint` hashes strategy, sizes, embed model/dim, contextual flag+backend, graph and OCR backends (`service.py:51-85`), so a stale-index detector exists.

### 5. Parent-child / multi-level units

Levels that exist today:
| Level | Exists? | Evidence |
|---|---|---|
| chunk (flat, FIXED/CLAUSE) | yes, the only searchable unit unless PARENT_CHILD | `chunking.py:115-369` |
| child (PARENT_CHILD) | yes, searchable and citable; `metadata["parent_id"]` | `parent_child.py:249`, `service.py:236` |
| parent (per-section, <=1800 chars) | yes, catalog-only context bank; `metadata["chunk_role"]="parent"`; created only if >=2 children | `parent_child.py:53-55,66,220-254`; grouping by same `section` and size, `257-279`; consumed by `_parent_expand` (`pipeline.py:1213`) |
| atomic unit (table/code/formula) | yes, but a chunking guard, not a retrieval level: emitted as one child | `atomic_units.py:435`, `parent_child.py:156-165` |
| section record / section summary | NO. The "parent" is just joined child text, not a summary, and is not retrievable | `parent_child.py:242` |
| doc-level card | NO (a whole-doc summary exists elsewhere, `RAG_WHOLE_DOC_SUMMARY_BATCH_CHUNKS` in `docker-compose.dev.yml:420`, not traced here) | n/a |
| row-level (table row) | NO | tables are one atomic child |
| proposition | NO | `grep -i proposition` over `*.py` returns nothing |

`atomic_units.py` is NOT proposition-like. It finds spans (table / code fence / formula line) that must not be split. It produces no standalone statements, and has no LLM. Its only importer is `parent_child.py:29` (`grep` confirmed), so it is wired into the INGEST path only when `chunk_strategy=parent_child`. It is not used on the query path. On the query side only the parent is consumed (`pipeline.py:3821,4088`).

### 6. Chunk IDs: stable and deterministic?

YES. Content-derived, not positional.
- `chunk_id(doc_id, text, seen) = f"{doc_id}::{sha256(text)[:12]}"` (`chunking.py:96-112`, hash at `:107`, `_ID_HEX=12` at `:93`). It is `…::<hex>` for the first occurrence and `…::<hex>.<n>` for the n-th identical text within one document (`:110-112`).
- IDs are rebuilt after merge and parent grouping so duplicate suffixes stay doc-scoped (`chunking.py:418-421`, `parent_child.py:250-253`). Parent ids are the same hash over the joined parent text (`parent_child.py:245`).
- Stability caveat: the ID moves whenever the chunk TEXT moves. A new chunk size, overlap or section boundary changes ids; the ID is not tied to the structure.
- Contextualized text does NOT enter the ID. The ID is computed in `_mk_chunk` (`chunking.py:44`) before contextualization, so toggling contextual retrieval keeps IDs.
- Qdrant point id is `_point_id(c.chunk_id)` (`qdrant_store.py:158`). Re-index upserts in place (same id) and `IndexResult.chunk_ids` drives stale-swap (`service.py:301`; test `test_parent_child_partition.py:90-113` asserts ids are identical across an identical re-chunk).
- The module docstring at `service.py:12` still says `doc_id::n` (positional). It is stale vs `chunking.py:84-112`.

### Chunk contract (`libs/schemas/src/rag_schemas/chunk.py`)
`Chunk` is frozen, `extra="forbid"` (`chunk.py:24-27`). Fields:
- `chunk_id: str` (`:29`)
- `doc_id: str` (`:30`)
- `tenant_id: str` (`:31`)
- `text: str`, PII already masked (`:32`)
- `modality: Modality = TEXT` (`:33`)
- `page: int|None >=1` (`:35`)
- `span: Span|None` (`:36`), where `Span{start,end}` are char offsets into the extracted doc text (`:15-21`)
- `source_uri: str|None` (`:37`)
- `acl: list[str]` (`:38`)
- `classification: int 0..3` (`:39-41`)
- `ocr_confidence: float|None` (`:43-45`)
- `embedding_ref: str|None` (`:46`)
- `lineage_id: str|None` (`:47`)
- `version: int=1` (`:48`)
- `metadata: dict[str,str]` (`:49`)

There is NO `parent_id`, `level`, `section_id`, `doc_type`, `breadcrumb`, `embed_text` or `chunk_hash` field. Structure lives in the free-form `metadata` dict: `section`, `parent_id`, `chunk_role`, `category`, `doc_date` (`chunking.py:78-82`, `parent_child.py:53-55`), plus all document tags copied from `doc.metadata` (`chunking.py:37`). `metadata` is `dict[str,str]`, so lists and numbers must be stringified. All of it flows into the Qdrant payload (`qdrant_store.py:162`).

### Gaps vs plan (Phase 5)
- ALREADY EXISTS — reuse: stable content-derived chunk ids, `{doc_id}::{sha256(text)[:12]}` (`chunking.py:96-112`). The plan's idempotence-by-`chunk_hash` can reuse the digest. A separate stored `chunk_hash` field is absent.
- ALREADY EXISTS — reuse: section-boundary splitting in ClauseChunker / ParentChildChunker (`chunking.py:346-351`), heading detection and breadcrumb path with ancestors (`sections.py:137-152,206-246`; separator `" > "` at `:42`), and the section path stamped on each chunk as `metadata["section"]` (`chunking.py:38-42`).
- ALREADY EXISTS — reuse: atomic table/code/formula protection (`atomic_units.py:435`, `parent_child.py:127-188`) and the parent-context bank with query-time expansion (`parent_child.py:220-279`, `pipeline.py:1213`). Extend the same machinery to the plan's `parent_id = section_id` links.
- ALREADY EXISTS — reuse: contextualize-before-embed seam with original text kept for citation (`service.py:246-258,285`) and the `Contextualizer` protocol + LLM implementation (`contextual.py:21-24`, `clients/contextual.py:17-25`). A new `embed_text` builder can be a third `Contextualizer`. The "answer-time LLM sees `text`, not `embed_text`" requirement is already satisfied by the catalog storing original text.
- ALREADY EXISTS — reuse: per-document `pipeline_fingerprint` stale detection (`service.py:51-85`). A `_v2` chunker/context change should be folded into it.
- ALREADY EXISTS — reuse: BGE-M3 multi-head embedder (dense + sparse + ColBERT) behind `multivector_enabled` (`bge_m3_embedder.py:121-142`). Deploy currently uses dense only.
- GAP: the contextual prefix is `[context] Document: <title|doc_id>, section <path>, page N, k=v; k=v`. The plan wants `[{doc_type}: {title} — {purpose}] [{breadcrumb}]`. There is no `doc_type` or `purpose` anywhere in the `Chunk` or `DocumentEnvelope` flow, so a doc-profile (Phase 1/2) input is needed. The order and format (`[context]` literal, comma list) must change. The tag blob (`tags`, `contextual.py:52`) dumps ALL `doc.metadata` into every embed text, noisy and unbounded.
- NOT A GAP (corrected): the contextual flag is OFF by code default, but production sets it true via `.env.prod-vm.example:88` on the `docker-compose.prod.yml` path. Still worth confirming the VM's actual `.env.prod` before baselining, since that file is filled in on the host.
- GAP: no multi-level model. There is no `doc`, `section` (summary), `row` or `proposition` level. The `Chunk` schema has no `level` / `parent_id` / `section_id` fields (only string metadata keys). Parents are not retrievable and are not summaries. Table rows are not linearised with header names (tables are single atomic children).
- GAP: breadcrumb/section detection is numbering-regex only. Markdown `#` / HTML `<h*>` / Chandra layout headings are not used (`sections.py:66-76` says layout signal is unavailable). Unnumbered headings and `> 2000`-char gaps yield an empty path. Phase 2's `sections` table output would need to replace `detect_sections`.
- GAP: no test that no chunk crosses a section boundary (Phase 5 acceptance). Known leaks: FIXED strategy, carry-over overlap across headings (`chunking.py:207-222`), and ClauseChunker not being table-aware.
- GAP: size model. Targets are chars (350 / 700 / 900), not 300-500 tokens, and there is no per-doc_type config. The token budget is bounded only for the TEI adapter. The deployed `openai_compat` adapter has no tokenizer hook (`openai_embedder.py` has no `token_budget`), so the char cap alone applies there. No mid-sentence guarantee beyond clause regex (`chunking.py:158`) and `_hard_wrap` word-boundary cuts (`:225-263`).
- GAP: no `Chunk` fields for the `_v2` payload (`doc_type`, `purpose`, `level`, `section_id`, `breadcrumb`, `embed_text`). The Qdrant payload is built from `metadata` (string-only) (`qdrant_store.py:156-163`).
- GAP: no proposition extraction. `atomic_units.py` is span protection, not proposition generation.
- Note: the LLM contextualizer's context sentence (when enabled) is not prefixed with `[context]` and sees only the first 2000 characters of the document (`contextual.py:60`). It is unsuitable as-is for long documents.
