## Config, Feature Flags & Evaluation

Paths relative to the repository root. Read-only discovery; nothing was executed.

### 1. Config system

**Mechanism.** 12-factor, env-driven pydantic-settings. One flat `Settings` class, not nested.
- `_ENV_PREFIX = "RAG_"` — `libs/common/src/rag_common/config.py:15`.
- `_Base(BaseSettings)` — `config.py:28-35`: `env_prefix=RAG_`, `env_file=".env"`, `extra="ignore"`, `frozen=True`.
- `class Settings(_Base)` — `config.py:38` (about 1,240 lines, grouped by `# --- section ---` comments, no sub-models).
- Accessor `get_settings()` — `config.py:1278-1281`, `@lru_cache(maxsize=1)`. Process-wide singleton. Invalidate with `get_settings.cache_clear()`; tests do this in `conftest.py:_clear_caches`.
- Field `foo_bar` is read from env `RAG_FOO_BAR`. A plan flag `RAG_V2_QUERY` therefore becomes field `rag_v2_query` and env `RAG_V2_QUERY`.
- Validation is at startup via `Field(ge=, le=)` and `pattern=`. Example: `rerank_top_k: int = Field(default=20, ge=1, le=100)` at `config.py:260`.

**Sources, in precedence order.**
1. Process env (`RAG_*`). In prod these are injected by the SecretsProvider (`.env.example:1-6`).
2. `.env` file.
3. Python defaults on the field.
4. Runtime admin toggles, which overwrite `os.environ` and rebuild settings (see §2).
5. Per-KB overrides, held outside `Settings`. They are rows in the catalog `tenant_config` table (key/value strings), read with `get_catalog().get_tenant_config(kb_id)` (`libs/clients/src/rag_clients/postgres_catalog.py:184`, setter at `:191`).
   - The allow-listed numeric keys are `_THRESHOLD_KEYS` (`services/control-plane/src/svc_control_plane/pipeline.py:737`).
   - `kb_threshold_overrides(kb_id)` returns `dict[str, float]` (`pipeline.py:1835`). A boolean is stored as 1.0/0.0; counts are cast to int by the consumer.
   - Keys include `abstain_threshold`, `faithful_threshold`, `flag_threshold`, `completeness_*`, `exact_boost`, `multivector_*`, `colbert_prefetch_pool`, `notation_grounding_enabled` and `parent_child_*`.
   - A Guardrails-policy overlay can override `faithful_threshold` (`pipeline.py:1850-1860`).
   - Per-tenant models (LLM, embedder) are resolved separately: `wiring.resolve_tenant_gateway` / `resolve_tenant_embedder` (used at `ragas_eval.py:~98`).

**Steps to add a new tunable correctly.**
1. Add a typed field with a default and bounds to `Settings` (`config.py`), with a comment stating why and how the value was measured. Follow the `query_top_k` precedent (`config.py:315-322`).
2. Document it in `.env.example` as `RAG_<NAME>=<default>`.
3. Read it with `get_settings().<name>` at call time, not at import time. Import-time reads break the cache-clear and toggle mechanism.
4. If it must differ per KB: add the key to `_THRESHOLD_KEYS` (`pipeline.py:737`), read it from `kb_threshold_overrides`, and fall back to the platform default when it is absent.
5. If an admin should be able to flip it at runtime: add a `TOGGLEABLE` entry (§2). Security and infrastructure settings go in `LOCKED`, read-only.
6. If it is shown in the console: expose it in the `/system/info` payload (`api.py:~5430`, `thresholds={...}`).
7. Add a unit test. Pattern: `monkeypatch.setenv("RAG_X")` plus `get_settings.cache_clear()`.

**Retrieval, generation and model settings in `Settings`** (`config.py`; defaults mirrored in `.env.example`).

| Setting | Default | `config.py` line |
|---|---|---|
| `embed_model` | `BAAI/bge-m3` (`.env.example:61` uses bge-small) | 147 |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 214 |
| `rerank_candidates` | 50 | 218 |
| `rerank_top_k` | 20 | 260 |
| `rerank_fallback_model` | "" | 272 |
| `llm_model` | `Qwen/Qwen2.5-3B-Instruct` | 288 |
| `llm_max_tokens` | 1024 | 301 |
| `llm_reasoning_effort` | medium | ~306 |
| `heavy_top_k` | 10 | 315 |
| `query_top_k` | 16 (measured; see §3) | 322 |
| `heavy_max_tokens` | 2048 | 323 |
| `chunk_max_chars` | 700 | 340 |
| `parent_child_{child,parent}_max_chars` | 350 / 1800 | 351 / 354 |
| `parent_child_max_parents` | 5 | 357 |
| `parent_child_context_max_chars` | 5000 | 362 |
| `near_duplicate_threshold` | 0.85 | 367 |
| `abstain_threshold` | 0.15 (calibrated for the cross-encoder scale) | 380 |
| `abstain_threshold_lexical` | 0.60 | 413 |
| `refusal_retry_min_score` | 0.0 | 398 |
| `faithful_threshold` / `flag_threshold` | 0.7 / 0.4 | 452 / 453 |
| `nli_model` | `cross-encoder/nli-deberta-v3-base` | 488 |
| `nli_support_threshold` | 0.5 | 489 |
| `nli_premise_max_chars` | 1000 | 473 |
| `coverage_min_score` / `coverage_max_spans` / `coverage_max_items` | 0.5 / 6 / 12 | 505 / 510 / 486 |
| `completeness_*` | 3 / 1.5 / 3 / 12 | 527-541 |
| `colbert_prefetch_pool` | 100 | 179 |
| `exact_term_boost` | 2.0 | 989 |
| `graph_retrieval_candidates` / `graph_retrieval_max_chunks` | 30 / 10 | 1197 / 1198 |
| `semantic_cache_threshold` | 0.97 | 1123 |
| `whole_doc_full_read_token_budget` | 96000 | 1055 |
| `whole_doc_summary_max_chunks` | 400 | 1068 |
| `llm_context_window` | 128000 | ~1060 |
| `doc_summary_max_docs` | 8 | 1034 |
| `ocr_min_confidence` | 0.5 | 635 |
| `chandra_model` | `datalab-to/chandra` | 774 |
| `vision_llm_model` | `Qwen/Qwen2.5-VL-7B-Instruct` | 754 |
| `continuous_eval_min_grounded_rate` / `continuous_eval_min_faithfulness` | 0.6 / 0.5 | 951 / 952 |
| `query_planner_enabled` | False (existing agentic planner, master flag) | 1046 |

- `.env.example` (209 lines) groups, in file order: App; Sovereignty guard; Auth; Secrets; Vector store (Qdrant); Sparse (OpenSearch); Object store (MinIO); Postgres catalog; Embeddings; LLM gateway; per-KB model attach and model CA; Chunking; Grounding (`RAG_ABSTAIN_THRESHOLD=0.15`, line 110); Faithfulness (`FAITHFUL_THRESHOLD=0.7`, `FLAG_THRESHOLD=0.4`, lines 115-116); Notation grounding; Observability (148); Quotas (153); P2 agentic: semantic cache (162); GraphRAG (173); Conversational (180); OCR routing (187); Reranking (194-209).
- `.env.example` only lists a subset of the Settings fields. Thresholds, k values and model names it contains: `EMBED_MODEL` (61), `LLM_MODEL` (72), `CHUNK_MAX_CHARS` (101), `ABSTAIN_THRESHOLD` (110), `FAITHFUL_THRESHOLD` (115), `FLAG_THRESHOLD` (116), `SEMANTIC_CACHE_THRESHOLD` / `_MAX_ENTRIES` (167-168), `RERANK_MODEL` (202), `RERANK_CANDIDATES=50`, `RERANK_TOP_K=20` (207), `RERANK_BATCH_SIZE=32`, `QUOTA_MAX_DOCUMENTS_PER_KB` (158).
- **Prompts are NOT in config or in prompt files.** They are inline module constants: `llm_gateway.py:59` (`_NO_REASONING_SYSTEM`), `doc-intelligence/ocr.py:796` (`_CHANDRA_OCR_PROMPT`), `multimodal.py:49,100,104`, and others. There is no prompt directory and no template loader (`find -iname '*prompt*'` finds only tests). This conflicts with the plan's "prompts in prompt files" constraint.

### 2. Feature flags

**Existing mechanism: `services/control-plane/src/svc_control_plane/feature_flags.py` (227 lines).**
- `TOGGLEABLE: dict[str, tuple[env_var, on_value, off_value, label]]` — `:36`. 16 entries: guardrails, faithfulness, semantic_cache, agentic_retry, contextual_retrieval, adaptive_router, query_transform, followups, auto_metadata, field_extraction, language_detection, translation, ocr, nl2sql, agentic_workflows, graphrag.
- `LOCKED` — `:57`. Read-only: sovereign_enforce, pii_masking, oidc_sso, mfa_required, multi_replica_scale, byok_kms.
- **Scope: global only, one set per deployment.** The module docstring says "Platform-wide (single override set per deployment)". There is no per-tenant or per-KB flag in this module.
- **Persistence:** Postgres table `"rag_identity".platform_flags (name TEXT PK, enabled BOOLEAN)` — DDL at `:173-176`, `PostgresFlagStore` at `:179`. The in-memory twin is `InMemoryFlagStore` (`:157`). Selection is by `wiring.feature_flag_store()` (`wiring.py:357`, in-memory when `inmemory`). The store is boolean-only (`put(name, enabled: bool)`).
- **Write path:**
  - `set_flag` (`:204`) calls `store.put`, then `_apply_env` (`:130`, writes `os.environ[RAG_X]`), then `_invalidate` (`:105`, `get_settings.cache_clear()` plus `wiring.doc_intel_service.cache_clear()`).
  - Startup: `load_overrides` (`:217`) is called from `apps/api/src/rag_api/app.py:107`. It first calls `capture_baseline()` (`:122`), which records the operator's env value so that "enable" restores the operator's backend choice (`_BASELINE`, `:119`).
  - HTTP: `GET /admin/features` and `PUT /admin/features/{name}` with body `{enabled: bool}` (`api.py:4860-4884`). The PUT needs the admin role plus the `_gate("security_change")` approval hook.
- **Read path.** Runtime code reads the ordinary settings attribute: `get_settings().graphrag_enabled` etc. `feature_flags.effective(name)` (`:78`) returns `bool(attr)` for bool settings and `str(val) != off_val` for string-backend settings.
- **Per-KB gating** is a separate mechanism: `tenant_config` keys in `_THRESHOLD_KEYS`, stored 1.0/0.0 and ANDed with the master setting. Example: `notation_grounding_enabled`, `multivector_colbert_enabled` (comments at `pipeline.py:737-775`).
- **Pattern for new features:** a master `Settings.<feature>_enabled: bool = False`, default off and "byte-identical when off" (CLAUDE.md "flag-gated, default OFF/safe"). Examples: `query_planner_enabled` (`config.py:1046`), `continuous_eval_enabled` (`config.py:~944`), `contextual_retrieval_enabled`.
- There is no existing tri-state or "shadow" flag. `grep shadow` in `config.py` returns nothing.

**How `RAG_V2_INGEST` / `RAG_V2_QUERY` / `RAG_V2_VERIFY` should be implemented to match conventions.**
1. In `Settings`, add:
   - `v2_ingest: bool = False`
   - `v2_verify: bool = False`
   - `rag_v2_query: Literal["off","shadow","on"] = "off"`, or a `StrEnum` like `AppEnum` at `config.py:22`. `kms_backend` already validates a string with `Field(pattern=...)` (`config.py:~103`), which is the same pattern.
   - Env vars become `RAG_V2_INGEST`, `RAG_V2_QUERY`, `RAG_V2_VERIFY`. Add the plan's alias via `validation_alias=AliasChoices("RAG_V2_QUERY","RAG_V2_QUERY")` if the unprefixed spelling is required.
2. Document them in `.env.example`. Keep call-time reads: `get_settings().rag_v2_query`. Branch in `pipeline.answer_query` (`pipeline.py:4913`, the public wrapper) and in `ingest_document` (`pipeline.py:~440`).
3. Runtime toggling for the booleans:
   - `v2_ingest` and `v2_verify` fit the existing tuple unchanged: `"v2_verify": ("RAG_V2_VERIFY","true","false","RAG v2 verification")`. `effective()` and `set_flag()` then work as is.
   - Tri-state limits:
     - `effective()` returns `str(val) != off_val`, so "shadow" would read as ON.
     - `FlagStore` and `platform_flags.enabled` are BOOLEAN.
     - `PUT /admin/features/{name}` accepts only `{enabled: bool}`.
   - Options for the tri-state:
     - Env-only (operator-managed, like `LOCKED`). Lowest risk, matches "ask before flipping".
     - Extend the flag mechanism with a string-valued mode: add `TOGGLEABLE` entries and a `platform_flag_values(name TEXT PK, value TEXT)` table, or change the column to TEXT.
     - A two-bool encoding (`rag_v2_query_enabled`, `rag_v2_query_shadow`) that fits the unchanged boolean store.
   - Recommendation: the env-only enum, with `shadow` meaning run v1 and v2, serve v1, and log both. Put `rag_v2_query` in `LOCKED` at first, then promote it after owner approval.
4. Per-KB canary (optional): add `rag_v2_query` to `_THRESHOLD_KEYS` as 0.0 / 0.5 (shadow) / 1.0, or store it as a plain tenant_config string. `kb_threshold_overrides` floats only, so a string needs a new reader. The v2 index `_v2` collection suffix is a new per-KB binding and does not exist yet.
5. Sub-settings for v2 (thresholds, k values, models, semaphores) must be added as ordinary `Settings` fields per §1. Prompts need a new `prompts/` directory plus loader, since none exists.

### 3. Existing eval harness (entry points)

There are three distinct families. All except the pytest gate drive the running stack over HTTP (`/v1/dev/token`, then the API).

| Entry point | Invoke | What it measures |
|---|---|---|
| **Golden-set release gate** (pytest) `tests/e2e/test_golden_eval.py` (`pytestmark = eval`, `:20`) | `make eval` (`Makefile:54`) = `pytest -m eval` | Ingests `sample-data/platform_brief.md`, runs `sample-data/golden/golden.json` (4 items) through `pipeline.answer_query` (top_k=3). Asserts citation_coverage == 1.0, abstain_correctness == 1.0, `report.passes()` (`:76-79`). Skips unless a real cross-encoder is configured (`:38-45`). Scoring function: `svc_guardrails_eval.evaluate` (`services/guardrails-eval/src/svc_guardrails_eval/evaluation.py:81`). |
| **Repeat-run live gate** `scripts/eval_gate.py` | `scripts/eval_gate.py --kb acme__research --repeat 3 [--top-k N] [--out f.json] [--compare baseline.json]` | Fetches the KB's saved golden set via `GET /v1/kb/{kb}/golden-set` (`api.py:1321`). Runs each question N times through `POST /v1/kb/{kb}/query`. Pass = grounded AND expected substring (NFKC, casefolded, `:_judge`); abstain-expected = `status == abstained`. Reports score, answerable_score, refusal_score, flaky, failing, errors. `--compare` returns exit 1 if the score delta is negative (`:_compare`). Baseline: `eval/baseline_sop.json`. |
| **Corpus retrieval golden** `scripts/run_corpus_golden.py` | `uv run python scripts/run_corpus_golden.py [--out before.json] [--top-k 10]` | Retrieval only (`/v1/kb/{kb}/retrieve`), over `sample-data/golden/corpus_golden.json`: section hit, quote hit, recall@k, MRR, would-abstain. Compare against `measured_{before,after}_reindex.json`. |
| **RAGAS governance gate** `scripts/ragas_gate.py` | `make ragas ARGS="--kbs a,b"` or `uv run python scripts/ragas_gate.py --base-url … --kbs … --report r.md --prometheus out.prom [--strict]` | Per KB, calls `POST /v1/kb/{id}/eval`. Scores with `ragas_metrics` on the tenant's own judge LLM and BGE-M3. Applies the contract `RagasFloors` (`ragas_governance.py:46`). Exit non-zero on FAIL (INCONCLUSIVE too with `--strict`). Writes `ragas_report.{json,md}`. Service core: `ragas_eval.run_ragas_eval_for_kb` (`ragas_eval.py:~83`), which records an `EVAL_RUN` audit event. Runbook: `docs/qa/RAGAS_GATE.md`. Grafana: `infra/grafana/dashboards/ragas-governance.json`. |
| **Continuous eval** `continuous_eval.py` | Scheduler loop, `RAG_CONTINUOUS_EVAL_ENABLED` (default False); `POST /kb/{id}/eval/run-now` | Golden plus 20% sampled live queries. grounded_rate, faithfulness_avg, hit_rate. Breach floors 0.6 / 0.5 (`breach_reasons`, `:82`). Writes `EVAL_RUN` / `EVAL_ALERT` audit events. |
| **Calibration sweep** `scripts/calibrate.py` (also `calibrate_abstain.py`, `calibrate_faithfulness.py`) | `make calibrate KB=… ARGS=…` = `scripts/calibrate.py --kb K --golden sample-data/golden/golden.json [--judge-url …] [--report f.md]` | Sweeps the abstain threshold and the faithful/flag thresholds over labelled answerable / unanswerable / unfaithful rows (template `golden-template.json`, `docs/golden-template.md`). Math in `rag_core.calibration`. Writes NOTHING to config; it produces evidence for a human to set values. |
| Latency benchmark `scripts/benchmark.py` | `make bench ARGS="scaling --kb … --peers 10,50"`, or `compare` | Federated scatter-gather p50/p95/p99 and vector-backend A/B. Stats in `rag_core/benchmark.py:41` (`summarize`). |
| Business value `business_value.py` (wall-clock per query, p50/p95/p99 over `pipeline.answer_query`, `:225-230`) | API only | Latency and cost, whole-query only. |
| Probes (no scoring) | `answer_probe.py` (prints retrieved vs cited spans per question); `shape_battery.py --kb K` (catalog-synonym, multi-point completeness, unanswerable-abstain; exit code usable in CI); `extraction_census.py` (per-doc extraction provenance, read-only); `load_soak.py`; `resolve_golden_quote.py` | Behaviour checks. Not quality metrics. |

Differences, plainly:
- **Golden release gate** (pytest and `eval_gate.py`) = binary behaviour: right fact, citation present, abstain correct. Deterministic assertions, no LLM judge. Zero tolerance in pytest.
- **RAGAS gate** = LLM-judge scores on four floors (§5), tied to the audit chain and Prometheus.
- **Calibration** = does not gate. It picks threshold values.
- Known caveats from `eval/README.md`: turn the semantic cache OFF (`RAG_SEMANTIC_CACHE_ENABLED=false`); `--repeat 1` is not evidence.
- `eval/FINDINGS.md` records adopted and rejected tunings: `top_k` 10 -> 16 adopted (91.7% -> 95.8%); `chunk_max_chars` 1400 rejected (-8.3 points).
- There is **no** `eval run --suite rag_v2` CLI. There is no `eval` console script and no suite concept. The `eval` Make target is just `pytest -m eval`.

### 4. Existing gold-set format

**`eval/golden_sop.json`** (the same shape is what `PUT /v1/kb/{kb}/golden-set` stores in `tenant_config["golden_set"]`).
- Schema: `{"questions": [ {"question": str, "expected": str | omitted } ]}`.
  - Pydantic model `EvalQuestion` (`services/control-plane/src/svc_control_plane/quality.py:107`): `question: str (min_length=1)`, `expected: str | None` ("a fact the answer should contain, case-insensitive").
  - API wrapper `GoldenSet.questions` has `max_length=50` (`api.py:1317`). `MAX_EVAL_QUESTIONS = 20` bounds a RAGAS run (`quality.py:26`).
- Semantics: with `expected`, the answer must be GROUNDED and contain the string. With no `expected`, the question is unanswerable and the platform must ABSTAIN (`eval_gate.py:_judge`).
- Size: **24 questions** (20 with `expected`, 4 without, i.e. refusals). Covers one document: "SOP Writing Guidance Pharma", on KB `acme__research` (per `baseline_sop.json: kb`).
- Verbatim entries (`eval/golden_sop.json:3-10`):
```json
{
  "question": "According to the SOP Writing Guidance Pharma document, who prepares and executes an SOP?",
  "expected": "Executive"
},
{
  "question": "According to the SOP Writing Guidance Pharma document, who approves an SOP?",
  "expected": "Asst. Manager"
}
```
- Baseline report `eval/baseline_sop.json` keys: `kb, repeat, top_k, questions, score, answerable_score, refusal_score, errors, flaky, failing, rows[{question, expected, pass_rate, statuses, faithfulness, errors}]`. Values: score 0.9583, answerable 0.95, refusal 1.0, one failing question.

**Two further gold formats exist.**
- `sample-data/golden/golden.json` (4 items, CI gate): `{"$comment", "doc_id", "items":[{question, expected_answer, expected_doc_ids[], expect_abstain}]}`. Loaded as `GoldenItem` (`evaluation.py:25`), which also has optional `expected_chunk_ids`.
- `sample-data/golden/corpus_golden.json` (**45 items**, 3 KBs, 38 answerable and 7 abstain). Each item has `kb_id, question, expected_answer, expect_abstain, expected_doc_ids, expected_chunk_ids, expected_quote, expected_section, tag, section_detected_at_build`. KBs: `acme__acme-content-governance-sop` 17, `acme__hr-policies` 16, `acme__rfp-analyser` 12. It was rebuilt with `resolve_golden_quote.py`; quote and section are the durable keys because chunk ids are positional.
- Calibration rows (`golden-template.json`): `{kind: answerable|unanswerable|unfaithful, question, expected_answer?, answer_to_judge?}`.

**Mapping to the plan's `eval/gold/<doc>.yaml`** (`intent`, `expected_answer` for lookup, `checklist[{id, facet, must_contain[], tier: must|should}]` for aggregate, `supersessions[{field, old, new}]`):

| Plan field | Today | Change needed |
|---|---|---|
| `query` | `question` | rename, or accept both |
| `intent` | missing | new. The router already emits an intent (`query_planner`, `intent_classifier`), but nothing in the gold stores one. |
| `expected_answer` (lookup) | `expected` (a substring) in golden_sop.json; `expected_answer` (prose) in `golden.json` and `corpus_golden.json` | the substring check is the "must_contain" analogue. Keep `expected` as the lookup check. |
| `checklist[].must_contain[] / tier / facet / id` | missing | new. There is no multi-item or recall-style scoring anywhere. |
| `supersessions[]` | missing | new |
| abstain | omit `expected` (golden_sop) or `expect_abstain` (the others) | formalise as `intent: unanswerable` or an `expect_abstain` key |
| provenance | `expected_quote`, `expected_section`, `expected_doc_ids` (corpus_golden only) | reuse for the checklist `evidence` |

- **Recommendation: add YAML alongside; do not break the JSON.** The existing JSON is read by three consumers: the API `GoldenSet` model (`extra` ignored, and `max_length=50` blocks aggregate checklists), `eval_gate.py`, and `continuous_eval` / `ragas_eval._load_golden`.
- The new loader should be a `rag_v2` suite that reads `eval/gold/*.yaml`. It should also adapt `golden_sop.json` and `corpus_golden.json` into lookup-intent items (`question` -> `query`, `expected` -> `must_contain`), so existing gold carries over as the lookup regression set.
- Note that PyYAML availability is not verified here. No YAML gold exists today; `eval/gold/` does not exist.

### 5. Metrics

**Implemented today.**
- Golden gate (`evaluation.py:40-57`): `citation_coverage` (cited / answered, over non-abstained answers), `abstain_correctness`, `context_hit_rate` (cited doc ∩ `expected_doc_ids`), `recall_at_k` and `mrr` (over items with `expected_chunk_ids`, `_retrieval_metrics`, `:129`). `passes()` defaults: citation >= 1.0, abstain >= 1.0, context >= 1.0.
- `eval_gate.py`: `score` (mean pass_rate over all questions × repeats), `answerable_score`, `refusal_score`, `flaky`, `failing`, and mean `faithfulness_score` per row. The only gate is `delta < 0` vs the baseline file.
- RAGAS (`ragas_metrics.py`):
  - `answer_relevancy` (`:62`, LLM-generated questions and embedding cosine, n=3)
  - `context_precision` (`:85`, judge relevance weighted by rank)
  - `context_recall` (`:127`, judge on ground-truth claims vs contexts)
  - `faithfulness` is passed in from the pipeline's own checker (`faithfulness_score`), not computed by ragas_metrics (`ragas_eval.py:~108` passes `None`).
- Floors (`RagasFloors`, `ragas_governance.py:46-62`): faithfulness >= 0.80, answer_relevancy >= 0.70, context_precision >= 0.70, context_recall >= 0.70, `min_sample` 8 (below it the verdict is INCONCLUSIVE), `max_flag_rate` optional. Zero grounded-but-uncited answers is required (`governance_statement`, `:223-232`). Env overrides `RAG_RAGAS_*` (`from_env`, `:65`).
- Continuous eval floors: grounded_rate 0.6, faithfulness 0.5 (`config.py:951-952`).
- Pipeline-level per-answer faithfulness (E4): lexical / llm / nli backends, three-way routing at 0.7 / 0.4 (`config.py:452-453`). NLI model is `cross-encoder/nli-deberta-v3-base` (`config.py:488`), NOT the plan's `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Numeric and notation verbatim checks exist (`notation_grounding`, Slice 4).

**Versus plan §0 success criteria.**

| Plan metric | Measured today? | Where / what name |
|---|---|---|
| Aggregate item recall (checklist, must / should) | **No** | No checklist gold and no item-recall scorer. Closest: `context_recall` (`ragas_metrics.py:127`), an LLM-judged single ground-truth statement; `recall_at_k` (`evaluation.py`) is chunk-level retrieval recall, not answer-item recall. |
| Value accuracy (numbers, units, codes) | **Partial, runtime only** | Not an eval metric. The platform has a runtime numeric check and notation grounding (flags unit mismatches, chemical notation), `config.py:~458` / `wiring.apply_notation_grounding`. `eval_gate` substring match on `expected` is the nearest offline proxy. |
| Claim faithfulness (NLI-entailed claims / total) | **Partial** | Per-answer `faithfulness_score` (nli / llm / lexical) and the RAGAS gate floor >= 0.80. Averaged over answers, not a per-claim entailed / total ratio. Different NLI model than the plan's. |
| Supersession correctness | **No** | Nothing in gold, metrics or pipeline for amended-value / old->new. |
| Lookup accuracy (no regression vs baseline) | **Yes** | `eval_gate.py` `score` / `answerable_score` + `--compare` (regression exit 1); `baseline_sop.json` is the reference. Also `run_corpus_golden` (section / quote / recall@k / MRR). |
| p95 latency, lookup | **Partial** | `benchmark.summarize` p50/p95/p99 (`rag_core/benchmark.py:41`); `business_value.run_workload` times `answer_query` end to end (`:225-230`). No intent split (lookup vs aggregate) and no gate target. |
| p95 latency, aggregate | **No** | No aggregate intent and no gate. Only whole-query timing via the same tools. |
| Router intent accuracy (Phase 11) | **No** | No intent labels in gold, no scorer. |
| Section coverage (Phase 11) | **No** | `coverage_*` settings exist for answer completeness (`config.py:486-510`), but no eval metric. |
| Abstain correctness (existing extra) | Yes | `abstain_correctness`, `refusal_score`. Reuse. |

### 6. Test layout, markers, CI

- `pyproject.toml:155-164` `[tool.pytest.ini_options]`: `asyncio_mode="auto"`, `testpaths=["tests","libs","services","apps","sdks"]`, `addopts="-ra --strict-markers"`. Markers (registered at `:159-163`):
  - `integration`: requires the docker-compose stores
  - `e2e`: full walking-skeleton slice
  - `eval`: golden-set release gate (needs the real cross-encoder)
- mypy: `strict=true`, `disallow_untyped_defs`, pydantic plugin, `mypy_path` for the four libs (`pyproject.toml:127-146`). Ruff lint plus format.
- Layout (test files): `tests/e2e` 90 (in-memory profile by default via `conftest.py` `inmemory` fixture, most carry `pytestmark = pytest.mark.e2e`); `tests/integration` 5 (`pytestmark = integration`); per-package unit tests under `libs/*/tests`, `services/*/tests` (control-plane 73, doc-intelligence 25, generation 4, guardrails-eval 1, indexing 1, ingestion 1), `libs/rag-core` 45, `libs/clients` 28, `libs/common` 15, `apps/mcp` 5.
- `RAG_TEST_REAL_STORES=true` makes the same test bodies run against the real stack (`conftest.py:1-14`).
- Makefile:
  - `make test` -> `pytest -m "not integration and not e2e and not eval"` (`:47`)
  - `make test-integration` -> `pytest -m integration` (`:51`)
  - `make eval` -> `pytest -m eval` (`:55`)
  - `make calibrate` -> `scripts/calibrate.py --kb $(KB) $(ARGS)` (`:59`)
  - `make ragas` -> `scripts/ragas_gate.py $(ARGS)` (`:63`)
  - `make ci` = `lint typecheck test` (`:98`)
  - `make lint` / `fmt` / `typecheck` (`mypy libs services apps sdks/python`)
- `.github/workflows/ci.yml` has three jobs.
  - `quality` (every push to main and every PR): `uv sync`, ruff check, ruff format --check, `mypy libs services apps`, `pytest -m "not integration and not e2e and not eval" --cov=libs --cov=services`.
  - `docker-build`: builds `apps/api/Dockerfile`.
  - `eval-gate` (needs `quality`): `make up`, `make migrate`, `pytest -m "integration or eval"`, then asserts the eval gate actually ran (`pytest -m eval -rs | grep passed`; a skip fails the job). The file's own comment says it can be marked required once dev stores are wired in CI.
- The e2e tests marked `e2e` are excluded from `make test` and CI `quality` (only in-memory-able ones run elsewhere). `eval_gate.py`, `ragas_gate.py`, `run_corpus_golden.py` and `calibrate.py` are NOT in CI. `ragas_gate.py` is a manual or QA command.
- Repo `CLAUDE.md` mentions the markers (`:101`), G9 continuous eval (`:274-287`) and the flag-gating convention ("each additive, flag-gated where it changes behaviour, default-safe", `:121`). It contains no rule for eval-set editing.

### 7. Latency instrumentation

- **Query path: no per-stage timing is recorded.**
  - `pipeline.answer_query` (`pipeline.py:4913`) and its inner path write `AuditAction.QUERY` events (about 10 sites, e.g. `:2117, :3073, :3622`). Their payloads carry `query`, `top_k`, `cache_hit` and similar, with no duration field. `grep` for `latency|perf_counter|duration_ms` in `pipeline.py` finds only the ingest `stage_ms` (`:453-475`) and graph timing (`:508-509`).
  - What exists is a per-answer `agent_trace: list[AgentStep]` (`libs/schemas/src/rag_schemas/citation.py:35, 145`; `pipeline.py:~3640`), which carries steps and details, not timings. It is explicitly non-authoritative.
  - OTel spans on the query path: only `federation.peer` and `federation.scatter_gather` (`pipeline.py:4599, 4625`). FastAPI instrumentation exists (`rag_common/tracing.py:51`, `instrument_app`) and gives whole-request spans.
  - Logging is structlog JSON with a request_id bound by middleware (`rag_common/middleware.py:27-33`); no timing fields are logged per stage.
- **Ingest path: per-stage timing exists.** `INDEX` audit event payload `stage_ms` (dict of stage -> completion offset ms) and `graph_ms` / `graph_at_ms`, built from `ingest_progress.completed_stages` (`pipeline.py:449-475, 508-509`) and consumed by `ingest_dag.py:293-315`.
- Whole-query latency: external only (`scripts/benchmark.py`; `business_value.run_workload`, `:225-230`, wall clock around `answer_query`).
- Other metrics: OTel meter gauges and counters exist only in `health_poller.py:56-63`. The Prometheus textfile from `ragas_gate --prometheus` carries the RAGAS gate numbers (`ragas_governance.prometheus_text`, `:237`).
- **There is no query-log table.** The tamper-evident audit chain (`AuditAction.QUERY`) is the closest. `continuous_eval._sample_live_queries` (`:105`) reads the last ~300 audit events to sample real queries, so the audit chain is the de facto query log, without timings.

### 8. Fixture corpora available

Repo-resident documents usable offline:
- `sample-data/platform_brief.md` (948 B, markdown): the only ingested fixture in a test (`test_golden_eval.py`), paired with `sample-data/golden/golden.json` (4 items).
- `sample-data/calibration/hr_policies_{answerable,unanswerable}.txt`: question lists (not documents) for the `acme__hr_policies` corpus.
- `demo/*.md`, `docs/*.md` and others: usable as markdown text, not purpose-built fixtures.
- **No PDF, DOCX or table-heavy fixture is committed.** The only PDFs in the repo are demo and product collateral (`demo/*.pdf`, `docs/product/*.pdf`), none used by tests.

Corpora referenced by gold but living only in the live stack's KBs (not in the repo):
- Acme Content Governance SOP, PDF (`Acme-MKT-CONT-SOP-001-V1.0-ContentGovernance.pdf`), 17 items. Document type: SOP / policy.
- Acme HR policy set (HR Policy Handbook, Mediclaim Handbook, Night Shift Allowance, Business Travel, Code of Conduct, Torch Bearers, Child Education Reimbursement, Customer Testimonial Reward), PDFs, 16 items. Document type: HR handbook / policy.
- CAG "Sovereign AI Platform" EOI tender, PDF, 12 items in `acme__rfp-analyser` (`corpus_golden.json`). Document type: tender.
- SOP Writing Guidance Pharma, KB `acme__research`, 24 questions (`eval/golden_sop.json`). Document type: SOP.
- The plan wants AEGCL corrigendum (Appendix A; not in repo), contract (not present anywhere), technical paper (Mistral-7B, not in repo; the plan says it "can be reused" but no copy was found). Document types available as gold today: SOP, HR policy, tender / EOI. Missing: contract, technical paper, corrigendum / amendment documents.

### Gaps vs plan (Phases 11-12 + hard constraints)

- **ALREADY EXISTS — reuse:** `Settings` / `get_settings()` env-driven config with validated bounds and per-KB `tenant_config` overrides (`config.py:38,1278`; `pipeline.py:737,1835`). New v2 settings go here.
- **ALREADY EXISTS — reuse:** global flag admin mechanism (`feature_flags.py`, `/admin/features`, `platform_flags` table) for the boolean `RAG_V2_INGEST` and `RAG_V2_VERIFY`.
- **ALREADY EXISTS — reuse:** `eval_gate.py` repeat-run harness plus `--compare` regression exit code; `baseline_sop.json` as the lookup no-regression baseline; `golden_sop.json` and `corpus_golden.json` (69 lookup / abstain items) as lookup gold.
- **ALREADY EXISTS — reuse:** `svc_guardrails_eval.evaluate` (citation, abstain, doc-hit, recall@k, MRR), RAGAS metrics and `RagasFloors` (env-overridable `RAG_RAGAS_*`), `rag_core.benchmark.summarize` (p50/p95/p99), `continuous_eval`, Prometheus textfile export, `EVAL_RUN` audit events.
- **ALREADY EXISTS — reuse:** the repo's "flag-gated, default OFF, byte-identical when off" idiom (`query_planner_enabled`, `contextual_retrieval_enabled`, etc.), and the ingest `stage_ms` pattern for stage timing.
- **GAP:** no tri-state / shadow flag. `effective()` treats any non-off value as ON; the store is BOOLEAN; the PUT body is a bool. `RAG_V2_QUERY=shadow` needs an env-only enum (simplest) or a flag-store extension (§2). There is also no shadow runner that runs v1 and v2, serves v1 and logs both.
- **GAP:** env prefix. The plan's `RAG_V2_*` resolves to `RAG_V2_*` unless an `AliasChoices` is added.
- **GAP:** gold format. No `eval/gold/` directory, no YAML, no `intent`, `checklist`, `must|should` tiers, `supersessions`, or facets. The API `GoldenSet` model caps at 50 questions and the `EvalQuestion` model lacks these fields. Add YAML alongside; do not break JSON (§4).
- **GAP:** metrics missing entirely: aggregate item recall (must / should), value accuracy as an eval metric, supersession correctness, router intent accuracy, section coverage, per-claim entailed ratio, latency split by intent. The 0.85 / 0.95 / 0.90 / 100% targets have no scorer to compare against.
- **GAP:** `eval run --suite rag_v2` does not exist. `make eval` is `pytest -m eval` over a 4-item doc. A new `eval` CLI (or `scripts/eval_run.py`) with a suite registry and a Make target is needed. Include a `rag_v2` pytest marker (and register it; `--strict-markers` is on) if tests are marked.
- **GAP:** CI runs none of the live gates except `eval-gate`'s `pytest -m "integration or eval"`; the repeat-run and RAGAS gates are manual. The `eval-gate` job is non-required and needs the real cross-encoder. A rag_v2 regression gate in CI needs the stack plus gold fixtures, and the pytest gate skips on the lexical reranker.
- **GAP:** per-stage latency (router / retrieval / map / reduce / verify) is not recorded for queries: no timings in QUERY audit payloads, no query log table, no stage spans, no metrics histograms. Phase 12 observability would be new work; reuse the `stage_ms` approach and structlog request_id.
- **GAP:** prompts and some thresholds are inline constants (`llm_gateway.py:59`, `ocr.py:796`, `multimodal.py:49,100,104`, ...). The plan's "prompts in prompt files" requires a new prompt directory and loader; there is no existing convention.
- **GAP:** NLI model differs. Config default is `cross-encoder/nli-deberta-v3-base` (`config.py:488`), not the mDeBERTa XNLI model named in the plan. No 512-token windowing evidence found in the config (only `nli_premise_max_chars=1000`).
- **GAP:** fixtures. Only the 948-byte `platform_brief.md` is committed. No AEGCL corrigendum, contract, or technical paper; HR / SOP / tender PDFs are only on the live stack. The plan's ≥4 document types need owner-supplied fixtures (plan's own [ASK]) and a committed `eval/fixtures/` path.
- **GAP:** v2 index suffix (`_v2`) and "no destructive re-index" have no existing config hook. The collection naming is not parameterised in config; the Qdrant collection settings are at `config.py:~140-180`. Not verified in this slice.
