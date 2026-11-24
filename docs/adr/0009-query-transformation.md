# ADR-0009 — Query transformation (FR-D6)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
A single embedding of a complex, multi-part query under-retrieves: "compare the uptime
and latency targets" matches neither fact well when they live in different chunks.
Transforming the query (decomposition / HyDE / rewrite) and retrieving each part lifts
recall — *if* the results are merged cleanly, otherwise it just adds duplicates, noise,
and latency.

## Decision
- A `QueryTransformer` Protocol (rag-core) with a deterministic **`DecompositionTransformer`**
  (CI default: split multi-question/comparative queries into sub-queries; ALWAYS keep the
  original) and LLM opt-ins **`HyDETransformer`** (retrieve on a hypothetical answer passage)
  and **`RewriteTransformer`**.
- Plugs into the **D5 router's heavy path**: transform → retrieve each sub-query → **merge**.
  If the router is off, the heavy gate is open so transform (when enabled) applies to all.
- **The merge is the quality step** (`merge_retrieved`, pure): union the sub-query results,
  **de-duplicate by `chunk_id` keeping the best score**, sort desc, take `top_k`. Keeping the
  max score ranks a chunk by the strongest evidence any sub-query found — recall up, no
  double-counting, `max_score` still feeds the grounded-or-abstain gate. Sub-queries are
  capped (4) to bound latency.
- **Pure-additive, no schema change.** Gated by `query_transform_enabled` (default OFF). The
  transform is recorded on the QUERY audit payload (`transform`, `subqueries`).

## Consequences
- New `get_query_transformer` factory + `wiring.query_transformer`; pipeline runs the
  transform+merge on the heavy path. Retrieval/grounding/faithfulness logic untouched.
- LLM transformers add one gateway call per query (HyDE ~160 tokens) — slow on the CPU dev
  model; deterministic decomposition is the default so CI and the dev loop stay fast.
