# ADR-0007 — Contextual Retrieval (FR-C2)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
Chunk-level retrieval loses document context: a chunk like "the target is 99.5%" is
ambiguous out of its document. Anthropic's "Contextual Retrieval" prepends a short
document-level context to each chunk before embedding so chunks are self-contained,
measurably lifting precision/recall — the KPI this Phase-1 slice targets.

## Decision
- Prepend a document-level **context** to each chunk **before embedding + BM25 indexing**,
  behind a `Contextualizer` Protocol (rag-core). Two backends:
  - **`MetadataContextualizer`** (default) — deterministic, no-LLM (doc title/id + page +
    tags). Cheap, sovereign, fast on CPU.
  - **`LLMContextualizer`** (opt-in) — the canonical LLM-writes-the-context technique via
    the sovereign gateway. One LLM call per chunk → slow on the 3B CPU model; reserved for
    vLLM/GPU.
- **Embed/index the contextualized text ONLY; store the ORIGINAL chunk** in the catalog.
  Citations and answers always show the original — the context prefix never leaks. No
  `Chunk` schema change (the contextualized string is ephemeral at index time).
- **Config-gated:** `contextual_retrieval_enabled` (default **OFF**, additive/reversible) +
  `contextual_retrieval_backend` (`metadata` | `llm`). Retrieval/grounding/faithfulness
  logic is **unchanged**; faithfulness thresholds are **not** tuned to compensate for any
  score shift.

## Re-indexing (operational note)
Turning the flag on is **new-ingest-only**: future ingestions get context; **existing
indexed tenants keep their current embeddings until re-ingested**. There is no silent
auto-migration — a bulk re-index job (re-run ingestion for a tenant's documents) is a
separate, explicit operation. Documented so operators aren't surprised by mixed indexes.

## Deferred (door kept open)
**Persisting `Chunk.context`** — i.e. storing *what was actually embedded* — is a deferred
option, NOT built now. Because RAG as a Service is an **audit platform**, if audit-traceability
of the embedding *input* is ever required, add an optional `Chunk.context` field and record
it alongside the original text. Deliberately left open; revisit when/if that requirement is
real.

## Consequences
- New `Contextualizer` seam + factory (`get_contextualizer`); indexing applies it behind the
  flag. BM25 index text may differ from the cited chunk text (by design — citation always
  hydrates the original from the catalog).
