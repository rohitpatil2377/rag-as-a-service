# ADR-0016 — Per-KB model roles: embedder & reranker attach (FR-E1 extended)

**Status:** accepted · 2026-08-03
**Context:** Acme deployed GPU-backed models on AIOps (`models.example.internal`, auth via a
raw `apikey:` header): Qwen 32B (LLM), bge-m3 (embedder), bge-reranker, with Sarvam 30B
coming. Product requirement: nothing hard-coded — every model attaches from the console.
The E1 registry covered LLMs only; embedder/reranker had no per-KB seam at all.

## Decisions

1. **`role` on the model-connection registry** (`generation | embedding | reranking`),
   per-ROLE defaults (first-of-role becomes default; promotion on delete stays within the
   role). The generation resolver filters by role, so an embedder row can never be picked
   as a chat model. Existing rows migrate as `generation` (`ADD COLUMN IF NOT EXISTS`).
   LLM attach (Qwen/Sarvam) needs zero further code — it was already complete.

2. **Probe-and-store, not runtime negotiation.** The Test button asks the RUNNING endpoint
   what it is: embeddings → the true dimension (`embed_dim`); rerank → which wire shape it
   speaks (`tei` or `cohere`, `+sigmoid` when it returns raw logits) plus a sample score.
   What the probe learns is stored on the connection and is REQUIRED to save (422
   otherwise); the runtime adapters (`OpenAICompatEmbedder`, `OpenAICompatReranker`) are
   then fully deterministic and fail loudly on drift — a changed dimension would poison
   the collection, a changed score scale would silently re-meaning the abstain threshold.

3. **Reranker attach is immediate; embedder attach is staged.** A reranker is stateless
   w.r.t. the index. An embedder is not: every stored vector lives in the old model's
   space. Invariant: *queries and ingests read `embedder_active` (tenant config); the
   connection store only ever feeds `embedder_pending`; only the explicit admin
   `POST /kb/{id}/embedder/apply` moves one to the other.* Empty KBs activate immediately.
   Deleting the active embedder while documents depend on it is a 409.

4. **Apply = recreate-in-place, not shadow-collection.** A dimension change forces a new
   Qdrant collection; an alias-flip swap would touch `tenancy.vector_collection` (the
   tenant-isolation source of truth) for a rare operator event. During the rebuild dense
   search returns empty for not-yet-re-embedded docs (BM25+rerank keeps serving), and each
   document goes through the write-then-swap reindex. Shadow-collection swap is the
   documented follow-up for large corpora.

5. **Default path is identity-preserving.** With nothing attached,
   `retrieval_service_for`/`indexing_service_for` return the very lru-cached platform
   singletons (asserted by identity in tests). Per-tenant retrievers share the SAME
   stores; only the two model seams differ. The semantic cache probes with the KB's
   active embedder and skips entries of a different vector width.

6. **TLS:** `RAG_MODEL_CA_BUNDLE` (PEM path) for private-CA model hosts, applied to the
   three model-endpoint clients only. No configuration yields `verify=False`.

## Consequences
- bge-reranker's (sigmoided) score distribution ≠ the platform cross-encoder's —
  re-calibrate the KB's abstain threshold after attaching (per-KB overrides exist).
- The cutover of existing KBs onto bge-m3 is an operator action (attach → apply) done
  with a golden-set before/after, not part of this change.
- Adding a third rerank wire shape is a contained adapter change (`_WIRES` + probe).

## Amendment (2026-09-15) — an ORG-WIDE model of any role is used, not just generation

Models became org-wide managed (one registry per org, KB managers request, org admin
approves). Resolution, however, only ever consulted that registry for **generation**:
an embedding or reranking entry added from the console was stored, listed, and then
never used by anything — inert configuration that reads as if it were in force. Live,
an operator added their `bge-m3` endpoint org-wide and the platform went on embedding
with the deployment env.

Resolution is now, per role:

| Role | Order |
|---|---|
| Generation | KB connection → org-wide default → platform env |
| Reranking | KB connection → **org-wide default** → platform env |
| Embedding | `embedder_active` (staged) → **org-wide default, IF COMPATIBLE** → platform env |

Reranking adopts immediately for the reason point 3 gives: nothing stored depends on it.

**Embedding keeps the asymmetry, narrowed rather than removed.** Adopting a *different*
embedder for a KB whose vectors were built elsewhere would not switch the embedder, it
would orphan every vector in the collection — so the org default is adopted only when it
AGREES with the embedder that built the corpus (same model id and dimension as the
platform embedder). That is the same vector space by construction, so using the org
connection changes only which endpoint and credential serve it. A mismatched one is
refused and logged (`org_embedder_not_adopted`) with the reason; changing the vector
space stays the staged per-KB attach + apply, which re-embeds.

`GET /system/model-health?kb_id=` probes the embedder and reranker that the KB actually
resolves (`_effective_infra_health`), and `BackendHealth.model` now carries the model the
probe reached — the console's roster reads CONFIG, and after this change a row's
configured model and its resolved one can differ. The same drift already produced a table
that graded one model while naming another (see `_effective_generation_health`).
