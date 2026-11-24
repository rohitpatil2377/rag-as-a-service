# ADR-0015 — Retrieval score semantics (FR-D2/E2)

**Status:** Accepted (Phase-1) · 2026-07-30

## Context

The grounded-or-abstain gate compares one number, `RetrievalResult.max_score`, against one
threshold, `RAG_ABSTAIN_THRESHOLD`. A threshold is only meaningful if that number means the
same thing in every query, on every deployment. It did not.

Five separate mechanisms produced or transformed the gate's input, and only one of them was
measuring relevance:

- **Rank fusion sat in `score`.** `HybridRetriever` hydrated candidates with the min-max
  normalised RRF value and the reranker later *moved* it to `fusion_score`. Because
  `normalize_scores` maps the best candidate of every result set to exactly 1.0 — and every
  candidate to 1.0 when the set is uniform — any path that skipped the cross-encoder handed the
  gate a confidence of 1.0 for a result set nothing had judged. `LexicalReranker`'s
  all-stopword short-circuit was such a path, and `RAG_RERANK_BACKEND=none` was an accepted
  config value that `build_reranker` silently ignored.
- **GraphRAG fabricated `score=0.5`.** Entity extraction is a proper-noun regex and the graph
  walk is a pure structural lookup, so the constant carried no relevance signal at all — yet it
  sat above the 0.35 default threshold. With `RAG_GRAPHRAG_ENABLED=true` (the dev compose
  default), any question naming one indexed entity could not abstain.
- **`max_score` was read off position 0** in four places. After `apply_recency` re-sorts, that
  is the freshness winner rather than the relevance winner, so a freshness weight could only
  ever tighten the gate: measured, a perfect 1.0 match on an older document reported 0.333.
- **The score was unbounded** on `RetrievedSpan` while bounded on `Citation`, and the clamp ran
  *after* the gate had already compared the raw value.
- **Nothing knew which model produced the score.** `rerank_model` was declared in config and
  never read; config said `bge-reranker-base`, compose ran `ms-marco-MiniLM-L-6-v2`.

Every quality number the platform reported was computed against some mixture of these.

## Decision

**One field, one meaning, one writer — enforced by the type rather than by convention.**

1. **`score` is absolute cross-encoder relevance in [0,1], comparable across queries.** It is
   the only gate input. `fusion_score` is the set-relative RRF value, diagnostic only, and
   lives in its own field permanently — the two never share one.
2. **Only a reranker writes `score`**, and it writes `scorer`
   (`cross_encoder | lexical | unscored`) in the same operation. Candidates are hydrated
   *unscored*. A span that never reached a reranker therefore arrives at the gate at 0.0 and
   abstains: **skipping the cross-encoder fails safe and stays queryable**, instead of passing
   a rank off as relevance.
3. **Graph-contributed candidates are scored by the same reranker**, not assigned a constant.
   They are fetched wide (`RAG_GRAPH_RETRIEVAL_CANDIDATES`) and the cross-encoder selects, so
   the choice is relevance-ordered rather than the lexicographic sample `linked_chunks`
   returns.
4. **`max_score` is `max(s.score for s in spans)`** everywhere — never `spans[0].score`, never
   a pre-merge maximum carried forward past a truncation.
5. **`RAG_RERANK_MODEL` is load-bearing.** The adapter asks the endpoint what it serves and
   refuses a mismatch, because the threshold is a claim about one model's score distribution.

## Alternatives rejected

- **Blending fusion and relevance (`α·rrf + (1-α)·ce`).** RRF is 1.0 by construction after
  min-max, so a query whose entire pool is irrelevant would still show a strong blended score —
  precisely the failure the gate exists to catch. And α would be a tunable with no principled
  value.
- **A score floor for graph spans (`max(ce_score, 0.4)`).** This re-creates the fabrication
  being removed, one step further down.
- **Excluding graph spans from `max_score` but keeping the constant.** Half a fix: those spans
  still become `Citation`s, so a fabricated 0.5 would still reach the console and `/retrieve`.
- **Reserving top-k slots for graph spans.** A cross-encoder rates a *bridge* passage low by
  construction — carrying no trace of the question is exactly why passage search missed it and
  the graph found it — so honest scoring could have made GraphRAG a no-op for the multi-hop
  queries it exists to serve. No new knob was needed: `merge_retrieved` already interleaves the
  lists round-robin, so multi-hop evidence keeps its place in the prompt while carrying a real
  number.
- **`score: float | None` instead of a `scorer` discriminator.** Genuinely unrepresentable
  illegal state, but it ripples through `Citation`, `transform`, `recency`, `grounding`, the
  `/retrieve` response shape, the console and both SDKs. The discriminator buys the same
  checkability additively.

## Consequences

- **`RAG_ABSTAIN_THRESHOLD=0.35` is not yet calibrated** and remains uncalibrated on purpose
  until measured against the repaired signal. It was originally set against a lexical overlap
  fraction. Calibration is a separate, evidence-first step.
- **Historical data splits at this deploy.** Stored `Citation.score` values and `max_score` on
  past audit rows mix normalized-RRF and cross-encoder scales. Anything trending them over time
  must split here. The audit chain is immutable by design, so this cannot be migrated.
- **Agentic retry behaviour changes.** It compares `max_score` across two retrieval calls; that
  comparison is now meaningful in a way it was not.
- **Graph-augmented queries cost more.** One extra reranker round trip, ~105 ms/candidate on
  the CPU MiniLM tier. `RAG_GRAPH_RETRIEVAL_CANDIDATES` is the knob; the GPU tier removes the
  concern.
- **The in-memory test profile still uses `LexicalReranker`**, whose scores are overlap
  fractions on a different scale. It is retained deliberately — several e2e tests are
  calibrated against it and swapping it would be large churn for no gate-semantics gain — but
  it means the default suite does not exercise cross-encoder gate behaviour. The three unit
  tests in `libs/rag-core/tests/test_hybrid.py` pin the invariant itself instead.
- **Switching to the GPU reranker is now a two-line change** (`RAG_RERANK_URL` *and*
  `RAG_RERANK_MODEL`) and requires re-calibrating the threshold. That is the intended cost:
  it was previously a one-line change that silently moved the gate.
