# ADR-0008 — Adaptive query router (FR-D5)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
Not every query needs the same effort: "what is the uptime target?" is cheap, while
"compare the ITR and the assessment order and reconcile them" is multi-hop and benefits
from deeper retrieval (and, later, query transformation). Routing lets us spend effort
where it helps and stay fast otherwise (FR-D5).

## Decision
- A `QueryRouter` Protocol (rag-core) with two backends: **`HeuristicRouter`** (default,
  deterministic, no-LLM — routes on length / multi-question / comparative-multi-hop
  markers) and **`LLMRouter`** (opt-in, one small classification call; reserve for
  vLLM/GPU).
- The router returns a `RouteDecision(path, top_k, reason)`. The **heavy** path currently
  deepens retrieval (boosts `top_k`, capped) and is the **seam D6 query-transformation
  plugs into**; the **standard** path is unchanged hybrid.
- **Pure-additive, no schema change.** Gated by `adaptive_router_enabled` (default **OFF**
  → identical behaviour). The decision is logged and folded into the existing **QUERY
  audit payload** (`route_path` / `route_reason` / `route_top_k`) — no new `AuditAction`,
  no new contract.

## Consequences
- New `get_query_router` factory + `wiring.query_router`; pipeline consults the router
  before retrieval when enabled. Retrieval/grounding/faithfulness logic untouched.
- D6 will branch on `RouteDecision.path == "heavy"` to apply query rewrite/decomposition/
  HyDE before retrieval.
