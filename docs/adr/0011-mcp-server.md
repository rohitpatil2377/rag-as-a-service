# ADR-0011 — Sovereign MCP server (FR-H2)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
The platform should be **agent-consumable** via the Model Context Protocol (FR-H2, TDD
§4.8). Agents shouldn't call raw stores — they should get the same safe path as the API:
guardrails, routing, transformation, grounded-or-abstain, faithfulness, and audit.

## Decision
- A new `apps/mcp` package (`rag_mcp`) using the **official `mcp` SDK** (`FastMCP`) —
  OSS/MIT, runs locally, **sovereign** (no external calls; egress guard unaffected).
- **Two tools wrap the EXISTING pipeline**, so agents inherit every safety control:
  - `rag.retrieve(query, top_k)` → ranked cited spans (`wiring.retrieval_service`).
  - `rag.query(query, top_k)` → grounded-or-abstain answer + citations
    (`pipeline.answer_query` — guardrails → router → transform → faithfulness → audit).
- **Transport: stdio** (first cut) — canonical MCP server shape, simplest to test,
  one process per session. HTTP/streamable transport + per-connection auth is a follow-up.
- **Tenancy (isolation preserved):** the server is bound to **one tenant at launch**
  (`RAG_MCP_TENANT`, with an optional `RAG_MCP_TOKEN` dev JWT validated against it). Tools
  ALWAYS scope to that tenant — never a client-supplied tenant — so a server instance can
  only ever act for its authorised tenant.
- **No schema/contract change** — it only wraps existing contracts. The `/v1/mcp/manifest`
  discoverability stub stays and matches the real tool names.

## Consequences
- New dependency `mcp` (in `apps/mcp` only, not the API image). New console script
  `rag-mcp`. Tool handlers are plain async callables (tested directly, in-memory).
- Multi-tenant networked MCP (HTTP transport, per-connection auth/OIDC) is the next step.
