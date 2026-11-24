# ADR-0001 — Modular Monolith First, Split at Phase-1/Scale

**Status:** Accepted · **Date:** 2026-07-02

## Context
The TDD (§3–§4) specifies 8 independently-deployable microservices (Connectors,
Document Intelligence, Indexing, Retrieval, Generation, Agentic Orchestration,
Guardrails/Eval, Control Plane). Building all 8 as separate deployables in Phase-0
would multiply operational surface (8 Dockerfiles, 8 sets of health checks, network
hops, service discovery) before the platform has a single proven user or workload.

## Decision
Phase-0 ships as **one deployable FastAPI application** (`apps/api`) that mounts each
service module's router in-process. Package boundaries are kept **exactly** as if the
services were already split:
- `services/<name>/` is its own Python package with its own `pyproject.toml`.
- Every cross-service call goes through a `rag-core` Protocol (`VectorStore`,
  `Retriever`, `LLMGateway`, etc.) — never a direct import of another service's
  internals.
- `services/control-plane/src/svc_control_plane/wiring.py` is the **one file** that
  wires concrete instances together; it is the seam a future split touches.

The TDD's microservices target is not abandoned — it's deferred until there's a real
scaling or team-ownership reason to pay for the operational cost of separate
deployables (Phase-1+).

## Consequences
- **Positive:** one container to build/deploy/debug in Phase-0; in-process calls have
  zero network latency; `docker-compose.dev.yml` has one `api` service instead of 8.
- **Positive:** splitting a layer later is a **deployment change**, not a rewrite —
  swap the in-process call in `wiring.py` for an HTTP/gRPC client implementing the
  same Protocol; the service's internal code does not change.
- **Negative:** until split, all layers share one process's failure domain, memory,
  and scaling profile. Acceptable for Phase-0 POC scale; revisit before GA if a single
  layer (e.g. OCR/embedding) needs independent horizontal scaling or GPU placement.
- **Negative:** the monolith's dependency graph pulls in every service's transitive
  deps (e.g. `pypdf`, `qdrant-client`, `opensearch-py`) into one image — larger image,
  slower cold start than a per-service image would be.

## Alternatives considered
- **8 microservices from day one** — rejected for Phase-0: highest fidelity to the TDD
  target, but the operational cost (8 CI pipelines, 8 sets of liveness probes, network
  auth between every hop) isn't justified before a single tenant is onboarded (PRD
  Phase-0 exit criteria is "clears the CAG POC," not "operates at scale").
- **Single flat script (no service boundaries)** — rejected: would need a full rewrite
  to reach the TDD's target architecture; defeats the "5-year product" mandate.
