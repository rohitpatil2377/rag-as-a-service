# rag-api — RAG as a Service modular-monolith API

**TDD:** whole-system entrypoint · **ADR:** [0001-monolith-first](../../docs/adr/0001-monolith-first.md)

One deployable FastAPI process that mounts every Phase-0 service module's router:

| Mounted router | Package | TDD component | PRD FRs |
|---|---|---|---|
| `/v1/*` (public) | `svc_control_plane` | §4.8 Control Plane | FR-H1, H5, G6, F6 |
| internal | `svc_ingestion` | §4.1 Connectors | FR-A1–A3, A7 |
| internal | `svc_doc_intelligence` | §4.2 Doc Intelligence | FR-B1–B3, B6, G2 |
| internal | `svc_indexing` | §4.3 Indexing | FR-C1, C3–C5, C8 |
| internal | `svc_retrieval` | §4.4 Retrieval | FR-D1, D2, D7, D8 |
| internal | `svc_generation` | §4.5 Generation | FR-E1–E3, E6 |
| internal | `svc_guardrails_eval` | §4.7 Guardrails/Eval | FR-G4, G5, G7 |

Package boundaries + rag-core Protocols mirror the microservice target, so each layer
can be split into its own container later without a rewrite (only `wiring.py` changes).

## Run
```bash
make up          # start stores + models + this API (docker-compose)
# or locally against a running stack:
uv run rag-api  # serves on :8000
```

Key endpoints (all tenant-scoped, RBAC/ABAC-enforced, OTel-traced):
`POST /v1/knowledge-bases`, `POST /v1/kb/{id}/ingest`, `POST /v1/kb/{id}/query`,
`POST /v1/kb/{id}/retrieve`, HITL `/v1/kb/{id}/hitl/...`, `GET /v1/mcp/manifest`.
