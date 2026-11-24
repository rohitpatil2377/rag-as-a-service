# Services

Seven service modules. They are packaged separately and their boundaries mirror the eight-layer
microservice target, but they all run **in one process** — `apps/api` mounts each one as a FastAPI
router. Splitting them out later is a deployment change, not a rewrite
([ADR-0001](../docs/adr/0001-monolith-first.md)).

| Directory | Module | Owns |
|---|---|---|
| `ingestion/` | `svc_ingestion` | Connectors: upload, object store, external buckets, web crawl, scheduled sync. |
| `doc-intelligence/` | `svc_doc_intelligence` | Extraction: OCR, layout and table reconstruction, boilerplate stripping, PII masking and the PII-driven classification floor. |
| `indexing/` | `svc_indexing` | Writing chunks into the dense and sparse indexes with their metadata and ACLs. |
| `retrieval/` | `svc_retrieval` | The retrieval entry point; the orchestration logic itself lives in `libs/rag-core`. |
| `generation/` | `svc_generation` | Prompt assembly and the call to the gateway. |
| `guardrails-eval/` | `svc_guardrails_eval` | Prompt-injection and toxicity guardrails, the evaluation harness and gold-suite loader. |
| `control-plane/` | `svc_control_plane` | By far the largest (80 modules): the whole `/v1` API, tenancy and KB lifecycle, RBAC, the query pipeline, HITL review, workflows, compliance, federation, NL→SQL. |

## Conventions

- A service module is **never imported outside the monolith wiring**. Cross-service calls go
  through `libs/rag-core` Protocols, not by importing a sibling `svc_*`.
- `control-plane/src/svc_control_plane/pipeline.py` is the spine: `ingest_document` and
  `answer_query` are the two functions most changes end up touching.
- `wiring.py` is where config becomes concrete dependencies. Add a seam there, not at the call site.
