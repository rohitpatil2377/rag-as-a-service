# RAG as a Service — Project Memory

A sovereign, multi-tenant, agentic RAG-as-a-Service platform. This file orients any future
session (human or AI) working in this repo. Read it before making architectural changes.

## What this is

A platform that turns a private document corpus into cited, trustworthy answers — or an
explicit abstention — with every component self-hostable, so no data and no inference leaves
the operator's own infrastructure.

Where to look:

- **This file** — the architecture map, the constraints that are not negotiable, and an
  honest inventory of what is built and what is broken.
- **[`docs/adr/`](docs/adr/)** — 26 decision records. Read the relevant one before reversing
  something that looks like an oversight; it usually is not.
- **[`docs/PLATFORM_GUIDE.md`](docs/PLATFORM_GUIDE.md)** — the operator-facing guide.
- **[`README.md`](README.md)** — quickstart, and how the in-memory and real-store profiles differ.

Requirement codes (`FR-A1`, `FR-E4`, …) appear throughout the code comments. They come from the
original requirement specification, which is not part of this repository; treat them as stable
labels for a capability, not as links.

**Layout note:** the package boundaries mirror an eight-layer microservice target, but
everything is deployed as a single process (ADR-0001). Splitting it is a deployment change,
not a rewrite.

## Hard constraints (non-negotiable, apply to every change)
- **Sovereign:** every component self-hostable. No foreign or managed AI/cloud calls at runtime. OSS models only, served locally (TEI/vLLM). Enforced in code
  by `rag_common.sovereignty.assert_sovereign` (egress allowlist), not just docs.
- **Multi-tenant by namespace, everywhere:** `tenant_id` is an explicit parameter on
  every store/retriever Protocol method — never inferred from ambient state for
  isolation. See "Tenancy" below.
- **Grounded-or-abstain:** every answer carries citations, or the platform abstains.
  Enforced at the **type level** — `GroundedAnswer` raises a validation error if
  `status=GROUNDED` has zero citations, or `status=ABSTAINED` has any.
- **No RAG framework lock-in:** `libs/rag-core` is hand-rolled (Protocols + pure
  logic). No LlamaIndex/LangChain. See ADR-0002.
- **Scope discipline:** unbuilt features are documented Protocol/interface stubs carrying a
  `TODO(Phase-N)` comment — never partial implementations.

## Architecture map (layer → package)

| Layer | Package(s) | Requirement codes |
|---|---|---|
| Connectors & Ingestion (§4.1) | `services/ingestion` (`svc_ingestion`) | A1–A3, A7 |
| Document Intelligence (§4.2) | `services/doc-intelligence` (`svc_doc_intelligence`) | B1–B3, B6, G2 |
| Indexing (§4.3) | `services/indexing` (`svc_indexing`) | C1, C3–C5, C8 |
| Retrieval Engine (§4.4) | `services/retrieval` (`svc_retrieval`), core in `libs/rag-core` | D1, D2, D7, D8 |
| Generation & Grounding (§4.5) | `services/generation` (`svc_generation`), gate in `libs/rag-core` | E1–E3, E6 |
| Agentic Orchestration (§4.6) | `services/control-plane/hitl.py` (HITL only) | F6 (rest = Phase-1/2 stub) |
| Guardrails & Eval (§4.7) | `services/guardrails-eval` (`svc_guardrails_eval`) | G4, G5, G7 |
| Control Plane (§4.8) | `services/control-plane` (`svc_control_plane`) — `/v1` API, tenancy, MCP stub | H1, G6 |
| Observability (§4.8) | `libs/common` (OTel) + `infra/otel`, `infra/prometheus`, `infra/grafana` | H5 |

All 7 service packages are mounted as routers in **one process**: `apps/api`
(`rag_api`), the modular monolith (see ADR-0001). `apps/demo-cli` drives the walking
skeleton end-to-end via ASGI (no network needed). `apps/console` is the **React + Vite +
TS + Tailwind** SPA (FR-H3) — a separate Node app (not in the uv workspace) consuming the
design tokens (`design/tokens`) and the `/v1` API; see `apps/console/README.md`.

## Shared libraries (`libs/`)
- **`rag_common`** — config (`Settings`, env-driven), structured logging (structlog,
  tenant-bound), OTel tracing, error taxonomy, **tenancy** (`tenant_scope`,
  `vector_collection`/`sparse_index`/`object_prefix`/`pg_schema` namespace helpers),
  dev-JWT auth (`Principal`, RBAC via `roles`, ABAC via `attrs`), `SecretsProvider`
  Protocol, sovereignty egress guard, FastAPI middleware/health helpers.
- **`rag_schemas`** — the 5 core contracts: `DocumentEnvelope`, `Chunk`,
  `RetrievalQuery`/`RetrievedSpan`/`RetrievalResult`, `Citation`/`GroundedAnswer`
  (grounded-or-abstain invariant enforced), `AuditEvent` (SHA-256 hash chain).
- **`rag_core`** — Protocols for every seam (`VectorStore`, `SparseIndex`,
  `Embedder`, `Reranker`, `LLMGateway`, `Chunker`, `ObjectStore`, `CatalogAdapter`,
  `AuditSink`, `Retriever`, plus Phase-1: `FaithfulnessChecker`, `Contextualizer`,
  `QueryRouter`) + pure logic: RRF fusion, chunking (FIXED/CLAUSE), grounded-or-abstain
  gate + citation assembly, the `HybridRetriever` orchestrator, faithfulness routing
  (`route_faithfulness`), contextual-retrieval helpers, the heuristic query router.
- **`rag_clients`** — concrete adapters implementing the Protocols: `QdrantVectorStore`,
  `OpenSearchSparseIndex`, `MinioObjectStore`, `PostgresCatalog`, `PostgresAuditSink`,
  `TeiEmbedder`, `OpenAICompatGateway` (serves both llama.cpp CPU + vLLM GPU) /
  `ExtractiveDevGateway`, `TeiReranker`/`LexicalReranker`, and Phase-1
  `Lexical/LLMFaithfulnessChecker`, `Metadata/LLMContextualizer`, `Heuristic/LLMRouter`,
  plus the **in-memory twins** in `memory.py`. `factory.py` is the ONE place that selects
  a backend from config (each seam has a `get_*` + a `build_*`) — services never import a
  concrete adapter. Config flags gate the additive Phase-1 features (all default OFF/safe).

## Tenancy (read before touching any store adapter)
`tenant_id` is **always an explicit parameter**, never inferred from a contextvar for
isolation decisions. `rag_common.tenancy` is the single source of truth for how a
`tenant_id` maps to each store's namespace:
- Qdrant: one **collection** per tenant (`vector_collection`).
- OpenSearch: one **index** per tenant (`sparse_index`).
- MinIO: one **key prefix** per tenant within a shared bucket (`object_prefix`).
- Postgres: one **schema** per tenant (`pg_schema`) — documents/chunks/tenant_config/
  audit_log/hitl_reviews tables all live per-schema.
The `contextvars`-based `current_tenant()`/`tenant_scope()` exists for logging
correlation and the demo CLI — it is defence-in-depth, not the isolation mechanism.
`tests/integration/test_tenant_isolation.py` is the correctness gate: it proves tenant
B cannot read tenant A's data across vector + BM25 + object + catalog.

## Conventions
- **Naming:** libs publish as `rag_common`/`rag_schemas`/`rag_core`/`rag_clients`;
  services are `svc_<name>` (not imported outside the monolith wiring); apps are
  `rag_api`/`demo_cli`.
- **Errors:** all domain errors subclass `rag_common.errors.PlatformError` (stable `code`
  + `http_status`); `TenantIsolationError` and `SovereigntyViolationError` are always
  fail-closed; `GroundingAbstain` is a 200, not a fault.
- **Logging:** structlog, JSON by default, tenant-bound via contextvars. Never log
  secret values (route them through `SecretsProvider`).
- **Testing:** `pytest` markers `integration`/`e2e`/`eval` gate on infra; default
  `make test` excludes them and runs the in-memory profile. See "Real stores vs
  in-memory" in the README for how to run the same tests against live Docker stores.
- **Type checking:** mypy strict across `libs services apps`. Zero tolerance —
  keep it green.
- **ADRs:** `docs/adr/` records deliberate deviations/decisions. Read them before
  reversing a decision that looks like an oversight — it probably isn't (check the ADR
  first).

## What is built

Grouped by capability rather than by the order it landed. Everything additive is flag-gated and
**default-safe**: with a flag off the code path is byte-identical to the one before it existed.
Flag names are the `RAG_*` env vars in [`.env.example`](.env.example); per-KB overrides live in
`tenant_config`.

### Retrieval and generation
- **Hybrid retrieval** — dense (Qdrant) + BM25 (OpenSearch) fused with reciprocal rank fusion,
  then a cross-encoder reranker selects the passages that reach the model.
- **Grounded-or-abstain** — enforced at the type level in `rag_schemas`, not by convention.
- **Faithfulness gate** — three-way routing GROUNDED / FLAGGED_FOR_REVIEW (→ human review) /
  ABSTAINED, with lexical and NLI backends and per-tenant thresholds. ADR-0005.
- **Contextual retrieval** — chunks are embedded with a generated context prefix; the original
  chunk is what gets cited. ADR-0007.
- **Adaptive router** — classifies a query simple/complex and picks a shallow or deep path.
  ADR-0008. (A measured A/B of router tuning came back a null result, so no tuning shipped.)
- **Query transformation + agentic retry** — decompose into sub-queries, retrieve each, merge by
  best score; one bounded retry when the first pass would abstain. ADR-0009.
- **Semantic cache** — per-KB LRU keyed on query embedding *and* caller ACL, invalidated on any
  write. Deliberately bypassed for recency- and metadata-filtered queries, whose keys it does
  not capture.
- **Recency and metadata filters** — optional time window plus half-life freshness boost;
  taxonomy/entity/year tags derived at ingest become retrieval filters.
- **Permission-aware retrieval** — source ACLs captured at ingest are enforced at query time; a
  span is visible only if its ACL is empty or intersects the caller's groups.
- **GraphRAG** — per-chunk entity/relationship extraction into per-KB Postgres adjacency tables,
  label-propagation communities with theme summaries, and both local (neighbour-expansion) and
  global (community-evidence) search. Heuristic by default, LLM extraction opt-in per KB.
- **RAG v2 pipeline** — document profiling (a reviewable prompt file builds a per-document card),
  section mapping and enrichment, map-reduce over selected sections, a card-driven router and a
  verification pass. Gated by `RAG_V2_*`.

### Ingestion and document intelligence
- **Connectors** — object store by reference (own namespace and external buckets with
  per-source credentials), bounded web crawl, scheduled delta sync, plus NoSQL/REST/SharePoint
  adapters. ADR-0012, ADR-0013.
- **Layout-aware extraction** — OCR, table reconstruction including merged cells, boilerplate and
  page-furniture stripping, heading/section detection feeding layout-aware chunking.
- **PII masking** — checksum-aware (Verhoeff, Luhn) so a 12-digit invoice number is not mistaken
  for an identity number, with redact / mask / hash strategies. Runs at ingest, so only
  de-identified text is chunked, embedded and indexed.
- **PII-driven classification floor** — detected PII can raise a document's classification, or
  block the ingest; a floor above the ingestor's own clearance is always blocked.
- **Delta and near-duplicate detection** — raw bytes are SHA-256 hashed, so re-ingesting
  unchanged content skips extract/embed/index and identical bytes under a new id are flagged.
- **Domain adaptation** — a KB's corpus is profiled (chemistry, legal, financial, code) and
  *suggests* per-KB knobs; an operator confirms them. Detection never moves a knob by itself.
  For chemistry, RDKit canonicalizes SMILES so equivalent molecules match at retrieval.

### Tenancy, access and governance
- **Namespace isolation** — one Qdrant collection, OpenSearch index, MinIO prefix and Postgres
  schema per tenant, inside shared store containers. `tenant_id` is an explicit parameter on
  every store method. This is namespace isolation, not a container per tenant.
- **RBAC** — a closed capability set (reader / ingestor / reviewer / admin / owner), org-wide
  custom role labels over those capabilities, per-KB roles carrying clearance, and KB managers
  with delegated member and model management scoped to one KB.
- **Multi-level security** — documents carry a classification, members a clearance; retrieval
  shows a span only when clearance ≥ classification. Cross-KB access requests let a manager grant
  additive, revocable, live access up to their own clearance.
- **Auth** — OIDC/SSO, a dev JWT mode, and API keys.
- **Encryption** — per-tenant data encryption keys wrapped by a KEK, with a KMS/HSM seam so
  revoking the key crypto-shreds the tenant's data; versioned rotation keeps old ciphertext readable.
- **Sovereignty** — `rag_common.sovereignty.assert_sovereign` refuses egress to any host not on
  the allowlist, with an in-product approval workflow for adding one.
- **Audit and compliance** — an append-only SHA-256 hash chain, right-to-erasure that cascades
  through every store and emits a tamper-evident certificate, per-KB retention, a residency
  attestation, and a declarative mapping of live controls onto several governance frameworks.
  Controls that are organisational rather than technical report MANUAL so nothing is over-claimed.
- **Governed NL→SQL** — model-written SQL is never trusted: single-statement SELECT only, every
  table checked against an allowlist, a function denylist, and a hard outer LIMIT, over a
  read-only connection. Violations raise rather than execute.
- **Agentic workflows** — a declarative, deterministic runtime whose branching is a transparent
  rules engine, not a black-box agent, with a recorded decision trace and a runaway guard. Each
  step re-checks the caller's role against the equivalent direct endpoint.

### Interfaces
- **`/v1` REST API** — the single entry point; `apps/api` mounts all seven service routers in one
  process (ADR-0001).
- **MCP server** — stdio (bound to one tenant at launch) and streamable HTTP (multi-tenant,
  mounted in the monolith at `/mcp`). Tools wrap the same pipeline, so agents inherit every
  safety control. ADR-0011.
- **SDKs and embeds** — Python and TypeScript clients that depend only on an HTTP library, a
  dependency-free embeddable chat widget whose session exchange keeps the API key out of the
  browser, and a chat-platform bot adapter that answers in the webhook response body (no
  outbound call, so no egress).
- **Console** — React + Vite + TypeScript SPA: KB management, ingest, a chat-style Ask with
  citations and a faithfulness meter, review queue, audit table, graph canvas, quality and
  compliance pages, plus a separate platform-operator shell for fleet lifecycle, residency
  attestation and air-gap preflight. Light and dark, WCAG-AA checked, English and Hindi.

### Evaluation and operations
- **Continuous evaluation** — a scheduler scores each enrolled KB's golden set plus a
  deterministic sample of recent live traffic, and alerts in-platform when grounded rate or
  faithfulness falls below a per-KB floor. Alerting is audit + logs + console only, by design.
- **Golden sets** — synthetic generation from a KB's own documents produces a *draft* for review,
  never auto-saved; `eval/gold/*.yaml` carries the typed suite format with per-tier item recall.
- **Metering** — usage counts aggregated from the audit trail with an operator-configurable rate
  card, plus budgets and caps.
- **Federation** — an opt-in libp2p mesh (`sovereign-node`, Go) where departments hold their own
  data and answer each other's scatter-gather queries under Ed25519-signed, revocable,
  delegatable grants.
- **Observability** — OpenTelemetry traces, Prometheus metrics, Grafana dashboards.
- **Air-gapped operation** — a self-contained install bundle, signed offline update bundles, and
  backup/restore scripts for the stateful stores.

## Known limitations

Kept here deliberately: these are real and some are load-bearing for how much to trust a result.

- **Audit events split across schemas.** Query-time events (`query`, `generate`, `faithfulness`,
  `guardrail`) are written under the principal's org `tenant_id`, while `ingest`/`index` events
  are written under the specific `kb_id`. Those are different Postgres schemas whenever
  `kb_id != tenant_id` — i.e. any org with more than one KB. Since the quality endpoint reads the
  KB's own schema, the Quality and Compliance dashboards are blind to query-time activity for
  those KBs. Needs a scope decision (query-time audit should probably move to `kb_id`) before
  fixing; see the `audit_service().record(...)` call sites in `pipeline.py`.
- **Output language is mitigated, not solved.** Naming the target language explicitly in the
  prompt took Hindi drift from ~78% to ~6%. It does not generalise: measured against Tamil,
  Bengali and Telugu, English questions stayed English only 10/15 times. Marathi has no distinct
  code and is silently labelled Hindi; Urdu's script is not in the detector at all.
- **Faithfulness checks entailment, not relevance.** A sentence can be entailed by *some*
  retrieved chunk that has nothing to do with the question, which is how a confidently wrong
  answer once scored 1.0.
- **Very long queries can overflow the sparse leg** (OpenSearch `maxClauseCount`).
- **Provisioning is not atomic.** A transient store timeout can leave an org created with no
  fleet registry record, and the retry then reports "already exists". Recovery is manual.
- **Heuristic graph extraction is noisy** on dense documents — co-occurrence produces giant
  components. Switch a KB to LLM extraction for a clean typed graph. The image-only junk filter
  reduces but does not eliminate this.
- **Chemistry normalization lifts recall only.** The reranker and the model still operate on
  text and do not know chemical equivalence, so a cross-form query will not always produce a
  grounded answer.
- **No ColPali-class visual retrieval.** It needs a GPU this project does not assume. The
  shipped approximation parses diagram structure into the knowledge graph instead.
- **Mesh onboarding is an ops action.** The identity, grant and transport machinery is
  department-count agnostic and was exercised at four nodes, but Compose has no loop construct,
  so each new node still needs a copied service block. There is no self-service endpoint.

## Working style

Commit at each green checkpoint. Plan-then-confirm only when a change (a) alters a schema or
contract, (b) adds a heavy dependency or service, or (c) changes what gets indexed or stored —
otherwise build additive logic straight through.

## How to run locally
See README.md "Quickstart" and "Real stores vs in-memory." Short version:
```bash
make install        # uv sync
make test            # in-memory profile, ~4s, no infra
make up              # docker-compose: stores, models, API, observability, AND the console
make migrate          # Alembic baseline against live Postgres
make demo             # walking skeleton (in-memory by default)
```
`make up` also builds and serves the **console** (nginx container, `apps/console/Dockerfile`)
at **http://localhost:8082** — nginx serves the built SPA and reverse-proxies `/v1` + `/health`
to the `api` service (same-origin, no CORS), so no separate dev server is needed to browse the
platform. Other ports: API `:8000`, Grafana `:3000` (admin/admin), Qdrant `:6333`, S3 `:9000`.

The default `up` is self-contained — no secrets, no model endpoint, no GPU. Generation is
`extractive_dev` and the two model servers are opt-in:

```bash
docker compose -f infra/compose/docker-compose.dev.yml --profile llm up -d llama   # real local LLM
docker compose -f infra/compose/docker-compose.dev.yml --profile nli up -d nli-checker
```

The object store is **SeaweedFS**, not MinIO — MinIO withdrew their public images, so a pinned
tag could no longer be pulled. The service is still named `minio` (it is the hostname in several
env vars) and `MinioObjectStore` is unchanged, since that adapter uses the MinIO *client*.
For live UI development you can still run the Vite dev server instead:
```bash
cd apps/console && npm install && npm run dev   # http://localhost:5173 (proxies /v1 to :8000)
```
Representative feature flags (all default OFF/safe unless noted): `RAG_CONTEXTUAL_RETRIEVAL_ENABLED`,
`RAG_ADAPTIVE_ROUTER_ENABLED`, `RAG_FAITHFULNESS_ENABLED` (on), `RAG_FAITHFULNESS_BACKEND`,
`RAG_LLM_BACKEND` (llamacpp|vllm|extractive_dev). See `.env.example`.
