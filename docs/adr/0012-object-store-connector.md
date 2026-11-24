# ADR-0012 — By-reference ingestion from object storage (FR-A2)

**Status:** Accepted (Phase-1) · 2026-07-03

## Context
Phase-0 ingested documents only as **inline bytes** (`content_b64`, the UPLOAD source).
Real tenants stage documents in **S3-compatible object storage** and want to ingest them
**by reference** (FR-A2) — pointing the platform at an object rather than base64-inlining
megabytes through the API. The connector seam (FR-A5) was built for exactly this: adding
a source must not touch doc-intelligence, indexing, or retrieval.

## Decision
- Add `ObjectStoreConnector` implementing the existing `Connector` Protocol
  (`services/ingestion`). It resolves an envelope's `uri` — an **object key within the
  tenant's own namespace** — via `ObjectStore.get(tenant_id, key)` and returns the raw
  bytes into the unchanged parse → chunk → index path. Registered under
  `source_kind="object_store"` alongside `"upload"`.
- **Two increments; this ADR ships A2a:**
  - **A2a (now):** pull from the **platform** object store (tenant's own MinIO namespace).
    No new dependency (MinIO SDK already present); the pull is tenant-isolated by
    construction (`ObjectStore.get` scopes the key under the tenant prefix) and sovereign
    (the platform store is on-allowlist).
  - **A2b (later):** pull from an **external**, tenant-provided S3 bucket — per-source
    endpoint/credentials via `SecretsProvider`, with the endpoint enforced through
    `assert_sovereign` so a foreign/non-allowlisted bucket is refused. Layers on the same
    seam; no core change.
- **API contract (the one contract touch):** `IngestRequest.content_b64` becomes optional
  and a `object_key` field is added. A `model_validator` enforces exactly one, keyed by
  `source_kind` (upload → `content_b64`; object_store → `object_key`; other kinds rejected
  422). Backward-compatible: existing inline callers/console/tests are unaffected.
- **Fail-closed:** a missing referenced object raises `NotFoundError` (404), not a 500.

## Alternatives considered
- **Separate `/v1/kb/{id}/ingest-ref` endpoint** — rejected: two endpoints to document and
  keep in sync; one endpoint with a source-kind-keyed validator is cleaner and keeps the
  inline path a strict subset.
- **Enforce the inline-XOR-reference invariant on the core `DocumentEnvelope`** — rejected:
  several unit tests legitimately build metadata-only UPLOAD envelopes with empty content
  (they exercise chunking/contextual logic directly), so the invariant lives at the request
  layer + the connector layer, where the payload is actually consumed. The envelope
  docstring remains the spec.

## Consequences
- New source with zero changes downstream of ingestion — validates the FR-A5 seam.
- By-reference ingest still **re-stores** the raw bytes under `raw/{doc_id}` (canonical
  lineage copy, identical to upload); the catalog keeps the source `uri` for provenance.
- A2b (external buckets) is the next connector step and carries the sovereignty-allowlist
  enforcement + per-source secret handling.
