# ADR-0013 — Ingestion from external tenant buckets (FR-A2b)

**Status:** Accepted (Phase-1) · 2026-07-03

## Context
ADR-0012 shipped A2a — by-reference ingest from the tenant's **own** platform object
store. Tenants also stage documents in **their own** S3-compatible buckets and want the
platform to pull from there (FR-A2b). Unlike A2a, this **crosses the egress boundary**:
the platform makes an outbound call to an endpoint it does not own. For a sovereign
platform that is the sensitive part — it must not become a way to exfiltrate to, or pull
from, a non-sovereign endpoint.

## Decision
- Add an `ExternalObjectFetcher` seam (Protocol + `MinioExternalFetcher` adapter +
  in-memory twin) distinct from `ObjectStore`: endpoint and credentials are supplied
  **per call**, not from platform config. `ObjectStoreConnector` gains an external branch,
  selected by an optional `external_source` on the envelope; A2a (platform pull) is
  unchanged.
- **Sovereignty is enforced first, and never loosened.** Before resolving any secret or
  opening any connection, the connector calls `assert_sovereign(endpoint)`. A non-
  allowlisted host raises `SovereigntyViolationError` (403) and nothing is fetched. So
  A2b reaches **only endpoints the operator has explicitly added to
  `RAG_ALLOWED_EGRESS_HOSTS`** — a deliberate, auditable operator action. A tenant/API
  caller cannot self-grant egress to an arbitrary bucket. The fetcher adapter re-asserts
  as defence-in-depth.
- **Credentials by reference only (no secrets in transit).** The request carries a
  `credentials_ref` (a name); the connector resolves `{ref}_access_key` /
  `{ref}_secret_key` via the `SecretsProvider` (env now, Vault later). Raw keys never
  appear in the request body, logs, or the persisted catalog envelope.
- **Contract:** optional typed `ExternalObjectSource {endpoint, bucket, secure,
  credentials_ref}` on `IngestRequest` + `DocumentEnvelope`, valid only with
  `source_kind=object_store`. Absent ⇒ A2a; present ⇒ A2b. No new dependency (the MinIO
  SDK already ships).

## Alternatives considered
- **Inline raw credentials in the request** — rejected: puts live secrets in the API
  payload, logs, and audit trail; violates the no-secrets-in-transit posture for a govt
  platform. Secret-reference keeps the 12-factor guarantee end-to-end.
- **A per-tenant "registered source" resource** (register endpoint+bucket+creds once,
  ingest by `source_id`) — deferred: cleaner API long-term but adds a new resource, CRUD
  surface, and persisted config. Inline-per-request ships the capability now; the registry
  can layer on without changing the connector.
- **Allowlist bypass / per-request egress override** — rejected outright: it would make
  the sovereignty guard advisory. The guard stays authoritative; enabling an endpoint is
  an operator config change, not a request parameter.

## Consequences
- External pull is available but **operator-gated** by the egress allowlist — the correct
  sovereign default. Onboarding a tenant's bucket = allowlist its endpoint + load its
  secret under the agreed `credentials_ref`.
- Validated in-memory incl. the refusal path (`tests/e2e/test_external_object_ingest.py`);
  the in-memory twin still runs `assert_sovereign`, so the guarantee is proven without a
  live external bucket. Full validation against a real remote MinIO/S3 is a real-store run.
- Pulled bytes still land in the platform `raw/{doc_id}` (canonical copy) and index via the
  unchanged downstream path; the catalog envelope records the external provenance
  (endpoint/bucket/ref — no secret) for lineage.
