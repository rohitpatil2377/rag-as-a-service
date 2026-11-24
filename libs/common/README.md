# `rag_common`

Cross-cutting concerns every other package depends on, and nothing domain-specific. If a change
here is wrong, it is wrong everywhere — treat this package as API surface.

24 modules. The ones worth knowing:

| Module | What it owns |
|---|---|
| `config.py` | `Settings` — the single, env-driven configuration object (prefix `RAG_`, 384 fields). Nothing reads `os.environ` directly. |
| `tenancy.py` | How a `tenant_id` becomes each store's namespace: `vector_collection`, `sparse_index`, `object_prefix`, `pg_schema`. The isolation primitive. |
| `auth.py` | `Principal`, dev-JWT issuing/verification, OIDC, API-key resolution, `require_roles`. |
| `rbac.py` | The closed capability set (reader / ingestor / reviewer / admin / owner) and the checks over it. |
| `sovereignty.py` | `assert_sovereign` — the egress allowlist. Fail-closed: an unlisted host raises rather than being called. |
| `crypto.py`, `kms.py` | Envelope encryption: per-tenant DEKs wrapped by a KEK, with a KMS/HSM seam. |
| `errors.py` | `PlatformError` and the domain error taxonomy, each with a stable `code` and `http_status`. |
| `logging.py`, `tracing.py` | structlog (JSON, tenant-bound via contextvars) and OpenTelemetry. |
| `middleware.py`, `health.py` | FastAPI wiring shared by every service router. |

## Two rules

**`tenant_id` is always an explicit parameter.** `current_tenant()`/`tenant_scope()` exist for log
correlation and the demo CLI. They are defence in depth, never the isolation mechanism — an
isolation decision taken from ambient state is a bug, and `tests/integration/test_tenant_isolation.py`
is the gate that proves it.

**Never log a secret.** Route credentials through `SecretsProvider`; `config.py` keeps them out of
`repr`.
