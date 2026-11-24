# ADR-0014 — Schema Migration Discipline: Alembic for Global, Runtime for Per-Tenant

**Status:** Accepted · **Date:** 2026-07-13

## Context
RAG as a Service's PostgreSQL state splits into two fundamentally different shapes:

1. **Global, fixed schema** — the `rag_identity` schema (organisations, users, service-account
   API keys) and the operational registries (continuous-eval + connector-sync schedules, runtime
   feature-flag overrides, egress-approval requests, the async bulk-ingestion job queue), plus the
   `public.tenants` baseline. These tables are the same for every deployment and change only when
   the code changes.
2. **Per-tenant dynamic schema** — one `t_<slug>` schema **per organisation**, created the moment a
   tenant is provisioned, holding that tenant's `documents`, `chunks`, `audit_log`, `tenant_config`,
   `knowledge_bases`, `conversations`, `hitl_reviews`, `model_connections`, and graph tables
   (schema-per-tenant isolation, FR-H1, ADR-0001).

Classic Alembic assumes a **known, fixed** set of schemas at migration time. It cannot pre-create
`t_<slug>` for tenants that don't exist yet — tenants are born at runtime. A naïve "put everything
in Alembic" approach does not fit a schema-per-tenant model.

The risk we are addressing: several global tables (`platform_flags`, `egress_requests`,
`ingest_jobs`, …) had been created **only** by the store adapters' runtime `ensure()` calls, so the
migration history did not describe the real global schema — a reproducibility/auditability gap for
production.

## Decision
Two mechanisms, each matched to the schema shape:

- **Global schema → Alembic migrations.** `0001_baseline` creates `public.tenants`;
  `0002_rag_identity` versions the entire `rag_identity` schema. To keep a **single source of
  truth and avoid drift**, `0002` executes the *same* idempotent DDL the store adapters use — it
  imports each store module's `_DDL` and runs it, rather than re-declaring the columns. A CI test
  (`test_migration_coverage.py`) asserts every expected global table is covered.

- **Per-tenant schema → runtime `ensure()`.** Each store adapter creates its tenant's tables
  idempotently (`CREATE … IF NOT EXISTS`) when the tenant is first touched. This is the correct
  model for dynamic per-tenant schemas and is **deliberately not** in Alembic.

- **Runtime `ensure()` stays as a self-healing safety net even for global tables.** Because every
  statement is `CREATE … IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`, a deployment that runs
  `make migrate` and one that relies on first-run `ensure()` converge to the identical schema, and
  the two never conflict.

## Consequences
- `make migrate` on a fresh database now produces the **complete, versioned** global schema
  deterministically (proven: an empty DB migrates to all nine `rag_identity` tables).
- Adding a new global table = add its `_DDL` to a store module **and** to `0002`'s module list; the
  coverage test fails if you forget.
- Per-tenant schema changes (e.g. a new column like `knowledge_bases.trashed_at`) ship as a
  migration-safe `ADD COLUMN IF NOT EXISTS` in the adapter's `ensure()` — applied lazily per tenant.
- **Not yet addressed:** a per-tenant migration *runner* (to apply a column change across all
  existing `t_<slug>` schemas in one operation) — today the change lands on next-touch. A
  fan-out migration tool over all tenant schemas is the documented follow-up when a per-tenant
  backfill becomes necessary.
