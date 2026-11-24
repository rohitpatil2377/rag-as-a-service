# ADR-0023 — `ensure_schema` memoises its per-tenant DDL pass, and serialises it with an advisory lock

**Status:** accepted · 2026-10-16
**Supersedes nothing. Amends the operational property relied on by** ADR-0004 (in-memory
adapters / real-adapter fixes) **and by the three schema-drift repairs listed below.**

## Context

`PostgresCatalog.ensure_schema(tenant_id)` is the single place per-tenant Postgres objects are
created, and **every** catalog read and write calls it first — `register_document`,
`list_documents`, `get_chunks`, `document_chunks`, `get_tenant_config`, `delete_document`. That was
cheap when the statement list was six statements, and the repo has leaned on it three times as a
*self-healing migration channel*: `documents.content_hash` (FR-A8/C9),
`federation_grants.parent_grant_id` and `eval_schedules.federate_from` were each added to code and
reached already-provisioned deployments only because an `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
re-ran on the next touch. `CREATE TABLE IF NOT EXISTS` alone silently no-ops on an existing table,
which is what made those three bugs possible in the first place.

RAG v2 Phase 1 (upgrade plan §1.1) grows that list from 6 statements to 33: ten new `documents`
columns, eight new `chunks` columns, six new tables and four indexes. Measured against the dev
Postgres (`ragaas-dev-postgres-1`), per pass:

| Statement list | p50 | p95 | statements |
|---|---|---|---|
| Pre-v2 (what it was) | 15.3 ms | 17.4 ms | 6 |
| Phase-1 full list, steady state (everything already satisfied) | 55.2 ms | 70.0 ms | 33 |
| — of which phase 1, tables + columns | 47.4 ms | 56.2 ms | 29 |
| — of which phase 2, indexes | 9.6 ms | 16.2 ms | 4 |
| Cold first touch on a fresh schema (both phases) | 419 ms | — | 33 |

A satisfied statement is not free: each is still a round trip, and each `ADD COLUMN IF NOT EXISTS`
still takes AccessExclusive on `documents`/`chunks` before discovering there is nothing to do. So
carrying the structure tables unconditionally would have put **~40 ms on every catalog operation** —
including chunk hydration inside the retrieval hot path, where it is paid per query.

## Decisions

1. **Memoise per process, per tenant.** `PostgresCatalog` keeps `self._ensured: set[str]` and
   returns immediately for a schema it has already brought up to date. The statements are
   idempotent, so once per process is sufficient.

2. **Keep the memo in-process, never persisted.** A fresh worker re-verifies from scratch. The
   drift-free property is preserved where it matters — a schema is still brought *fully* up to date
   the first time any worker touches it, with no dependence on which env var was set — and a
   persisted flag would have cached a stale shape forever.

3. **Take `pg_advisory_xact_lock(hashtext(:schema))` as the FIRST statement of the pass.** This is
   a correctness fix, not an optimisation, and it is independent of the memo:

   - `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` **is** concurrency-safe. It serialises on the
     table's AccessExclusive lock, and the loser re-checks after acquiring it and emits a skip
     NOTICE.
   - `CREATE SCHEMA/TABLE IF NOT EXISTS` is **not**. Its existence check takes no lock — there is no
     relation to lock yet — so every concurrent worker passes it and the losers fail on a system
     catalog's unique index: `pg_namespace_nspname_index` for the schema,
     `pg_type_typname_nsp_index` for a table. Reproduced on this Postgres: 4 cold workers racing one
     fresh schema, **3 failed**. Because the pass is a single transaction, that rolls back the
     *entire* `ensure_schema` and surfaces as a failed request in whatever catalog call triggered
     it.

   This race **pre-dates** this change — the pre-v2 six-statement list fails identically, verified —
   so the lock repairs a latent defect rather than one Phase 1 introduced. It bites on new-tenant
   provisioning across 4 api instances x 2 workers. The advisory lock is used rather than a table
   lock precisely because the objects may not exist yet; the `xact` flavour is released by
   COMMIT/ROLLBACK, so a crashed worker cannot leak it, and `hashtext` keys it per schema so two
   tenants never block each other. Cost is one extra round trip, paid once per worker per tenant
   because of the memo.

4. **Split the pass into two transactions: tables/columns, then indexes.** An index build needs only
   ShareLock, but the phase-1 ALTERs hold AccessExclusive for the whole transaction, which blocks
   **readers** too. Together, the full 419 ms first touch held every reader of `documents` and
   `chunks` off for its entire duration, and an index build's cost scales with row count — so it got
   worse on bigger tenants. Split, readers are only held off for the brief ALTERs. Both halves stay
   idempotent, both take the advisory lock, and `self._ensured` is set **only after both commit**, so
   a failure in either leaves the schema to be retried on the next call.

5. **`drop_v2_tables` discards the memo** (and takes the same lock). Otherwise the process would go
   on believing tables it had just dropped still exist.

## Consequences

- **Accepted cost — the self-healing channel is now per-process, not per-call.** Hand-editing a
  tenant schema with `psql` against a *warm* worker no longer repairs itself on the next request:
  that worker still believes the schema is current. **Such surgery now requires an api restart to be
  picked up.** This is the real behaviour change, and the reason this ADR exists. Code-driven
  additive migration is unaffected — a deploy is a new process, so the next worker runs the full
  pass, which is exactly how the `content_hash` / `parent_grant_id` / `federate_from` repairs
  reached live deployments.
- ~40 ms comes off every catalog operation after the first per tenant, retrieval included.
- First touch of a tenant costs ~419 ms once per worker, during which only the brief phase-1 ALTERs
  block readers.
- Tests: the memo is observable without a database via a counting engine stub
  (`libs/clients/tests/test_v2_catalog_schema.py`). Because a warm memo issues **zero** SQL, every
  live idempotency assertion must use a **second, cold** `PostgresCatalog` (or clear `_ensured`) or
  it proves nothing — `tests/integration/test_v2_catalog_ddl.py` does the former, which also mirrors
  the real two-worker shape, and carries the 4-worker concurrency regression guard for decision 3.

## Alternatives rejected

- **Leave it unmemoised.** Correct and simple, but ~40 ms per catalog call, on the retrieval path.
- **Gate the v2 DDL on `v2_ingest`.** Makes a tenant's schema *shape* depend on which env was
  set the last time anything touched it — the exact drift class the three repairs above had to fix.
- **Persist an "ensured" marker in `tenant_config`.** Removes the per-process re-verification that
  makes the whole channel trustworthy, and would need its own migration.
- **Alembic for per-tenant schemas.** The real fix long-term, but Alembic manages only `public` /
  `rag_identity` (head `0007_tenants`) and would have to learn to enumerate 25 live tenant schemas
  and every future one; out of scope for Phase 1 and a separate decision.
- **Advisory lock only, no memo.** Serialising workers makes the pass correct but not cheap: the
  ~40 ms stays, and the lock adds contention between workers on the same hot tenant.
