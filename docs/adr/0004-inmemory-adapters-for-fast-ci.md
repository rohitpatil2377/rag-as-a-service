# ADR-0004 — In-Memory Adapter Profile for Fast CI/Dev; Real Stores Verified Separately

**Status:** Accepted · **Date:** 2026-07-02

## Context
The Phase-0 walking skeleton and its correctness tests (tenant isolation, golden RAGAS
eval, full vertical slice) need to run fast and deterministically in CI, but also need
to be proven against the real sovereign stack (Qdrant, OpenSearch, MinIO, Postgres,
TEI) before anyone trusts them. Standing up Docker for every CI run is slower and adds
a class of flakiness (image pulls, health-check races) that has nothing to do with the
correctness of the RAG logic being tested.

## Decision
`libs/clients/src/rag_clients/memory.py` provides in-memory implementations of every
`rag-core` Protocol (`InMemoryVectorStore`, `InMemorySparseIndex`, `InMemoryObjectStore`,
`InMemoryCatalog`, `InMemoryAuditSink`) plus a deterministic, dependency-free
`LocalHashEmbedder`. Each enforces the **same per-tenant namespacing** as its real
counterpart (via `rag_common.tenancy`), so tenant-isolation guarantees are provable
without infrastructure.

Selection is one setting: `RAG_INMEMORY=true` (also `RAG_TEST_REAL_STORES=true` in
the test fixture, see `tests/conftest.py`). The in-memory profile is:
- the **default** for `make test` / CI (`tests/conftest.py`'s `inmemory` fixture),
- **forbidden in `RAG_ENV=prod`** (the factory raises `ConfigError` if both are set),
- **not a substitute** for real-store validation — it is a fast first gate.

The same 24 tests run unmodified against the real stack by exporting
`RAG_TEST_REAL_STORES=true` plus the real backend URLs (see README "Real stores vs
in-memory"). Both runs were confirmed green on 2026-07-02 (Docker validation pass).

## Consequences
- **Positive:** `make test` and CI run in ~4 seconds with zero infra, so unit-level
  feedback is instant.
- **Positive:** because the SAME test bodies run against both profiles, this doubles
  as an adapter-conformance test — divergent behaviour between an in-memory adapter
  and its real counterpart surfaces as a test failure (see "Real-adapter findings"
  below), not a silent bug shipped to a tenant.
- **Real-adapter findings from the Docker validation pass** (fixed in the real
  adapters, Protocols left untouched):
  1. `QdrantVectorStore.search` raised `404 UnexpectedResponse` for a tenant with no
     collection yet; the in-memory twin returned `[]`. Fixed: the adapter now catches
     404 and returns `[]` — an unprovisioned tenant has "no results," not an error.
  2. `OpenSearchSparseIndex.search` raised `NotFoundError` for the same reason; fixed
     identically.
  3. `PostgresCatalog.get_chunks` queried a tenant's schema without calling
     `ensure_schema` first (every other `CatalogAdapter` method does); a
     never-ingested tenant raised an undefined-table error instead of returning `{}`.
     Fixed by adding the same `ensure_schema` call other methods already had.
  4. `MinioObjectStore.get` raised `minio.error.S3Error` on a missing key; the
     in-memory twin raises `KeyError`. Normalised the real adapter to raise `KeyError`
     so callers catch one exception type regardless of backend — documented on the
     `ObjectStore.get` Protocol docstring.
  5. Real adapters (`aiohttp`/`httpx` sessions, SQLAlchemy engines) leaked connections
     across test runs because nothing closed them — invisible with in-memory adapters,
     which hold no real resources. Added `rag_clients.factory.close_all()`, wired into
     the FastAPI app's `lifespan` shutdown and the test fixture's teardown.
- **Negative:** the in-memory profile's `LocalHashEmbedder` is a bag-of-words hash, not
  a semantic model — cosine-similarity behaviour is a coarse proxy for real embeddings.
  Tests that assert *ranking quality* (not just "isolation holds" or "citations exist")
  must run in real-store mode to be meaningful; the golden-eval test currently checks
  citation coverage and abstain-correctness, which hold in both profiles by design.

## Alternatives considered
- **Mock every Protocol with `unittest.mock`** — rejected: mocks don't enforce the
  namespacing contract, so a tenant-isolation bug in real wiring could pass a mocked
  test trivially. The in-memory adapters are real (if simple) implementations of the
  same Protocol, not mocks.
- **Require Docker for all tests, always** — rejected: too slow for the inner dev loop
  and not necessary for testing pure logic (fusion math, chunking boundaries, the
  abstain gate), which already has dedicated unit tests with no I/O at all.
