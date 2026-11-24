# ADR-0003 — Qdrant for Phase-0 Dev; Milvus-at-Scale Decision Deferred

**Status:** Accepted · **Date:** 2026-07-02

## Context
PRD §15 (Open Question 1) and TDD §15 (Open Technical Decision 1) both leave
Milvus-vs-Qdrant as an explicit open decision, to be settled by a benchmark under
CAG-scale load (PB-scale, 20K peak concurrent users). That benchmark has not run.

## Decision
Phase-0 implements `QdrantVectorStore` as the concrete `VectorStore` adapter. Every
service depends on the `VectorStore` Protocol (`libs/rag-core/interfaces.py`), never
on the Qdrant client directly. Isolation is one **collection per tenant**, named via
`rag_common.tenancy.vector_collection`, so a caller can only ever address the
collection derived from its own authenticated `tenant_id`.

The Milvus-at-scale decision **stays open** — this ADR does not resolve it, it only
records that Qdrant is the working default until the benchmark referenced in the
PRD/TDD runs.

## Consequences
- **Positive:** simple single-node local dev/CI story; the Protocol boundary means
  adding `MilvusVectorStore` later is a new adapter file plus a config switch in
  `rag_clients/factory.py` — zero service-layer changes.
- **Positive:** Phase-0's `tests/integration/test_tenant_isolation.py` validates the
  namespacing contract against the real Qdrant server, so the isolation guarantee is
  proven for whichever backend implements the same Protocol correctly.
- **Operational note (found during Docker validation, 2026-07-02):** Qdrant's on-disk
  segment format is **not** guaranteed cross-version-compatible — a v1.12 data volume
  failed to load under v1.18 (`unknown variant "on_disk"`). The compose file pins the
  server image to match the `qdrant-client` library version and documents that
  bumping either requires wiping the dev volume or following Qdrant's official
  migration path. This has no bearing on the Milvus-vs-Qdrant decision itself, but is
  a real constraint on how Qdrant upgrades must be operated.
- **Negative:** if the eventual benchmark favours Milvus for PB-scale sharding, Phase-0
  data does not migrate automatically — re-indexing from source documents (not a
  Qdrant→Milvus data migration) is the expected path, consistent with FR-C8
  (versioned, replayable indices).

## Alternatives considered
- **Milvus from day one** — rejected for Phase-0: heavier operational footprint
  (dependency on etcd + MinIO/S3 + Pulsar/Kafka for its own internals) for no
  Phase-0-scale benefit; better decided by the PRD/TDD's planned benchmark.
- **pgvector** (TDD §12 alternative) — rejected: adequate for small scale but weaker
  ANN performance/feature set (filtering, HNSW tuning) than a dedicated vector DB at
  the scale this product targets.
