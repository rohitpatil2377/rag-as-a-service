# `rag_clients`

The concrete adapters behind every Protocol in [`rag_core`](../rag-core/README.md) — and, just as
importantly, their **in-memory twins**.

42 modules, 44 test files.

## Every seam has two implementations

| Protocol | Real adapter | In-memory twin |
|---|---|---|
| `VectorStore` | `QdrantVectorStore` | `memory.py` |
| `SparseIndex` | `OpenSearchSparseIndex` | `memory.py` |
| `ObjectStore` | `MinioObjectStore` (S3; the dev stack serves it from SeaweedFS) | `memory.py` |
| `CatalogAdapter`, `AuditSink` | `PostgresCatalog`, `PostgresAuditSink` | `memory.py` |
| `Embedder` | `TeiEmbedder`, `OpenAIEmbedder`, `BgeM3Embedder` | `LocalHashEmbedder` |
| `Reranker` | `TeiReranker`, `OpenAICompatReranker` | `LexicalReranker` |
| `LLMGateway` | `OpenAICompatGateway` (serves llama.cpp and vLLM alike) | `ExtractiveDevGateway` |

The twins carry the **same per-tenant namespacing** as the real thing, which is what lets the same
test bodies run against both profiles — a real adapter that drifts from its Protocol contract fails
a test rather than failing in production. See [ADR-0004](../../docs/adr/0004-inmemory-adapters-for-fast-ci.md).

## `factory.py` is the only place a backend is chosen

Services never import a concrete adapter. Each seam has a `get_*` and a `build_*`; config decides
which implementation comes back. Two consequences:

- Swapping Qdrant for something else is a change in one file.
- In-memory is **forbidden when `RAG_ENV=prod`** — the factory raises rather than silently serving
  a toy store in production.

## Migrations

`migrations/` holds the Alembic revisions (`uv run alembic -c libs/clients/alembic.ini upgrade head`,
or `make migrate`). Schema-per-tenant: see [ADR-0014](../../docs/adr/0014-schema-migrations.md) and
[ADR-0023](../../docs/adr/0023-ensure-schema-ddl-memo.md). Lazy `ensure()` helpers use
`CREATE TABLE IF NOT EXISTS`, which **no-ops on an existing table and will not add a new column** —
adding one needs an explicit `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. That exact trap has bitten
this codebase more than once.
