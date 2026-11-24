"""Parent/child indexing partition (FR-C2, Slice 3): parents are CATALOG-ONLY.

The three highest-risk storage invariants, pinned end-to-end through the in-memory stores:
  * a parent record reaches the catalog but is NEVER embedded (dense) or BM25-indexed (sparse),
    so it can never be retrieved/ranked/cited as if it were a child;
  * IndexResult.chunk_ids INCLUDES the parent ids, so the write-then-swap reindex
    ("stale = old - fresh") does not purge the parents it just wrote;
  * children carry metadata["parent_id"] pointing at a real parent record.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_common import sparse_index
from rag_core.interfaces import CatalogAdapter, Embedder, SparseIndex, VectorStore
from rag_core.parent_child import META_CHUNK_ROLE, META_PARENT_ID, PARENT_ROLE, is_parent
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind

_HEADER = "| Metric | Value |"
_ROWS = "\n".join(f"| Metric {i} | {i * 100} |" for i in range(5))
_PROSE = (
    "2.0 Metrics. This section records the platform's service metrics in detail. "
    "Every metric below is tracked continuously by the operations team and reviewed weekly. "
    "The following table lists each metric and its committed value for the current period."
)
_TEXT = f"{_PROSE}\n{_HEADER}\n| --- | --- |\n{_ROWS}\n"


def _doc(tenant: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id="metrics",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 9, 18, tzinfo=UTC),
    )


def _stores() -> tuple[Embedder, VectorStore, SparseIndex, CatalogAdapter]:
    from rag_clients.memory import (
        InMemoryCatalog,
        InMemorySparseIndex,
        InMemoryVectorStore,
        LocalHashEmbedder,
    )

    return LocalHashEmbedder(), InMemoryVectorStore(), InMemorySparseIndex(), InMemoryCatalog()


async def test_parents_are_catalog_only_and_children_are_searchable(inmemory: None) -> None:
    from svc_indexing import IndexingService

    tenant = "acme"
    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog)
    result = await svc.index_document(_doc(tenant), _TEXT, strategy=ChunkStrategy.PARENT_CHILD, child_max_chars=120)

    cataloged = await catalog.document_chunks(tenant, "metrics")
    parents = [c for c in cataloged if is_parent(c)]
    children = [c for c in cataloged if not is_parent(c)]
    assert parents, "no parent record was written to the catalog"
    assert children, "no children were written"

    parent_ids = {p.chunk_id for p in parents}
    dense_ids = set(vs._c[vs._ns(tenant)].keys())  # type: ignore[attr-defined]
    sparse_ids = set(sparse._i[sparse_index(tenant)].keys())  # type: ignore[attr-defined]

    # A parent is present in the catalog but NEVER in the dense or sparse index.
    assert not (parent_ids & dense_ids), "a parent leaked into the vector index"
    assert not (parent_ids & sparse_ids), "a parent leaked into the sparse (BM25) index"
    # Every child IS in both search indexes.
    child_ids = {c.chunk_id for c in children}
    assert child_ids <= dense_ids and child_ids <= sparse_ids

    # chunk_ids includes BOTH children and parents.
    assert parent_ids <= set(result.chunk_ids)
    assert child_ids <= set(result.chunk_ids)
    assert result.chunk_count == len(children)  # searchable count excludes parents

    # Children point at real parent records; parents are marked.
    stamped = [c for c in children if c.metadata.get(META_PARENT_ID)]
    assert stamped, "no child carried a parent_id"
    assert all(c.metadata[META_PARENT_ID] in parent_ids for c in stamped)
    assert all(p.metadata[META_CHUNK_ROLE] == PARENT_ROLE for p in parents)


async def test_reindex_swap_does_not_purge_parents(inmemory: None) -> None:
    """Content-derived parent ids are stable across a re-chunk, and chunk_ids includes them, so the
    write-then-swap ``stale = old - fresh`` computes to empty for parents — they survive reindex."""
    from svc_indexing import IndexingService

    tenant = "acme"
    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog)

    first = await svc.index_document(_doc(tenant), _TEXT, strategy=ChunkStrategy.PARENT_CHILD, child_max_chars=120)
    old_ids = {c.chunk_id for c in await catalog.document_chunks(tenant, "metrics")}
    parent_ids = {c.chunk_id for c in await catalog.document_chunks(tenant, "metrics") if is_parent(c)}
    assert parent_ids

    # Re-chunk the SAME text (a reindex). Fresh ids include the parents again.
    second = await svc.index_document(_doc(tenant), _TEXT, strategy=ChunkStrategy.PARENT_CHILD, child_max_chars=120)
    fresh = set(second.chunk_ids)
    stale = old_ids - fresh  # what the swap would delete

    assert not (parent_ids & stale), "the reindex swap would purge parents"
    assert set(first.chunk_ids) == fresh, "ids drifted across an identical re-chunk"
    # Parents still queryable in the catalog after the (no-op) swap.
    still_there = {c.chunk_id for c in await catalog.document_chunks(tenant, "metrics") if is_parent(c)}
    assert still_there == parent_ids
