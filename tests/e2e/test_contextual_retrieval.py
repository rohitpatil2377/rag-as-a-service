"""E2E: Contextual Retrieval (FR-C2) — deterministic, in-memory.

Proves the core C2 invariant: with the flag ON, the BM25 index (and embeddings) carry
the CONTEXTUALIZED text, but the catalog stores the ORIGINAL chunk — so citations never
leak the context prefix. With the flag OFF, indexing is unchanged. Retrieval/grounding
contracts are untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_common import sparse_index
from rag_core import MetadataContextualizer
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.e2e

_TEXT = "Uptime target is 99.5 percent baseline. Latency P95 is 3 seconds or less."


def _doc(tenant: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id="brief",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64="",
        metadata={"title": "Sovereign Platform Brief"},
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


def _stores():  # type: ignore[no-untyped-def]
    from rag_clients.memory import (
        InMemoryCatalog,
        InMemorySparseIndex,
        InMemoryVectorStore,
        LocalHashEmbedder,
    )

    return LocalHashEmbedder(), InMemoryVectorStore(), InMemorySparseIndex(), InMemoryCatalog()


async def test_contextual_on_indexes_context_but_cites_original(inmemory: None) -> None:
    from svc_indexing import IndexingService

    tenant = "acme"
    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog, MetadataContextualizer(), contextual_enabled=True)
    await svc.index_document(_doc(tenant), _TEXT, strategy="clause")  # type: ignore[arg-type]

    # BM25 index text carries the context prefix (contextualized embedding input)...
    idx = sparse._i[sparse_index(tenant)]  # type: ignore[attr-defined]
    assert any("[context]" in t and "Sovereign Platform Brief" in t for t, _meta, _doc_id, _level in idx.values())

    # ...but the catalog stores the ORIGINAL chunk (no context leak into citations).
    chunk_ids = list(idx.keys())
    stored = await catalog.get_chunks(tenant, chunk_ids)
    assert stored and all("[context]" not in c.text for c in stored.values())

    # A query on a context-only token (document title) now finds the chunk (C2 benefit).
    hits = await sparse.search(tenant, "sovereign brief", top_k=5)
    assert hits


async def test_contextual_off_is_unchanged(inmemory: None) -> None:
    from svc_indexing import IndexingService

    tenant = "beta"
    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog, MetadataContextualizer(), contextual_enabled=False)
    await svc.index_document(_doc(tenant), _TEXT, strategy="clause")  # type: ignore[arg-type]

    idx = sparse._i[sparse_index(tenant)]  # type: ignore[attr-defined]
    assert all("[context]" not in t for t, _meta, _doc_id, _level in idx.values())  # no context when flag off
