"""Tenant-isolation correctness gate (FR-H1/D8) — a first-class test.

Proves that after tenant ALPHA ingests a document, tenant BETA cannot read ALPHA's
data across EVERY store: vector, BM25, object, and metadata catalog. Isolation is
enforced by per-tenant namespacing derived from the explicit ``tenant_id`` at each
adapter (never ambient), so a cross-tenant read is structurally impossible.

Runs on the in-memory profile (same Protocols/namespacing as the live stores).
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_common import tenant_scope
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.integration

ALPHA, BETA = "alpha-bank", "beta-gov"
_DOC = "Acme sovereign cloud uptime target is 99.5 percent baseline. Secret alpha ledger balance."


def _envelope(tenant: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id="doc1",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64=base64.b64encode(_DOC.encode()).decode(),
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


async def test_beta_cannot_read_alpha_across_all_stores(inmemory: None) -> None:
    from svc_control_plane import pipeline

    from rag_clients import factory

    # ALPHA ingests; BETA ingests nothing.
    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    vs = factory.get_vector_store()
    sparse = factory.get_sparse_index()
    catalog = factory.get_catalog()
    objects = factory.get_object_store()
    embedder = factory.get_embedder()

    [qvec] = await embedder.embed(["uptime target"])

    # 1) Vector store: BETA's collection is empty; ALPHA's is not.
    assert await vs.search(BETA, qvec, top_k=5) == []
    assert await vs.search(ALPHA, qvec, top_k=5) != []

    # 2) BM25: BETA sees nothing.
    assert await sparse.search(BETA, "uptime", top_k=5) == []
    assert await sparse.search(ALPHA, "uptime", top_k=5) != []

    # 3) Catalog: BETA cannot resolve ALPHA's chunk ids (cross-tenant hydration denied).
    alpha_ids = [cid for cid, _ in await vs.search(ALPHA, qvec, top_k=5)]
    assert await catalog.get_chunks(BETA, alpha_ids) == {}
    assert set(await catalog.get_chunks(ALPHA, alpha_ids)) == set(alpha_ids)

    # 4) Object store: BETA cannot read ALPHA's raw bytes.
    with pytest.raises(KeyError):
        await objects.get(BETA, "raw/doc1")
    assert await objects.get(ALPHA, "raw/doc1")  # ALPHA can

    # 5) End-to-end: BETA's grounded query returns no ALPHA content (abstains).
    from svc_control_plane import wiring

    with tenant_scope(BETA):
        result = await wiring.retrieval_service().retrieve(
            RetrievalQuery(tenant_id=BETA, text="uptime target", top_k=5)
        )
    assert result.spans == []
    assert "alpha ledger" not in " ".join(s.text for s in result.spans)


async def test_beta_cannot_read_alpha_via_multivector_search_methods(inmemory: None) -> None:
    """Slice-2b: the named dense/sparse/colbert search methods are isolated per tenant exactly like
    the legacy `search` — a caller only ever touches the collection its explicit tenant_id names."""
    from rag_clients import factory
    from rag_schemas.chunk import Span

    vs = factory.get_vector_store()
    emb = factory.get_embedder()
    # `embed_multi` is the MultiVectorEmbedder capability, and only a multivector-capable
    # embedder has it: the in-memory LocalHashEmbedder and BgeM3Embedder (`bge_m3_multi`).
    # Under a single-vector real-store profile the factory hands back a TeiEmbedder, which
    # correctly has no such method — so there is no multivector path to isolate and the
    # assertion below has nothing to say. Skip rather than fail: this used to raise
    # AttributeError in the real-store pass and read like an isolation defect, which it is not.
    if not hasattr(emb, "embed_multi"):
        pytest.skip(f"{type(emb).__name__} is single-vector; the 2b multivector legs do not apply")
    chunk = __import__("rag_schemas", fromlist=["Chunk"]).Chunk(
        chunk_id="a1", doc_id="d1", tenant_id=ALPHA, text="alpha ledger balance secret", span=Span(start=0, end=5)
    )
    [out] = await emb.embed_multi([chunk.text])
    await vs.upsert_multi(ALPHA, [chunk], [out])

    # ALPHA sees its own vectors on every 2b leg; BETA (never upserted) sees nothing.
    assert await vs.search_dense(ALPHA, out.dense, 5) != []
    assert await vs.search_dense(BETA, out.dense, 5) == []
    assert await vs.search_sparse(ALPHA, out.sparse, 5) != []
    assert await vs.search_sparse(BETA, out.sparse, 5) == []
    assert await vs.search_colbert(ALPHA, out.dense, out.colbert, 5, prefetch=10) != []
    assert await vs.search_colbert(BETA, out.dense, out.colbert, 5, prefetch=10) == []
