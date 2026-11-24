"""Per-level reconciliation against LIVE Qdrant, OpenSearch and Postgres (Phase 6 acceptance).

ADR-0024: v2 chunks are indexed into the tenant's OWN namespace, so the plan's "counts per level
reconcile across Qdrant, OpenSearch, Postgres" is read within the tenant namespace, grouped by
`level` — and `count_by_level` is the gate. The twins prove the gate's logic
(`libs/clients/tests/test_count_by_level.py`); only real servers prove the three things that
decide whether it can be trusted on a real corpus:

1. **Each store's own counting path.** Qdrant counts with a payload filter and derives the
   level-less bucket as a REMAINDER; OpenSearch uses one `terms` aggregation with a `missing`
   bucket; Postgres a `GROUP BY` over a `COALESCE` of the column and the serialised record. Three
   different mechanisms, one answer — a twin cannot show they agree.
2. **The payload/field is actually written.** The level has to survive the real Qdrant payload and
   the real OpenSearch `_source` + mapping, not just a dict in a twin.
3. **An index that predates the `level` field still counts correctly** — the case every live
   deployment is in today.

Marked ``integration``. From the host (the default URLs name the in-compose hostnames):

    RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" \\
    RAG_QDRANT_URL="http://localhost:6333" RAG_OPENSEARCH_URL="http://localhost:9200" \\
    uv run pytest -m integration tests/integration/test_v2_index_reconcile.py

Every namespace is a throwaway ``zzrecon-*`` dropped in a ``finally`` — no existing tenant's
collection, index or schema is read or written.
"""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_common import pg_schema
from rag_core.layout import LayoutTree, parse_layout
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkLevel, ChunkStrategy, SourceKind

pytestmark = pytest.mark.integration

_DOC_ID = "fixture-reconcile-live.pdf"

_MARKUP = """
<h1>Attachment-1</h1>
<h2>13. Video Wall Technical Requirements</h2>
<p>The controller shall comply with IEC 62443-4-2.</p>
<p>Certificates shall be furnished with the bid.</p>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Brightness</td><td>700 nits</td></tr>
  <tr><td>Contrast ratio</td><td>4000:1</td></tr>
</table>
"""


def _reconcile() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "reconcile_v2_index.py"
    spec = importlib.util.spec_from_file_location("reconcile_v2_index", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconcile


def _tree() -> LayoutTree:
    overlay = {
        "schema_version": 2,
        "extractor": "chandra",
        "content_hash": "",
        "pages": [{"page_no": 1, "markup": _MARKUP, "markup_format": "html", "degradations": []}],
    }
    return parse_layout(overlay, _DOC_ID)


def _doc(tenant: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id=_DOC_ID,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


@pytest.fixture
async def live() -> AsyncIterator[tuple[Any, str]]:
    """An IndexingService over the three REAL adapters plus a throwaway tenant; skips unless all
    three stores are reachable. Collection, index and schema are all dropped in a ``finally``."""
    from sqlalchemy import text
    from svc_indexing import IndexingService

    from rag_clients.memory import LocalHashEmbedder
    from rag_clients.opensearch_index import OpenSearchSparseIndex
    from rag_clients.postgres_catalog import PostgresCatalog
    from rag_clients.qdrant_store import QdrantVectorStore

    tenant = f"zzrecon-{uuid.uuid4().hex[:10]}"
    vs, sparse, catalog = QdrantVectorStore(), OpenSearchSparseIndex(), PostgresCatalog()
    try:
        await vs._client.get_collections()
        await sparse._client.info()
        async with catalog._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        for store in (vs, sparse, catalog):
            await store.aclose()
        pytest.skip(f"a live store is not reachable ({type(exc).__name__}); integration test skipped")

    svc = IndexingService(LocalHashEmbedder(), vs, sparse, catalog)
    try:
        yield svc, tenant
    finally:
        try:
            await vs.delete_tenant(tenant)
            await sparse.delete_tenant(tenant)
            async with catalog._engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{pg_schema(tenant)}" CASCADE'))
        finally:
            for store in (vs, sparse, catalog):
                await store.aclose()


async def _index(svc: Any, doc: DocumentEnvelope, text_body: str, **kwargs: Any) -> Any:
    """Register, then index — the order the real pipeline uses. `record_chunks` deliberately does
    not create the schema itself (see ADR-0023's caller list); `register_document` is what does."""
    await svc._catalog.register_document(doc)
    return await svc.index_document(doc, text_body, **kwargs)


async def _counts(svc: Any, tenant: str) -> dict[str, dict[str | None, int]]:
    return {
        "catalog": await svc._catalog.count_by_level(tenant),
        "vector": await svc._vs.count_by_level(tenant),
        "sparse": await svc._sparse.count_by_level(tenant),
    }


async def test_a_v2_document_reconciles_across_the_three_real_stores(live: tuple[Any, str]) -> None:
    svc, tenant = live
    result = await _index(svc, _doc(tenant), "irrelevant", layout=_tree(), mask=lambda t: t)
    assert result.chunk_count > 0

    counts = await _counts(svc, tenant)
    assert _reconcile()(counts)["reconciled"], counts
    # Both of the levels Phase 5 emits are present, in all three stores, and nothing else is.
    assert set(counts["catalog"]) == {ChunkLevel.CHUNK.value, ChunkLevel.ROW.value}, counts
    assert sum(counts["catalog"].values()) == result.chunk_count


async def test_a_v1_document_lands_wholly_in_the_level_less_bucket(live: tuple[Any, str]) -> None:
    """The remainder (Qdrant) and the `missing` bucket (OpenSearch) are different mechanisms from
    the `GROUP BY` the catalog uses; this is where they are shown to agree on real servers. It is
    also the state of every live index today — created before the `level` field existed."""
    svc, tenant = live
    text_body = "1. Scope. The contractor shall supply the system. 2. Delivery. Within ninety days."
    result = await _index(svc, _doc(tenant), text_body, strategy=ChunkStrategy.CLAUSE)

    counts = await _counts(svc, tenant)
    assert _reconcile()(counts)["reconciled"], counts
    assert list(counts["catalog"]) == [None]
    assert counts["catalog"][None] == result.chunk_count


async def test_a_mixed_corpus_reconciles_level_by_level(live: tuple[Any, str]) -> None:
    """A KB mid-conversion holds v1 and v2 chunks in ONE collection and index — the consequence
    ADR-0024 accepts. The gate has to stay readable in exactly that state."""
    svc, tenant = live
    v1 = await _index(
        svc,
        _doc(tenant).model_copy(update={"doc_id": "legacy.txt"}),
        "1. Scope. The contractor shall supply the system. 2. Delivery. Within ninety days.",
        strategy=ChunkStrategy.CLAUSE,
    )
    v2 = await _index(svc, _doc(tenant), "irrelevant", layout=_tree(), mask=lambda t: t)

    counts = await _counts(svc, tenant)
    assert _reconcile()(counts)["reconciled"], counts
    assert counts["catalog"][None] == v1.chunk_count
    assert sum(n for lvl, n in counts["catalog"].items() if lvl is not None) == v2.chunk_count
