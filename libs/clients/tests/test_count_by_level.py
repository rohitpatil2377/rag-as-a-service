"""Per-level reconciliation across the three stores (RAG v2 Phase 6 acceptance; ADR-0024).

v2 chunks are indexed into the tenant's OWN collection/index/schema — there is no parallel `_v2`
namespace — so the plan's "counts per level reconcile across Qdrant, OpenSearch, Postgres" is read
within the tenant namespace, grouped by `level`. That makes `count_by_level` the gate, and these
tests pin the two things a gate needs: that the three stores agree when they should, and that it
still reports agreement in the one case where the stores legitimately hold DIFFERENT numbers of
rows (FR-C2 parents, which are catalog-only by design).

The twins are what `make test` can reach; `tests/integration/test_v2_index_reconcile.py` runs the
same assertions against live Qdrant, OpenSearch and Postgres, because a reconciliation that only
ever ran against the twins would be proving the twins.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_core.layout import LayoutTree, parse_layout
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkLevel, ChunkStrategy, SourceKind

_DOC_ID = "fixture-reconcile.pdf"
_TENANT = "zzreconcile"

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


def _reconcile_script() -> Any:
    """The script is the acceptance gate an operator runs, so the comparison it makes is tested —
    not a second copy of the comparison written for the test."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "reconcile_v2_index.py"
    spec = importlib.util.spec_from_file_location("reconcile_v2_index", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree() -> LayoutTree:
    overlay = {
        "schema_version": 2,
        "extractor": "chandra",
        "content_hash": "",
        "pages": [{"page_no": 1, "markup": _MARKUP, "markup_format": "html", "degradations": []}],
    }
    return parse_layout(overlay, _DOC_ID)


def _doc(doc_id: str = _DOC_ID) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=_TENANT,
        doc_id=doc_id,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


def _stores() -> tuple[Any, Any, Any, Any]:
    from rag_clients.memory import (
        InMemoryCatalog,
        InMemorySparseIndex,
        InMemoryVectorStore,
        LocalHashEmbedder,
    )

    return LocalHashEmbedder(), InMemoryVectorStore(), InMemorySparseIndex(), InMemoryCatalog()


async def _counts(vs: Any, sparse: Any, catalog: Any) -> dict[str, dict[str | None, int]]:
    return {
        "catalog": await catalog.count_by_level(_TENANT),
        "vector": await vs.count_by_level(_TENANT),
        "sparse": await sparse.count_by_level(_TENANT),
    }


async def test_the_three_stores_agree_per_level_on_a_v2_document(inmemory: None) -> None:
    from svc_indexing import IndexingService

    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog)
    await svc.index_document(_doc(), "irrelevant", layout=_tree(), mask=lambda t: t)

    counts = await _counts(vs, sparse, catalog)
    report = _reconcile_script().reconcile(counts)
    assert report["reconciled"], counts
    # Phase 5 emits exactly two levels, and both must be present or this is not reconciling v2.
    assert set(counts["catalog"]) == {ChunkLevel.CHUNK.value, ChunkLevel.ROW.value}, counts["catalog"]
    assert all(n > 0 for n in counts["catalog"].values())


async def test_a_v1_document_reconciles_in_the_level_less_bucket(inmemory: None) -> None:
    """The gate has to work on the corpus as it is today — every chunk level-less — or it cannot be
    run before a conversion to tell a real discrepancy from an unconverted document."""
    from svc_indexing import IndexingService

    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog)
    text = "1. Scope. The contractor shall supply the system. 2. Delivery. Within ninety days."
    await svc.index_document(_doc(), text, strategy=ChunkStrategy.CLAUSE)

    counts = await _counts(vs, sparse, catalog)
    assert list(counts["catalog"]) == [None]
    assert _reconcile_script().reconcile(counts)["reconciled"], counts


async def test_parent_records_do_not_break_a_correct_parent_child_kb(inmemory: None) -> None:
    """A parent is a context-only record that is NEVER embedded or BM25-indexed, so the catalog
    genuinely holds more rows than the indexes. Counting parents would make this KB — which is
    exactly right — fail the gate, and the failure would look like real index loss."""
    from svc_indexing import IndexingService

    from rag_core.parent_child import is_parent

    embedder, vs, sparse, catalog = _stores()
    svc = IndexingService(embedder, vs, sparse, catalog)
    prose = (
        "2.0 Metrics. This section records the platform's service metrics in detail. "
        "Every metric below is tracked continuously by the operations team and reviewed weekly."
    )
    rows = "\n".join(f"| Metric {i} | {i * 100} |" for i in range(5))
    text = f"{prose}\n| Metric | Value |\n| --- | --- |\n{rows}\n"
    result = await svc.index_document(_doc(), text, strategy=ChunkStrategy.PARENT_CHILD, child_max_chars=120)

    stored = await catalog.get_chunks(_TENANT, result.chunk_ids)
    assert any(is_parent(c) for c in stored.values()), "fixture produced no parent to test with"
    counts = await _counts(vs, sparse, catalog)
    assert _reconcile_script().reconcile(counts)["reconciled"], counts


async def test_a_namespace_that_was_never_written_counts_nothing(inmemory: None) -> None:
    """ "Nothing indexed" is not an error — the gate must be runnable against a fresh KB."""
    _embedder, vs, sparse, catalog = _stores()
    assert await vs.count_by_level("zzempty") == {}
    assert await sparse.count_by_level("zzempty") == {}
    assert await catalog.count_by_level("zzempty") == {}


def test_a_store_that_is_missing_rows_is_reported_not_rounded_off() -> None:
    """The gate's whole job. A level where one store holds fewer units — an interrupted ingest, a
    half-finished swap — must read as a MISMATCH on that level, not as a total that happens to add
    up elsewhere."""
    reconcile = _reconcile_script().reconcile
    counts = {"catalog": {"chunk": 10, "row": 4}, "vector": {"chunk": 10, "row": 4}, "sparse": {"chunk": 7, "row": 4}}
    report = reconcile(counts)
    assert not report["reconciled"]
    assert [r["level"] for r in report["levels"] if not r["agrees"]] == ["chunk"]
    # ...and a level present in one store and absent from another is a mismatch, not an omission.
    report = reconcile({"catalog": {"row": 4}, "vector": {}, "sparse": {"row": 4}})
    assert not report["reconciled"]
    assert [r["level"] for r in report["levels"] if not r["agrees"]] == ["row"]


# --- the `level` mapping is not optional (the fault the live run found) -------------------------


def test_a_fresh_index_always_maps_level_as_a_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenSearch CANNOT RETYPE A FIELD. A v2 chunk reaching an index with no `level` mapping is
    dynamically typed as analysed text, and then the per-level aggregation fails outright with
    "Text fields are not optimised for operations that require per-document field data" — observed
    live against this deployment's OpenSearch when the mapping was flag-gated — and every `term`
    filter under-matches, for the life of that index.

    So the mapping is UNCONDITIONAL, for the same reason the v2 Postgres DDL is: an index shape
    that depends on which env var was set when it happened to be created is schema drift, and here
    the drift is unrecoverable without dropping the index."""
    from rag_clients.opensearch_index import _index_mapping
    from rag_common.config import get_settings

    # Both flags are varied: `chem` IS gated, so this also pins that trimming it never takes
    # `level` with it.
    for v2 in ("true", "false"):
        for chem in ("true", "false"):
            monkeypatch.setenv("RAG_V2_INGEST", v2)
            monkeypatch.setenv("RAG_CHEM_NORMALIZATION_ENABLED", chem)
            get_settings.cache_clear()
            props = _index_mapping()["mappings"]["properties"]
            assert props["level"] == {"type": "keyword"}, f"v2_ingest={v2} chem={chem}"
            assert ("chem" in props) is (chem == "true")
    get_settings.cache_clear()


def test_both_stores_carry_the_level_only_when_the_chunk_has_one() -> None:
    """The level has to be IN the index for either store to count it — and absent from a v1 record,
    so turning v2 on cannot rewrite every point and document in a live namespace."""
    from rag_clients.opensearch_index import OpenSearchSparseIndex
    from rag_clients.qdrant_store import _payload
    from rag_schemas import Chunk

    v1 = Chunk(chunk_id="c1", doc_id="d", tenant_id=_TENANT, text="plain")
    v2 = v1.model_copy(update={"chunk_id": "c2", "level": ChunkLevel.ROW})

    assert "level" not in _payload(v1)
    assert _payload(v2)["level"] == ChunkLevel.ROW.value
    assert "level" not in OpenSearchSparseIndex._source(v1)
    assert OpenSearchSparseIndex._source(v2)["level"] == ChunkLevel.ROW.value
