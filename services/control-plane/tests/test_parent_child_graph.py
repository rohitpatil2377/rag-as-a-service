"""Regression (FR-C2): a PARENT context record must never enter the knowledge graph.

If `_index_graph` read parents, their ids would land in graph_nodes and then be fetched by
`_graph_augment`/`_global_augment` and MERGED into retrieval.spans as ranked, CITABLE spans — an
1800-char parent blob answering in place of the precise child. This pins that parents are filtered
out of graph indexing (the one document_chunks consumer that was missed in the first pass).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from svc_control_plane import pipeline, wiring

from rag_common import get_settings
from rag_core.graph import GraphTriple
from rag_schemas import Chunk


@pytest.fixture
def _graph_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chunk(cid: str, *, role: str | None = None, parent_id: str | None = None) -> Chunk:
    meta: dict[str, str] = {}
    if role:
        meta["chunk_role"] = role
    if parent_id:
        meta["parent_id"] = parent_id
    return Chunk(chunk_id=cid, doc_id="d", tenant_id="kb1", text=f"Acme Corporation ships {cid}.", metadata=meta)


async def test_parent_ids_never_enter_the_graph(_graph_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_clients.graph_store import InMemoryGraphStore

    child_a = _chunk("d::c1", parent_id="d::p")
    child_b = _chunk("d::c2", parent_id="d::p")
    parent = _chunk("d::p", role="parent")
    all_chunks = [child_a, child_b, parent]

    class _Catalog:
        async def document_chunks(self, _tenant: str, _doc: str) -> list[Chunk]:
            return all_chunks

    class _Extractor:
        async def extract(self, _text: str) -> GraphTriple:
            # Every chunk that REACHES the extractor yields a node — so if a parent were read, its
            # id would appear in the graph. It must not.
            return GraphTriple(entities=["Acme Corporation"], edges=[])

    store = InMemoryGraphStore()

    async def _resolver(_kb: str) -> _Extractor:
        return _Extractor()

    def _get_catalog() -> _Catalog:
        return _Catalog()

    def _get_store() -> InMemoryGraphStore:
        return store

    monkeypatch.setattr(pipeline, "get_catalog", _get_catalog)
    monkeypatch.setattr(pipeline, "get_graph_store", _get_store)
    monkeypatch.setattr(wiring, "resolve_graph_extractor", _resolver)

    entities, _edges = await pipeline._index_graph("kb1", "d")
    assert entities > 0, "the extractor should have produced nodes for the children"

    # Every chunk id referenced by any graph node must be a CHILD — the parent id is absent.
    graphed_ids = set().union(*(cids for _display, cids in store._nodes.get("kb1", {}).values()))
    assert "d::p" not in graphed_ids, "a parent id leaked into the knowledge graph"
    assert {"d::c1", "d::c2"} <= graphed_ids, "children should have been graphed"
