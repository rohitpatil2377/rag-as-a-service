"""Declared relations reach the knowledge graph (RAG v2 Phase 6 wiring).

`rag_core` decides WHICH chunk proves a relation; this is the other half — that the pipeline
reads the rows, writes directed labeled edges into the graph store the GraphRAG retrieval path
reads, and inherits that stage's gates. Exercised through the in-memory twins, which is why the
twin keeps a relations map at all: nothing in this build WRITES `doc_relations` or
`sections.amends` (Phases 3-4 do, and the plan's rollout order puts them after this one), so a
test fills them directly or the whole wiring would be untested until those phases land.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rag_core.graph import DeclaredRelation
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import SourceKind

_KB = "aegcl"
_DOC = "corrigendum-v"
_QUOTE = "The MAAT requirement is revised from Rs. 35 Crores to Rs. 19.44 Crores."


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="c1", doc_id=_DOC, tenant_id=_KB, text="1. Scope. This corrigendum applies to the tender."),
        Chunk(chunk_id="c2", doc_id=_DOC, tenant_id=_KB, text=f"7. Financial qualification. {_QUOTE}"),
    ]


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=_KB,
        doc_id=_DOC,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


async def _seeded(monkeypatch: pytest.MonkeyPatch, relations: list[DeclaredRelation]) -> tuple[Any, int]:
    """Run the seeding with `relations` declared, returning the graph store and the edge count.

    Patches the two names the pipeline holds — the factory's relation read and the graph-store
    getter — rather than the factory's own `lru_cache`d getters, which the `inmemory` fixture owns
    and clears."""
    from svc_control_plane import pipeline

    from rag_clients.graph_store import InMemoryGraphStore

    store = InMemoryGraphStore()

    async def _relations(kb_id: str, doc_id: str) -> list[DeclaredRelation]:
        assert (kb_id, doc_id) == (_KB, _DOC)
        return list(relations)

    monkeypatch.setattr(pipeline, "document_relations", _relations)
    monkeypatch.setattr(pipeline, "get_graph_store", lambda: store)
    written = await pipeline._seed_declared_relations(_KB, _DOC, _chunks())
    return store, written


async def test_a_declared_amendment_becomes_a_directed_labeled_edge(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, written = await _seeded(
        monkeypatch,
        [
            DeclaredRelation(
                rel_type="amends", source_ref="Corrigendum-V", target_ref="Corrigendum-II", evidence_quote=_QUOTE
            )
        ],
    )

    assert written == 1
    # Direction and label survive: `amends` points FROM the amending document TO the amended one,
    # which is the whole content of the edge.
    [(pair, relation)] = store._edge_relations[_KB].items()
    assert relation == "amends"
    assert set(pair) == {"corrigendum-v", "corrigendum-ii"}
    # ...and the edge is anchored to the chunk that proves it, so a walk reaching it lands on
    # citable text rather than on a node with no evidence.
    assert store._edges[_KB][pair] == {"c2"}
    assert await store.linked_chunks(_KB, ["Corrigendum-II"]) == ["c2"]


async def test_a_relation_no_chunk_proves_writes_no_edge(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    store, written = await _seeded(
        monkeypatch,
        [
            DeclaredRelation(
                rel_type="supersedes",
                source_ref="Corrigendum-V",
                target_ref="Corrigendum-IV",
                evidence_quote="A sentence that appears in no chunk of this document.",
            )
        ],
    )

    assert written == 0
    assert not store._edges.get(_KB)


async def test_several_relations_on_one_chunk_are_written_together(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrigendum commonly amends one document and references another in the same sentence. Both
    edges hang off that one chunk, and both node names must be registered or the edge endpoints
    would have no node to walk from."""
    store, written = await _seeded(
        monkeypatch,
        [
            DeclaredRelation(
                rel_type="amends", source_ref="Corrigendum-V", target_ref="Corrigendum-II", evidence_quote=_QUOTE
            ),
            DeclaredRelation(
                rel_type="references", source_ref="Corrigendum-V", target_ref="NIT AEGCL/T-161", evidence_quote=_QUOTE
            ),
        ],
    )

    assert written == 2
    assert len(store._edge_relations[_KB]) == 2
    assert sorted(store._nodes[_KB]) == ["corrigendum-ii", "corrigendum-v", "nit aegcl/t-161"]


async def test_a_document_that_declares_nothing_writes_nothing(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every document in this build, until Phases 3-4 land."""
    store, written = await _seeded(monkeypatch, [])
    assert written == 0
    assert not store._nodes.get(_KB)


async def test_the_graph_stage_gate_covers_the_seeding(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The seeding runs INSIDE `_index_graph`, so a KB with GraphRAG off must not get relation
    edges either — the edges live in the store only GraphRAG retrieval reads, so writing them for
    a graph-disabled KB would be invisible state with no consumer."""
    from svc_control_plane import pipeline

    from rag_common import get_settings

    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "false")
    get_settings.cache_clear()
    try:
        calls: list[str] = []
        monkeypatch.setattr(
            pipeline,
            "_seed_declared_relations",
            lambda *a, **k: calls.append("called") or 0,  # type: ignore[func-returns-value]
        )
        assert await pipeline._index_graph(_KB, _DOC) == (0, 0)
        assert calls == [], "the seeding ran despite GraphRAG being off"
    finally:
        get_settings.cache_clear()


async def test_the_factory_seam_reads_the_catalogs_relations(inmemory: None) -> None:
    """The other half of the wiring: `factory.document_relations` is a v2-only catalog operation
    that is NOT on the `CatalogAdapter` Protocol, so the factory is the one place that knows which
    adapter is in play (the same shape as `persist_document_structure`). Exercised through the real
    seam so the twin's own read is covered rather than assumed."""
    from rag_clients import factory

    catalog = factory.get_catalog()
    declared = DeclaredRelation(
        rel_type="supersedes", source_ref="Corrigendum-V", target_ref="Corrigendum-IV", evidence_quote=_QUOTE
    )
    catalog._relations[_KB][_DOC] = [declared]  # type: ignore[attr-defined]
    try:
        assert await factory.document_relations(_KB, _DOC) == [declared]
        assert await factory.document_relations(_KB, "another-doc") == []
    finally:
        catalog._relations[_KB].pop(_DOC, None)  # type: ignore[attr-defined]


async def test_an_unknown_catalog_backend_is_refused_rather_than_answering_nothing(inmemory: None) -> None:
    """Every document declares nothing today, so "no relations" is the correct answer everywhere —
    which is exactly why a backend that CANNOT answer must not be allowed to say the same thing.
    It fails closed, and `_index_graph` turns that into a logged graph-stage failure, never a
    failed ingest.

    Swapped by assignment rather than monkeypatch, matching
    `test_layout_structure_persist.test_an_unknown_catalog_adapter_fails_closed`: the getter is
    `lru_cache`d and the fixture's teardown calls `cache_info()` on whatever is bound to the name.
    """
    from rag_clients import factory
    from rag_common.errors import ConfigError

    class _UnknownCatalog:
        """Neither Postgres nor the in-memory twin — a future backend."""

    def _unknown() -> _UnknownCatalog:
        return _UnknownCatalog()

    factory.get_catalog.cache_clear()
    original = factory.get_catalog
    factory.get_catalog = _unknown  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError) as ei:
            await factory.document_relations(_KB, _DOC)
        assert "_UnknownCatalog" in str(ei.value), "the error must name the backend that was not handled"
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()
