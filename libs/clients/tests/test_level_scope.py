"""Retrieval grain is its own parameter, not a metadata filter (``RetrievalQuery.levels``).

Phase 6 indexed ``level`` as an exact-match field in BOTH stores and said, in both adapters, that
a level-scoped query leg would follow. This is that leg — and it is deliberately NOT the
``filters`` channel, for exactly the reason ``doc_ids`` is not (see ``test_document_scope``): the
dense payload is flat, so ``filters={"level": "section"}`` would have matched there, while the
sparse leg namespaces every filter key under ``meta.*`` and would have matched nothing. Half a
predicate, silently. ``level`` is now a reserved filter key, so that shape fails loudly instead.

The one claim worth stating twice: EMPTY IS NO FILTER, not "every level". A v1 chunk has no
level at all, and every document in every deployment today is v1 — a strategy that asked for the
three v2 levels explicitly would make them invisible, so an unscoped caller must keep seeing them.

These tests drive the twins AND the real adapters from one body, the repo's convention.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import RequestError

from rag_clients.memory import InMemorySparseIndex, InMemoryVectorStore
from rag_clients.opensearch_index import OpenSearchSparseIndex
from rag_clients.qdrant_store import _build_filter
from rag_common import get_settings
from rag_schemas import Chunk, RetrievalQuery
from rag_schemas.chunk import Span
from rag_schemas.common import ChunkLevel

_TENANT = "levelscope"
_QUERY = "earnest money deposit"
_DOC = "corrigendum-v.pdf"
_DIM = 4


def _chunk(cid: str, text: str, level: ChunkLevel | None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=_DOC,
        tenant_id=_TENANT,
        text=text,
        span=Span(start=0, end=len(text)),
        level=level,
        section_id="s1" if level is not None else None,
    )


def _corpus() -> list[Chunk]:
    """One unit per level plus a LEVEL-LESS v1 chunk, all matching ``_QUERY`` — so an empty
    scoped result can only come from the predicate, never from the query failing to match."""
    return [
        _chunk("sec", "Eligibility: the earnest money deposit section summary.", ChunkLevel.SECTION),
        _chunk("chk", "The earnest money deposit shall be furnished with the bid.", ChunkLevel.CHUNK),
        _chunk("row", "Earnest money deposit | Rs. 5 lakh", ChunkLevel.ROW),
        _chunk("prop", "The bidder furnishes an earnest money deposit.", ChunkLevel.PROPOSITION),
        _chunk("v1", "An earnest money deposit is required from every bidder.", None),
    ]


# --- the reserved key --------------------------------------------------------------------------


def test_level_as_a_filter_key_is_refused() -> None:
    with pytest.raises(ValueError, match="levels"):
        RetrievalQuery(tenant_id=_TENANT, text=_QUERY, filters={"level": "chunk"})


# --- sparse twin -------------------------------------------------------------------------------


async def _sparse_twin() -> InMemorySparseIndex:
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(_TENANT, _corpus())
    return index


async def test_levels_scopes_the_sparse_twin_to_one_grain() -> None:
    index = await _sparse_twin()
    hits = await index.search(_TENANT, _QUERY, 10, levels=(ChunkLevel.ROW,))
    assert [cid for cid, _ in hits] == ["row"]


async def test_levels_is_an_or_over_the_set_on_the_sparse_twin() -> None:
    index = await _sparse_twin()
    hits = await index.search(_TENANT, _QUERY, 10, levels=(ChunkLevel.CHUNK, ChunkLevel.ROW))
    assert sorted(cid for cid, _ in hits) == ["chk", "row"]


async def test_a_v1_chunk_is_excluded_by_any_grain_request_on_the_sparse_twin() -> None:
    """A level-less chunk is not "every level" — it is a chunk whose grain is unknown, and a
    caller asking for a grain has said that is not what it wants."""
    index = await _sparse_twin()
    hits = await index.search(_TENANT, _QUERY, 10, levels=tuple(ChunkLevel))
    assert "v1" not in {cid for cid, _ in hits}


async def test_empty_levels_changes_nothing_on_the_sparse_twin() -> None:
    index = await _sparse_twin()
    unscoped = await index.search(_TENANT, _QUERY, 10)
    assert await index.search(_TENANT, _QUERY, 10, levels=()) == unscoped
    assert "v1" in {cid for cid, _ in unscoped}, "and the v1 chunk is in it"


async def test_levels_and_doc_ids_both_apply_on_the_sparse_twin() -> None:
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    other = Chunk(
        chunk_id="other_row",
        doc_id="other.pdf",
        tenant_id=_TENANT,
        text="Earnest money deposit | Rs. 9 lakh",
        span=Span(start=0, end=10),
        level=ChunkLevel.ROW,
    )
    await index.index(_TENANT, [*_corpus(), other])
    hits = await index.search(_TENANT, _QUERY, 10, doc_ids=(_DOC,), levels=(ChunkLevel.ROW,))
    assert [cid for cid, _ in hits] == ["row"]


# --- dense twin --------------------------------------------------------------------------------


def _vec(i: int) -> list[float]:
    v = [0.0] * _DIM
    v[i % _DIM] = 1.0
    return v


async def _dense_twin() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.ensure_collection(_TENANT, _DIM)
    chunks = _corpus()
    await store.upsert(_TENANT, chunks, [_vec(0) for _ in chunks])
    return store


async def test_levels_scopes_the_dense_twin_to_one_grain() -> None:
    store = await _dense_twin()
    hits = await store.search(_TENANT, _vec(0), 10, levels=(ChunkLevel.SECTION,))
    assert [cid for cid, _ in hits] == ["sec"]


async def test_a_v1_point_is_excluded_by_any_grain_request_on_the_dense_twin() -> None:
    store = await _dense_twin()
    hits = await store.search(_TENANT, _vec(0), 10, levels=tuple(ChunkLevel))
    assert "v1" not in {cid for cid, _ in hits}


async def test_empty_levels_changes_nothing_on_the_dense_twin() -> None:
    store = await _dense_twin()
    assert await store.search(_TENANT, _vec(0), 10, levels=()) == await store.search(_TENANT, _vec(0), 10)


# --- the Qdrant filter body ----------------------------------------------------------------------


def test_the_grain_is_a_match_any_on_the_top_level_key() -> None:
    """One ``MatchAny`` on the RECORD key, ANDed with whatever else was asked for."""
    built = _build_filter({"category": "SOP"}, ("d1",), (ChunkLevel.CHUNK, ChunkLevel.ROW))
    assert built is not None
    keys = [c.key for c in built.must]  # type: ignore[union-attr]
    assert keys == ["category", "doc_id", "level"]
    assert built.must[-1].match.any == ["chunk", "row"]  # type: ignore[union-attr,index]


def test_no_predicate_is_no_filter_at_all() -> None:
    """Not an empty ``Filter``: a query with nothing to scope by must send the same body it sent
    before this parameter existed."""
    assert _build_filter(None, (), ()) is None


# --- the OpenSearch query body (mocked client — no live server, so every branch is reachable) ---


@pytest.fixture
def _exact_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _exact_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_index() -> OpenSearchSparseIndex:
    index = OpenSearchSparseIndex(url="http://localhost:9200")
    index._client = AsyncMock()
    return index


async def test_the_grain_is_a_terms_clause_on_the_top_level_field(_exact_off: None) -> None:
    index = _fake_index()
    index._client.search = AsyncMock(return_value={"hits": {"hits": []}})
    await index.search(_TENANT, _QUERY, 5, doc_ids=(_DOC,), levels=(ChunkLevel.CHUNK, ChunkLevel.ROW))
    body = index._client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["must"] == [
        {"match": {"text": _QUERY}},
        {"terms": {"doc_id": [_DOC]}},
        {"terms": {"level": ["chunk", "row"]}},
    ]


async def test_the_degraded_fallback_keeps_the_grain(_exact_on: None) -> None:
    """Same claim the document scope makes, for the same reason: a maxClauseCount degradation
    drops the SCORING clauses, and a predicate that fell out with them would answer a narrowed
    question from the whole index without saying so."""
    index = _fake_index()
    clause_err = RequestError(
        400,
        "search_phase_execution_exception",
        {"error": {"root_cause": [{"type": "too_many_clauses", "reason": "maxClauseCount is set to 1024"}]}},
    )
    index._client.search = AsyncMock(
        side_effect=[clause_err, {"hits": {"hits": [{"_source": {"chunk_id": "row"}, "_score": 1.0}]}}]
    )
    hits = await index.search(_TENANT, _QUERY, 5, levels=(ChunkLevel.ROW,), exact_boost=3.0)
    assert hits == [("row", 1.0)]
    fallback = index._client.search.call_args_list[1].kwargs["body"]
    assert {"terms": {"level": ["row"]}} in fallback["query"]["bool"]["must"]


async def test_an_unscoped_body_is_byte_identical_to_before_the_parameter(_exact_off: None) -> None:
    index = _fake_index()
    index._client.search = AsyncMock(return_value={"hits": {"hits": []}})
    await index.search(_TENANT, _QUERY, 5)
    without = index._client.search.call_args.kwargs["body"]
    await index.search(_TENANT, _QUERY, 5, levels=())
    assert index._client.search.call_args.kwargs["body"] == without
