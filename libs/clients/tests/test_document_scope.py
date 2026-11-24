"""Document scoping is its own parameter, not a metadata filter (``RetrievalQuery.doc_ids``).

Two callers — document-scoped compare and per-document summary — used to scope a query by
stuffing ``doc_id`` into the FR-D8 metadata ``filters`` dict. The two legs read that dict
differently, and neither of them is wrong:

* **Sparse.** Every filter key is namespaced under ``meta.<key>`` (``opensearch_index.meta_path``),
  so a filter key names a document/auto TAG. That is deliberate and tested — see
  ``test_sparse_metadata.test_a_filter_on_a_reserved_name_matches_the_tag_not_the_field``.
  A ``doc_id`` filter therefore asked for the tag and matched NOTHING. Measured live on
  ``t_public__company_wide_bm25``: ``term`` on ``doc_id`` -> 181 hits, on ``meta.doc_id`` -> 0.
* **Dense.** The Qdrant payload is flat, so the record's own ``doc_id`` IS filterable there.

Net effect: document-scoped retrieval worked on the dense leg by accident and returned zero on
the sparse leg, so compare/summary silently ran dense-only with no BM25 and nothing said so.
``doc_ids`` is the record-field predicate, kept apart from the tag channel so BOTH keep working.

These tests drive the twins AND the real adapters from one body, the repo's convention, because
the twin matching the record's doc_id while production did not is exactly how this stayed hidden.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import RequestError

from rag_clients.memory import InMemorySparseIndex, InMemoryVectorStore
from rag_clients.opensearch_index import OpenSearchSparseIndex
from rag_clients.qdrant_store import QdrantVectorStore
from rag_common import get_settings
from rag_schemas import Chunk
from rag_schemas.chunk import Span

_TENANT = "docscope"
_QUERY = "night shift allowance"
_DOC_A = "doc-a"
_DOC_B = "doc-b"
_DIM = 4


def _chunk(cid: str, doc_id: str, text: str, meta: dict[str, str] | None = None) -> Chunk:
    # The tags go in via an explicit dict, not **kwargs: this file deliberately plants tags
    # NAMED `doc_id`/`chunk_id`, which a kwargs helper cannot express alongside the record field
    # of the same name — the very collision under test.
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        tenant_id=_TENANT,
        text=text,
        span=Span(start=0, end=len(text)),
        metadata=meta or {},
    )


def _corpus() -> list[Chunk]:
    """Two documents whose text both match ``_QUERY`` — so an empty scoped result can only come
    from the scope predicate, never from the query failing to match."""
    return [
        _chunk("a1", _DOC_A, "The night shift allowance is paid to operations engineers."),
        _chunk("a2", _DOC_A, "A night shift allowance claim closes with the shift roster."),
        _chunk("b1", _DOC_B, "The night shift allowance for contractors is settled monthly."),
    ]


# --- sparse twin ------------------------------------------------------------------------------


async def _sparse_twin() -> InMemorySparseIndex:
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(_TENANT, _corpus())
    return index


async def test_doc_ids_scopes_the_sparse_twin_to_one_document() -> None:
    index = await _sparse_twin()
    hits = await index.search(_TENANT, _QUERY, 10, doc_ids=(_DOC_A,))
    assert sorted(cid for cid, _ in hits) == ["a1", "a2"]


async def test_a_scoped_sparse_search_is_not_silently_empty() -> None:
    """The regression that was live. Scoping a BM25 search to one document of a multi-document KB
    must return THAT document's chunks — the old call shape (the same intent expressed through the
    metadata channel) returns nothing at all, which reads as "no results", never as an error."""
    index = await _sparse_twin()
    through_the_tag_channel = await index.search(_TENANT, _QUERY, 10, {"doc_id": _DOC_A})
    scoped = await index.search(_TENANT, _QUERY, 10, doc_ids=(_DOC_A,))
    assert through_the_tag_channel == [], "a record id is not a tag value; the old shape matched nothing"
    assert scoped, "a document-scoped BM25 search must not be silently empty"
    assert {cid for cid, _ in scoped} == {"a1", "a2"}


async def test_a_doc_id_filter_still_means_the_tag_on_the_sparse_twin() -> None:
    """The deliberate FR-D8 semantics are untouched: a ``doc_id`` FILTER selects the chunk carrying
    that TAG, and must NOT scope by the record — the chunk whose record doc_id is the value stays
    out, and the chunk merely tagged with it comes back."""
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(
        _TENANT,
        [
            _chunk("tagged", _DOC_A, "The night shift allowance is paid monthly.", {"doc_id": _DOC_B}),
            _chunk("b1", _DOC_B, "The night shift allowance for contractors is settled monthly."),
        ],
    )
    hits = await index.search(_TENANT, _QUERY, 10, {"doc_id": _DOC_B})
    assert [cid for cid, _ in hits] == ["tagged"]


async def test_doc_ids_and_filters_both_apply_on_the_sparse_twin() -> None:
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(
        _TENANT,
        [
            _chunk("a_policy", _DOC_A, "The night shift allowance is paid monthly.", {"doc_type": "policy"}),
            _chunk("a_memo", _DOC_A, "The night shift allowance was discussed.", {"doc_type": "memo"}),
            _chunk("b_policy", _DOC_B, "The night shift allowance for contractors.", {"doc_type": "policy"}),
        ],
    )
    hits = await index.search(_TENANT, _QUERY, 10, {"doc_type": "policy"}, doc_ids=(_DOC_A,))
    assert [cid for cid, _ in hits] == ["a_policy"]


async def test_empty_doc_ids_changes_nothing_on_the_sparse_twin() -> None:
    index = await _sparse_twin()
    assert await index.search(_TENANT, _QUERY, 10, doc_ids=()) == await index.search(_TENANT, _QUERY, 10)


async def test_a_doc_id_tag_is_not_in_the_sparse_twins_filterable_payload() -> None:
    """The record id is carried out of band precisely so it cannot become a filterable tag: a
    chunk with no ``doc_id`` tag must not be matched by a ``doc_id`` filter on its record id."""
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(_TENANT, [_chunk("a1", _DOC_A, "The night shift allowance is paid monthly.")])
    assert await index.search(_TENANT, _QUERY, 10, {"doc_id": _DOC_A}) == []


# --- dense twin -------------------------------------------------------------------------------


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


async def test_doc_ids_scopes_the_dense_twin_to_one_document() -> None:
    store = await _dense_twin()
    hits = await store.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A,))
    assert sorted(cid for cid, _ in hits) == ["a1", "a2"]


async def test_doc_ids_is_an_or_over_the_set_on_the_dense_twin() -> None:
    store = await _dense_twin()
    hits = await store.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A, _DOC_B))
    assert sorted(cid for cid, _ in hits) == ["a1", "a2", "b1"]


async def test_empty_doc_ids_changes_nothing_on_the_dense_twin() -> None:
    store = await _dense_twin()
    assert await store.search(_TENANT, _vec(0), 10, doc_ids=()) == await store.search(_TENANT, _vec(0), 10)


async def test_a_metadata_tag_cannot_redefine_the_record_in_the_dense_twin() -> None:
    """The vector payload is FLAT (tags filterable at the top level, which every stored D8 filter
    relies on), so a tag called ``doc_id`` and the record's own ``doc_id`` are the same key. Spread
    last, the tag silently replaced the record — and the chunk then became unreachable by its real
    document. B7 auto-metadata can emit exactly such a key."""
    store = InMemoryVectorStore()
    await store.ensure_collection(_TENANT, _DIM)
    chunk = _chunk("a1", _DOC_A, "The night shift allowance.", {"doc_id": "tag-wrong", "chunk_id": "tag-wrong"})
    await store.upsert(_TENANT, [chunk], [_vec(0)])
    assert [cid for cid, _ in await store.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A,))] == ["a1"]
    assert await store.search(_TENANT, _vec(0), 10, doc_ids=("tag-wrong",)) == []


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
    monkeypatch.setenv("RAG_EXACT_TERM_BOOST", "2.0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_index() -> OpenSearchSparseIndex:
    index = OpenSearchSparseIndex(url="http://localhost:9200")
    index._client = AsyncMock()
    return index


async def test_the_scope_is_a_terms_clause_on_the_top_level_field(_exact_off: None) -> None:
    """One ``terms`` clause on the RECORD field — not a ``meta.*`` term, and not one clause per
    value (the clause budget above it counts a ``terms`` query as a single clause)."""
    index = _fake_index()
    index._client.search = AsyncMock(return_value={"hits": {"hits": []}})
    await index.search(_TENANT, _QUERY, 5, {"category": "SOP"}, doc_ids=(_DOC_A, _DOC_B))
    body = index._client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["must"] == [
        {"match": {"text": _QUERY}},
        {"term": {"meta.category": "SOP"}},
        {"terms": {"doc_id": [_DOC_A, _DOC_B]}},
    ]


async def test_the_scope_rides_in_must_on_the_exact_branch_too(_exact_on: None) -> None:
    """The ``should``/``must`` branch must scope as well — a leg that forgets it widens the query
    to the whole KB whenever an operator turns the exact-match flag on."""
    index = _fake_index()
    index._client.search = AsyncMock(return_value={"hits": {"hits": []}})
    await index.search(_TENANT, _QUERY, 5, doc_ids=(_DOC_A,), exact_boost=3.0)
    bool_q = index._client.search.call_args.kwargs["body"]["query"]["bool"]
    assert bool_q["minimum_should_match"] == 1
    assert bool_q["must"] == [{"terms": {"doc_id": [_DOC_A]}}], "the scope is a filter, never a `should`"


async def test_the_degraded_fallback_keeps_the_document_scope(_exact_on: None) -> None:
    """A maxClauseCount overflow degrades the SCORING clauses to a plain ``match``. If it dropped
    the scope too, a question asked about ONE document would quietly be answered from the whole
    KB — a worse failure than the one being degraded around."""
    index = _fake_index()
    clause_err = RequestError(
        400,
        "search_phase_execution_exception",
        {"error": {"root_cause": [{"type": "too_many_clauses", "reason": "maxClauseCount is set to 1024"}]}},
    )
    index._client.search = AsyncMock(
        side_effect=[clause_err, {"hits": {"hits": [{"_source": {"chunk_id": "a1"}, "_score": 1.0}]}}]
    )
    hits = await index.search(_TENANT, _QUERY, 5, {"category": "SOP"}, doc_ids=(_DOC_A,), exact_boost=3.0)
    assert hits == [("a1", 1.0)]
    assert index._client.search.call_count == 2
    fallback = index._client.search.call_args_list[1].kwargs["body"]
    assert fallback["query"]["bool"] == {
        "must": [
            {"match": {"text": _QUERY}},
            {"term": {"meta.category": "SOP"}},
            {"terms": {"doc_id": [_DOC_A]}},
        ]
    }


async def test_an_empty_scope_leaves_the_body_byte_identical(_exact_off: None) -> None:
    index = _fake_index()
    index._client.search = AsyncMock(return_value={"hits": {"hits": []}})
    await index.search(_TENANT, _QUERY, 5, {"category": "SOP"})
    without = index._client.search.call_args.kwargs["body"]
    await index.search(_TENANT, _QUERY, 5, {"category": "SOP"}, doc_ids=())
    assert index._client.search.call_args.kwargs["body"] == without


# --- the real adapters ------------------------------------------------------------------------


def _real_sparse() -> OpenSearchSparseIndex:
    """A live OpenSearch adapter, or skip. The dev stack may publish no host ports."""
    url = os.environ.get("RAG_OPENSEARCH_URL", "http://localhost:9200")
    try:
        import httpx

        httpx.get(url, timeout=5).raise_for_status()
    except Exception as exc:
        pytest.skip(f"no OpenSearch at {url} ({type(exc).__name__}); integration test skipped")
    return OpenSearchSparseIndex(url=url)


@pytest.fixture
async def real_sparse() -> AsyncIterator[OpenSearchSparseIndex]:
    index = _real_sparse()
    await index.delete_tenant(_TENANT)
    await index.ensure_index(_TENANT)
    await index.index(_TENANT, _corpus())
    yield index
    try:
        await index.delete_tenant(_TENANT)
    finally:
        await index.aclose()


def _ns(suffix: str) -> Callable[[str], str]:
    """A private collection per run so repeats cannot collide (isolation itself is covered by
    test_tenant_isolation)."""
    name = f"docscope_it_{suffix}_{uuid.uuid4().hex[:8]}"
    return lambda _tenant: name


@pytest.fixture
async def real_dense() -> AsyncIterator[QdrantVectorStore]:
    url = os.environ.get("RAG_QDRANT_URL", "http://localhost:6333")
    store = QdrantVectorStore(url=url, namespace=_ns("dense"))
    try:
        await store._client.get_collections()
    except Exception as exc:
        await store.aclose()
        pytest.skip(f"no Qdrant at {url} ({type(exc).__name__}); integration test skipped")
    await store.ensure_collection(_TENANT, _DIM)
    yield store
    try:
        await store.delete_tenant(_TENANT)
    finally:
        await store.aclose()


@pytest.mark.integration
async def test_doc_ids_scopes_opensearch_to_one_document(real_sparse: OpenSearchSparseIndex) -> None:
    hits = await real_sparse.search(_TENANT, _QUERY, 10, doc_ids=(_DOC_A,))
    assert sorted(cid for cid, _ in hits) == ["a1", "a2"]


@pytest.mark.integration
async def test_a_scoped_opensearch_search_is_not_silently_empty(real_sparse: OpenSearchSparseIndex) -> None:
    """The measured regression: ``term`` on ``meta.doc_id`` matched 0 of 181 documents, so a
    document-scoped BM25 leg returned nothing and the answer was built from the dense leg alone."""
    through_the_tag_channel = await real_sparse.search(_TENANT, _QUERY, 10, {"doc_id": _DOC_A})
    scoped = await real_sparse.search(_TENANT, _QUERY, 10, doc_ids=(_DOC_A,))
    assert through_the_tag_channel == [], "the old shape asked for a tag that does not exist"
    assert {cid for cid, _ in scoped} == {"a1", "a2"}, "BM25 must survive document scoping"


@pytest.mark.integration
async def test_doc_ids_and_filters_both_apply_in_opensearch(real_sparse: OpenSearchSparseIndex) -> None:
    await real_sparse.index(
        _TENANT,
        [
            _chunk("a_policy", _DOC_A, "The night shift allowance is paid monthly.", {"doc_type": "policy"}),
            _chunk("b_policy", _DOC_B, "The night shift allowance for contractors.", {"doc_type": "policy"}),
        ],
    )
    hits = await real_sparse.search(_TENANT, _QUERY, 10, {"doc_type": "policy"}, doc_ids=(_DOC_A,))
    assert [cid for cid, _ in hits] == ["a_policy"]


@pytest.mark.integration
async def test_a_doc_id_filter_still_means_the_tag_in_opensearch(real_sparse: OpenSearchSparseIndex) -> None:
    await real_sparse.index(
        _TENANT, [_chunk("tagged", "doc-c", "The night shift allowance is paid monthly.", {"doc_id": _DOC_B})]
    )
    hits = await real_sparse.search(_TENANT, _QUERY, 10, {"doc_id": _DOC_B})
    assert [cid for cid, _ in hits] == ["tagged"], "a doc_id FILTER selects the tag, never the record"


@pytest.mark.integration
async def test_empty_doc_ids_changes_nothing_in_opensearch(real_sparse: OpenSearchSparseIndex) -> None:
    assert await real_sparse.search(_TENANT, _QUERY, 10, doc_ids=()) == await real_sparse.search(_TENANT, _QUERY, 10)


@pytest.mark.integration
async def test_doc_ids_scopes_qdrant_to_one_document(real_dense: QdrantVectorStore) -> None:
    chunks = _corpus()
    await real_dense.upsert(_TENANT, chunks, [_vec(0) for _ in chunks])
    hits = await real_dense.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A,))
    assert sorted(cid for cid, _ in hits) == ["a1", "a2"]
    both = await real_dense.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A, _DOC_B))
    assert sorted(cid for cid, _ in both) == ["a1", "a2", "b1"]


@pytest.mark.integration
async def test_doc_ids_and_filters_both_apply_in_qdrant(real_dense: QdrantVectorStore) -> None:
    chunks = [
        _chunk("a_policy", _DOC_A, "night shift allowance", {"doc_type": "policy"}),
        _chunk("a_memo", _DOC_A, "night shift allowance", {"doc_type": "memo"}),
        _chunk("b_policy", _DOC_B, "night shift allowance", {"doc_type": "policy"}),
    ]
    await real_dense.upsert(_TENANT, chunks, [_vec(0) for _ in chunks])
    hits = await real_dense.search(_TENANT, _vec(0), 10, {"doc_type": "policy"}, doc_ids=(_DOC_A,))
    assert [cid for cid, _ in hits] == ["a_policy"]


@pytest.mark.integration
async def test_empty_doc_ids_changes_nothing_in_qdrant(real_dense: QdrantVectorStore) -> None:
    chunks = _corpus()
    await real_dense.upsert(_TENANT, chunks, [_vec(0) for _ in chunks])
    assert await real_dense.search(_TENANT, _vec(0), 10, doc_ids=()) == await real_dense.search(_TENANT, _vec(0), 10)


@pytest.mark.integration
async def test_a_metadata_tag_cannot_redefine_the_record_in_qdrant(real_dense: QdrantVectorStore) -> None:
    """The payload stays flat so tags remain filterable where they always were, but the record
    fields are written LAST and win — a ``doc_id``/``chunk_id`` tag can no longer replace the
    record it only describes, and the chunk is still reachable by its real document."""
    chunk = _chunk("a1", _DOC_A, "night shift allowance", {"doc_id": "tag-wrong", "chunk_id": "tag-wrong"})
    await real_dense.upsert(_TENANT, [chunk], [_vec(0)])
    assert [cid for cid, _ in await real_dense.search(_TENANT, _vec(0), 10, doc_ids=(_DOC_A,))] == ["a1"]
    assert await real_dense.search(_TENANT, _vec(0), 10, doc_ids=("tag-wrong",)) == []
