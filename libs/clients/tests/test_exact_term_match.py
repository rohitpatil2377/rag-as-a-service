"""Slice 2a — exact technical-token BM25 matching.

The `standard` analyzer shreds chemical formulas (H₂SO₄), SMILES (CC(=O)Oc1ccccc1C(=O)O) and
identifiers (SOP-BULK-001) into fragments, so a query naming one under-retrieves the chunk that
contains it. An additive, flag-gated, per-KB-adaptive whitespace `text.exact` path fixes that.

These tests assert (in the in-memory twin, no live OpenSearch needed):
* an exact formula/ID/SMILES query ranks the containing chunk above a fragment-only competitor;
* a per-KB `exact_boost` override changes ranking versus the global default;
* flag OFF is byte-identical to fragment-only behaviour;
and, against a MOCKED OpenSearch client:
* the query body is byte-identical when the flag is off, gains the boosted `text.exact` clause
  when on, and an index missing the sub-field is upgraded live via put_mapping with a warning.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import RequestError

from rag_clients.memory import InMemorySparseIndex
from rag_common import get_settings, sparse_index
from rag_schemas import Chunk
from rag_schemas.chunk import Span

_TENANT = "s2a"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id=_TENANT, text=text, span=Span(start=0, end=len(text)))


@pytest.fixture
def _exact_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "true")
    monkeypatch.setenv("RAG_EXACT_TERM_BOOST", "2.0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _exact_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- in-memory twin -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exact_query",
    ["SOP-BULK-001", "H₂SO₄", "CC(=O)Oc1ccccc1C(=O)O"],
    ids=["identifier", "formula", "smiles"],
)
async def test_exact_token_query_ranks_the_containing_chunk_first(_exact_on: None, exact_query: str) -> None:
    """A query that IS a technical token surfaces the chunk containing it verbatim above a chunk
    that only shares the fragments the `standard`-style tokenizer would produce."""
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    # `target` contains the exact token; `decoy` shares only shredded fragments / prose so a
    # fragment-only match would rank it comparably or higher.
    await index.index(
        _TENANT,
        [
            _chunk("target", f"The batch was processed per {exact_query} under standard conditions."),
            _chunk("decoy", "The standard operating procedure batch was processed under conditions " * 5),
        ],
    )
    hits = await index.search(_TENANT, exact_query, 10)
    assert hits, "the exact query matched nothing"
    assert hits[0][0] == "target", f"expected the verbatim-containing chunk first, got {hits}"


async def test_per_kb_exact_boost_override_changes_ranking(_exact_on: None) -> None:
    """The 'not static' requirement: a higher per-KB `exact_boost` lifts the exact-match chunk over
    a chunk that a plain lexical overlap otherwise scores higher — while the global default does not."""
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    await index.index(
        _TENANT,
        [
            # Shares the exact identifier once, little else.
            _chunk("exact", "See SOP-BULK-001 for the mixing sequence."),
            # No exact token, but heavy lexical overlap with the query's prose words.
            _chunk("lexical", "the mixing sequence for the bulk batch is the standard mixing sequence"),
        ],
    )
    query = "SOP-BULK-001 mixing sequence"
    # Global default (exact_boost=None -> uses the modest global setting).
    default_hits = dict(await index.search(_TENANT, query, 10))
    # A pharma-weighted per-KB boost.
    boosted_hits = dict(await index.search(_TENANT, query, 10, exact_boost=25.0))

    assert boosted_hits["exact"] > default_hits["exact"], "a higher per-KB boost must raise the exact chunk's score"
    # With a strong per-KB boost the exact chunk wins; it need not win under the modest default.
    assert max(boosted_hits, key=lambda k: boosted_hits[k]) == "exact"


async def test_flag_off_is_byte_identical_in_the_twin(_exact_off: None) -> None:
    """Flag OFF: the exact params are ignored and scores are exactly the fragment-only scores."""
    index = InMemorySparseIndex()
    await index.ensure_index(_TENANT)
    chunks = [
        _chunk("a", "SOP-BULK-001 mixing sequence for the bulk batch"),
        _chunk("b", "an unrelated safety note about goggles"),
    ]
    await index.index(_TENANT, chunks)
    # Passing an exact_boost while the flag is off must have NO effect.
    with_boost = await index.search(_TENANT, "SOP-BULK-001 mixing", 10, exact_boost=99.0)
    without = await index.search(_TENANT, "SOP-BULK-001 mixing", 10)
    assert with_boost == without


# --- OpenSearch adapter (mocked client) ---------------------------------------------------------


def _fake_index(monkeypatch: pytest.MonkeyPatch) -> object:
    """An OpenSearchSparseIndex whose transport is a mock — no live server."""
    from rag_clients.opensearch_index import OpenSearchSparseIndex

    idx = OpenSearchSparseIndex(url="http://localhost:9200")
    idx._client = AsyncMock()
    return idx


async def test_query_body_is_byte_identical_when_flag_off(_exact_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = _fake_index(monkeypatch)
    idx._client.search = AsyncMock(return_value={"hits": {"hits": []}})  # type: ignore[attr-defined]
    await idx.search(_TENANT, "H₂SO₄ purity", 5, {"category": "SOP"})  # type: ignore[attr-defined]
    body = idx._client.search.call_args.kwargs["body"]  # type: ignore[attr-defined]
    assert body == {
        "size": 5,
        "query": {"bool": {"must": [{"match": {"text": "H₂SO₄ purity"}}, {"term": {"meta.category": "SOP"}}]}},
    }


async def test_query_body_adds_boosted_exact_clause_when_flag_on(
    _exact_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = _fake_index(monkeypatch)
    idx._client.search = AsyncMock(return_value={"hits": {"hits": []}})  # type: ignore[attr-defined]
    await idx.search(_TENANT, "H₂SO₄ purity", 5, {"category": "SOP"}, exact_boost=3.5)  # type: ignore[attr-defined]
    body = idx._client.search.call_args.kwargs["body"]  # type: ignore[attr-defined]
    bool_q = body["query"]["bool"]
    assert bool_q["minimum_should_match"] == 1
    assert {"match": {"text": "H₂SO₄ purity"}} in bool_q["should"]
    assert {"match": {"text.exact": {"query": "H₂SO₄ purity", "boost": 3.5}}} in bool_q["should"]
    # filters stay in `must`, never in `should`.
    assert bool_q["must"] == [{"term": {"meta.category": "SOP"}}]


async def test_over_budget_query_degrades_to_text_only_instead_of_erroring(
    _exact_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathological, clause-count-blowing query must not 500 the answer: catch OpenSearch's
    too_many_clauses and retry with the flag-off text-only body (M1)."""
    idx = _fake_index(monkeypatch)
    clause_err = RequestError(
        400,
        "search_phase_execution_exception",
        {"error": {"root_cause": [{"type": "too_many_clauses", "reason": "maxClauseCount is set to 1024"}]}},
    )
    idx._client.search = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[clause_err, {"hits": {"hits": [{"_source": {"chunk_id": "c1"}, "_score": 1.0}]}}]
    )
    hits = await idx.search(_TENANT, "H₂SO₄ purity", 5, {"category": "SOP"}, exact_boost=3.0)  # type: ignore[attr-defined]
    assert hits == [("c1", 1.0)], "the degraded text-only retry should still return results"
    assert idx._client.search.call_count == 2  # type: ignore[attr-defined]
    fallback_body = idx._client.search.call_args_list[1].kwargs["body"]  # type: ignore[attr-defined]
    # The retry is the exact flag-off body: single `text` match + the term filter in `must`.
    assert fallback_body["query"]["bool"] == {
        "must": [{"match": {"text": "H₂SO₄ purity"}}, {"term": {"meta.category": "SOP"}}]
    }


async def test_a_non_clause_request_error_is_not_swallowed(_exact_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the clause-count overflow degrades; any other RequestError propagates."""
    idx = _fake_index(monkeypatch)
    other = RequestError(400, "parsing_exception", {"error": {"reason": "malformed query"}})
    idx._client.search = AsyncMock(side_effect=other)  # type: ignore[attr-defined]
    with pytest.raises(RequestError):
        await idx.search(_TENANT, "H₂SO₄", 5, exact_boost=3.0)  # type: ignore[attr-defined]


async def test_existing_index_missing_subfield_is_upgraded_with_warning(
    _exact_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index that predates Slice 2a: ensure_index adds text.exact via put_mapping (live, no drop)
    AND warns that existing documents must be reindexed to populate it."""
    import structlog

    idx = _fake_index(monkeypatch)
    name = sparse_index(_TENANT)
    idx._client.indices.exists = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    # Namespaced tags present (so _warn_if_stale is quiet) but no text.exact sub-field.
    idx._client.indices.get_mapping = AsyncMock(  # type: ignore[attr-defined]
        return_value={
            name: {
                "mappings": {
                    "dynamic_templates": [{"tags_are_keywords": {"path_match": "meta.*"}}],
                    "properties": {"text": {"type": "text", "analyzer": "standard"}},
                }
            }
        }
    )
    idx._client.indices.put_mapping = AsyncMock()  # type: ignore[attr-defined]

    with structlog.testing.capture_logs() as logs:
        await idx.ensure_index(_TENANT)  # type: ignore[attr-defined]

    # Scoped to the put this test is about. `ensure_index` runs every live mapping upgrade in turn
    # (text.exact, chem, the RAG v2 `level` keyword), and this fixture's mapping predates all of
    # them, so a sibling upgrade legitimately puts its own field — counting ALL puts would make
    # this test fail for a change it is not about.
    exact_puts = [
        call.kwargs["body"]
        for call in idx._client.indices.put_mapping.call_args_list  # type: ignore[attr-defined]
        if "text" in call.kwargs["body"]["properties"]
    ]
    assert len(exact_puts) == 1
    assert exact_puts[0]["properties"]["text"]["fields"]["exact"]["analyzer"] == "whitespace"
    assert [e for e in logs if e.get("event") == "opensearch_exact_subfield_added"]


async def test_existing_index_with_subfield_is_not_re_put(_exact_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotent: an index that already has text.exact is left alone (no put_mapping)."""
    idx = _fake_index(monkeypatch)
    name = sparse_index(_TENANT)
    idx._client.indices.exists = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    idx._client.indices.get_mapping = AsyncMock(  # type: ignore[attr-defined]
        return_value={
            name: {
                "mappings": {
                    "dynamic_templates": [{"tags_are_keywords": {"path_match": "meta.*"}}],
                    "properties": {
                        "text": {"type": "text", "analyzer": "standard", "fields": {"exact": {"type": "text"}}}
                    },
                }
            }
        }
    )
    idx._client.indices.put_mapping = AsyncMock()  # type: ignore[attr-defined]
    await idx.ensure_index(_TENANT)  # type: ignore[attr-defined]
    assert not [
        call
        for call in idx._client.indices.put_mapping.call_args_list  # type: ignore[attr-defined]
        if "text" in call.kwargs["body"]["properties"]
    ], "text.exact already exists — it must not be re-put"
