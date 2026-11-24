"""Metadata filters must match what was stored (P20).

Two faults in the sparse leg, both silent.

**Filters under-matched.** The index mapping declared no metadata fields, so OpenSearch mapped
them dynamically as analysed ``text``. A ``term`` query needs an exact keyword, so it matched
only values that happen to survive analysis unchanged. Measured against the running server:

    term lang="en"                        -> 1 hit   (single lowercase token, by luck)
    term category="Business Travel Policy" -> 0 hits
    term doc_type="policy"                 -> 1 hit

B7 auto-metadata produces exactly the multi-word, capitalised values that return nothing —
and a filtered query that under-matches looks like "no results", never like an error.

**Metadata could overwrite the record.** The tags were spread into the document AFTER
``text``/``doc_id``, so a chunk carrying a metadata key called ``text`` replaced the indexed
body with its own tag value.

The in-memory twin matched exactly and so passed while production failed — which is why these
tests drive BOTH profiles from one body.
"""

from __future__ import annotations

import os

import pytest

from rag_clients.memory import InMemorySparseIndex
from rag_schemas import Chunk
from rag_schemas.chunk import Span

_TENANT = "p20"
_TEXT = "The night shift allowance is paid to operations engineers."


def _chunk(cid: str, **metadata: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        tenant_id=_TENANT,
        text=_TEXT,
        span=Span(start=0, end=len(_TEXT)),
        metadata=metadata,
    )


async def _round_trip(index: object, chunks: list[Chunk], filters: dict[str, str]) -> list[str]:
    await index.ensure_index(_TENANT)  # type: ignore[attr-defined]
    await index.index(_TENANT, chunks)  # type: ignore[attr-defined]
    hits = await index.search(_TENANT, "night shift allowance", 10, filters)  # type: ignore[attr-defined]
    return [cid for cid, _ in hits]


def _real_index() -> object:
    """A live OpenSearch adapter, or skip. The in-memory twin cannot reproduce a mapping bug."""
    from rag_clients.opensearch_index import OpenSearchSparseIndex

    url = os.environ.get("RAG_OPENSEARCH_URL", "http://localhost:9200")
    try:
        import httpx

        httpx.get(url, timeout=5).raise_for_status()
    except Exception:
        pytest.skip(f"no OpenSearch at {url}")
    return OpenSearchSparseIndex(url=url)


# --- the twin (fast, every run) ---------------------------------------------------------------


async def test_a_multi_word_value_filters_in_the_twin() -> None:
    got = await _round_trip(
        InMemorySparseIndex(),
        [_chunk("c1", category="Business Travel Policy")],
        {"category": "Business Travel Policy"},
    )
    assert got == ["c1"]


async def test_metadata_cannot_overwrite_the_indexed_body() -> None:
    """A tag called `text` replaced the body it was supposed to describe."""
    index = InMemorySparseIndex()
    got = await _round_trip(index, [_chunk("c1", text="policy", doc_id="wrong")], {})
    assert got == ["c1"], "the body was replaced, so the query no longer matches it"


async def test_a_filter_on_a_reserved_name_matches_the_tag_not_the_field() -> None:
    """`doc_id` as a TAG is a different thing from the record's own doc_id, and asking for one
    must not silently answer with the other."""
    index = InMemorySparseIndex()
    got = await _round_trip(index, [_chunk("c1", doc_id="tag-value")], {"doc_id": "tag-value"})
    assert got == ["c1"]


# --- the real adapter (the only place a mapping bug is reproducible) ---------------------------


@pytest.mark.integration
async def test_a_multi_word_value_filters_in_opensearch() -> None:
    """The measured failure: this returned 0 hits because the field was analysed text."""
    index = _real_index()
    await index.delete_tenant(_TENANT)  # type: ignore[attr-defined]
    got = await _round_trip(
        index, [_chunk("c1", category="Business Travel Policy")], {"category": "Business Travel Policy"}
    )
    await index.aclose()  # type: ignore[attr-defined]
    assert got == ["c1"]


@pytest.mark.integration
async def test_provenance_tags_are_filterable_in_opensearch() -> None:
    """The claim made three times and never verified: extraction provenance is queryable.
    `extraction_degradation` is comma-joined and `extraction_extractor` can be comma-joined
    too, so both are exactly the multi-token values that used to fail."""
    index = _real_index()
    await index.delete_tenant(_TENANT)  # type: ignore[attr-defined]
    chunks = [
        _chunk("clean", extraction_degraded="false", extraction_extractor="pdfplumber"),
        _chunk("degraded", extraction_degraded="true", extraction_extractor="pdfplumber,pypdf"),
    ]
    got = await _round_trip(index, chunks, {"extraction_extractor": "pdfplumber,pypdf"})
    await index.aclose()  # type: ignore[attr-defined]
    assert got == ["degraded"]


@pytest.mark.integration
async def test_metadata_cannot_overwrite_the_body_in_opensearch() -> None:
    index = _real_index()
    await index.delete_tenant(_TENANT)  # type: ignore[attr-defined]
    got = await _round_trip(index, [_chunk("c1", text="policy")], {})
    await index.aclose()  # type: ignore[attr-defined]
    assert got == ["c1"]


@pytest.mark.integration
async def test_an_index_predating_the_mapping_says_so() -> None:
    """OpenSearch cannot retype an existing field, so re-indexing into an old index does NOT
    fix the filters — it has to be dropped and rebuilt. An operator who re-indexes and still
    sees empty filter results needs to be told why, not left to find it."""
    import structlog

    index = _real_index()
    await index.delete_tenant(_TENANT)  # type: ignore[attr-defined]
    # Recreate it the way it used to be: no dynamic template for tags.
    from rag_common import sparse_index

    await index._client.indices.create(  # type: ignore[attr-defined]
        index=sparse_index(_TENANT), body={"mappings": {"properties": {"text": {"type": "text"}}}}
    )
    with structlog.testing.capture_logs() as logs:
        await index.ensure_index(_TENANT)  # type: ignore[attr-defined]
    await index.delete_tenant(_TENANT)  # type: ignore[attr-defined]
    await index.aclose()  # type: ignore[attr-defined]
    assert [e for e in logs if e.get("event") == "opensearch_index_mapping_stale"]
