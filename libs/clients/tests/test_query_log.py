"""The ``query_log`` through the catalog seam: the in-memory twin and the factory (Phase 12).

The Postgres half is ``tests/integration/test_v2_query_log.py`` — real ``JSONB``, a real
``TEXT[]`` and an array-overlap erasure predicate are not things a dict can demonstrate. What
belongs here is the behaviour both adapters must share, and one of them is a correction worth
naming: ``factory.purge_query_log_citing`` used to answer 0 for the in-memory profile, truthfully,
because nothing wrote the log there. Something does now, so the in-memory branch purges for real
— an erasure path that silently did nothing under the profile the unit and e2e gates run on
would be untested exactly where it matters.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_clients import factory
from rag_clients.memory import InMemoryCatalog
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TENANT = "acme"


async def _catalog(*doc_ids: str) -> InMemoryCatalog:
    catalog = InMemoryCatalog()
    for doc_id in doc_ids:
        await catalog.register_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=doc_id,
                source_id="s1",
                source_kind=SourceKind.UPLOAD,
                mime="application/pdf",
                ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
            )
        )
    return catalog


async def _row(catalog: InMemoryCatalog, *, cites: list[str], query: str = "what is the EMD?") -> int:
    return await catalog.record_query_log(
        _TENANT,
        query=query,
        answer="The EMD is Rs. 5 lakh.",
        cited_doc_ids=cites,
        latency_ms={"retrieve": 40, "generate": 900, "total": 950},
    )


async def test_a_row_is_appended_and_read_back() -> None:
    catalog = await _catalog("d1")
    await _row(catalog, cites=["d1"])

    rows = await catalog.query_log(_TENANT)
    assert len(rows) == 1
    assert rows[0]["query"] == "what is the EMD?"
    assert rows[0]["cited_doc_ids"] == ["d1"]
    assert rows[0]["latency_ms"]["total"] == 950


async def test_the_log_appends_rather_than_upserts() -> None:
    """A query log is a log. Two identical questions are two rows, because the thing being
    measured is the request, not the question."""
    catalog = await _catalog("d1")
    first = await _row(catalog, cites=["d1"])
    second = await _row(catalog, cites=["d1"])

    assert second != first
    assert len(await catalog.query_log(_TENANT)) == 2


async def test_the_phase_seven_to_ten_columns_default_to_none() -> None:
    """Left NULL rather than filled with today's nearest equivalent: a dashboard reading "no
    sections selected" where the truth is "nothing selects sections yet" is worse than empty."""
    catalog = await _catalog("d1")
    await _row(catalog, cites=["d1"])

    row = (await catalog.query_log(_TENANT))[0]
    assert row["router_json"] is None and row["selected_sections"] is None and row["verification"] is None


# --- erasure ---------------------------------------------------------------------------------------


async def test_a_row_is_purged_when_any_document_it_cited_is_erased() -> None:
    """Array OVERLAP, not containment: a partially-erased answer still quotes the erased document."""
    catalog = await _catalog("d1", "d2")
    await _row(catalog, cites=["d1", "d2"])

    assert await catalog.delete_query_log_citing(_TENANT, {"d2"}) == 1
    assert await catalog.query_log(_TENANT) == []


async def test_a_row_citing_nothing_is_never_collaterally_purged() -> None:
    """What Postgres's `&&` does with an empty array, and the twin must do the same."""
    catalog = await _catalog("d1")
    await _row(catalog, cites=[])

    assert await catalog.delete_query_log_citing(_TENANT, {"d1"}) == 0
    assert len(await catalog.query_log(_TENANT)) == 1


async def test_deleting_a_document_takes_the_rows_that_quoted_it() -> None:
    """This was the gap `PostgresCatalog.delete_document` flagged in Phase 1 and left recorded:
    nothing wrote the table, so nothing could leak. The moment a row carries an answer quoting a
    document, a document-scoped delete that left it behind makes the erasure certificate false."""
    catalog = await _catalog("d1", "d2")
    await _row(catalog, cites=["d1"])
    await _row(catalog, cites=["d2"])

    await catalog.delete_document(_TENANT, "d1")

    remaining = await catalog.query_log(_TENANT)
    assert [r["cited_doc_ids"] for r in remaining] == [["d2"]]


# --- the factory seam --------------------------------------------------------------------------------


async def test_the_factory_routes_the_write_to_the_configured_catalog() -> None:
    catalog = await _catalog("d1")
    original = factory.get_catalog
    factory.get_catalog = lambda: catalog  # type: ignore[assignment]
    try:
        assert await factory.record_query_log(
            _TENANT, query="q", answer="a", cited_doc_ids=["d1"], latency_ms={"total": 1}
        )
        assert await factory.purge_query_log_citing(_TENANT, {"d1"}) == 1
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


async def test_an_unknown_adapter_fails_open_on_the_write_and_closed_on_the_purge() -> None:
    """The asymmetry is the decision. The purge issues a certificate asserting an erasure
    happened, so an unreachable store has to stop it. The write is an observability record on the
    live query path, and failing it would make every ANSWER fail because its latency could not be
    logged."""

    class _Other:
        pass

    other = _Other()

    def _unknown() -> _Other:
        return other

    original = factory.get_catalog
    factory.get_catalog = _unknown  # type: ignore[assignment]
    try:
        assert await factory.record_query_log(_TENANT, query="q", answer="a", cited_doc_ids=[], latency_ms={}) is None
        with pytest.raises(Exception, match="certificate"):
            await factory.purge_query_log_citing(_TENANT, {"d1"})
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()
