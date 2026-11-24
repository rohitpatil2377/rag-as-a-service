"""The RAG v2 ``query_log`` WRITE path against a LIVE Postgres (upgrade plan §1.1, Phase 12).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_query_log.py``).

What only a real server can prove:

1. **The round trip onto the right types** — ``latency_ms`` as real ``JSONB`` a dashboard can
   index into, ``cited_doc_ids`` as a real ``TEXT[]``, ``ts`` as a real ``TIMESTAMPTZ``, and the
   Phase-7/8/10 columns genuinely NULL rather than empty JSON.
2. **``BIGSERIAL`` append semantics** — two identical questions are two rows with two ids. A log
   is a log.
3. **The erasure predicate** — ``cited_doc_ids && ARRAY[...]`` is array OVERLAP, which a dict twin
   cannot demonstrate: a row goes if it cited the document AT ALL, and a row citing nothing is
   never collaterally purged by it.
4. **``delete_document`` reaches the log** — the gap Phase 1 flagged at that method and left
   recorded ("needs a decision when the table starts being written"). This is the decision, and
   it has to hold inside the same transaction as the rest of the erasure.

The tenant id is a per-test throwaway that the fixture drops in a ``finally``; no existing tenant
schema is read or written.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "emd-policy.pdf"
_OTHER = "unrelated-tender.pdf"
_LATENCY = {"guard": 1, "cache": 3, "route": 180, "retrieve": 420, "generate": 5200, "post": 11, "total": 5815}


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable.

    The schema is dropped in a ``finally`` so a passing OR failing run leaves no residue, and the
    id is a random ``zzv2qlog-*`` so it can never collide with a real tenant."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2qlog-{uuid.uuid4().hex[:10]}"
    try:
        async with catalog._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await catalog.aclose()
        pytest.skip(f"live Postgres not reachable ({type(exc).__name__}); integration test skipped")
    try:
        yield catalog, tenant
    finally:
        try:
            async with catalog._engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{pg_schema(tenant)}" CASCADE'))
        finally:
            await catalog.aclose()


async def _register(catalog: PostgresCatalog, tenant: str, *doc_ids: str) -> None:
    for doc_id in doc_ids:
        await catalog.register_document(
            DocumentEnvelope(
                tenant_id=tenant,
                doc_id=doc_id,
                source_id="s1",
                source_kind=SourceKind.UPLOAD,
                mime="application/pdf",
                ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
            )
        )


async def _write(catalog: PostgresCatalog, tenant: str, *, cites: list[str], query: str = "what EMD applies?") -> int:
    return await catalog.record_query_log(
        tenant,
        query=query,
        answer="The Earnest Money Deposit is Rs. 5 lakh.",
        cited_doc_ids=cites,
        latency_ms=_LATENCY,
    )


async def _rows(catalog: PostgresCatalog, tenant: str) -> list[dict[str, Any]]:
    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        result = await conn.execute(text(f'SELECT * FROM "{schema}".query_log ORDER BY id'))
        return [dict(r) for r in result.mappings().all()]


# --- the round trip ---------------------------------------------------------------------------------


async def test_a_row_lands_with_the_right_column_types(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    row_id = await _write(catalog, tenant, cites=[_DOC])
    assert row_id > 0

    rows = await _rows(catalog, tenant)
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "what EMD applies?"
    assert row["answer"].startswith("The Earnest Money Deposit")
    # Real JSONB: psycopg hands back parsed Python, not the text that was sent.
    assert isinstance(row["latency_ms"], dict) and row["latency_ms"]["generate"] == 5200
    # Real TEXT[].
    assert row["cited_doc_ids"] == [_DOC]
    # Real TIMESTAMPTZ.
    assert isinstance(row["ts"], datetime) and row["ts"].tzinfo is not None
    # The Phase-7/8/10 columns are NULL, not empty structures — see `record_query_log`.
    assert row["router_json"] is None and row["selected_sections"] is None and row["verification"] is None


async def test_the_latency_is_queryable_as_jsonb(live: tuple[PostgresCatalog, str]) -> None:
    """The point of the column. A JSON STRING would still read back as something; it would not
    answer this query, which is how a dashboard graphs a p95 per stage."""
    catalog, tenant = live
    await _write(catalog, tenant, cites=[_DOC])

    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        slowest = (
            await conn.execute(
                text(f"SELECT (latency_ms->>'generate')::int AS ms FROM \"{schema}\".query_log ORDER BY ms DESC")
            )
        ).scalar_one()
    assert slowest == 5200


async def test_a_verification_report_round_trips_as_jsonb(live: tuple[PostgresCatalog, str]) -> None:
    """Phase 10 fills this column (plan §10 item 6). It has existed since Phase 1 and had no
    writer until Phase 12; a JSON STRING stored in it would read back as something but would not
    answer the query a reviewer actually runs — "which answers had a claim fail, and why"."""
    catalog, tenant = live
    report = (
        '{"verify_version": 1, "checked": 3, "failed": 1, '
        '"failures": [{"claim": "The EMD is Rs. 5 Crores", "ordinal": 1, '
        '"reasons": ["value_not_in_quote"], "missing_values": ["qty:5 crore"], '
        '"omitted_items": [], "uncited": false}]}'
    )
    await catalog.record_query_log(
        tenant,
        query="what EMD applies?",
        answer="The EMD is Rs. 5 Crores.",
        cited_doc_ids=[_DOC],
        latency_ms=_LATENCY,
        verification=report,
    )

    row = (await _rows(catalog, tenant))[0]
    assert isinstance(row["verification"], dict)
    assert row["verification"]["failed"] == 1
    assert row["verification"]["failures"][0]["reasons"] == ["value_not_in_quote"]

    schema = pg_schema(tenant)
    sql = f"SELECT (verification->>'failed')::int FROM \"{schema}\".query_log ORDER BY id DESC LIMIT 1"
    async with catalog._engine.connect() as conn:
        failed = (await conn.execute(text(sql))).scalar_one()
    assert failed == 1


async def test_the_log_appends_rather_than_upserting(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    first = await _write(catalog, tenant, cites=[_DOC])
    second = await _write(catalog, tenant, cites=[_DOC])

    assert second > first
    assert len(await _rows(catalog, tenant)) == 2


async def test_a_row_can_cite_nothing(live: tuple[PostgresCatalog, str]) -> None:
    """An abstain cites nothing, and the row must record that rather than inventing an attribution."""
    catalog, tenant = live
    await _write(catalog, tenant, cites=[])

    assert (await _rows(catalog, tenant))[0]["cited_doc_ids"] == []


# --- erasure ----------------------------------------------------------------------------------------


async def test_the_scrub_is_array_overlap_not_containment(live: tuple[PostgresCatalog, str]) -> None:
    """A partially-erased answer still quotes the erased document."""
    catalog, tenant = live
    await _write(catalog, tenant, cites=[_DOC, _OTHER])

    assert await catalog.delete_query_log_citing(tenant, {_OTHER}) == 1
    assert await _rows(catalog, tenant) == []


async def test_a_row_citing_nothing_is_never_collaterally_purged(live: tuple[PostgresCatalog, str]) -> None:
    """``&&`` never matches an empty array, which is the behaviour being relied on."""
    catalog, tenant = live
    await _write(catalog, tenant, cites=[])

    assert await catalog.delete_query_log_citing(tenant, {_DOC}) == 0
    assert len(await _rows(catalog, tenant)) == 1


async def test_deleting_a_document_purges_the_rows_that_quoted_it(live: tuple[PostgresCatalog, str]) -> None:
    """THE GAP PHASE 1 FLAGGED, closed. ``delete_document`` is the path DPDP erasure cascades
    through; a row whose answer quotes the erased document left behind would make the certificate
    false."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC, _OTHER)
    await _write(catalog, tenant, cites=[_DOC])
    await _write(catalog, tenant, cites=[_OTHER])

    await catalog.delete_document(tenant, _DOC)

    assert [r["cited_doc_ids"] for r in await _rows(catalog, tenant)] == [[_OTHER]]


async def test_the_purge_is_scoped_to_the_tenants_own_schema(live: tuple[PostgresCatalog, str]) -> None:
    """One schema per tenant is the isolation mechanism; a log row is no exception to it."""
    catalog, tenant = live
    other_tenant = f"{tenant}x"
    try:
        await _write(catalog, tenant, cites=[_DOC])
        await _write(catalog, other_tenant, cites=[_DOC])

        await catalog.delete_query_log_citing(tenant, {_DOC})

        assert await _rows(catalog, tenant) == []
        assert len(await _rows(catalog, other_tenant)) == 1
    finally:
        async with catalog._engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{pg_schema(other_tenant)}" CASCADE'))
