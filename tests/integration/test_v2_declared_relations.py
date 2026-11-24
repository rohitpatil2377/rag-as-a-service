"""Reading a document's declared relations from a LIVE Postgres (RAG v2 Phase 6).

`PostgresCatalog.document_relations` is the input to the graph seeding, and it is the one piece of
that feature a twin cannot validate: the rows live in two different shapes (a `doc_relations` table
and a JSONB column on `sections`), the document-level read LEFT JOINs `documents` twice to resolve
names, and the section-level read extracts from JSONB. All three are SQL, so a real server is the
only thing that can say they work.

**Nothing in this build writes these rows** — Phase 3 writes `doc_relations`, Phase 4 writes
`sections.amends`, and the plan's rollout order puts both after this phase. So the tests INSERT
them directly, which is the only way to exercise the read before its producers exist, and the
reason this read is a complete implementation rather than a stub.

    RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" \\
    uv run pytest -m integration tests/integration/test_v2_declared_relations.py
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_core.graph import DeclaredRelation
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"
_AMENDED = "corrigendum-ii"
_QUOTE = "The MAAT requirement is revised from Rs. 35 Crores to Rs. 19.44 Crores."


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2rel-{uuid.uuid4().hex[:10]}"
    try:
        async with catalog._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await catalog.aclose()
        pytest.skip(f"live Postgres not reachable ({type(exc).__name__}); integration test skipped")
    try:
        for doc_id in (_DOC, _AMENDED):
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
        yield catalog, tenant
    finally:
        try:
            async with catalog._engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{pg_schema(tenant)}" CASCADE'))
        finally:
            await catalog.aclose()


async def _insert_relation(catalog: PostgresCatalog, tenant: str, **row: object) -> None:
    columns = ", ".join(row)
    values = ", ".join(f":{k}" for k in row)
    async with catalog._engine.begin() as conn:
        await conn.execute(text(f'INSERT INTO "{pg_schema(tenant)}".doc_relations ({columns}) VALUES ({values})'), row)


async def _set_title(catalog: PostgresCatalog, tenant: str, doc_id: str, title: str) -> None:
    async with catalog._engine.begin() as conn:
        await conn.execute(
            text(f'UPDATE "{pg_schema(tenant)}".documents SET title = :t WHERE doc_id = :d'),
            {"t": title, "d": doc_id},
        )


async def test_a_resolved_reference_is_named_by_the_targets_own_title(live: tuple[PostgresCatalog, str]) -> None:
    """When Phase 3 resolved the reference to a document this KB holds, both ends are named by
    their titles — so the edge lands on the SAME graph node any other mention of that document
    does, instead of a second node spelled differently."""
    catalog, tenant = live
    await _set_title(catalog, tenant, _DOC, "Corrigendum-V")
    await _set_title(catalog, tenant, _AMENDED, "Corrigendum-II")
    await _insert_relation(
        catalog,
        tenant,
        doc_id=_DOC,
        rel_type="amends",
        target_ref="the second corrigendum",
        target_doc_id=_AMENDED,
        evidence_quote=_QUOTE,
    )

    assert await catalog.document_relations(tenant, _DOC) == [
        DeclaredRelation(
            rel_type="amends", source_ref="Corrigendum-V", target_ref="Corrigendum-II", evidence_quote=_QUOTE
        )
    ]


async def test_an_unresolved_reference_keeps_the_words_the_document_used(
    live: tuple[PostgresCatalog, str],
) -> None:
    """The NORMAL case for a corrigendum: it amends a tender this KB may never hold. The edge must
    still exist, named as the document named it — and when that document is ingested later and
    Phase 3 gives it that title, the two collapse onto one node by name."""
    catalog, tenant = live
    await _insert_relation(
        catalog,
        tenant,
        doc_id=_DOC,
        rel_type="supersedes",
        target_ref="NIT AEGCL/CGM/SLDC/T-161",
        target_doc_id=None,
        evidence_quote=_QUOTE,
    )

    [relation] = await catalog.document_relations(tenant, _DOC)
    assert relation.target_ref == "NIT AEGCL/CGM/SLDC/T-161"
    # No Phase-3 title yet, so the source falls back to the doc id rather than reading as empty.
    assert relation.source_ref == _DOC


async def test_a_section_amendment_is_read_out_of_its_jsonb(live: tuple[PostgresCatalog, str]) -> None:
    """Phase 4 writes `sections.amends` as `{target_ref, old, new, quote}`. The read lifts the two
    fields an edge needs out of the JSONB and implies the relation type from the column."""
    catalog, tenant = live
    schema = pg_schema(tenant)
    amends = {
        "target_ref": "Clause 7.2 of the bidding document",
        "old": "Rs. 35 Crores",
        "new": "Rs. 19.44 Crores",
        "quote": _QUOTE,
    }
    async with catalog._engine.begin() as conn:
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".sections (section_id, doc_id, breadcrumb, amends) '
                "VALUES (:s, :d, :b, CAST(:a AS JSONB))"
            ),
            {"s": f"{_DOC}::s1", "d": _DOC, "b": "Attachment-1 > 7. Financial qualification", "a": json.dumps(amends)},
        )

    assert await catalog.document_relations(tenant, _DOC) == [
        DeclaredRelation(
            rel_type="amends",
            source_ref="Attachment-1 > 7. Financial qualification",
            target_ref="Clause 7.2 of the bidding document",
            evidence_quote=_QUOTE,
        )
    ]


async def test_a_document_that_declares_nothing_reads_as_nothing(live: tuple[PostgresCatalog, str]) -> None:
    """The state of every document in every deployment today: both tables exist and are empty, so
    the seeding has nothing to do and does nothing."""
    catalog, tenant = live
    assert await catalog.document_relations(tenant, _DOC) == []
    assert await catalog.document_relations(tenant, "no-such-document") == []


async def test_relations_are_scoped_to_their_own_document(live: tuple[PostgresCatalog, str]) -> None:
    """Both reads are per-document, and a relation leaking across documents would seed an edge onto
    a chunk from the wrong document — citing text that does not say it."""
    catalog, tenant = live
    await _insert_relation(
        catalog,
        tenant,
        doc_id=_AMENDED,
        rel_type="references",
        target_ref="Annexure-III",
        target_doc_id=None,
        evidence_quote="Refer to Annexure-III for the bill of materials.",
    )

    assert await catalog.document_relations(tenant, _DOC) == []
    [relation] = await catalog.document_relations(tenant, _AMENDED)
    assert relation.target_ref == "Annexure-III"
