"""The Phase-8 section-card READ against a LIVE Postgres (upgrade plan §8, step 1).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_section_cards.py``).

What only a real server can prove:

1. **The join is one row, not two reads.** Postgres holds the Phase-2 identity columns and the
   Phase-4 derived ones on the same ``sections`` row; the twin keeps them in two maps and joins
   them in Python. If the SQL named a column that moved, only this would catch it.
2. **``amends`` is read as a SQL boolean** out of a JSONB column — ``(amends IS NOT NULL)``, so
   the amended clause TEXT never leaves the database. A dict twin cannot show the difference
   between the projection and the payload.
3. **``facet_ids`` arrives as a real ``TEXT[]``**, not a string the driver happened to split.
4. **The ordering is the server's**, including ``ordinal NULLS LAST`` — a section whose ordinal
   never got written must sort after the numbered ones, not before them, and Python's sort has
   the opposite instinct about None.
5. **Erasure reaches it** — ``delete_document`` drops the rows, so a purged document contributes
   no cards to a later aggregate.

The tenant id is a per-test throwaway that the fixture drops in a ``finally``; no existing tenant
schema is read or written.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import SectionAmends, SectionEnrichment
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"
_OTHER = "nit"
_CRUMBS = ("Sl.1-9 Amended Clauses", "Attachment-1 > Switches", "Attachment-1 > Firewalls")


def _sid(crumb: str, doc_id: str = _DOC) -> str:
    return f"{doc_id}::{crumb}"


def _tree(doc_id: str = _DOC, *, crumbs: tuple[str, ...] = _CRUMBS) -> LayoutTree:
    def section(crumb: str, ordinal: int) -> LayoutSection:
        sid = _sid(crumb, doc_id)
        return LayoutSection(
            section_id=sid,
            breadcrumb=crumb,
            title=crumb,
            depth=1,
            ordinal=ordinal,
            text=f"body of {crumb}",
            text_hash=f"h{ordinal}",
            page=ordinal + 1,
            blocks=(
                LayoutBlock(
                    block_id=f"{sid}-b0", block_type="paragraph", text=f"body of {crumb}", page=ordinal + 1, ordinal=0
                ),
            ),
        )

    return LayoutTree(doc_id=doc_id, sections=tuple(section(c, i) for i, c in enumerate(crumbs)), source="markup")


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2card-{uuid.uuid4().hex[:10]}"
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


async def _prepare(catalog: PostgresCatalog, tenant: str, doc_id: str = _DOC, **kw: object) -> None:
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s1",
            source_kind=SourceKind.UPLOAD,
            mime="application/pdf",
            ingested_at=datetime(2026, 10, 22, tzinfo=UTC),
        )
    )
    await catalog.replace_document_structure(tenant, _tree(doc_id, **kw))  # type: ignore[arg-type]


_AMENDS = SectionAmends(
    target_ref="Appendix to ITB-2",
    old="Rs. 35 Crores",
    new="Rs. 19.44 Crores",
    quote="Is amended as follows",
)


# --- the read ---------------------------------------------------------------------------------------


async def test_every_section_comes_back_in_document_order(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)

    cards = await catalog.document_sections(tenant, [_DOC])

    assert [c.breadcrumb for c in cards] == list(_CRUMBS)
    assert [c.ordinal for c in cards] == [0, 1, 2]
    assert all(c.doc_id == _DOC and c.depth == 1 for c in cards)


async def test_the_phase_4_columns_join_onto_the_same_row(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(
        tenant,
        _DOC,
        {
            _sid(_CRUMBS[1]): SectionEnrichment(
                section_id=_sid(_CRUMBS[1]),
                summary="Switch certifications: UL 62368, CSA 22.2, CE, FCC Part 15 Class A.",
                facet_ids=("certifications", "supply_chain"),
            )
        },
    )

    cards = {c.section_id: c for c in await catalog.document_sections(tenant, [_DOC])}
    enriched = cards[_sid(_CRUMBS[1])]
    assert enriched.summary.startswith("Switch certifications")
    assert enriched.facet_ids == ("certifications", "supply_chain"), "a real TEXT[], tuple-ordered as stored"
    assert enriched.breadcrumb == _CRUMBS[1], "and the Phase-2 identity columns are on the same row"


async def test_an_unenriched_section_still_comes_back(live: tuple[PostgresCatalog, str]) -> None:
    """The OUTER-join claim against real SQL. Enrichment is fail-safe per section, so a document
    where only some sections were summarised is the ORDINARY state, and a card missing from this
    read is a section the aggregate strategy can never select."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(
        tenant, _DOC, {_sid(_CRUMBS[0]): SectionEnrichment(section_id=_sid(_CRUMBS[0]), summary="Amends nine clauses.")}
    )

    cards = {c.section_id: c for c in await catalog.document_sections(tenant, [_DOC])}

    assert len(cards) == 3
    assert cards[_sid(_CRUMBS[2])].summary == ""
    assert cards[_sid(_CRUMBS[2])].facet_ids == ()


async def test_amends_is_projected_to_a_boolean_and_the_clause_text_stays_in_the_database(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(
        tenant,
        _DOC,
        {_sid(_CRUMBS[0]): SectionEnrichment(section_id=_sid(_CRUMBS[0]), summary="x", amends=_AMENDS)},
    )

    cards = {c.section_id: c for c in await catalog.document_sections(tenant, [_DOC])}

    assert cards[_sid(_CRUMBS[0])].amends is True
    assert cards[_sid(_CRUMBS[1])].amends is False
    # The payload IS stored — this read simply does not carry it.
    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        stored = (
            await conn.execute(
                text(f'SELECT amends FROM "{schema}".sections WHERE section_id = :s'), {"s": _sid(_CRUMBS[0])}
            )
        ).scalar_one()
    assert stored["old"] == "Rs. 35 Crores"
    assert not any("35 Crores" in (c.summary + c.breadcrumb) for c in cards.values())


async def test_an_unordered_section_sorts_after_the_numbered_ones(live: tuple[PostgresCatalog, str]) -> None:
    """``ordinal NULLS LAST``. Postgres sorts NULL first by default and Python sorts None first
    too if it can compare at all — both are wrong here: a section with no ordinal is one the
    extractor could not place, and placing it at the TOP of a compliance answer is the worst
    possible guess."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    schema = pg_schema(tenant)
    async with catalog._engine.begin() as conn:
        await conn.execute(
            text(f'UPDATE "{schema}".sections SET ordinal = NULL WHERE section_id = :s'), {"s": _sid(_CRUMBS[0])}
        )

    cards = await catalog.document_sections(tenant, [_DOC])

    assert [c.breadcrumb for c in cards] == [_CRUMBS[1], _CRUMBS[2], _CRUMBS[0]]
    assert cards[-1].ordinal == 0, "and a NULL ordinal reads as 0, not as a crash"


# --- the scope --------------------------------------------------------------------------------------


async def test_the_scope_selects_documents_and_an_empty_scope_selects_none(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await _prepare(catalog, tenant, _OTHER, crumbs=("Notice Inviting Tender",))

    assert {c.doc_id for c in await catalog.document_sections(tenant, [_DOC])} == {_DOC}
    assert {c.doc_id for c in await catalog.document_sections(tenant, [_DOC, _OTHER])} == {_DOC, _OTHER}
    assert await catalog.document_sections(tenant, []) == [], "no scope is not the whole tenant"


async def test_several_documents_come_back_grouped_by_document(live: tuple[PostgresCatalog, str]) -> None:
    """`compare` runs the aggregate recipe per target, so the cards of two documents must not
    interleave by ordinal — a section 0 of each is two different places."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await _prepare(catalog, tenant, _OTHER, crumbs=("A", "B"))

    docs = [c.doc_id for c in await catalog.document_sections(tenant, [_DOC, _OTHER])]

    assert docs == sorted(docs), "grouped, because the server orders by doc_id first"


# --- erasure ----------------------------------------------------------------------------------------


async def test_deleting_the_document_removes_its_cards(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(
        tenant, _DOC, {_sid(_CRUMBS[0]): SectionEnrichment(section_id=_sid(_CRUMBS[0]), summary="Amends nine clauses.")}
    )
    assert await catalog.document_sections(tenant, [_DOC])

    await catalog.delete_document(tenant, _DOC)

    assert await catalog.document_sections(tenant, [_DOC]) == []
