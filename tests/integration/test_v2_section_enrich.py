"""The RAG v2 section-enrichment WRITE path against a LIVE Postgres (upgrade plan §1.1, Phase 4).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_section_enrich.py``).

What only a real server can prove:

1. **The round trip onto the right columns** — ``summary`` as TEXT, ``facet_ids`` as a real
   ``TEXT[]`` the GIN index can be asked about, ``key_values`` and ``amends`` as real ``JSONB``,
   all on the ``sections`` row the structure tree wrote and without touching its identity columns.
2. **UPDATE, never INSERT** — an enrichment naming a ``section_id`` this document has no row for
   writes nothing and is counted as a miss. The twin's dict cannot show a predicate.
3. **The ordering constraint that matters** — ``replace_document_structure`` DELETEs and
   re-INSERTs every section, so an enrichment written BEFORE it is gone. Only a real
   delete+insert can demonstrate that.
4. **``document_relations`` reads ``sections.amends``** — the half of Phase 6's graph-seeding
   query that had no producer until this phase. That SQL (a three-table read plus a second query
   over ``sections``) has no twin at all.
5. **Erasure** — ``delete_document`` drops the ``sections`` rows and the summaries and verbatim
   quotes inside them, which is the path DPDP right-to-erasure cascades through.

The tenant id is a per-test throwaway that the fixture drops in a ``finally``; no existing tenant
schema is read or written.

NOT PROVEN HERE, and not provable without the AEGCL corrigendum (discovery §7, §9.1): that a
model reads the MAAT section as amending Corrigendum-II from Rs. 35 Crores to Rs. 19.44 Crores,
or that the router section's propositions include HBOM and SBOM, or that >= 95% of its quotes
validate. Every enrichment below is one this file composed. See the header of
``libs/rag-core/tests/test_section_enrich.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import KeyValue, Proposition, SectionAmends, SectionEnrichment
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"
_MAAT = "4.2 Financial qualification"
_SCOPE = "1. Scope"


def _tree(doc_id: str = _DOC, *, crumbs: tuple[str, ...] = (_SCOPE, _MAAT)) -> LayoutTree:
    def section(crumb: str, ordinal: int) -> LayoutSection:
        sid = f"{doc_id}::{crumb.split(maxsplit=1)[0]}"
        return LayoutSection(
            section_id=sid,
            breadcrumb=crumb,
            title=crumb,
            depth=0,
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

    return LayoutTree(
        doc_id=doc_id, sections=tuple(section(c, i) for i, c in enumerate(crumbs)), source="markup", pages=(1, 2)
    )


def _section_id(crumb: str, doc_id: str = _DOC) -> str:
    return f"{doc_id}::{crumb.split(maxsplit=1)[0]}"


def _enrichment(
    crumb: str = _MAAT,
    *,
    summary: str = "Sets the minimum average annual turnover.",
    facet_ids: tuple[str, ...] = ("financial_qualification", "eligibility"),
    amends: SectionAmends | None = None,
    key_values: tuple[KeyValue, ...] | None = None,
) -> SectionEnrichment:
    return SectionEnrichment(
        section_id=_section_id(crumb),
        summary=summary,
        facet_ids=facet_ids,
        key_values=key_values
        if key_values is not None
        else (KeyValue("average annual turnover", "19.44", "Rs. Crores", "turnover of Rs. 19.44 Crores"),),
        amends=amends,
        propositions=(Proposition("The bidder must meet the turnover floor.", "turnover of Rs. 19.44"),),
    )


_AMENDS = SectionAmends(
    target_ref="Corrigendum-II",
    old="Rs. 35 Crores",
    new="Rs. 19.44 Crores",
    quote="supersedes the figure of Rs. 35 Crores stated in Corrigendum-II",
)


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable.

    The schema is dropped in a ``finally`` so a passing OR failing run leaves no residue, and the
    id is a random ``zzv2enr-*`` so it can never collide with a real tenant."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2enr-{uuid.uuid4().hex[:10]}"
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


async def _prepare(catalog: PostgresCatalog, tenant: str, doc_id: str = _DOC) -> None:
    """A registered document with its structure tree stored — the state Phase 4 always runs in."""
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
    await catalog.replace_document_structure(tenant, _tree(doc_id))


async def _rows(catalog: PostgresCatalog, tenant: str, doc_id: str = _DOC) -> dict[str, dict[str, Any]]:
    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        result = await conn.execute(text(f'SELECT * FROM "{schema}".sections WHERE doc_id = :d'), {"d": doc_id})
        return {row["section_id"]: dict(row) for row in result.mappings().all()}


# --- the round trip ---------------------------------------------------------------------------------


async def test_the_enrichment_lands_on_the_existing_sections_row(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)

    assert await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()}) == 1

    rows = await _rows(catalog, tenant)
    row = rows[_section_id(_MAAT)]
    assert row["summary"] == "Sets the minimum average annual turnover."
    # A real TEXT[], not a string — the column the plan puts a GIN index on.
    assert row["facet_ids"] == ["financial_qualification", "eligibility"]
    # Real JSONB: psycopg hands back parsed Python, not the text we sent.
    assert isinstance(row["key_values"], list) and row["key_values"][0]["value"] == "19.44"
    assert row["amends"] is None
    # The identity columns the structure tree wrote are untouched.
    assert row["breadcrumb"] == _MAAT and row["page_start"] == 2 and row["text_hash"] == "h1"


async def test_the_other_sections_are_left_alone(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()})

    untouched = (await _rows(catalog, tenant))[_section_id(_SCOPE)]
    assert untouched["summary"] is None and untouched["facet_ids"] is None and untouched["amends"] is None


async def test_the_facet_ids_are_queryable_through_the_gin_index(live: tuple[PostgresCatalog, str]) -> None:
    """A TEXT[] written as a string would still read back as something; it would not answer this
    query, which is what Phase 8 groups an aggregate answer with."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()})

    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        sql = f'SELECT section_id FROM "{schema}".sections WHERE facet_ids && ARRAY[:facet]'
        found = (await conn.execute(text(sql), {"facet": "financial_qualification"})).scalars().all()
    assert list(found) == [_section_id(_MAAT)]


# --- UPDATE, never INSERT ------------------------------------------------------------------------------


async def test_an_enrichment_for_a_section_with_no_row_writes_nothing(live: tuple[PostgresCatalog, str]) -> None:
    """It means the tree was re-parsed between enriching and persisting — every summary would be
    attached to the wrong text or to nothing. A conjured row with a summary and no text would be
    worse than the miss."""
    catalog, tenant = live
    await _prepare(catalog, tenant)

    hit = await catalog.replace_section_enrichment(
        tenant,
        _DOC,
        {_section_id(_MAAT): _enrichment(), "ghost-section": SectionEnrichment("ghost-section", "nowhere")},
    )

    assert hit == 1
    assert set(await _rows(catalog, tenant)) == {_section_id(_SCOPE), _section_id(_MAAT)}


async def test_an_empty_map_writes_nothing_and_purges_nothing(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()})

    assert await catalog.replace_section_enrichment(tenant, _DOC, {}) == 0
    assert (await _rows(catalog, tenant))[_section_id(_MAAT)]["summary"] is not None


async def test_re_enriching_replaces_all_four_columns_including_back_to_null(
    live: tuple[PostgresCatalog, str],
) -> None:
    """A section that amended something last run and amends nothing now must not keep the old
    `amends` — the whole row's derived half is this run's answer, not a merge of two runs'."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment(amends=_AMENDS)})
    assert (await _rows(catalog, tenant))[_section_id(_MAAT)]["amends"] is not None

    await catalog.replace_section_enrichment(
        tenant, _DOC, {_section_id(_MAAT): _enrichment(summary="Rewritten.", facet_ids=(), key_values=())}
    )

    row = (await _rows(catalog, tenant))[_section_id(_MAAT)]
    assert row["summary"] == "Rewritten."
    assert row["amends"] is None and row["facet_ids"] == [] and row["key_values"] == []


async def test_replacing_the_structure_tree_clears_the_enrichment_with_it(
    live: tuple[PostgresCatalog, str],
) -> None:
    """This is WHY the pipeline persists the tree first and the enrichment second. The tree
    replace is a DELETE and an INSERT, so an enrichment written before it is simply gone."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()})

    await catalog.replace_document_structure(tenant, _tree())

    assert (await _rows(catalog, tenant))[_section_id(_MAAT)]["summary"] is None


# --- the second half of Phase 6's graph seeding -------------------------------------------------------


async def test_a_section_amends_is_returned_by_document_relations(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment(amends=_AMENDS)})

    relations = await catalog.document_relations(tenant, _DOC)

    assert len(relations) == 1
    rel = relations[0]
    assert rel.rel_type == "amends"
    # The SOURCE is the section's BREADCRUMB — a section-level amendment names the clause that
    # declares it, not the document.
    assert rel.source_ref == _MAAT
    assert rel.target_ref == "Corrigendum-II"
    assert rel.evidence_quote == _AMENDS.quote


async def test_a_document_whose_sections_amend_nothing_declares_nothing(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment()})

    assert await catalog.document_relations(tenant, _DOC) == []


# --- erasure ---------------------------------------------------------------------------------------------


async def test_deleting_a_document_takes_its_summaries_and_quotes_with_it(
    live: tuple[PostgresCatalog, str],
) -> None:
    """`summary` and the quotes inside `key_values`/`amends` are derived from verbatim document
    text, so leaving them behind would make an erasure certificate false."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    await catalog.replace_section_enrichment(tenant, _DOC, {_section_id(_MAAT): _enrichment(amends=_AMENDS)})

    await catalog.delete_document(tenant, _DOC)

    assert await _rows(catalog, tenant) == {}
    assert await catalog.document_relations(tenant, _DOC) == []
