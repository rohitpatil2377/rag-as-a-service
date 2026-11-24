"""The RAG v2 profile-card WRITE path against a LIVE Postgres (upgrade plan §1.1, Phase 3).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_doc_profile.py``).

What only a real server can prove:

1. **The round trip** — the card's columns land on the EXISTING ``documents`` row (they are
   `ALTER TABLE ... ADD COLUMN` additions, not a rival table), ``issue_date`` arrives as a real
   ``DATE`` and ``profile_json`` as real ``JSONB``, and the Phase-4 columns beside them stay NULL.
2. **Replace, not append** — re-profiling (what every reindex does) leaves one taxonomy, not two.
   The in-memory twin cannot show this: its "replace" is an assignment, while the real one is a
   DELETE and an INSERT inside one transaction, and a primary-key collision on
   ``facets(doc_id, facet_id)`` or a missed predicate would only ever surface here.
3. **The join `document_relations` actually runs** — a three-table query with an INNER JOIN for
   the source title and a LEFT JOIN for an unresolved target. That SQL has no twin at all.
4. **Erasure** — ``delete_document`` empties ``facets`` and ``doc_relations`` for the document,
   which is the path DPDP right-to-erasure cascades through; ``doc_relations.evidence_quote``
   holds verbatim document text.
5. **A card cannot be hung on a document that is not there** — the UPDATE matches no row and
   nothing is written, so `document_relations`' INNER JOIN can never be starved of its title.

The tenant id is a per-test throwaway that the fixture drops in a ``finally``; no existing tenant
schema is read or written.

NOT PROVEN HERE, and not provable without the AEGCL corrigendum (discovery §7, §9.1): that a model
reads that document as a corrigendum, finds the amends relation to Corrigendum-II, or produces the
ten facets the plan's Phase-3 acceptance names. Every card below is one this file composed. See
the header of ``libs/rag-core/tests/test_doc_profile.py``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_core.doc_profile import PROFILE_VERSION, DocProfile, ProfileFacet, ProfileRelation
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"
_OTHER = "unrelated-tender"


def _profile(
    *,
    title: str = "Corrigendum-V",
    doc_type: str = "corrigendum",
    facets: tuple[str, ...] = ("eligibility", "financial_qualification"),
    relations: tuple[tuple[str, str, str], ...] = (("amends", "Corrigendum-II", "amends Corrigendum-II"),),
) -> DocProfile:
    return DocProfile(
        doc_type=doc_type,
        purpose="Amends the bidding document.",
        title=title,
        issuer="AEGCL",
        issue_date=date(2026, 9, 24),
        audience="prospective bidders",
        lang="en",
        relations=tuple(ProfileRelation(t, ref, quote) for t, ref, quote in relations),
        facets=tuple(ProfileFacet(f, f.replace("_", " ").title(), "d") for f in facets),
    )


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable.

    The schema is dropped in a ``finally`` so a passing OR failing run leaves no residue, and the
    id is a random ``zzv2prof-*`` so it can never collide with a real tenant."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2prof-{uuid.uuid4().hex[:10]}"
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


async def _register(catalog: PostgresCatalog, tenant: str, doc_id: str) -> None:
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


async def _row(catalog: PostgresCatalog, tenant: str, doc_id: str) -> dict[str, Any]:
    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        result = await conn.execute(text(f'SELECT * FROM "{schema}".documents WHERE doc_id = :d'), {"d": doc_id})
        return dict(result.mappings().one())


async def _rows(catalog: PostgresCatalog, tenant: str, table: str, doc_id: str) -> list[dict[str, Any]]:
    schema = pg_schema(tenant)
    order = "facet_id" if table == "facets" else "id"
    async with catalog._engine.connect() as conn:
        result = await conn.execute(
            text(f'SELECT * FROM "{schema}".{table} WHERE doc_id = :d ORDER BY {order}'), {"d": doc_id}
        )
        return [dict(r) for r in result.mappings().all()]


async def test_the_router_reads_the_card_back(live: tuple[PostgresCatalog, str]) -> None:
    """THE FIRST READ OF A CARD OUT OF POSTGRES (RAG v2 Phase 7).

    When Phase 3 wrote `replace_document_profile`, its note recorded that nothing read a card
    back — the in-memory twin had a reader for tests and the real adapter had none, because no
    consumer existed. The Phase-7 router is that consumer, and `document_cards` is a real query:
    a filter on `profile_version IS NOT NULL`, an ordering, and a second read joining the
    `facets` table. None of it has a twin."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile(facets=("eligibility", "financial_qualification")))

    cards = await catalog.document_cards(tenant)

    assert len(cards) == 1
    card = cards[0]
    assert card.doc_id == _DOC and card.title == "Corrigendum-V" and card.doc_type == "corrigendum"
    # Facets come from the TABLE, not out of `profile_json`: the table is the queryable form and
    # the one the router's validation is checked against.
    assert card.facet_ids == ("eligibility", "financial_qualification")


async def test_an_unprofiled_document_is_not_offered_to_the_router(live: tuple[PostgresCatalog, str]) -> None:
    """A card is the router's whole input; a cardless document in scope would let it believe it
    had been shown a profile it had not."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await _register(catalog, tenant, _OTHER)
    await catalog.replace_document_profile(tenant, _DOC, _profile())

    assert [c.doc_id for c in await catalog.document_cards(tenant)] == [_DOC]


async def test_the_router_scope_can_be_narrowed_to_named_documents(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await _register(catalog, tenant, _OTHER)
    await catalog.replace_document_profile(tenant, _DOC, _profile())
    await catalog.replace_document_profile(tenant, _OTHER, _profile(title="Unrelated"))

    assert [c.doc_id for c in await catalog.document_cards(tenant, doc_ids=[_OTHER])] == [_OTHER]
    assert len(await catalog.document_cards(tenant)) == 2


async def test_the_card_sample_is_bounded(live: tuple[PostgresCatalog, str]) -> None:
    """The cards go into a prompt on the latency path."""
    catalog, tenant = live
    for i in range(4):
        await _register(catalog, tenant, f"doc-{i}")
        await catalog.replace_document_profile(tenant, f"doc-{i}", _profile(title=f"Doc {i}"))

    assert len(await catalog.document_cards(tenant, limit=2)) == 2


async def test_the_card_lands_on_the_existing_documents_row(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)

    assert await catalog.replace_document_profile(tenant, _DOC, _profile()) is True

    row = await _row(catalog, tenant, _DOC)
    assert row["title"] == "Corrigendum-V"
    assert row["doc_type"] == "corrigendum"
    assert row["purpose"] == "Amends the bidding document."
    assert row["issuer"] == "AEGCL"
    # A real DATE, not a string — the column type is what recency filtering and supersession
    # ordering will read.
    assert row["issue_date"] == date(2026, 9, 24)
    assert row["lang"] == "en"
    assert row["profile_version"] == PROFILE_VERSION
    # The pre-v2 columns on the same row are untouched by the card.
    assert row["source_id"] == "s1" and row["mime"] == "application/pdf"
    # ...and the Phase-2 column beside them is still NULL: a profile is not a layout overlay.
    assert row["minio_layout_key"] is None


async def test_profile_json_is_real_jsonb_and_carries_the_whole_card(live: tuple[PostgresCatalog, str]) -> None:
    """Phase 4's section-enrichment prompt is handed ``{profile_json}`` as one blob, so it has to
    be queryable as JSON rather than stored as a string in a JSONB column."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile())

    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        value = (
            (
                await conn.execute(
                    text(
                        f"SELECT profile_json->>'doc_type' AS t, profile_json->'facet_taxonomy' AS f "
                        f'FROM "{schema}".documents WHERE doc_id = :d'
                    ),
                    {"d": _DOC},
                )
            )
            .mappings()
            .one()
        )
    assert value["t"] == "corrigendum"
    facets = value["f"] if isinstance(value["f"], list) else json.loads(value["f"])
    assert [f["facet_id"] for f in facets] == ["eligibility", "financial_qualification"]


async def test_the_facets_and_relations_rows_are_written(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile())

    facets = await _rows(catalog, tenant, "facets", _DOC)
    assert [f["facet_id"] for f in facets] == ["eligibility", "financial_qualification"]
    assert facets[0]["name"] == "Eligibility"

    relations = await _rows(catalog, tenant, "doc_relations", _DOC)
    assert len(relations) == 1
    assert relations[0]["rel_type"] == "amends"
    assert relations[0]["target_ref"] == "Corrigendum-II"
    assert relations[0]["evidence_quote"] == "amends Corrigendum-II"
    # Phase 3 records what the document SAYS; resolving the reference is not this phase's act.
    assert relations[0]["target_doc_id"] is None
    assert relations[0]["target_section_id"] is None


async def test_re_profiling_replaces_the_taxonomy_rather_than_colliding_on_its_primary_key(
    live: tuple[PostgresCatalog, str],
) -> None:
    """``facets`` is keyed ``(doc_id, facet_id)``. An append would either raise on the unchanged
    ids or, under an upsert, leave both runs' taxonomies on one document."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile(facets=("eligibility", "financial_qualification")))

    await catalog.replace_document_profile(tenant, _DOC, _profile(facets=("eligibility", "supply_chain")))

    assert [f["facet_id"] for f in await _rows(catalog, tenant, "facets", _DOC)] == [
        "eligibility",
        "supply_chain",
    ]
    assert len(await _rows(catalog, tenant, "doc_relations", _DOC)) == 1, "relations replaced, not doubled"


async def test_re_profiling_is_idempotent_when_the_card_is_unchanged(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile())
    first = await _row(catalog, tenant, _DOC)

    await catalog.replace_document_profile(tenant, _DOC, _profile())

    assert await _row(catalog, tenant, _DOC) == first
    assert len(await _rows(catalog, tenant, "facets", _DOC)) == 2


async def test_the_card_touches_only_its_own_document(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await _register(catalog, tenant, _OTHER)
    await catalog.replace_document_profile(tenant, _OTHER, _profile(title="Other", facets=("scope",)))

    await catalog.replace_document_profile(tenant, _DOC, _profile())

    assert [f["facet_id"] for f in await _rows(catalog, tenant, "facets", _OTHER)] == ["scope"]
    assert (await _row(catalog, tenant, _OTHER))["title"] == "Other"


async def test_a_card_for_a_document_with_no_row_writes_nothing_anywhere(
    live: tuple[PostgresCatalog, str],
) -> None:
    """The UPDATE matches no row, so the facets and relations are not written either — rows
    without a `documents` row would be invisible to `document_relations`' INNER JOIN while still
    being counted by anything that trusts the tables."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)  # a DIFFERENT document, so the schema exists

    assert await catalog.replace_document_profile(tenant, "never-ingested", _profile()) is False

    assert await _rows(catalog, tenant, "facets", "never-ingested") == []
    assert await _rows(catalog, tenant, "doc_relations", "never-ingested") == []


async def test_document_relations_joins_the_title_and_leaves_an_unresolved_target_as_written(
    live: tuple[PostgresCatalog, str],
) -> None:
    """The Phase-6 graph seeding's input, now that it has a producer (discovery §8.12). Names, not
    ids: the source is the document's own title and the target is the reference verbatim, because
    the amended tender is typically never ingested."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile())

    relations = await catalog.document_relations(tenant, _DOC)

    assert len(relations) == 1
    assert relations[0].rel_type == "amends"
    assert relations[0].source_ref == "Corrigendum-V"
    assert relations[0].target_ref == "Corrigendum-II"
    assert relations[0].evidence_quote == "amends Corrigendum-II"


async def test_a_second_declared_relation_type_round_trips_too(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(
        tenant,
        _DOC,
        _profile(
            relations=(
                ("amends", "Corrigendum-II", "amends Corrigendum-II"),
                ("attaches", "Attachment-1", "Appendix-D is replaced by Attachment-1"),
            )
        ),
    )

    relations = {r.rel_type: r.target_ref for r in await catalog.document_relations(tenant, _DOC)}
    assert relations == {"amends": "Corrigendum-II", "attaches": "Attachment-1"}


async def test_erasing_the_document_empties_facets_and_doc_relations(
    live: tuple[PostgresCatalog, str],
) -> None:
    """``doc_relations.evidence_quote`` holds verbatim document text, so leaving it behind after an
    erasure would be a compliance defect, not stale data."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await _register(catalog, tenant, _OTHER)
    await catalog.replace_document_profile(tenant, _DOC, _profile())
    await catalog.replace_document_profile(tenant, _OTHER, _profile(title="Other", facets=("scope",)))

    await catalog.delete_document(tenant, _DOC)

    assert await _rows(catalog, tenant, "facets", _DOC) == []
    assert await _rows(catalog, tenant, "doc_relations", _DOC) == []
    assert await catalog.document_relations(tenant, _DOC) == []
    # ...and the other document's rows are untouched.
    assert len(await _rows(catalog, tenant, "facets", _OTHER)) == 1


# --- the facet vocabulary, pinned to a document revision ----------------------------------------
#
# `revision` and `aliases` are `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` additions to a table
# every live KB already has, so only a real server shows that they land, that JSONB aliases
# survive the DELETE+INSERT the rewrite performs, and that the read refuses a vocabulary it
# cannot attribute to one set of bytes.

_REV = "3dc59d958833024d/p1"


async def test_the_revision_lands_on_the_facet_rows_and_reads_back_as_a_vocabulary(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile(), revision=_REV)

    rows = await _rows(catalog, tenant, "facets", _DOC)
    assert {r["revision"] for r in rows} == {_REV}

    vocab = await catalog.facet_vocabulary(tenant, _DOC)
    assert vocab is not None
    assert vocab.revision == _REV and vocab.ids == ("eligibility", "financial_qualification")
    assert vocab.matches(_REV) and not vocab.matches("OTHERBYTES/p1")


async def test_rows_written_without_a_revision_are_not_reusable(live: tuple[PostgresCatalog, str]) -> None:
    """Every facet row in every live KB predates this column, and reusing one would attribute a
    vocabulary to bytes nobody checked."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile())

    assert await catalog.facet_vocabulary(tenant, _DOC) is None


async def test_approved_aliases_are_real_jsonb_and_survive_the_rewrite(live: tuple[PostgresCatalog, str]) -> None:
    """The rewrite is a DELETE+INSERT, so an alias registry that did not round-trip would be
    silently emptied by the next ingest — and an alias is only ever added by a controlled update,
    so a lost one cannot be re-earned by a model."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    card = DocProfile(
        doc_type="corrigendum",
        purpose="Amends the bidding document.",
        title="Corrigendum-V",
        facets=(
            ProfileFacet(
                "performance_and_scalability",
                "Performance And Scalability",
                "d",
                ("performance_and_scalability_metrics", "performance_metrics"),
            ),
        ),
    )
    await catalog.replace_document_profile(tenant, _DOC, card, revision=_REV)

    rows = await _rows(catalog, tenant, "facets", _DOC)
    assert rows[0]["aliases"] == ["performance_and_scalability_metrics", "performance_metrics"]

    vocab = await catalog.facet_vocabulary(tenant, _DOC)
    assert vocab is not None
    assert vocab.resolve("performance_metrics") == "performance_and_scalability"
    assert vocab.resolve("performance_and_scalability_requirements") is None, "no fuzzy matching"


async def test_erasing_the_document_erases_its_vocabulary(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await catalog.replace_document_profile(tenant, _DOC, _profile(), revision=_REV)
    assert await catalog.facet_vocabulary(tenant, _DOC) is not None

    await catalog.delete_document(tenant, _DOC)

    assert await catalog.facet_vocabulary(tenant, _DOC) is None


async def test_a_facets_table_created_before_these_columns_migrates_in_place(
    live: tuple[PostgresCatalog, str],
) -> None:
    """ADR-0004's hazard, on the table this change touches.

    `CREATE TABLE IF NOT EXISTS` silently NO-OPS on an existing table, so a deployment that
    already has `facets` would never get `revision`/`aliases` from the CREATE — which is why they
    are also declared as `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. This builds the OLD shape by
    hand, then lets `ensure_schema` run, which is exactly what the first catalog operation after a
    deploy does. It was prompted by a real observation: the three preserved reference corpora,
    ingested before this commit, do not have the columns until something touches their schema.

    The pre-existing row must then read as NOT REUSABLE rather than as a vocabulary with an empty
    revision — a vocabulary that cannot say which bytes it describes must never be reused."""
    catalog, tenant = live
    schema = pg_schema(tenant)
    async with catalog._engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        await conn.execute(
            text(
                f'''CREATE TABLE "{schema}".facets (
                    doc_id TEXT NOT NULL,
                    facet_id TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    PRIMARY KEY (doc_id, facet_id)
                )'''
            )
        )
        await conn.execute(
            text(f'INSERT INTO "{schema}".facets (doc_id, facet_id, name) VALUES (:d, :f, :n)'),
            {"d": _DOC, "f": "legacy_facet", "n": "Legacy"},
        )

    await catalog.ensure_schema(tenant)

    async with catalog._engine.connect() as conn:
        cols = {
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = :s AND table_name = 'facets'"
                    ),
                    {"s": schema},
                )
            ).all()
        }
    assert {"revision", "aliases"} <= cols, "the ALTERs did not reach a pre-existing table"

    # The legacy row survived the migration and is correctly not reusable.
    rows = await _rows(catalog, tenant, "facets", _DOC)
    assert [r["facet_id"] for r in rows] == ["legacy_facet"]
    assert rows[0]["revision"] is None and rows[0]["aliases"] is None
    assert await catalog.facet_vocabulary(tenant, _DOC) is None
