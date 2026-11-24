"""The RAG v2 structure WRITE path against a LIVE Postgres (upgrade plan §1.1, Phase 2 task 5).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_structure_roundtrip.py``).

What only a real server can prove:

1. **The round trip** — a parsed tree inserted by ``replace_document_structure`` comes back out of
   ``sections``/``blocks`` with its parent links, page spans and table->row relation intact, and
   with the later-phase columns genuinely NULL rather than defaulted.
2. **Replace, not append** — re-running it (what every reindex does) leaves the same rows, not two
   copies. The in-memory twin cannot show this: its "replace" is an assignment, while the real one
   is a DELETE and an INSERT inside one transaction, and a primary-key collision or a missed
   predicate would only ever surface here.
3. **Erasure** — ``delete_document`` empties BOTH tables for the document, which is the path DPDP
   right-to-erasure cascades through and the first one that has to reach real document text
   (``blocks.text``).

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
from rag_core.layout import SUPPORTED_SCHEMA_VERSIONS, LayoutTree, parse_layout
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkLevel, SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "night-shift-policy"
_OTHER = "unrelated-policy"
_VERSION = max(SUPPORTED_SCHEMA_VERSIONS)

# Two pages: a heading hierarchy, a table with a declared header, and that table continuing over
# the page break with no header of its own (so the parser merges it).
_PAGE_1 = (
    "<h1>Night shift policy</h1><p>Applies to all operations staff.</p>"
    "<h2>Allowance</h2><p>Paid monthly with salary.</p>"
    "<table><tr><th>Grade</th><th>Allowance</th></tr><tr><td>M1</td><td>Rs.200</td></tr></table>"
)
_PAGE_2 = (
    "<table><tr><td>M2</td><td>Rs.250</td></tr><tr><td>M3</td><td>Rs.300</td></tr></table>"
    "<h2>Eligibility</h2><p>Six months of service.</p>"
)
_REVISED = "<h1>Night shift policy (revised)</h1><p>The allowance is now Rs.400 per day.</p>"


def _overlay(*pages: str) -> dict[str, Any]:
    return {
        "schema_version": _VERSION,
        "extractor": "chandra",
        "pages": [
            {"page_no": i, "markup": markup, "markup_format": "html", "degradations": []}
            for i, markup in enumerate(pages, start=1)
        ],
    }


def _tree(doc_id: str, *pages: str) -> LayoutTree:
    return parse_layout(_overlay(*pages), doc_id)


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable.

    The schema is dropped in a ``finally`` so a passing OR failing run leaves no residue, and the
    id is a random ``zzv2struct-*`` so it can never collide with a real tenant."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2struct-{uuid.uuid4().hex[:10]}"
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


async def _rows(catalog: PostgresCatalog, tenant: str, table: str, doc_id: str) -> list[dict[str, Any]]:
    schema = pg_schema(tenant)
    order = "ordinal, section_id" if table == "sections" else "section_id, ordinal"
    async with catalog._engine.connect() as conn:
        result = await conn.execute(
            text(f'SELECT * FROM "{schema}".{table} WHERE doc_id = :d ORDER BY {order}'), {"d": doc_id}
        )
        return [dict(r) for r in result.mappings().all()]


async def _register(catalog: PostgresCatalog, tenant: str, doc_id: str) -> None:
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=doc_id,
            source_id="s1",
            source_kind=SourceKind.UPLOAD,
            mime="application/pdf",
            ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
        )
    )


async def test_the_tree_round_trips_through_sections_and_blocks(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    tree = _tree(_DOC, _PAGE_1, _PAGE_2)

    written = await catalog.replace_document_structure(tenant, tree)

    sections = await _rows(catalog, tenant, "sections", _DOC)
    blocks = await _rows(catalog, tenant, "blocks", _DOC)
    assert written == (len(sections), len(blocks)) == (3, 10)

    # The hierarchy survived as parent links, and every link resolves.
    by_id = {r["section_id"]: r for r in sections}
    root = next(r for r in sections if r["parent_section_id"] is None)
    assert root["heading"] == "Night shift policy"
    assert root["depth"] == 0
    assert {r["heading"] for r in sections if r["parent_section_id"] == root["section_id"]} == {
        "Allowance",
        "Eligibility",
    }
    assert all(r["parent_section_id"] in by_id for r in sections if r["parent_section_id"] is not None)
    # The breadcrumb column is the normalised form that was hashed into the id.
    allowance = next(r for r in sections if r["heading"] == "Allowance")
    assert allowance["breadcrumb"].endswith("Allowance")
    assert allowance["text_hash"]

    # A merged cross-page table: the `page` column holds where it STARTS, and the span survives in
    # its rows and in the owning section's page_start/page_end.
    table = next(r for r in blocks if r["block_type"] == "table")
    rows = [r for r in blocks if r["parent_block_id"] == table["block_id"]]
    assert table["page"] == 1
    assert {r["page"] for r in rows} == {1, 2}
    assert (allowance["page_start"], allowance["page_end"]) == (1, 2)
    assert "Rs.300" in table["text"], "the continuation's rows are part of the merged table"
    assert table["html"], "a table keeps the engine's own markup"

    # Every block hangs off a section row of the same document.
    assert {r["section_id"] for r in blocks} <= set(by_id)

    # The later-phase columns are NULL, not defaulted to something a reader could mistake for
    # derived content; `bbox` is permanently NULL on this path (no coordinates exist).
    assert all(r[col] is None for r in sections for col in ("summary", "key_values", "facet_ids", "amends"))
    assert all(r["token_count"] is None for r in sections), "NULL means not counted; a guess would read as real"
    assert all(r["bbox"] is None for r in blocks)
    # And this slice does NOT touch chunking (Phase 5): the chunk columns stay NULL.
    async with catalog._engine.connect() as conn:
        chunk_rows = (
            await conn.execute(text(f'SELECT section_id, level, parent_id FROM "{pg_schema(tenant)}".chunks'))
        ).all()
    assert all(r.section_id is None and r.level is None and r.parent_id is None for r in chunk_rows)


async def test_re_running_the_write_replaces_rather_than_doubling(live: tuple[PostgresCatalog, str]) -> None:
    """What every reindex does. An append would raise on the primary key or, under an upsert, keep
    every section and block the new extraction no longer produces."""
    catalog, tenant = live
    tree = _tree(_DOC, _PAGE_1, _PAGE_2)

    await catalog.replace_document_structure(tenant, tree)
    first = (await _rows(catalog, tenant, "sections", _DOC), await _rows(catalog, tenant, "blocks", _DOC))
    await catalog.replace_document_structure(tenant, tree)

    assert (await _rows(catalog, tenant, "sections", _DOC), await _rows(catalog, tenant, "blocks", _DOC)) == first


async def test_a_different_extraction_leaves_none_of_the_previous_tree(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await catalog.replace_document_structure(tenant, _tree(_DOC, _PAGE_1, _PAGE_2))
    await catalog.replace_document_structure(tenant, _tree(_OTHER, _PAGE_1))

    await catalog.replace_document_structure(tenant, _tree(_DOC, _REVISED))

    sections = await _rows(catalog, tenant, "sections", _DOC)
    blocks = await _rows(catalog, tenant, "blocks", _DOC)
    assert [r["heading"] for r in sections] == ["Night shift policy (revised)"]
    assert [r["block_type"] for r in blocks] == ["heading", "paragraph"]
    assert await _rows(catalog, tenant, "sections", _OTHER), "another document's tree is untouched"


async def test_the_write_is_safe_on_a_document_with_no_stored_structure_and_on_an_empty_tree(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live

    # Nothing stored yet: the deletes match nothing and the inserts are the whole tree.
    assert await catalog.replace_document_structure(tenant, _tree(_DOC, _PAGE_1)) == (2, 6)
    # An empty tree is a legitimate instruction — a re-extraction that produced no structure.
    assert await catalog.replace_document_structure(tenant, LayoutTree(doc_id=_DOC, sections=(), source="markup")) == (
        0,
        0,
    )
    assert await _rows(catalog, tenant, "sections", _DOC) == []
    assert await _rows(catalog, tenant, "blocks", _DOC) == []


async def test_delete_document_empties_both_tables(live: tuple[PostgresCatalog, str]) -> None:
    """The DPDP erasure path. `blocks.text` holds verbatim document text, so an erasure that left
    these rows behind would make the signed certificate false."""
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)
    await _register(catalog, tenant, _OTHER)
    await catalog.replace_document_structure(tenant, _tree(_DOC, _PAGE_1, _PAGE_2))
    await catalog.replace_document_structure(tenant, _tree(_OTHER, _PAGE_1))
    assert any("Rs.200" in r["text"] for r in await _rows(catalog, tenant, "blocks", _DOC))

    await catalog.delete_document(tenant, _DOC)

    assert await _rows(catalog, tenant, "sections", _DOC) == []
    assert await _rows(catalog, tenant, "blocks", _DOC) == []
    assert await _rows(catalog, tenant, "sections", _OTHER), "only the erased document's rows go"
    assert await _rows(catalog, tenant, "blocks", _OTHER)


async def test_map_cache_rows_do_not_outlive_the_sections_they_point_at(
    live: tuple[PostgresCatalog, str],
) -> None:
    """`map_cache` is keyed by `section_id` with no FK, so both the replace and the delete have to
    purge it through `sections` — and before them. Nothing writes the table yet (Phase 3), so this
    seeds a row by hand to prove the ordering now rather than after it starts mattering."""
    catalog, tenant = live
    schema = pg_schema(tenant)
    await catalog.replace_document_structure(tenant, _tree(_DOC, _PAGE_1))
    section_id = (await _rows(catalog, tenant, "sections", _DOC))[0]["section_id"]

    async def _seed() -> None:
        async with catalog._engine.begin() as conn:
            await conn.execute(
                text(
                    f'INSERT INTO "{schema}".map_cache (cache_key, section_id, query_sig) '
                    "VALUES (:k, :s, :q) ON CONFLICT (cache_key) DO NOTHING"
                ),
                {"k": f"k-{section_id}", "s": section_id, "q": "sig"},
            )

    async def _count() -> int:
        async with catalog._engine.connect() as conn:
            return int(
                (
                    await conn.execute(
                        text(f'SELECT count(*) AS n FROM "{schema}".map_cache WHERE section_id = :s'),
                        {"s": section_id},
                    )
                ).scalar_one()
            )

    await _seed()
    assert await _count() == 1
    await catalog.replace_document_structure(tenant, _tree(_DOC, _REVISED))
    assert await _count() == 0, "a replace must not leave a cached map pointing at a section it dropped"

    await catalog.replace_document_structure(tenant, _tree(_DOC, _PAGE_1))
    await _seed()
    await catalog.delete_document(tenant, _DOC)
    assert await _count() == 0, "an erasure must not leave a cached map pointing at an erased section"


async def test_the_blocks_table_carries_the_parent_block_id_column_and_a_doc_index(
    live: tuple[PostgresCatalog, str],
) -> None:
    """Both are added by this slice: Phase 1's `blocks` had no column for the table->row relation,
    and the replace path purges by `doc_id` on every ingest and reindex of every document, which
    without an index is a sequential scan of the tenant's whole `blocks` table each time."""
    catalog, tenant = live
    schema = pg_schema(tenant)
    await catalog.ensure_schema(tenant)
    columns_sql = text(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = :s AND table_name = 'blocks'"
    )
    indexes_sql = text("SELECT indexname FROM pg_indexes WHERE schemaname = :s")
    async with catalog._engine.connect() as conn:
        columns = {str(r.column_name) for r in (await conn.execute(columns_sql, {"s": schema})).all()}
        indexes = {str(r.indexname) for r in (await conn.execute(indexes_sql, {"s": schema})).all()}
    assert "parent_block_id" in columns
    assert "blocks_by_doc" in indexes


async def test_document_chunks_returns_spanless_v2_chunks_in_document_order(
    live: tuple[PostgresCatalog, str],
) -> None:
    """Only a real server can prove this, which is exactly how the defect survived.

    `document_chunks` orders by `page NULLS FIRST, span.start NULLS FIRST`. Every v2 chunk has
    `span=None` — markup-derived text has no offset into the extracted text layer — so all of them
    tied on NULL and intra-page order became whatever Postgres chose to return. The in-memory twin
    sorted stably on `span.start or 0`, so `make test` was deterministic while production was not.

    The rows are INSERTED in deliberately scrambled order here, because inserting them in document
    order would let a server that ignores the ORDER BY entirely still pass.
    """
    catalog, tenant = live
    await _register(catalog, tenant, _DOC)

    def chunk(seq: int, page: int, body: str) -> Chunk:
        return Chunk(
            chunk_id=f"{_DOC}::c{seq}",
            doc_id=_DOC,
            tenant_id=tenant,
            text=body,
            page=page,
            span=None,
            source_uri="u",
            acl=[],
            classification=0,
            lineage_id="l1",
            level=ChunkLevel.CHUNK,
            section_id=f"{_DOC}::s0",
            sequence=seq,
        )

    ordered = [chunk(0, 1, "first"), chunk(1, 1, "second"), chunk(2, 1, "third"), chunk(3, 2, "fourth")]
    await catalog.record_chunks(tenant, [ordered[2], ordered[0], ordered[3], ordered[1]])

    back = await catalog.document_chunks(tenant, _DOC)
    assert [c.text for c in back] == ["first", "second", "third", "fourth"]
    assert [c.sequence for c in back] == [0, 1, 2, 3]
    # And the order is TOTAL: a repeat read cannot disagree with the first.
    assert [c.chunk_id for c in await catalog.document_chunks(tenant, _DOC)] == [c.chunk_id for c in back]


async def test_a_v1_chunk_still_orders_on_its_span(live: tuple[PostgresCatalog, str]) -> None:
    """The added tiebreaks must not have moved a v1 chunk, whose `sequence` is NULL."""
    catalog, tenant = live
    await _register(catalog, tenant, _OTHER)

    def chunk(start: int, body: str) -> Chunk:
        return Chunk(
            chunk_id=f"{_OTHER}::{start}",
            doc_id=_OTHER,
            tenant_id=tenant,
            text=body,
            page=1,
            span={"start": start, "end": start + len(body)},
            source_uri="u",
            acl=[],
            classification=0,
            lineage_id="l1",
        )

    await catalog.record_chunks(tenant, [chunk(40, "later"), chunk(0, "earlier"), chunk(20, "middle")])

    back = await catalog.document_chunks(tenant, _OTHER)
    assert [c.text for c in back] == ["earlier", "middle", "later"]
    assert all(c.sequence is None for c in back)
