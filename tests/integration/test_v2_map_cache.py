"""The Phase-9 ``map_cache`` against a LIVE Postgres (upgrade plan §9.1).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box (the default DSN names the in-compose hostname, so from the host:
``RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" uv run pytest -m integration
tests/integration/test_v2_map_cache.py``).

What only a real server can prove:

1. **The items round-trip through JSONB** — a list of objects with nulls, nested value lists and
   non-ASCII text, out exactly as it went in. A dict twin stores the object itself and so cannot
   show a serialisation fault.
2. **UPSERT on the key** — two answers to the same question over the same document race here by
   construction. A second write must replace, not raise and not duplicate.
3. **The purge paths reach it**, which is the whole reason ``section_id`` is a column rather than
   only an input to the key hash: ``replace_document_structure`` drops a section's cached items
   when it replaces that section's text, and ``delete_document`` drops them on erasure. A cache
   that outlived the text it describes would serve extracted quotes from a deleted document —
   a DPDP failure, not just a stale read.

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
from rag_core.mapreduce import MapCacheEntry, map_cache_key
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"
_SECTIONS = ("21. Switches", "22. Firewalls")
_SIG = "a" * 32
_MODEL = "gemma4-31b"


def _sid(crumb: str) -> str:
    return f"{_DOC}::{crumb}"


def _tree(crumbs: tuple[str, ...] = _SECTIONS) -> LayoutTree:
    return LayoutTree(
        doc_id=_DOC,
        sections=tuple(
            LayoutSection(
                section_id=_sid(crumb),
                breadcrumb=crumb,
                title=crumb,
                depth=0,
                ordinal=i,
                text=f"body of {crumb}",
                text_hash=f"h{i}",
                blocks=(
                    LayoutBlock(
                        block_id=f"{_sid(crumb)}-b0", block_type="paragraph", text=f"body of {crumb}", page=1, ordinal=0
                    ),
                ),
            )
            for i, crumb in enumerate(crumbs)
        ),
        source="markup",
    )


def _items(statement: str = "The switches shall comply with UL 62368.") -> list[dict[str, object]]:
    return [
        {
            "item_id": "m1",
            "section_id": _sid(_SECTIONS[0]),
            "doc_id": _DOC,
            "facet_id": "certifications",
            "statement": statement,
            "quote": "comply with UL 62368",
            "values": [{"name": "standard", "value": "UL 62368", "unit": None}],
            "clause_ref": "Sl. 5",
            "is_amendment": True,
            "old_value": None,
            "new_value": "Rs. 19.44 Crores — ₹ non-ASCII",
        }
    ]


def _entry(crumb: str, items: list[dict[str, object]] | None = None) -> MapCacheEntry:
    section_id = _sid(crumb)
    return MapCacheEntry(
        cache_key=map_cache_key(section_id, _SIG, _MODEL),
        section_id=section_id,
        query_sig=_SIG,
        model_version=_MODEL,
        items=_items() if items is None else items,
    )


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    """A ``PostgresCatalog`` plus a throwaway tenant id; skips if Postgres is unreachable."""
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2map-{uuid.uuid4().hex[:10]}"
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


async def _prepare(catalog: PostgresCatalog, tenant: str) -> None:
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id=_DOC,
            source_id="s1",
            source_kind=SourceKind.UPLOAD,
            mime="application/pdf",
            ingested_at=datetime(2026, 10, 22, tzinfo=UTC),
        )
    )
    await catalog.replace_document_structure(tenant, _tree())


# --- the round trip ---------------------------------------------------------------------------------


async def test_items_survive_jsonb_intact(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entry = _entry(_SECTIONS[0])

    assert await catalog.write_map_cache(tenant, [entry]) == 1

    read = await catalog.read_map_cache(tenant, [entry.cache_key])
    [item] = read[entry.cache_key]
    assert item["statement"] == "The switches shall comply with UL 62368."
    assert item["old_value"] is None, "a JSON null is not the string 'None'"
    assert item["is_amendment"] is True
    assert item["values"] == [{"name": "standard", "value": "UL 62368", "unit": None}]
    assert "₹" in str(item["new_value"]), "non-ASCII survives"


async def test_an_empty_item_list_is_cached_and_read_back_as_empty(live: tuple[PostgresCatalog, str]) -> None:
    """ "This section has nothing relevant to this question" is the expected result for most
    sections of a long document, and it costs exactly as much to compute as a full one. A cache
    that declined to store it would re-read every irrelevant section on every question."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entry = _entry(_SECTIONS[0], items=[])

    await catalog.write_map_cache(tenant, [entry])

    read = await catalog.read_map_cache(tenant, [entry.cache_key])
    assert entry.cache_key in read, "a hit, not a miss"
    assert read[entry.cache_key] == []


async def test_a_miss_is_simply_absent(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    assert await catalog.read_map_cache(tenant, ["nothing-was-ever-written-here"]) == {}
    assert await catalog.read_map_cache(tenant, []) == {}


async def test_a_second_write_replaces_rather_than_duplicating(live: tuple[PostgresCatalog, str]) -> None:
    """Two answers to the same question over the same document race here by construction, and
    neither result is wrong — they are the same extraction."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entry = _entry(_SECTIONS[0])

    await catalog.write_map_cache(tenant, [entry])
    await catalog.write_map_cache(tenant, [_entry(_SECTIONS[0], items=_items("A revised statement."))])

    read = await catalog.read_map_cache(tenant, [entry.cache_key])
    assert len(read[entry.cache_key]) == 1
    assert read[entry.cache_key][0]["statement"] == "A revised statement."

    schema = pg_schema(tenant)
    async with catalog._engine.connect() as conn:
        rows = (
            await conn.execute(
                text(f'SELECT count(*) FROM "{schema}".map_cache WHERE cache_key = :k'), {"k": entry.cache_key}
            )
        ).scalar_one()
    assert rows == 1


async def test_several_keys_are_read_in_one_round_trip(live: tuple[PostgresCatalog, str]) -> None:
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entries = [_entry(crumb) for crumb in _SECTIONS]

    await catalog.write_map_cache(tenant, entries)

    read = await catalog.read_map_cache(tenant, [e.cache_key for e in entries])
    assert set(read) == {e.cache_key for e in entries}


# --- the purge paths --------------------------------------------------------------------------------


async def test_replacing_the_structure_purges_the_cached_items(live: tuple[PostgresCatalog, str]) -> None:
    """A section surviving the replace with the same id has still had its text replaced, so a
    cached extraction over it is stale either way. This is why ``section_id`` is a column and
    not merely an input to the key hash."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entries = [_entry(crumb) for crumb in _SECTIONS]
    await catalog.write_map_cache(tenant, entries)

    await catalog.replace_document_structure(tenant, _tree())

    assert await catalog.read_map_cache(tenant, [e.cache_key for e in entries]) == {}


async def test_deleting_the_document_purges_the_cached_items(live: tuple[PostgresCatalog, str]) -> None:
    """DPDP erasure. The cached items hold verbatim quotes from the document, so a cache that
    outlived it would serve extracted document text after the document was erased."""
    catalog, tenant = live
    await _prepare(catalog, tenant)
    entries = [_entry(crumb) for crumb in _SECTIONS]
    await catalog.write_map_cache(tenant, entries)

    await catalog.delete_document(tenant, _DOC)

    assert await catalog.read_map_cache(tenant, [e.cache_key for e in entries]) == {}
