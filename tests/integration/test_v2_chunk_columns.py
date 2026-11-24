"""The promoted `chunks` columns, against a LIVE Postgres (RAG v2 Phase 6).

Phase 1 added eight columns to `chunks` and Phase 5 put the values on the `Chunk` contract, but
until Phase 6 nothing wrote them: every structure coordinate lived only inside the `record` JSONB,
so the `chunks(section_id, level)` index could not be used and every per-chunk read had to
deserialise a document's worth of JSON. This pins what only a real server can show:

1. **The columns are populated** — and with the same values the record carries, so the promotion
   cannot drift from the source of truth.
2. **A v1 chunk is NULL where it has no value**, and still gets `chunk_hash`/`page_start`/
   `page_end`, which is what makes the re-embed skip work on the corpus as it is today.
3. **The upsert REFRESHES them.** Unchanged text keeps its id, so a re-index takes the ON CONFLICT
   path — and `sequence`, `embed_text` and `token_count` all move even when the text does not.
4. **`chunk_hashes` reads back what was written**, since the whole re-embed skip rests on it.

Run from the host (the default DSN names the in-compose hostname):

    RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" \\
    uv run pytest -m integration tests/integration/test_v2_chunk_columns.py

The tenant is a throwaway `zzv2cols-*` whose schema is dropped in a `finally`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from rag_common import pg_schema
from rag_core.chunking import text_hash
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.chunk import Span
from rag_schemas.common import ChunkLevel, SourceKind

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

_DOC = "corrigendum-v"


@pytest.fixture
async def live() -> AsyncIterator[tuple[PostgresCatalog, str]]:
    from rag_clients.postgres_catalog import PostgresCatalog

    catalog = PostgresCatalog()
    tenant = f"zzv2cols-{uuid.uuid4().hex[:10]}"
    try:
        async with catalog._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await catalog.aclose()
        pytest.skip(f"live Postgres not reachable ({type(exc).__name__}); integration test skipped")
    try:
        await catalog.register_document(
            DocumentEnvelope(
                tenant_id=tenant,
                doc_id=_DOC,
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


def _v2_chunk(tenant: str, *, chunk_id: str, body: str, sequence: int, level: ChunkLevel) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=_DOC,
        tenant_id=tenant,
        text=body,
        page=4,
        level=level,
        section_id=f"{_DOC}::s0bb1e0000001",
        parent_id=f"{_DOC}::s0bb1e0000000",
        embed_text=f"[Attachment-1 > 13. Video wall] {body}",
        sequence=sequence,
        token_count=11,
    )


async def _rows(catalog: PostgresCatalog, tenant: str) -> dict[str, dict[str, Any]]:
    async with catalog._engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT chunk_id, page, section_id, level, parent_id, embed_text, sequence, "
                f'page_start, page_end, token_count, chunk_hash FROM "{pg_schema(tenant)}".chunks'
            )
        )
        return {row["chunk_id"]: dict(row) for row in result.mappings().all()}


async def test_a_v2_chunk_lands_in_the_columns_and_agrees_with_its_record(
    live: tuple[PostgresCatalog, str],
) -> None:
    catalog, tenant = live
    body = "The controller shall comply with IEC 62443-4-2."
    chunk = _v2_chunk(tenant, chunk_id=f"{_DOC}::c000000000001", body=body, sequence=7, level=ChunkLevel.CHUNK)
    await catalog.record_chunks(tenant, [chunk])

    row = (await _rows(catalog, tenant))[chunk.chunk_id]
    assert row["level"] == "chunk"
    assert row["section_id"] == chunk.section_id
    assert row["parent_id"] == chunk.parent_id
    assert row["embed_text"] == chunk.embed_text
    assert row["sequence"] == 7
    assert row["token_count"] == 11
    assert row["chunk_hash"] == text_hash(body)
    # Single-page by construction, so the range is degenerate — but present, so a page predicate
    # reads the same on `chunks` as on `sections`.
    assert row["page_start"] == row["page_end"] == 4

    # ...and the promotion did not invent anything: what the columns say, the record says.
    [stored] = (await catalog.get_chunks(tenant, [chunk.chunk_id])).values()
    assert (stored.level, stored.section_id, stored.parent_id) == (chunk.level, chunk.section_id, chunk.parent_id)
    assert (stored.sequence, stored.token_count, stored.embed_text) == (7, 11, chunk.embed_text)


async def test_a_v1_chunk_is_null_where_it_has_no_value(live: tuple[PostgresCatalog, str]) -> None:
    """A v1 chunk has no level, section or sequence — NULL is what that means, and the write must
    not fabricate one. It DOES get a hash and a page, which is what lets the re-embed skip work on
    a corpus that was never v2."""
    catalog, tenant = live
    body = "1. Scope. This corrigendum amends the bidding document."
    v1 = Chunk(
        chunk_id=f"{_DOC}::aaaabbbbcccc",
        doc_id=_DOC,
        tenant_id=tenant,
        text=body,
        page=1,
        span=Span(start=0, end=len(body)),
    )
    await catalog.record_chunks(tenant, [v1])

    row = (await _rows(catalog, tenant))[v1.chunk_id]
    assert [row[c] for c in ("level", "section_id", "parent_id", "embed_text", "sequence", "token_count")] == [None] * 6
    assert row["chunk_hash"] == text_hash(body)
    assert row["page_start"] == row["page_end"] == 1


async def test_re_recording_the_same_chunk_refreshes_the_columns(live: tuple[PostgresCatalog, str]) -> None:
    """A re-index of a document whose text survived takes the ON CONFLICT path for every surviving
    chunk. `sequence` is the case that matters: insert a section ahead of this one and the position
    moves while the text does not, so a DO UPDATE that forgot the column would leave the document
    reading back in its OLD order — exactly the defect `sequence` was added to fix."""
    catalog, tenant = live
    body = "Dual redundant feeds shall be provided to each rack."
    cid = f"{_DOC}::c000000000002"
    await catalog.record_chunks(tenant, [_v2_chunk(tenant, chunk_id=cid, body=body, sequence=2, level=ChunkLevel.ROW)])

    moved = _v2_chunk(tenant, chunk_id=cid, body=body, sequence=5, level=ChunkLevel.ROW)
    await catalog.record_chunks(tenant, [moved.model_copy(update={"token_count": 12})])

    rows = await _rows(catalog, tenant)
    assert len(rows) == 1, "the same id must UPDATE, never insert a second row"
    assert rows[cid]["sequence"] == 5
    assert rows[cid]["token_count"] == 12
    assert rows[cid]["chunk_hash"] == text_hash(body), "unchanged text keeps its hash"


async def test_chunk_hashes_reads_back_what_was_written(live: tuple[PostgresCatalog, str]) -> None:
    """The re-embed skip is only as good as this read. It must return the hash for every chunk of
    the document and nothing from any other document."""
    catalog, tenant = live
    a = _v2_chunk(tenant, chunk_id=f"{_DOC}::c00000000000a", body="Clause A.", sequence=0, level=ChunkLevel.CHUNK)
    b = _v2_chunk(tenant, chunk_id=f"{_DOC}::c00000000000b", body="Clause B.", sequence=1, level=ChunkLevel.CHUNK)
    other = b.model_copy(update={"chunk_id": "other-doc::c1", "doc_id": "other-doc", "text": "Clause C."})
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id="other-doc",
            source_id="s1",
            source_kind=SourceKind.UPLOAD,
            mime="application/pdf",
            ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
        )
    )
    await catalog.record_chunks(tenant, [a, b, other])

    hashes = await catalog.chunk_hashes(tenant, _DOC)
    assert hashes == {a.chunk_id: text_hash("Clause A."), b.chunk_id: text_hash("Clause B.")}
    assert await catalog.chunk_hashes(tenant, "no-such-doc") == {}
