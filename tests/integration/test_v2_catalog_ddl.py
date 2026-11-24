"""RAG v2 per-tenant DDL against a LIVE Postgres (upgrade plan §1.1 acceptance).

Marked ``integration`` so the default unit gate skips it; run it against the docker-compose
Postgres on the box. It proves the three things that can only be proven against a real server:

1. **Server-side idempotency** — the DDL runs twice against one schema with no error and no
   duplicate objects. The second pass MUST come from a SECOND ``PostgresCatalog`` with a cold memo:
   re-calling ``ensure_schema`` on the same instance returns from the memo having issued no SQL at
   all, so it would assert nothing. A cold second instance is also exactly the real shape — 4 api
   instances x 2 workers, each with its own memo.
2. **Concurrency** — two cold-memo workers racing one fresh schema both succeed. ``IF NOT EXISTS``
   is not a concurrency primitive (the existence check takes no lock), so without the advisory lock
   in ``ensure_schema`` the loser dies on a system-catalog unique violation and the whole pass rolls
   back into a failed request.
3. **Additive upgrade of an EXISTING schema** — the real risk of this change. The test builds a
   schema in its PRE-v2 shape (the DDL as it stood before the v2 columns), puts real rows in it,
   then runs ``ensure_schema`` and asserts the new columns appeared while every existing value is
   byte-identical.
4. **Teardown** — ``drop_v2_tables`` removes exactly the six new tables and leaves
   ``documents``/``chunks``/``tenant_config`` and their rows alone.

The tenant id is a per-test throwaway that the fixture drops afterwards; no existing tenant schema
is read or written.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from rag_common import pg_schema

if TYPE_CHECKING:
    from rag_clients.postgres_catalog import PostgresCatalog

pytestmark = pytest.mark.integration

# The `documents` / `chunks` / `tenant_config` DDL as it stood BEFORE the v2 columns — a verbatim
# fixture of the old shape, so "ensure_schema upgrades an existing schema" is tested against the
# schema real deployments actually have, not against one the new code just built.
_PRE_V2_DDL = (
    'CREATE SCHEMA IF NOT EXISTS "{s}"',
    """CREATE TABLE IF NOT EXISTS "{s}".documents (
        doc_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        mime TEXT NOT NULL,
        uri TEXT,
        lineage_id TEXT,
        source_acl JSONB NOT NULL DEFAULT '[]',
        metadata JSONB NOT NULL DEFAULT '{{}}',
        content_hash TEXT,
        ingested_at TIMESTAMPTZ NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS "{s}".chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        page INTEGER,
        version INTEGER NOT NULL DEFAULT 1,
        record JSONB NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS "{s}".tenant_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
)

_V2_COLUMNS_ON_DOCUMENTS = frozenset(
    {
        "title",
        "doc_type",
        "purpose",
        "issuer",
        "issue_date",
        "lang",
        "profile_json",
        "profile_version",
        "status",
        "minio_layout_key",
    }
)
_V2_COLUMNS_ON_CHUNKS = frozenset(
    {"section_id", "level", "parent_id", "embed_text", "page_start", "page_end", "token_count", "chunk_hash"}
)
_V2_TABLE_NAMES = frozenset({"doc_relations", "facets", "sections", "blocks", "map_cache", "query_log"})
_PRE_V2_TABLE_NAMES = frozenset({"documents", "chunks", "tenant_config"})


@pytest.fixture
async def live() -> AsyncIterator[tuple[Callable[[], PostgresCatalog], str]]:
    """A FACTORY of `PostgresCatalog` instances plus a throwaway tenant id; skips if Postgres is
    unreachable.

    A factory rather than one catalog because each instance carries its own `_ensured` memo, and
    several of these tests need a COLD one to make the server do any work at all. The fixture closes
    every instance it handed out and drops the whole throwaway schema afterwards, so a passing OR
    failing run leaves no residue — and it never names an existing tenant."""
    from rag_clients.postgres_catalog import PostgresCatalog

    made: list[PostgresCatalog] = []

    def make() -> PostgresCatalog:
        cat = PostgresCatalog()
        made.append(cat)
        return cat

    tenant = f"zzv2ddl-{uuid.uuid4().hex[:10]}"
    probe = make()
    try:
        async with probe._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await probe.aclose()
        pytest.skip(f"live Postgres not reachable ({type(exc).__name__}); integration test skipped")
    try:
        yield make, tenant
    finally:
        try:
            async with probe._engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{pg_schema(tenant)}" CASCADE'))
        finally:
            for cat in made:
                await cat.aclose()


async def _tables(cat: PostgresCatalog, schema: str) -> set[str]:
    async with cat._engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = :s"), {"s": schema}
            )
        ).all()
    return {str(r.table_name) for r in rows}


async def _columns(cat: PostgresCatalog, schema: str, table: str) -> set[str]:
    async with cat._engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_schema = :s AND table_name = :t"),
                {"s": schema, "t": table},
            )
        ).all()
    return {str(r.column_name) for r in rows}


async def _indexes(cat: PostgresCatalog, schema: str) -> set[str]:
    async with cat._engine.connect() as conn:
        rows = (await conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = :s"), {"s": schema})).all()
    return {str(r.indexname) for r in rows}


async def test_ensure_schema_is_idempotent_and_the_teardown_leaves_no_residue(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    make, tenant = live
    cat = make()
    schema = pg_schema(tenant)

    await cat.ensure_schema(tenant)
    first_tables = await _tables(cat, schema)
    first_indexes = await _indexes(cat, schema)

    # Run 2 from a SECOND instance, so the memo is cold and the server really re-runs the whole
    # statement list against the now-populated schema. Re-calling `cat.ensure_schema` instead would
    # return from the memo without issuing a single statement, and assert nothing.
    second = make()
    await second.ensure_schema(tenant)
    assert await _tables(second, schema) == first_tables
    assert await _indexes(second, schema) == first_indexes

    # And a third time with the memo cleared on the original instance, to pin that it is the DDL
    # that is idempotent and not merely the memo hiding it.
    cat._ensured.clear()
    await cat.ensure_schema(tenant)
    assert await _tables(cat, schema) == first_tables
    assert await _indexes(cat, schema) == first_indexes

    assert first_tables >= _V2_TABLE_NAMES
    assert first_tables >= _PRE_V2_TABLE_NAMES
    assert first_indexes >= {"sections_by_doc", "sections_by_facet", "chunks_by_section_level", "documents_by_hash"}
    assert await _columns(cat, schema, "documents") >= _V2_COLUMNS_ON_DOCUMENTS
    assert await _columns(cat, schema, "chunks") >= _V2_COLUMNS_ON_CHUNKS

    # Teardown, twice — it has to be idempotent too, or a retried purge would fail.
    await cat.drop_v2_tables(tenant)
    await cat.drop_v2_tables(tenant)
    after = await _tables(cat, schema)
    assert not (_V2_TABLE_NAMES & after), f"v2 residue left behind: {sorted(_V2_TABLE_NAMES & after)}"
    assert after >= _PRE_V2_TABLE_NAMES, "teardown must not touch the pre-existing tables"


async def test_ensure_schema_upgrades_an_existing_schema_without_touching_its_data(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    """The real risk of this change: 25 live per-KB schemas already hold data in `documents` and
    `chunks`, and `ensure_schema` runs against them on the very next query."""
    make, tenant = live
    cat = make()
    schema = pg_schema(tenant)

    # 1. Build the schema in its PRE-v2 shape and put real rows in it.
    async with cat._engine.begin() as conn:
        for stmt in _PRE_V2_DDL:
            await conn.execute(text(stmt.format(s=schema)))
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".documents '
                "(doc_id, source_id, source_kind, mime, uri, lineage_id, source_acl, metadata, "
                " content_hash, ingested_at) "
                "VALUES ('d1', 's1', 'upload', 'application/pdf', 'file:///tender.pdf', 'lin-1', "
                " CAST('[\"hr\"]' AS JSONB), CAST('{\"year\": \"2026\"}' AS JSONB), 'abc123', "
                " TIMESTAMPTZ '2026-01-02 03:04:05+00')"
            )
        )
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".chunks (chunk_id, doc_id, page, version, record) '
                "VALUES ('d1::deadbeefcafe', 'd1', 7, 3, CAST('{\"text\": \"clause 4.2\"}' AS JSONB))"
            )
        )
        await conn.execute(text(f"INSERT INTO \"{schema}\".tenant_config (key, value) VALUES ('k', 'v')"))

    assert not (_V2_COLUMNS_ON_DOCUMENTS & await _columns(cat, schema, "documents"))
    assert not (_V2_TABLE_NAMES & await _tables(cat, schema))

    # 2. The upgrade.
    await cat.ensure_schema(tenant)

    assert await _columns(cat, schema, "documents") >= _V2_COLUMNS_ON_DOCUMENTS
    assert await _columns(cat, schema, "chunks") >= _V2_COLUMNS_ON_CHUNKS
    assert await _tables(cat, schema) >= _V2_TABLE_NAMES

    # 3. Every pre-existing value is untouched, and the new columns are NULL (nothing populates
    #    them in this phase).
    async with cat._engine.connect() as conn:
        doc = (await conn.execute(text(f"SELECT * FROM \"{schema}\".documents WHERE doc_id = 'd1'"))).mappings().one()
        chunk = (await conn.execute(text(f'SELECT * FROM "{schema}".chunks'))).mappings().one()
        cfg = (await conn.execute(text(f'SELECT * FROM "{schema}".tenant_config'))).mappings().one()
        counts = (
            (
                await conn.execute(
                    text(
                        f'SELECT (SELECT count(*) FROM "{schema}".sections) AS sections, '
                        f'(SELECT count(*) FROM "{schema}".blocks) AS blocks, '
                        f'(SELECT count(*) FROM "{schema}".facets) AS facets, '
                        f'(SELECT count(*) FROM "{schema}".doc_relations) AS rels, '
                        f'(SELECT count(*) FROM "{schema}".map_cache) AS cache, '
                        f'(SELECT count(*) FROM "{schema}".query_log) AS qlog'
                    )
                )
            )
            .mappings()
            .one()
        )

    assert doc["source_id"] == "s1"
    assert doc["source_kind"] == "upload"
    assert doc["mime"] == "application/pdf"
    assert doc["uri"] == "file:///tender.pdf"  # the plan's `source_uri` IS this column
    assert doc["lineage_id"] == "lin-1"
    assert doc["source_acl"] == ["hr"]
    assert doc["metadata"] == {"year": "2026"}
    assert doc["content_hash"] == "abc123"
    assert doc["ingested_at"].isoformat() == "2026-01-02T03:04:05+00:00"
    assert all(doc[c] is None for c in _V2_COLUMNS_ON_DOCUMENTS)

    assert chunk["chunk_id"] == "d1::deadbeefcafe"
    assert chunk["page"] == 7
    assert chunk["version"] == 3
    assert chunk["record"] == {"text": "clause 4.2"}
    assert all(chunk[c] is None for c in _V2_COLUMNS_ON_CHUNKS)

    assert (cfg["key"], cfg["value"]) == ("k", "v")
    assert all(v == 0 for v in counts.values()), "Phase 1 populates nothing"


async def test_concurrent_cold_memo_workers_on_one_fresh_schema_all_succeed(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    """The regression guard for `ensure_schema`'s advisory lock.

    `CREATE SCHEMA/TABLE IF NOT EXISTS` is NOT concurrency-safe: the existence check takes no lock,
    because there is no relation to lock yet, so every worker passes it and the losers fail on a
    system-catalog unique index (`pg_namespace_nspname_index` / `pg_type_typname_nsp_index`).
    Because the pass is one transaction, that rolls the entire `ensure_schema` back and surfaces as
    a failed request in whichever catalog call triggered it.

    Verified to FAIL without the `pg_advisory_xact_lock` line: 3 of these 4 workers raise
    IntegrityError. (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` is safe by contrast — it serialises
    on the table's AccessExclusive lock and the loser gets a skip NOTICE — so this bites hardest on
    a brand-new tenant, where nothing exists to lock.)"""
    make, tenant = live
    workers = [make() for _ in range(4)]

    results = await asyncio.gather(*(w.ensure_schema(tenant) for w in workers), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, f"{len(failures)}/4 concurrent workers failed: {failures!r}"

    # One schema, built once, complete — not four half-built ones.
    schema = pg_schema(tenant)
    assert await _tables(make(), schema) >= _V2_TABLE_NAMES | _PRE_V2_TABLE_NAMES
    assert await _indexes(make(), schema) >= {"sections_by_doc", "sections_by_facet", "chunks_by_section_level"}


async def test_delete_document_purges_the_v2_structure_rows(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    """`delete_document` is the path DPDP right-to-erasure cascades through, and the v2 tables hold
    verbatim document text (`blocks.text`/`html`) and derived content (`sections.summary`). There
    are no FK cascades, so if `delete_document` does not name these tables the text survives an
    erasure — a compliance defect, not just stale data."""
    make, tenant = live
    cat = make()
    schema = pg_schema(tenant)
    await cat.ensure_schema(tenant)

    async with cat._engine.begin() as conn:
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".documents (doc_id, source_id, source_kind, mime, ingested_at) '
                "VALUES ('doomed', 's', 'upload', 'application/pdf', now()), "
                "       ('keeper', 's', 'upload', 'application/pdf', now())"
            )
        )
        for doc in ("doomed", "keeper"):
            await conn.execute(
                text(
                    f'INSERT INTO "{schema}".sections (section_id, doc_id, breadcrumb, summary) '
                    "VALUES (:sid, :d, 'Attachment-1 > 13', 'derived summary text')"
                ),
                {"sid": f"{doc}::s0", "d": doc},
            )
            await conn.execute(
                text(f'INSERT INTO "{schema}".blocks (block_id, section_id, doc_id, text) VALUES (:b, :s, :d, :t)'),
                {"b": f"{doc}::b0", "s": f"{doc}::s0", "d": doc, "t": "verbatim document text"},
            )
            await conn.execute(
                text(f"INSERT INTO \"{schema}\".facets (doc_id, facet_id, name) VALUES (:d, 'f1', 'Compliance')"),
                {"d": doc},
            )
            await conn.execute(
                text(
                    f"INSERT INTO \"{schema}\".doc_relations (doc_id, rel_type, target_ref) VALUES (:d, 'amends', 'X')"
                ),
                {"d": doc},
            )
            await conn.execute(
                text(f"INSERT INTO \"{schema}\".map_cache (cache_key, section_id, items) VALUES (:k, :s, '[]')"),
                {"k": f"{doc}-key", "s": f"{doc}::s0"},
            )

    await cat.delete_document(tenant, "doomed")

    async with cat._engine.connect() as conn:
        for table in ("sections", "blocks", "facets", "doc_relations"):
            rows = (await conn.execute(text(f'SELECT doc_id FROM "{schema}".{table}'))).all()
            assert [str(r.doc_id) for r in rows] == ["keeper"], f"{table} not purged for the deleted doc"
        cached = (await conn.execute(text(f'SELECT cache_key FROM "{schema}".map_cache'))).all()
        assert [str(r.cache_key) for r in cached] == ["keeper-key"]
        docs = (await conn.execute(text(f'SELECT doc_id FROM "{schema}".documents'))).all()
        assert [str(r.doc_id) for r in docs] == ["keeper"]


async def test_erasure_purges_query_log_rows_by_citation(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    """The DPDP erasure scrub of `query_log`, against real seeded rows.

    Seeding matters: the purge is a no-op against the empty table it is today, so a test that only
    showed "0 rows deleted from an empty table" would prove nothing about the overlap semantics —
    which are the whole point, since a row must die if it cited ANY erased document.
    """
    make, tenant = live
    cat = make()
    schema = pg_schema(tenant)
    await cat.ensure_schema(tenant)

    rows = [
        ("cites-target-only", ["doomed"]),
        ("cites-target-and-other", ["doomed", "keeper"]),  # partially erased: still quotes `doomed`
        ("cites-other-only", ["keeper"]),
        ("cites-nothing-empty", []),
        ("cites-nothing-null", None),
    ]
    async with cat._engine.begin() as conn:
        for query, cited in rows:
            await conn.execute(
                text(
                    f'INSERT INTO "{schema}".query_log (ts, query, answer, cited_doc_ids) '
                    "VALUES (now(), :q, 'an answer quoting the document', :c)"
                ),
                {"q": query, "c": cited},
            )

    async def surviving() -> list[str]:
        async with cat._engine.connect() as conn:
            got = (await conn.execute(text(f'SELECT query FROM "{schema}".query_log ORDER BY query'))).all()
        return [str(r.query) for r in got]

    assert len(await surviving()) == 5

    purged = await cat.delete_query_log_citing(tenant, {"doomed"})
    assert purged == 2, "both the target-only row and the partially-erased row must go"
    assert await surviving() == ["cites-nothing-empty", "cites-nothing-null", "cites-other-only"], (
        "a row citing nothing in the erased set — including NULL/empty citations — must survive"
    )

    # Idempotent: a retried erasure must not fail or double-count.
    assert await cat.delete_query_log_citing(tenant, {"doomed"}) == 0
    assert len(await surviving()) == 3

    # An empty target set purges nothing, even with rows present.
    assert await cat.delete_query_log_citing(tenant, set()) == 0
    assert len(await surviving()) == 3

    # The remaining document can still be erased on its own.
    assert await cat.delete_query_log_citing(tenant, {"keeper"}) == 1
    assert await surviving() == ["cites-nothing-empty", "cites-nothing-null"]


async def test_the_cited_doc_ids_purge_uses_the_gin_index(
    live: tuple[Callable[[], PostgresCatalog], str],
) -> None:
    """The index exists so the purge is not a full sweep of a growing log. Asserted on the real
    planner rather than on the DDL string, with a row estimate high enough that a seq scan is not
    simply the cheaper plan."""
    make, tenant = live
    cat = make()
    schema = pg_schema(tenant)
    await cat.ensure_schema(tenant)

    async with cat._engine.begin() as conn:
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".query_log (ts, query, cited_doc_ids) '
                "SELECT now(), 'q' || g, ARRAY['doc' || mod(g, 500)] FROM generate_series(1, 5000) g"
            )
        )
        await conn.execute(text(f'ANALYZE "{schema}".query_log'))
        plan = (
            await conn.execute(
                text(f"EXPLAIN SELECT id FROM \"{schema}\".query_log WHERE cited_doc_ids && ARRAY['doc7']")
            )
        ).all()

    rendered = "\n".join(str(r[0]) for r in plan)
    assert "query_log_by_cited_doc" in rendered, f"the GIN index is not used:\n{rendered}"
