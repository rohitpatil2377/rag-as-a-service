"""The RAG v2 per-tenant DDL (upgrade plan §1.1) — shape and idempotency, no infra needed.

`ensure_schema` runs on EVERY catalog read and write, against the 25 live per-KB schemas, so the
one property the DDL cannot lose is that each statement is a no-op the second time. This asserts
that structurally — on the statement list itself — so a future statement that forgets its
`IF NOT EXISTS` guard fails here rather than on a live schema. The live round trip (run it twice
against a real Postgres, then tear down) is `tests/integration/test_v2_catalog_ddl.py`.
"""

from __future__ import annotations

import re

from rag_clients.postgres_catalog import _V2_TABLES, _ddl, _index_ddl, _table_ddl

_SCHEMA = "t_unit_test"
_STATEMENTS = _ddl(_SCHEMA)
# What one cold `ensure_schema` actually executes: the DDL, plus the advisory lock taken as the
# FIRST statement of each of its two transactions (tables/columns, then indexes).
_PASS_STATEMENTS = len(_STATEMENTS) + 2


def test_every_statement_is_guarded_so_a_second_run_is_a_no_op() -> None:
    allowed = (
        "CREATE SCHEMA IF NOT EXISTS",
        "CREATE TABLE IF NOT EXISTS",
        "CREATE INDEX IF NOT EXISTS",
    )
    for stmt in _STATEMENTS:
        head = " ".join(stmt.split())
        guarded = head.startswith(allowed) or re.match(r"ALTER TABLE \S+ ADD COLUMN IF NOT EXISTS ", head)
        assert guarded, f"statement is not idempotent: {head[:120]}"


def test_the_pre_existing_tables_are_extended_not_duplicated() -> None:
    """The plan's §1.1 proposes fresh `documents`/`chunks` tables; tables of both names already
    exist in every tenant schema, so there must be exactly ONE CREATE for each — the original —
    and the v2 fields must arrive as ALTERs."""
    for table in ("documents", "chunks", "tenant_config"):
        creates = [s for s in _STATEMENTS if f'CREATE TABLE IF NOT EXISTS "{_SCHEMA}".{table} ' in s]
        assert len(creates) == 1, f"{table} should be created exactly once, got {len(creates)}"


def test_the_v2_document_profile_columns_are_added_to_the_existing_documents_table() -> None:
    alters = {
        m.group(1)
        for s in _STATEMENTS
        if (m := re.match(rf'ALTER TABLE "{_SCHEMA}"\.documents ADD COLUMN IF NOT EXISTS (\w+) ', s))
    }
    assert {
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
        "content_hash",  # pre-existing (FR-A8/C9), still migration-safe
    } <= alters


def test_the_plans_source_uri_is_mapped_onto_the_existing_uri_column_not_duplicated() -> None:
    assert not any("source_uri" in s for s in _STATEMENTS)
    assert any("uri TEXT" in s and ".documents" in s for s in _STATEMENTS)


def test_the_v2_structure_columns_are_added_to_the_existing_chunks_table() -> None:
    alters = {
        m.group(1)
        for s in _STATEMENTS
        if (m := re.match(rf'ALTER TABLE "{_SCHEMA}"\.chunks ADD COLUMN IF NOT EXISTS (\w+) ', s))
    }
    assert alters == {
        "section_id",
        "level",
        "parent_id",
        "embed_text",
        "page_start",
        "page_end",
        "token_count",
        "chunk_hash",
        # Not one of the plan's nine. `Chunk.sequence` is document order, and it is the ONLY
        # ordering key a v2 chunk has (markup-derived text has no span), so Phase 6 promotes it
        # onto a column with the rest rather than leaving the ORDER BY to deserialise `record`.
        "sequence",
    }, "the plan's eight promoted chunk fields (not `text`, which stays in `record`), plus `sequence`"


def test_the_chunk_text_is_not_denormalised_out_of_record() -> None:
    """`record` is the serialised `Chunk` and the source of truth for citations; a second copy of
    the text as a column could disagree with it."""
    assert not any(re.match(rf'ALTER TABLE "{_SCHEMA}"\.chunks ADD COLUMN IF NOT EXISTS text ', s) for s in _STATEMENTS)


def test_the_created_tables_are_exactly_the_declared_ones() -> None:
    """`_V2_TABLES` drives the teardown, so it drifting from the DDL would silently leave a table
    undroppable. Asserted as SET EQUALITY over the names regexed out of the DDL, in both directions
    — iterating `_V2_TABLES` alone (the earlier version of this test) could not catch a table added
    to the DDL and forgotten in the tuple."""
    created = {
        m.group(1) for stmt in _STATEMENTS if (m := re.search(rf'CREATE TABLE IF NOT EXISTS "{_SCHEMA}"\.(\w+) ', stmt))
    }
    assert created == set(_V2_TABLES) | {"documents", "chunks", "tenant_config"}


def test_the_two_phases_partition_the_statement_list_in_order() -> None:
    """`ensure_schema` runs `_table_ddl` then `_index_ddl` in separate transactions; `_ddl` is the
    whole-list view the guard test above asserts over. They must not drift apart."""
    assert _table_ddl(_SCHEMA) + _index_ddl(_SCHEMA) == _STATEMENTS


def test_no_index_is_built_in_the_table_and_column_phase() -> None:
    """An index build only needs ShareLock, but phase 1 holds AccessExclusive on
    `documents`/`chunks` for its whole transaction — which blocks READS too. Keeping the builds out
    of that transaction is what bounds how long readers are held off, and the index build's cost
    scales with row count, so this matters more on bigger tenants."""
    assert not any("CREATE INDEX" in stmt for stmt in _table_ddl(_SCHEMA))
    assert all("CREATE INDEX" in stmt for stmt in _index_ddl(_SCHEMA))


def test_phase_one_creates_every_table_the_indexes_need() -> None:
    """Phase 2 commits after phase 1, so an index on a table phase 1 forgot would fail every call."""
    created = {
        m.group(1)
        for stmt in _table_ddl(_SCHEMA)
        if (m := re.search(rf'CREATE TABLE IF NOT EXISTS "{_SCHEMA}"\.(\w+) ', stmt))
    }
    indexed = {m.group(1) for stmt in _index_ddl(_SCHEMA) if (m := re.search(rf'ON "{_SCHEMA}"\.(\w+)', stmt))}
    assert indexed <= created, f"index on a table phase 1 does not create: {sorted(indexed - created)}"


def test_the_plan_indexes_are_created_in_the_existing_naming_style() -> None:
    joined = " ".join(" ".join(s.split()) for s in _STATEMENTS)
    assert f'CREATE INDEX IF NOT EXISTS sections_by_doc ON "{_SCHEMA}".sections (doc_id)' in joined
    assert f'CREATE INDEX IF NOT EXISTS sections_by_facet ON "{_SCHEMA}".sections USING GIN (facet_ids)' in joined
    assert f'CREATE INDEX IF NOT EXISTS chunks_by_section_level ON "{_SCHEMA}".chunks (section_id, level)' in joined


def test_no_v2_collection_or_index_namespace_is_introduced() -> None:
    """Phase 1 is Postgres only — the plan's §1.2/§1.3 `_v2` Qdrant collection and OpenSearch index
    are deferred (discovery §5), so nothing here may name one."""
    assert not any("_v2" in s for s in _STATEMENTS)


# --- the DDL must not be paid on every catalog call ---------------------------------------------


class _CountingConn:
    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    async def execute(self, *_: object, **__: object) -> None:
        self._counter[0] += 1


class _CountingEngine:
    """Counts DDL statements executed, so the memo is observable without a database."""

    def __init__(self) -> None:
        self.executed = [0]

    def begin(self) -> _CountingEngine:
        return self

    async def __aenter__(self) -> _CountingConn:
        return _CountingConn(self.executed)

    async def __aexit__(self, *_: object) -> None:
        return None


def _catalog() -> tuple[object, _CountingEngine]:
    from rag_clients.postgres_catalog import PostgresCatalog

    engine = _CountingEngine()
    return PostgresCatalog(engine=engine), engine  # type: ignore[arg-type]


def test_the_ddl_runs_once_per_tenant_not_once_per_catalog_call() -> None:
    """`ensure_schema` is what every catalog read and write funnels through. Measured against the
    dev Postgres, a steady-state run of this statement list — every statement already satisfied —
    costs p50 61ms, because each is a round-trip and each ALTER takes its lock before finding
    nothing to do; the pre-v2 list cost ~16ms. Unmemoised, the structure tables would therefore put
    ~45ms on EVERY catalog operation, chunk hydration during retrieval included."""
    import asyncio

    cat, engine = _catalog()
    asyncio.run(cat.ensure_schema("acme"))  # type: ignore[attr-defined]
    first = engine.executed[0]
    assert first == _PASS_STATEMENTS, "the first call must actually run the DDL"
    for _ in range(10):
        asyncio.run(cat.ensure_schema("acme"))  # type: ignore[attr-defined]
    assert engine.executed[0] == first, f"the DDL re-ran {engine.executed[0] - first} extra statements"


def test_each_tenant_is_ensured_on_its_own() -> None:
    """A memo keyed too broadly would leave a second KB's schema unbuilt."""
    import asyncio

    cat, engine = _catalog()
    asyncio.run(cat.ensure_schema("acme"))  # type: ignore[attr-defined]
    asyncio.run(cat.ensure_schema("globex"))  # type: ignore[attr-defined]
    assert engine.executed[0] == 2 * _PASS_STATEMENTS


def test_dropping_the_v2_tables_invalidates_the_memo() -> None:
    """Otherwise the teardown would leave the process believing tables it just deleted exist."""
    import asyncio

    cat, engine = _catalog()
    asyncio.run(cat.ensure_schema("acme"))  # type: ignore[attr-defined]
    asyncio.run(cat.drop_v2_tables("acme"))  # type: ignore[attr-defined]
    before = engine.executed[0]
    asyncio.run(cat.ensure_schema("acme"))  # type: ignore[attr-defined]
    assert engine.executed[0] == before + _PASS_STATEMENTS, "ensure_schema did not rebuild after the drop"


def test_query_log_rows_are_attributable_to_the_documents_they_quoted() -> None:
    """Without `cited_doc_ids` a log row cannot be purged on erasure, and `query_log.answer` will
    hold text lifted from the corpus — so this column is what makes the erasure certificate
    truthful, not a convenience. Declared in the CREATE and re-applied as an ALTER, like
    `documents.content_hash`, because the table already exists on anything that ran the first
    Phase-1 commit and `CREATE TABLE IF NOT EXISTS` would no-op."""
    create = next(s for s in _STATEMENTS if f'CREATE TABLE IF NOT EXISTS "{_SCHEMA}".query_log ' in s)
    assert "cited_doc_ids TEXT[]" in create
    assert f'ALTER TABLE "{_SCHEMA}".query_log ADD COLUMN IF NOT EXISTS cited_doc_ids TEXT[]' in _STATEMENTS, (
        "the column must also be added to a query_log table that already exists"
    )


def test_the_citation_purge_is_index_backed() -> None:
    """The purge matches with array containment (`&&`), which needs GIN to avoid sweeping a growing
    log on every erasure. (That the planner actually picks it is asserted against a live Postgres in
    tests/integration/test_v2_catalog_ddl.py.)"""
    assert (
        f'CREATE INDEX IF NOT EXISTS query_log_by_cited_doc ON "{_SCHEMA}".query_log USING GIN (cited_doc_ids)'
        in _index_ddl(_SCHEMA)
    )
