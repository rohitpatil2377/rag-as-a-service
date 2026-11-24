"""Postgres metadata/lineage catalog adapter (implements :class:`CatalogAdapter`).

FR-G7 (lineage raw->chunk->answer) + FR-H1 (schema-per-tenant isolation). Each tenant
gets its own Postgres **schema** (``t_<slug>``) holding ``documents``, ``chunks``,
``tenant_config`` and the RAG v2 structure tables (see :func:`_v2_ddl`). The schema name is derived
from
:func:`rag_common.tenancy.pg_schema`, which validates the slug, so the interpolated
identifier is safe and every read/write is structurally tenant-scoped.

OpenMetadata/DataHub is the deferred richer catalog (Phase-1) behind this same Protocol.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_clients.layout_rows import flatten_layout_tree
from rag_common import get_logger, get_settings, pg_schema
from rag_core.chunking import text_hash
from rag_core.doc_profile import DocProfile
from rag_core.facet_vocab import CanonicalFacet, FacetVocabulary
from rag_core.graph import DeclaredRelation
from rag_core.layout import LayoutTree
from rag_core.mapreduce import MapCacheEntry
from rag_core.router import DocCard
from rag_core.section_enrich import EnrichmentManifest, SectionEnrichment
from rag_core.strategies import SectionCard
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import SourceKind

_log = get_logger(__name__)


# Serialises concurrent first-touch DDL per schema — see `ensure_schema`. An ADVISORY XACT lock is
# used rather than a table lock because the objects being created may not exist yet (so there is
# nothing to lock), and the `xact` flavour is released by COMMIT/ROLLBACK, so a crashed worker
# cannot leak it. `hashtext` keys it on the schema name, so two different tenants never block each
# other.
_ADVISORY_LOCK = text("SELECT pg_advisory_xact_lock(hashtext(:schema))")


def _ddl(schema: str) -> list[str]:
    """Every statement `ensure_schema` runs, in order. It executes them as the two phases below;
    this whole-list view is what the structural guard test asserts over."""
    return _table_ddl(schema) + _index_ddl(schema)


def _table_ddl(schema: str) -> list[str]:
    """Phase 1: the schema, its tables and its columns. These take AccessExclusive on
    `documents`/`chunks` for the duration of the transaction, so the list is kept as short as it can
    be and the index builds are split out into :func:`_index_ddl`."""
    return [
        f'CREATE SCHEMA IF NOT EXISTS "{schema}"',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".documents (
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
        )''',
        # Migration-safe for schemas created before FR-A8/C9 (content-hash delta + near-dup).
        f'ALTER TABLE "{schema}".documents ADD COLUMN IF NOT EXISTS content_hash TEXT',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            page INTEGER,
            version INTEGER NOT NULL DEFAULT 1,
            record JSONB NOT NULL
        )''',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".tenant_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )''',
        *_v2_table_ddl(schema),
    ]


def _index_ddl(schema: str) -> list[str]:
    """Phase 2: the indexes, run in their OWN transaction AFTER phase 1 commits.

    Kept separate because an index build takes only ShareLock on its table, while the phase-1
    ALTERs hold AccessExclusive — which blocks READS as well as writes. Running them together meant
    the whole first-touch pass (~390ms on a fresh schema) blocked every reader of `documents` and
    `chunks` for its full duration, and the duration of an index build scales with row count, so it
    got worse the bigger the tenant. Split, readers are only held off for the brief ALTERs."""
    return [
        f'CREATE INDEX IF NOT EXISTS documents_by_hash ON "{schema}".documents (content_hash)',
        # Plan §1.1's indexes, named in the existing style (cf. `documents_by_hash`); Postgres puts
        # an index in its table's schema, so the bare name is already tenant-scoped.
        f'CREATE INDEX IF NOT EXISTS sections_by_doc ON "{schema}".sections (doc_id)',
        f'CREATE INDEX IF NOT EXISTS sections_by_facet ON "{schema}".sections USING GIN (facet_ids)',
        # Not in the plan's index list, added with the structure WRITE path: `blocks` is purged by
        # `doc_id` on every ingest and every reindex of every document (see
        # `replace_document_structure`, which replaces rather than appends), so without this each
        # one costs a sequential scan of the tenant's whole `blocks` table and re-indexing a corpus
        # is quadratic in it. `sections` is already covered by `sections_by_doc`.
        f'CREATE INDEX IF NOT EXISTS blocks_by_doc ON "{schema}".blocks (doc_id)',
        f'CREATE INDEX IF NOT EXISTS chunks_by_section_level ON "{schema}".chunks (section_id, level)',
        # `chunks` is read and purged BY DOCUMENT on every path that touches one — `document_chunks`
        # (citation viewer, whole-document answering, neighbour expansion), `delete_document`,
        # `chunk_hashes` (the re-embed skip below) — and had no index for it, so each was a
        # sequential scan of the tenant's whole chunk table and got worse the bigger the tenant.
        # Same reasoning as `blocks_by_doc` above, which was added with the structure write path.
        f'CREATE INDEX IF NOT EXISTS chunks_by_doc ON "{schema}".chunks (doc_id)',
        # Array CONTAINMENT (`&&`) is how the erasure purge finds rows citing an erased document,
        # and a GIN index is what makes that an index scan instead of a full sweep of the log.
        f'CREATE INDEX IF NOT EXISTS query_log_by_cited_doc ON "{schema}".query_log USING GIN (cited_doc_ids)',
    ]


# --- RAG v2 data model (upgrade plan §1.1) ------------------------------------------------------
# The plan writes §1.1 as a fresh `documents`/`chunks` schema, but tables of BOTH names already
# exist in every one of the live per-KB schemas above, so a rival pair would collide. The v2 columns
# are therefore added to the EXISTING tables with `ADD COLUMN IF NOT EXISTS` — the same additive
# idiom `documents.content_hash` already uses, and the one that works on the already-provisioned
# deployments where `CREATE TABLE IF NOT EXISTS` silently no-ops (see ADR-0004).
#
# Alembic is not involved: it manages only the shared `public` / `rag_identity` schemas (head
# `0007_tenants`). Per-tenant tables have always come from this runtime DDL, called by
# `ensure_schema`, which every catalog read and write funnels through.
#
# Every added column is NULLABLE with no DEFAULT, which is what makes this safe to run against the
# live schemas: Postgres records the column in the catalog and does not rewrite the heap. Verified
# on this deployment's PG 16.15 over a 20 000-row table — relfilenode unchanged, total relation size
# unchanged, every row intact. A column with a DEFAULT, or NOT NULL, would not have that property.
#
# These statements are UNCONDITIONAL — deliberately NOT gated on `settings.v2_ingest`. A
# conditional DDL would make a tenant's schema SHAPE depend on which env happened to be set the
# last time any code path touched that tenant, which is precisely the schema-drift class of bug
# that `documents.content_hash`, `federation_grants.parent_grant_id` and
# `eval_schedules.federate_from` each had to be repaired for. The new tables and columns are empty
# and additive, and nothing reads or writes them while the flag is off, so creating them always
# costs one DDL pass per tenant per worker (see `ensure_schema`'s memo and ADR-0023) and buys a
# single schema shape everywhere.
_V2_TABLES = ("doc_relations", "facets", "sections", "blocks", "map_cache", "query_log")


def _v2_table_ddl(schema: str) -> list[str]:
    return [
        # Document profile card (plan §1.1 + §3). `source_uri` from the plan is NOT added: the
        # existing `documents.uri` column already IS it, so the plan's field maps onto that rather
        # than being duplicated. `minio_raw_key` is likewise absent — the raw object key is derived
        # deterministically from the tenant's `object_prefix` + doc id, so there is nothing to store.
        *(
            f'ALTER TABLE "{schema}".documents ADD COLUMN IF NOT EXISTS {col}'
            for col in (
                "title TEXT",
                "doc_type TEXT",
                "purpose TEXT",
                "issuer TEXT",
                "issue_date DATE",
                "lang TEXT",
                "profile_json JSONB",
                "profile_version INTEGER",
                "status TEXT",
                "minio_layout_key TEXT",
                # WHAT PHASE 4 MANAGED TO DO TO THIS DOCUMENT, bound to the bytes it describes.
                # Not reconstructible after the fact: a `sections` row with no summary cannot say
                # whether the minimum-length rule DECLINED the section or a model call FAILED on
                # it, and only one of those is a defect. Without it an answer served a day later
                # cannot know that part of the document was never readable.
                "enrichment_manifest JSONB",
            )
        ),
        # The plan's §1.1 `chunks` adds nine fields to the existing table; all of them are added
        # here EXCEPT `text`. The structure coordinates are promoted onto real columns because they
        # are QUERIED and FILTERED on (`chunks(section_id, level)`), and the rest are the per-chunk
        # facts the v2 retrieval path needs without deserialising `record`. `text` is deliberately
        # omitted: `record` is the serialised `Chunk` and the single source of truth for citations,
        # so a second copy of the body could disagree with it. `embed_text` is NOT the same thing —
        # it is the contextualised string actually sent to the embedder, which differs from the
        # displayed text by design (ADR-0007) and is not stored anywhere today.
        *(
            f'ALTER TABLE "{schema}".chunks ADD COLUMN IF NOT EXISTS {col}'
            for col in (
                "section_id TEXT",
                "level TEXT",
                "parent_id TEXT",
                "embed_text TEXT",
                "page_start INTEGER",
                "page_end INTEGER",
                "token_count INTEGER",
                "chunk_hash TEXT",
                # NOT one of the plan's nine: `Chunk.sequence` is document order, which the plan
                # did not foresee needing because it assumed v2 chunks would keep a span. They do
                # not — markup-derived text has no offset into the extracted text — so `sequence`
                # IS the ordering key for every v2 chunk and belongs beside the rest of them.
                # `sequence` is a non-reserved keyword in Postgres and is legal unquoted as a
                # column name (verified on this deployment's PG 16.15), so it needs no quoting
                # here or in the ORDER BY that reads it.
                "sequence INTEGER",
            )
        ),
        # No FOREIGN KEY clauses anywhere below, matching the existing tables: `chunks.doc_id` has
        # never been a declared FK either, and a hard FK would make a partially indexed document
        # un-cleanable by the delete paths. What replaces the FK is NOT inherited behaviour — the
        # existing purge paths knew nothing about these tables — so `delete_document` below was
        # extended to delete the per-document rows explicitly. See the note there about `query_log`,
        # the one new table a document-scoped delete cannot reach.
        f'''CREATE TABLE IF NOT EXISTS "{schema}".doc_relations (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            rel_type TEXT NOT NULL,
            target_ref TEXT,
            target_doc_id TEXT,
            target_section_id TEXT,
            evidence_quote TEXT
        )''',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".facets (
            doc_id TEXT NOT NULL,
            facet_id TEXT NOT NULL,
            name TEXT,
            description TEXT,
            revision TEXT,
            aliases JSONB,
            PRIMARY KEY (doc_id, facet_id)
        )''',
        # `revision` PINS THE VOCABULARY TO THE BYTES IT DESCRIBES (content_hash + profile
        # version), so a re-ingest of unchanged bytes reuses these ids instead of letting Phase 3
        # invent new ones -- measured: three ingests of one document produced three 10-facet
        # vocabularies sharing ONE id, and a facet id is Phase 8's facet-union join key and,
        # de-underscored, its cross-encoder probe. `aliases` is the APPROVED alias registry for a
        # facet; it is matched exactly and never fuzzily. See `rag_core.facet_vocab`.
        #
        # Declared in the CREATE above AND as ALTERs, like `documents.content_hash`: a deployment
        # that already has this table would silently no-op on CREATE IF NOT EXISTS (ADR-0004).
        f'ALTER TABLE "{schema}".facets ADD COLUMN IF NOT EXISTS revision TEXT',
        f'ALTER TABLE "{schema}".facets ADD COLUMN IF NOT EXISTS aliases JSONB',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".sections (
            section_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            parent_section_id TEXT,
            depth INTEGER,
            ordinal INTEGER,
            heading TEXT,
            breadcrumb TEXT,
            page_start INTEGER,
            page_end INTEGER,
            summary TEXT,
            key_values JSONB,
            facet_ids TEXT[],
            amends JSONB,
            text_hash TEXT,
            token_count INTEGER
        )''',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".blocks (
            block_id TEXT PRIMARY KEY,
            section_id TEXT,
            doc_id TEXT NOT NULL,
            block_type TEXT,
            page INTEGER,
            bbox JSONB,
            ordinal INTEGER,
            text TEXT,
            html TEXT,
            parent_block_id TEXT
        )''',
        # The `table` a `table_row` came from (`LayoutBlock.parent_block_id`). Phase 1 had no column
        # for it, so a row would have landed addressable but orphaned and Phase 2b would have had to
        # re-derive the relation from adjacency — which stops being derivable the moment a merged
        # cross-page table interleaves rows from two pages. Declared in the CREATE above AND as an
        # ALTER here, exactly like `documents.content_hash` and `query_log.cited_doc_ids`: a
        # deployment that already ran the Phase-1 commit has the table without this column, and
        # `CREATE TABLE IF NOT EXISTS` would silently no-op (ADR-0004).
        f'ALTER TABLE "{schema}".blocks ADD COLUMN IF NOT EXISTS parent_block_id TEXT',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".map_cache (
            cache_key TEXT PRIMARY KEY,
            section_id TEXT,
            query_sig TEXT,
            items JSONB,
            model_version TEXT,
            created_at TIMESTAMPTZ
        )''',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".query_log (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ,
            query TEXT,
            router_json JSONB,
            selected_sections TEXT[],
            answer TEXT,
            verification JSONB,
            latency_ms JSONB,
            cited_doc_ids TEXT[]
        )''',
        # `cited_doc_ids` is what makes a log row ATTRIBUTABLE to the documents it quoted, so DPDP
        # erasure can purge it by citation — the same contract the derived chat history already
        # honours (`ConversationStore.delete_citing`). See `delete_query_log_citing` and
        # `docs/automated-erasure.md`. Declared in the CREATE above AND as an ALTER here, exactly
        # like `documents.content_hash`: a deployment that already ran the first Phase-1 commit has
        # the table without this column, and `CREATE TABLE IF NOT EXISTS` would silently no-op.
        f'ALTER TABLE "{schema}".query_log ADD COLUMN IF NOT EXISTS cited_doc_ids TEXT[]',
    ]


class PostgresCatalog:
    """Metadata/lineage catalog, schema-per-tenant."""

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
        # Schemas this PROCESS has already brought up to date — see ensure_schema.
        self._ensured: set[str] = set()

    async def ensure_schema(self, tenant_id: str) -> None:
        """Create the tenant schema + tables if absent (idempotent), ONCE per process per tenant.

        Every catalog read and write funnels through here, so the DDL's cost is paid on the hot
        path — including chunk hydration during retrieval. Measured against the dev Postgres, a
        steady-state run of the full statement list (where every statement is already satisfied and
        does nothing) costs p50 61ms / p95 88ms, because each one is still a round-trip and each
        `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` still takes its lock on `documents`/`chunks`
        before discovering there is nothing to do. The same list trimmed to its pre-v2 contents
        costs ~16ms, so carrying the structure tables unconditionally would have put ~45ms on every
        catalog call.

        The statements are idempotent, so running them once per process is sufficient: the memo
        keeps the drift-free property (a schema is still brought fully up to date the first time
        anything touches it, with no dependency on which env var was set) while costing one DDL
        pass per tenant per worker instead of one per operation. See ADR-0023 for the measurements
        and the trade-off.

        CONCURRENCY. `IF NOT EXISTS` is not a concurrency primitive, and the two halves differ:

        * `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` IS concurrency-safe — it serialises on the
          table's AccessExclusive lock, and the loser re-checks after acquiring it and emits a
          skip NOTICE.
        * `CREATE SCHEMA/TABLE IF NOT EXISTS` is NOT. The existence check takes no lock (there is
          no relation to lock yet), so two workers both pass it and the loser fails on the system
          catalog's unique index — `pg_namespace_nspname_index` for the schema,
          `pg_type_typname_nsp_index` for a table. Reproduced on this Postgres: 4 racing workers on
          one fresh schema, 3 failed. Because the pass is one transaction, that rolls the WHOLE
          `ensure_schema` back and surfaces as a failed request in whichever catalog call triggered
          it. (This pre-dates the v2 tables — the pre-v2 statement list fails identically — but 4
          api instances x 2 workers provisioning a new tenant hit it for real.)

        `_ADVISORY_LOCK` is therefore the FIRST statement in each transaction, serialising workers
        per schema so the loser's `IF NOT EXISTS` runs AFTER the winner commits and correctly finds
        nothing to do. One extra round trip, paid once per worker per tenant thanks to the memo.

        The pass is split into two transactions (tables/columns, then indexes) so the AccessExclusive
        ALTERs are not held for the index builds — see :func:`_index_ddl`. The memo is set only after
        BOTH commit, so a failure in either leaves the schema to be retried on the next call.

        OPERATOR NOTE: the memo trades "self-heals on the next catalog call" for "self-heals on the
        next process". Hand-editing a tenant schema with `psql` against a warm worker no longer
        repairs itself on the next request — the worker still believes the schema is current — so
        such surgery now needs an api restart to be picked up."""
        schema = pg_schema(tenant_id)
        if schema in self._ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            for stmt in _table_ddl(schema):
                await conn.execute(text(stmt))
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            for stmt in _index_ddl(schema):
                await conn.execute(text(stmt))
        self._ensured.add(schema)
        _log.info("catalog_schema_ready", schema=schema)

    async def drop_v2_tables(self, tenant_id: str) -> None:
        """Documented TEARDOWN for the RAG v2 tables created by :func:`_v2_ddl` (plan §1.1
        acceptance: "migrations run cleanly up and down").

        There is no Alembic down-path for per-tenant tables because there is no Alembic up-path
        either, so the revert is this explicit helper. It drops exactly the six NEW tables and
        nothing else; it is idempotent (``IF EXISTS``), and it is the only thing the DDL test uses
        to clean up after itself.

        It deliberately does NOT drop the additive COLUMNS on ``documents``/``chunks``. Dropping an
        empty table that this code created is reversible by re-running ``ensure_schema``; dropping a
        column off a live, populated table that holds a tenant's corpus is not the same act, and an
        all-NULL column is not residue. It is also deliberately NOT reachable from any API route:
        this method is not on the ``CatalogAdapter`` Protocol, so ``get_catalog()`` — which every
        service holds as that Protocol — cannot see it. The KB-delete path is unchanged and still
        never drops a tenant's Postgres schema (see ``pipeline.delete_knowledge_base``, which drops
        the Qdrant collection and OpenSearch index but purges Postgres per document); this helper is
        strictly narrower than that path and is not wired into it."""
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            # Same lock `ensure_schema` takes, for the same reason: a DROP interleaved with another
            # worker's CREATE would leave the schema half-built with no error anywhere.
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            for table in _V2_TABLES:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{schema}".{table}'))
        # The schema no longer matches what ensure_schema's memo says it does, so forget it: the
        # next catalog touch rebuilds the tables instead of trusting a stale "already ensured".
        self._ensured.discard(schema)
        _log.info("catalog_v2_tables_dropped", schema=schema, tables=len(_V2_TABLES))

    async def set_layout_key(self, tenant_id: str, doc_id: str, key: str) -> bool:
        """Point ``documents.minio_layout_key`` at this document's stored layout overlay.

        An UPDATE, not part of ``register_document``: the overlay is written by the ingest path
        AFTER the row exists, and ``register_document``'s ``ON CONFLICT`` deliberately refreshes
        only ``ingested_at``/``content_hash``/``metadata``, so a later re-register of the same
        document cannot blank this column back to NULL.

        Returns False when the document has no row (nothing to point at) so the caller can tell
        "recorded" from "silently did nothing"; the UPDATE itself is idempotent.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.record_layout_key`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f'UPDATE "{schema}".documents SET minio_layout_key = :k WHERE doc_id = :d'),
                {"k": key, "d": doc_id},
            )
        return bool(result.rowcount)

    async def replace_document_structure(self, tenant_id: str, tree: LayoutTree) -> tuple[int, int]:
        """REPLACE this document's stored structure tree with ``tree``. Returns ``(sections, blocks)``
        written.

        REPLACE, not append — and that is the whole contract. ``reindex_document`` re-runs this for
        a document that already has a tree, and the ids are content-derived, so an append would
        re-insert the unchanged rows (primary-key violation) or, worse under an upsert, leave behind
        every section and block the new extraction no longer produces: a tree that grows on every
        reindex and describes text that is not being served. So the document's rows are DELETED
        first and the new tree inserted whole.

        ONE transaction, so there is no window in which the document has partial structure or none:
        a reader either sees the whole old tree or the whole new one. It takes the same
        ``_ADVISORY_LOCK`` the rest of this module takes, keyed on the schema — two workers
        re-indexing the same document (a retried job, or an ingest racing a reindex) would otherwise
        interleave their delete and insert halves and produce a tree that is half of each.

        Safe to run twice (the second run deletes exactly what the first inserted and re-inserts it)
        and safe on a document with no stored structure yet (the deletes match nothing). An empty
        ``tree`` is a legitimate instruction: it PURGES the document's structure, which is what a
        re-extraction that produced none means.

        ``map_cache`` goes first and for the same reason ``delete_document`` purges it first: its
        rows are keyed by ``section_id``, so they must not outlive the sections they point at. A
        section surviving the replace with the same id has still had its text replaced, so a cached
        map over it is stale either way — a cache is the one thing it is always safe to drop.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.persist_document_structure`` is the seam services call."""
        doc_id = tree.doc_id
        sections, blocks = flatten_layout_tree(tree)
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            await conn.execute(
                text(
                    f'DELETE FROM "{schema}".map_cache WHERE section_id IN '
                    f'(SELECT section_id FROM "{schema}".sections WHERE doc_id = :d)'
                ),
                {"d": doc_id},
            )
            for table in ("blocks", "sections"):
                await conn.execute(text(f'DELETE FROM "{schema}".{table} WHERE doc_id = :d'), {"d": doc_id})
            if sections:
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".sections
                            (section_id, doc_id, parent_section_id, depth, ordinal, heading, breadcrumb,
                             page_start, page_end, text_hash)
                            VALUES (:section_id, :doc_id, :parent_section_id, :depth, :ordinal, :heading,
                                    :breadcrumb, :page_start, :page_end, :text_hash)'''
                    ),
                    [asdict(row) for row in sections],
                )
            if blocks:
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".blocks
                            (block_id, section_id, doc_id, block_type, page, ordinal, text, html,
                             parent_block_id)
                            VALUES (:block_id, :section_id, :doc_id, :block_type, :page, :ordinal, :text,
                                    :html, :parent_block_id)'''
                    ),
                    [asdict(row) for row in blocks],
                )
        _log.info(
            "document_structure_replaced",
            schema=schema,
            doc_id=doc_id,
            sections=len(sections),
            blocks=len(blocks),
            source=tree.source,
        )
        return len(sections), len(blocks)

    async def replace_document_profile(
        self, tenant_id: str, doc_id: str, profile: DocProfile, *, revision: str = ""
    ) -> bool:
        """REPLACE this document's profile card — ``documents`` columns + ``facets`` +
        ``doc_relations`` (RAG v2 Phase 3). Returns False when the document has no catalog row.

        REPLACE, for the reason :meth:`replace_document_structure` states about the structure
        tables and with one addition of its own. A reindex re-profiles, and a facet taxonomy is
        generated per document by a model: the ids a second run produces are NOT the ids the first
        produced (the prompt asks for facets drawn from this document's content, not from a fixed
        vocabulary), so an upsert would accumulate both runs' taxonomies on one document and Phase
        4 would label sections against a union that no single profile call ever described. The
        document's rows are deleted and the new card inserted whole.

        ``revision`` is what makes that churn STOP MATTERING rather than merely be contained. It
        pins these rows to the bytes they describe (`facet_vocab.document_revision`), so the next
        ingest of the same bytes can read them back and REUSE the ids instead of accepting a fresh
        invention — which is what Phase 8's facet union joins on and what its cross-encoder is
        probed with. Empty when the caller cannot name a revision (no content hash); such rows are
        written as before and are simply never reused, because
        :meth:`facet_vocabulary` refuses a vocabulary that cannot say which bytes it describes.

        ONE transaction under the same ``_ADVISORY_LOCK`` the rest of this module takes, so a
        reader never sees a card whose ``profile_json`` names facets that are not in ``facets``
        yet, and two workers re-profiling the same document cannot interleave their halves.

        ``documents`` is UPDATEd, never inserted: the row is written by ``register_document``,
        which this follows on both the ingest and the reindex path. The columns it refreshes are
        exactly the card's; ``register_document``'s ``ON CONFLICT`` does not touch them, so a later
        re-register cannot blank the card back to NULL — the same arrangement
        :meth:`set_layout_key` relies on.

        ``doc_relations.target_doc_id`` / ``target_section_id`` are left NULL. Phase 3 records what
        the document SAYS ("Corrigendum-II"), and resolving that reference to a document in this KB
        is a different act with a different failure mode — a wrong resolution points an amendment
        edge at the wrong document, which is worse than an unresolved one. :meth:`document_relations`
        already LEFT JOINs on it and falls back to the literal ``target_ref``, which is the correct
        node name either way (its own docstring says so: "the day its document is ingested the two
        collapse onto one node by name").

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.persist_document_profile`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            updated = await conn.execute(
                text(
                    f'''UPDATE "{schema}".documents SET
                            title = :title, doc_type = :doc_type, purpose = :purpose, issuer = :issuer,
                            issue_date = :issue_date, lang = :lang, profile_json = CAST(:profile_json AS JSONB),
                            profile_version = :profile_version
                        WHERE doc_id = :doc_id'''
                ),
                {
                    "title": profile.title or None,
                    "doc_type": profile.doc_type,
                    "purpose": profile.purpose,
                    "issuer": profile.issuer,
                    "issue_date": profile.issue_date,
                    "lang": profile.lang,
                    "profile_json": profile.as_json(),
                    "profile_version": profile.profile_version,
                    "doc_id": doc_id,
                },
            )
            if not updated.rowcount:
                # No row to hang the card on. The facets and relations are NOT written in that
                # case: `document_relations` INNER JOINs `documents` for the source node's name, so
                # rows without one would be unreadable by their only consumer while still being
                # counted by anything that trusts the tables.
                return False
            for table in ("facets", "doc_relations"):
                await conn.execute(text(f'DELETE FROM "{schema}".{table} WHERE doc_id = :d'), {"d": doc_id})
            if profile.facets:
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".facets
                            (doc_id, facet_id, name, description, revision, aliases)
                            VALUES (:doc_id, :facet_id, :name, :description, :revision,
                                    CAST(:aliases AS JSONB))'''
                    ),
                    [
                        {
                            "doc_id": doc_id,
                            "facet_id": f.facet_id,
                            "name": f.name,
                            "description": f.description,
                            "revision": revision,
                            # Carried through UNCHANGED. The approved aliases of a reused
                            # vocabulary must survive the rewrite, and a newly adopted taxonomy
                            # has none -- an alias is only ever added by a controlled update.
                            "aliases": json.dumps(list(f.aliases)),
                        }
                        for f in profile.facets
                    ],
                )
            if profile.relations:
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".doc_relations
                            (doc_id, rel_type, target_ref, target_doc_id, target_section_id, evidence_quote)
                            VALUES (:doc_id, :rel_type, :target_ref, NULL, NULL, :evidence_quote)'''
                    ),
                    [
                        {
                            "doc_id": doc_id,
                            "rel_type": r.rel_type,
                            "target_ref": r.target_ref,
                            "evidence_quote": r.evidence_quote,
                        }
                        for r in profile.relations
                    ],
                )
        _log.info(
            "document_profile_replaced",
            schema=schema,
            doc_id=doc_id,
            doc_type=profile.doc_type,
            facets=len(profile.facets),
            relations=len(profile.relations),
            profile_version=profile.profile_version,
        )
        return True

    async def replace_enrichment_manifest(self, tenant_id: str, doc_id: str, manifest: EnrichmentManifest) -> bool:
        """Write what Phase 4 managed to do to this document. Returns whether a row was hit.

        ONE ROW, ONE COLUMN, and the manifest replaces whatever was there. A re-ingest that failed
        must not leave a previous run's success in place -- which is the whole reason the manifest
        carries `content_hash` and the reason this is an unconditional overwrite rather than a
        merge.

        A document row that does not exist is NOT created. The manifest describes a document the
        catalog is supposed to already know, and conjuring a row from an ingest-time side effect
        would put a document in the catalog with no envelope behind it.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations --
        ``rag_clients.factory.persist_enrichment_manifest`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            result = await conn.execute(
                text(f'UPDATE "{schema}".documents SET enrichment_manifest = CAST(:m AS JSONB) WHERE doc_id = :doc_id'),
                {"m": json.dumps(manifest.as_dict()), "doc_id": doc_id},
            )
        hit = bool(result.rowcount)
        _log.info(
            "enrichment_manifest_stored",
            schema=schema,
            doc_id=doc_id,
            stored=hit,
            sections_failed=manifest.sections_failed,
            sections_skipped_policy=manifest.sections_skipped_policy,
            failure_classes=dict(manifest.failure_classes) or None,
        )
        return hit

    async def facet_vocabulary(self, tenant_id: str, doc_id: str) -> FacetVocabulary | None:
        """The stored facet vocabulary for one document, or None when there is nothing to REUSE.

        None, not an empty vocabulary, for: a document with no `facets` rows, rows written before
        `revision` existed, and rows carrying MORE THAN ONE revision. The last is the interesting
        one -- it can only happen if a write interleaved two profiles, and a vocabulary that cannot
        say which single set of bytes it describes must not be reused as if it could. An empty
        vocabulary would instead read as "this document has no facets" and silently disable Phase
        8's facet union for it.

        THE REVISION IS RETURNED, NOT CHECKED, exactly as :meth:`enrichment_manifests` returns the
        content hash: staleness is the caller's judgement, and `facet_vocab.reconcile` is where it
        is made.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations --
        ``rag_clients.factory.facet_vocabulary`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"SELECT facet_id, coalesce(name, '') AS name, coalesce(description, '') AS description, "
                        f"coalesce(revision, '') AS revision, aliases "
                        f'FROM "{schema}".facets WHERE doc_id = :doc_id ORDER BY facet_id'
                    ),
                    {"doc_id": doc_id},
                )
            ).all()
        if not rows:
            return None
        revisions = {r.revision for r in rows}
        if len(revisions) != 1:
            _log.warning(
                "facet_vocabulary_mixed_revisions",
                tenant_id=tenant_id,
                doc_id=doc_id,
                revisions=sorted(revisions)[:4],
                detail="not reused: a vocabulary that cannot name one set of bytes is not evidence about them",
            )
            return None
        revision = revisions.pop()
        if not revision:
            return None
        return FacetVocabulary(
            revision=revision,
            facets=tuple(
                CanonicalFacet(
                    facet_id=r.facet_id,
                    name=r.name,
                    description=r.description,
                    aliases=tuple(str(a) for a in r.aliases) if isinstance(r.aliases, list) else (),
                )
                for r in rows
            ),
        )

    async def enrichment_manifests(
        self, tenant_id: str, doc_ids: Sequence[str]
    ) -> dict[str, tuple[EnrichmentManifest, str]]:
        """The stored manifest for each of ``doc_ids`` that HAS a readable one.

        A document absent from the result has no manifest this code can interpret -- never
        ingested under a version that wrote one, written by a newer version, or stored malformed.
        That is deliberately NOT the same as a manifest saying nothing failed: a caller must
        report it as unknown rather than as complete, or every legacy document silently inherits
        a completeness guarantee it never earned.

        THE HASH IS NOT CHECKED HERE, it is RETURNED. Each value is the stored manifest paired
        with the document row's CURRENT `content_hash`, because this query is the only cheap place
        that knows both and staleness is the caller's judgement to make -- a manifest describing
        bytes the document no longer has is not evidence about the document in front of you.
        :meth:`EnrichmentManifest.matches` is how the caller asks."""
        if not doc_ids:
            return {}
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"SELECT doc_id, enrichment_manifest, coalesce(content_hash, '') AS current_hash "
                        f'FROM "{schema}".documents '
                        f"WHERE doc_id = ANY(:ids) AND enrichment_manifest IS NOT NULL"
                    ),
                    {"ids": list(doc_ids)},
                )
            ).all()
        out: dict[str, tuple[EnrichmentManifest, str]] = {}
        for row in rows:
            manifest = EnrichmentManifest.from_dict(row.enrichment_manifest)
            if manifest is not None:
                out[row.doc_id] = (manifest, row.current_hash)
        return out

    async def replace_section_enrichment(
        self, tenant_id: str, doc_id: str, enrichments: Mapping[str, SectionEnrichment]
    ) -> int:
        """Write Phase-4 enrichment onto this document's ``sections`` rows. Returns rows updated.

        UPDATE, never INSERT. A ``sections`` row is created by :meth:`replace_document_structure`
        from the parsed tree, which is what owns the row's identity columns; this writes only the
        four DERIVED ones the plan lists for Phase 4 — ``summary``, ``facet_ids``, ``key_values``
        and ``amends``. An enrichment naming a ``section_id`` this document has no row for updates
        nothing and is counted as such (the caller logs the shortfall), rather than conjuring a
        row with a summary and no text.

        PER SECTION, EACH UPDATE REPLACES ALL FOUR COLUMNS, including setting them NULL where this
        run found nothing — a section that amended something last run and amends nothing now must
        not keep the old ``amends``. What this does NOT do is blank the columns of sections absent
        from ``enrichments``: the pipeline runs :meth:`replace_document_structure` immediately
        before this, which DELETEs and re-INSERTs every one of the document's rows, so an absent
        section's columns are already NULL and there is no stale value to purge. Adding a blanket
        clear would buy nothing and would import exactly the hazard
        :meth:`replace_document_profile` refuses — a transient model failure turning into data
        loss, and here at section granularity, where enrichment is fail-safe PER SECTION and a
        partial result is the ordinary outcome rather than the exceptional one.

        AN EMPTY MAP WRITES NOTHING AND TAKES NO LOCK, for that same reason: "the model produced
        nothing for this document" is not an instruction to erase what is stored.

        ONE transaction under the same ``_ADVISORY_LOCK`` the rest of this module takes, so two
        workers re-enriching one document cannot interleave their updates, and a reader never sees
        half a document's sections carrying this run's summaries and half the previous run's.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.persist_section_enrichment`` is the seam services call."""
        if not enrichments:
            return 0
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        rows = [
            {
                "section_id": section_id,
                "doc_id": doc_id,
                "summary": e.summary,
                "facet_ids": list(e.facet_ids),
                "key_values": e.key_values_json(),
                "amends": e.amends_json(),
            }
            for section_id, e in enrichments.items()
        ]
        updated = 0
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            statement = text(
                f'''UPDATE "{schema}".sections SET
                        summary = :summary, facet_ids = :facet_ids,
                        key_values = CAST(:key_values AS JSONB), amends = CAST(:amends AS JSONB)
                    WHERE section_id = :section_id AND doc_id = :doc_id'''
            )
            # Executed one row at a time rather than as one executemany, because `rowcount` on a
            # batched execute is the TOTAL and cannot say which sections matched — and "the
            # enrichment named sections this document does not have" is the one failure here that
            # is otherwise invisible (it means the tree was re-parsed between enriching and
            # persisting, so every summary is attached to the wrong text or to nothing).
            for row in rows:
                updated += (await conn.execute(statement, row)).rowcount or 0
        _log.info(
            "section_enrichment_replaced",
            schema=schema,
            doc_id=doc_id,
            sections=len(rows),
            updated=updated,
            amends=sum(1 for e in enrichments.values() if e.amends is not None),
        )
        if updated < len(rows):
            _log.warning(
                "section_enrichment_rows_missing", schema=schema, doc_id=doc_id, offered=len(rows), updated=updated
            )
        return updated

    async def document_cards(
        self, tenant_id: str, *, doc_ids: Sequence[str] | None = None, limit: int = 20
    ) -> list[DocCard]:
        """The Phase-3 profile cards the Phase-7 router routes on, flattened (RAG v2 Phase 7).

        THE FIRST READ OF A CARD OUT OF POSTGRES. Phase 3 wrote the card and
        :meth:`replace_document_profile`'s own note recorded that nothing read one back — the
        in-memory twin had a reader for tests, and the real adapter had none because no consumer
        existed. The router is that consumer, and it cannot be built without this.

        ONLY PROFILED DOCUMENTS. A row with ``profile_version`` NULL has no card, and a card is
        the whole input: routing on a title alone is what the existing planner already does
        (``query_plan.PlanContext``), and returning a cardless document here would let the router
        believe it had been shown a profile it had not. On a KB where no document is profiled —
        which is every KB today, since the card is written only on the v2 ingest path — this
        returns nothing and the router is handed an empty scope, which it is built to handle.

        ``doc_ids`` narrows to a caller-named scope. Without it, the newest ``limit`` profiled
        documents are returned, which is a SAMPLE and not a ranking: the plan asks for "top docs
        by doc-level retrieval" and there is no doc-level retrieval to ask (the ``doc`` chunk level
        has no producer — discovery §8.17). ``RouterScope.explicit`` is what carries that
        distinction to the decision.

        Facets come from the ``facets`` table rather than out of ``profile_json``, because the
        table is the queryable form and the one the router's validation is checked against — a
        facet id the router returns has to join to a row, and reading the blob instead would let
        the two disagree.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.document_cards`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        where = "profile_version IS NOT NULL" + (" AND doc_id = ANY(:ids)" if doc_ids else "")
        params: dict[str, Any] = {"n": max(1, limit)}
        if doc_ids:
            params["ids"] = list(doc_ids)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT doc_id, title, doc_type, purpose "
                        f'FROM "{schema}".documents WHERE {where} '
                        "ORDER BY ingested_at DESC NULLS LAST, doc_id LIMIT :n"
                    ),
                    params,
                )
            ).all()
            if not rows:
                return []
            ids = [str(r.doc_id) for r in rows]
            facet_rows = (
                await conn.execute(
                    text(
                        f'SELECT doc_id, facet_id FROM "{schema}".facets '
                        "WHERE doc_id = ANY(:ids) ORDER BY doc_id, facet_id"
                    ),
                    {"ids": ids},
                )
            ).all()
        facets: dict[str, list[str]] = {}
        for row in facet_rows:
            facets.setdefault(str(row.doc_id), []).append(str(row.facet_id))
        return [
            DocCard(
                doc_id=str(r.doc_id),
                title=str(r.title or r.doc_id),
                doc_type=str(r.doc_type or ""),
                purpose=str(r.purpose or ""),
                facet_ids=tuple(facets.get(str(r.doc_id), ())),
            )
            for r in rows
        ]

    async def document_sections(self, tenant_id: str, doc_ids: Sequence[str]) -> list[SectionCard]:
        """Every section of ``doc_ids``, as the Phase-8 selector sees it (RAG v2 Phase 8).

        THE AGGREGATE STRATEGY'S WHOLE INPUT, and the plan's step 1 verbatim: "load all
        section-level units for doc_ids from Postgres (summaries + breadcrumbs)". Not a search —
        every section, because an aggregate answer's failure mode is the requirement it never
        looked at, and a retrieval step that decided which sections to load would reintroduce
        exactly the recall problem the section pass exists to avoid.

        ORDERED BY (doc_id, ordinal, section_id) so the selection and the answer read in document
        order. ``ordinal`` is the tree's own ordering field; ``section_id`` breaks a tie
        deterministically rather than leaving it to the heap.

        ONLY MASKED COLUMNS ARE READ. ``summary``, ``heading`` and ``breadcrumb`` come from the
        Phase-4 enrichment, whose prompts were given masked text, and ``facet_ids``. ``blocks.text``
        and the text behind ``sections.text_hash`` are unmasked PII (FR-G2) and are deliberately
        not on this path — ``amends`` is read as a BOOLEAN for the same reason, since the plan's
        step 5 asks only whether the section amends something.

        An empty ``doc_ids`` returns nothing rather than the whole KB: "no document in scope" must
        not silently become "every document in the tenant", which on a large KB would load every
        section of every document onto the latency path.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.document_sections`` is the seam services call."""
        if not doc_ids:
            return []
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT section_id, doc_id, breadcrumb, heading, summary, facet_ids, "
                        "(amends IS NOT NULL) AS amends, ordinal, depth, token_count "
                        f'FROM "{schema}".sections WHERE doc_id = ANY(:ids) '
                        "ORDER BY doc_id, ordinal NULLS LAST, section_id"
                    ),
                    {"ids": list(doc_ids)},
                )
            ).all()
        return [
            SectionCard(
                section_id=str(r.section_id),
                doc_id=str(r.doc_id),
                breadcrumb=str(r.breadcrumb or ""),
                heading=str(r.heading or ""),
                summary=str(r.summary or ""),
                facet_ids=tuple(r.facet_ids or ()),
                amends=bool(r.amends),
                ordinal=int(r.ordinal or 0),
                depth=int(r.depth or 0),
                token_count=int(r.token_count or 0),
            )
            for r in rows
        ]

    async def stale_profiles(self, tenant_id: str, *, below: int) -> list[str]:
        """Documents whose stored card predates ``below`` (RAG v2 Phase 12 re-enrichment).

        The reason ``PROFILE_VERSION`` exists, finally given a reader. Its own docstring says it
        answers "which documents hold a card this build would no longer produce", and until now
        nothing asked: the column was written at ingest and never queried.

        A document with NO card (``profile_version IS NULL``) is included. It is the commonest
        stale case by far — every document ingested before the v2 flag was turned on — and a
        query that skipped it would make the CLI useless on exactly the corpus an operator
        wants to bring forward.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.stale_profiles`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f'SELECT doc_id FROM "{schema}".documents '
                        "WHERE profile_version IS NULL OR profile_version < :v ORDER BY doc_id"
                    ),
                    {"v": below},
                )
            ).all()
        return [str(r.doc_id) for r in rows]

    async def read_map_cache(self, tenant_id: str, keys: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Cached Phase-9 map items for ``keys``, keyed by cache key (RAG v2 Phase 9).

        A MISS IS THE NORMAL CASE and costs nothing: the caller maps the section. What the cache
        buys is the second identical question over the same document, which on a tender is the
        difference between one model call per selected section and none — the plan's §9.1
        "cache by (section_id, hash(query + facets), model_version)".

        The key already carries ``MAP_VERSION``, the section id, the query signature and the
        model, so a hit is an exact match on all four, and nothing here re-validates the items:
        ``replace_document_structure`` purges a section's cache rows whenever its text is
        replaced, and ``delete_document`` purges them on erasure.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.read_map_cache`` is the seam services call."""
        if not keys:
            return {}
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(f'SELECT cache_key, items FROM "{schema}".map_cache WHERE cache_key = ANY(:k)'),
                    {"k": list(keys)},
                )
            ).all()
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            stored = row.items
            if isinstance(stored, list):
                out[str(row.cache_key)] = [item for item in stored if isinstance(item, dict)]
        return out

    async def write_map_cache(self, tenant_id: str, entries: Sequence[MapCacheEntry]) -> int:
        """Store Phase-9 map items. Returns the number of rows written.

        UPSERT on the key, because two concurrent answers to the same question over the same
        document race here by construction and neither result is wrong — they are the same
        extraction. ``section_id`` is stored beside the key rather than only hashed into it,
        because it is what the PURGE paths join on: ``replace_document_structure`` and
        ``delete_document`` both delete by section, and a key they could not decompose would
        leave cached items describing text that no longer exists.

        AN EMPTY ITEM LIST IS WORTH CACHING. "This section has nothing relevant to this
        question" is the expected result for most sections of a long document, and it costs
        exactly as much to compute as a full one.

        It CAN raise; the fail-open lives in ``rag_clients.factory.write_map_cache``, because a
        cache write that fails must cost the cache and not the answer — and putting the swallow
        here would hide a broken cache from the integration tests too.

        Not on the ``CatalogAdapter`` Protocol — ``rag_clients.factory.write_map_cache`` is the
        seam services call."""
        if not entries:
            return 0
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        now = datetime.now(UTC)
        statement = (
            f'INSERT INTO "{schema}".map_cache '
            "(cache_key, section_id, query_sig, items, model_version, created_at) "
            "VALUES (:cache_key, :section_id, :query_sig, CAST(:items AS JSONB), "
            ":model_version, :created_at) "
            "ON CONFLICT (cache_key) DO UPDATE SET "
            "items = EXCLUDED.items, created_at = EXCLUDED.created_at"
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                text(statement),
                [
                    {
                        "cache_key": e.cache_key,
                        "section_id": e.section_id,
                        "query_sig": e.query_sig,
                        "items": json.dumps(e.items, ensure_ascii=False),
                        "model_version": e.model_version,
                        "created_at": now,
                    }
                    for e in entries
                ],
            )
        return len(entries)

    async def record_query_log(
        self,
        tenant_id: str,
        *,
        query: str,
        answer: str,
        cited_doc_ids: Sequence[str],
        latency_ms: Mapping[str, int],
        router_json: str | None = None,
        selected_sections: Sequence[str] | None = None,
        verification: str | None = None,
    ) -> int:
        """Append one row to ``query_log`` (RAG v2 Phase 12). Returns its id.

        The plan's §1.1 table, written for the first time here: Phase 1 created it, Phase 10's
        erasure scrub was built against it, and until now nothing produced a row. What it is FOR
        is the plan's own §0 latency criteria, which discovery found unmeasurable (§8.2) — hence
        ``latency_ms``, which holds per-stage durations and not just a round trip.

        ``cited_doc_ids`` IS THE ERASURE KEY and is the reason this method takes it as a required
        argument rather than deriving nothing. ``answer`` quotes the corpus, so a row is a derived
        record of document text in exactly the way a chat turn is (``ConversationStore``), and the
        only way a document-scoped delete can reach it is the array it cited. A row written
        without it would be unreachable by erasure and invisible as a defect.

        ``router_json``, ``selected_sections`` and ``verification`` are the Phase-7, Phase-8 and
        Phase-10 columns, and all three now have writers — the router decision, the section
        selection and the verification report respectively. They still DEFAULT to None, and the
        default is meaningful rather than lazy: NULL records that the phase did not run for this
        question (its flag is off, or the intent was not one it applies to), which is a different
        fact from an empty value. `selected_sections` makes the distinction sharpest — NULL is
        "no section pass ran" and `{}` is "it ran and selected nothing".

        APPEND, not upsert: a query log is a log. Two identical questions are two rows, because
        the thing being measured is the request, not the question.

        No advisory lock, deliberately — unlike every other write in this module. This is a plain
        single-row INSERT into a table with a ``BIGSERIAL`` key and no cross-row invariant, and
        taking the schema-wide lock on the query path would serialise every answer in the KB
        behind every ingest.

        Not on the ``CatalogAdapter`` Protocol, like the other v2-only catalog operations —
        ``rag_clients.factory.record_query_log`` is the seam services call."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            row_id = (
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".query_log
                            (ts, query, router_json, selected_sections, answer, verification,
                             latency_ms, cited_doc_ids)
                            VALUES (:ts, :query, CAST(:router_json AS JSONB), :selected_sections, :answer,
                                    CAST(:verification AS JSONB), CAST(:latency_ms AS JSONB), :cited_doc_ids)
                            RETURNING id'''
                    ),
                    {
                        "ts": datetime.now(UTC),
                        "query": query,
                        "router_json": router_json,
                        # `is not None`, NOT a truthiness test: Phase 8 made the empty case meaningful.
                        # An EMPTY array says the section pass RAN and selected nothing; NULL says no pass
                        # ran at all. Collapsing the two would make a dashboard read "nothing selects
                        # sections yet" for a strategy that ran and found nothing — the exact confusion the
                        # TODO(Phase-8) note here was left to avoid.
                        "selected_sections": list(selected_sections) if selected_sections is not None else None,
                        "answer": answer,
                        "verification": verification,
                        "latency_ms": json.dumps(dict(latency_ms)),
                        "cited_doc_ids": list(cited_doc_ids),
                    },
                )
            ).scalar_one()
        return int(row_id)

    async def delete_query_log_citing(self, tenant_id: str, doc_ids: set[str]) -> int:
        """Delete every ``query_log`` row that cites any of ``doc_ids`` (DPDP erasure scrub —
        a derived record quoting erased data must go too). Returns rows removed.

        Deliberately mirrors ``ConversationStore.delete_citing``, because ``query_log`` is the same
        class of artifact: a derived record holding text quoted out of documents.
        ``query_log.answer`` will carry passages lifted from the corpus and ``selected_sections``
        names the structure they came from, so an erasure that left those rows behind would leave
        the erased text readable — the certificate would be false. Matching is array OVERLAP
        (``&&``) on ``cited_doc_ids``, so a row is removed if it cited ANY erased document, not only
        if every citation was erased: a partially-erased answer still quotes the erased document.

        A row with NULL or empty ``cited_doc_ids`` is never matched by ``&&`` and so is never
        collaterally purged.

        RETAINED, as a reasoned position rather than an oversight: the row's own ``query`` text. It
        is the data principal's *question*, not content derived from the erased document, and the
        platform already takes exactly this position elsewhere — the tamper-evident audit chain
        stores query text in its QUERY payloads and is append-only by design. Erasing a subject's
        own query history is therefore a separate, subject-scoped question with an existing answer,
        not something this document-scoped purge is failing to do. See
        ``docs/automated-erasure.md``.

        Called from the ERASURE path only (``compliance.erase``), never from a plain
        ``delete_document``. That asymmetry is deliberate and mirrors conversations: deleting a
        document purges the stores and its own derived rows, while an erasure additionally scrubs
        derived history that merely quoted it.

        Idempotent, and a no-op returning 0 for an empty ``doc_ids``. Kept OFF the
        ``CatalogAdapter`` Protocol like ``drop_v2_tables``; ``rag_clients.factory`` exposes it to
        callers so no service imports a concrete adapter."""
        if not doc_ids:
            return 0
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            rows = (
                await conn.execute(
                    text(f'DELETE FROM "{schema}".query_log WHERE cited_doc_ids && :docs RETURNING id'),
                    {"docs": list(doc_ids)},
                )
            ).all()
        if rows:
            _log.info("query_log_erasure_scrub", schema=schema, rows=len(rows), docs=len(doc_ids))
        return len(rows)

    async def register_document(self, doc: DocumentEnvelope) -> None:
        schema = pg_schema(doc.tenant_id)
        await self.ensure_schema(doc.tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'''INSERT INTO "{schema}".documents
                        (doc_id, source_id, source_kind, mime, uri, lineage_id, source_acl, metadata,
                         content_hash, ingested_at)
                        VALUES (:doc_id, :source_id, :source_kind, :mime, :uri, :lineage_id,
                                CAST(:source_acl AS JSONB), CAST(:metadata AS JSONB), :content_hash, :ingested_at)
                        ON CONFLICT (doc_id) DO UPDATE SET ingested_at = EXCLUDED.ingested_at,
                            content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata'''
                ),
                {
                    "doc_id": doc.doc_id,
                    "source_id": doc.source_id,
                    "source_kind": doc.source_kind.value,
                    "mime": doc.mime,
                    "uri": doc.uri,
                    "lineage_id": doc.lineage_id,
                    "source_acl": _json(doc.source_acl),
                    "metadata": _json(doc.metadata),
                    "content_hash": doc.content_hash,
                    "ingested_at": doc.ingested_at,
                },
            )

    async def list_documents(self, tenant_id: str) -> list[DocumentEnvelope]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f'SELECT * FROM "{schema}".documents ORDER BY ingested_at DESC'))
            rows = result.mappings().all()
        return [_row_to_doc(dict(r), tenant_id) for r in rows]

    async def record_chunks(self, tenant_id: str, chunks: list[Chunk]) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            for c in chunks:
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".chunks (
                                chunk_id, doc_id, page, version, record,
                                section_id, level, parent_id, embed_text, sequence,
                                page_start, page_end, token_count, chunk_hash
                            )
                            VALUES (
                                :chunk_id, :doc_id, :page, :version, CAST(:record AS JSONB),
                                :section_id, :level, :parent_id, :embed_text, :sequence,
                                :page_start, :page_end, :token_count, :chunk_hash
                            )
                            ON CONFLICT (chunk_id) DO UPDATE SET record = EXCLUDED.record,
                                version = EXCLUDED.version, page = EXCLUDED.page,
                                section_id = EXCLUDED.section_id, level = EXCLUDED.level,
                                parent_id = EXCLUDED.parent_id, embed_text = EXCLUDED.embed_text,
                                sequence = EXCLUDED.sequence, page_start = EXCLUDED.page_start,
                                page_end = EXCLUDED.page_end, token_count = EXCLUDED.token_count,
                                chunk_hash = EXCLUDED.chunk_hash'''
                    ),
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "page": c.page,
                        "version": c.version,
                        # The plan's §1.1 columns, promoted out of `record` (Phase 6). The record
                        # stays the source of truth for a citation — these are the same facts in a
                        # form that can be JOINED, FILTERED and ORDERED without deserialising every
                        # row, which is what `chunks(section_id, level)` was indexed for in Phase 1
                        # and what nothing could use until now.
                        #
                        # Written for EVERY chunk, not only v2 ones: a v1 chunk simply has None for
                        # the structure fields, which is what it means. That is a column write only
                        # — `record` keeps using `exclude_none`, so what the previous image reads
                        # back is unchanged (see the note below it).
                        "section_id": c.section_id,
                        "level": c.level.value if c.level is not None else None,
                        "parent_id": c.parent_id,
                        "embed_text": c.embed_text,
                        "sequence": c.sequence,
                        # A chunk is single-page BY CONSTRUCTION — no unit may own content from two
                        # pages, which `test_a_body_chunk_never_owns_content_from_two_pages` pins —
                        # so the pair is a degenerate range rather than a span. Both are written
                        # anyway because the plan names them and because a page-range predicate
                        # (`page_start >= 3 AND page_end <= 5`) should not have to special-case
                        # chunks against `sections`, where the range is real.
                        "page_start": c.page,
                        "page_end": c.page,
                        "token_count": c.token_count,
                        # Derived HERE, at the one place a chunk is written, so it cannot disagree
                        # with the text stored beside it — and so it exists for a v1 chunk too,
                        # which is what makes the re-embed skip work on the corpus as it is today.
                        # Full-width sha256: it is a change detector compared for equality, never a
                        # name (see `rag_core.chunking.text_hash`).
                        "chunk_hash": text_hash(c.text),
                        # `exclude_none` is a COMPATIBILITY requirement, not tidiness. `Chunk` is
                        # `extra="forbid"`, so every key this record carries is a key an older
                        # build must already know. The RAG v2 fields (`level`/`section_id`/
                        # `parent_id`/`embed_text`) default to None and are None on every v1 chunk,
                        # yet a plain dump writes them as explicit nulls — and the PREVIOUS image
                        # then raises `ValidationError: Extra inputs are not permitted` reading its
                        # own corpus back. That is not gated by `v2_ingest`: it happens on any
                        # ingest or reindex with the flag OFF, and it takes out the citation viewer,
                        # whole-document answering, neighbour expansion, the reindex swap's
                        # `old_chunk_ids` read and — the reason this is not merely an annoyance —
                        # the DPDP erasure subject scan (`compliance.py`), which would 500 on an
                        # erasure request. Omitting the nulls keeps a new record readable by both
                        # builds, so a rollback or a staged multi-container update stays safe.
                        "record": c.model_dump_json(exclude_none=True),
                    },
                )

    async def delete_document(self, tenant_id: str, doc_id: str) -> list[str]:
        """Remove a document and everything derived from it. This is the path DPDP right-to-erasure
        cascades through (``svc_control_plane.compliance``), so "everything" has to include the RAG
        v2 structure tables: ``blocks.text``/``html`` hold verbatim document text and
        ``sections.summary``/``key_values`` hold derived content, and leaving them behind after an
        erasure would be a compliance defect, not just stale data. There are no FK cascades (see
        ``_v2_table_ddl``), so each one is deleted explicitly.

        ``map_cache`` is purged through its ``section_id``, which is why it must go BEFORE
        ``sections``.

        ``query_log`` WAS the flagged gap here, and Phase 12 closed it. The table has no
        ``doc_id`` — it records QUERIES, not documents — so the decision recorded at this line was
        "either scope rows by the doc ids they cited, or keep the log free of answer text". The
        first was taken: :meth:`record_query_log` stamps ``cited_doc_ids``, and this purge matches
        on it with the same array-overlap predicate :meth:`delete_query_log_citing` uses. A row is
        removed if it cited this document at ALL, not only if every citation was to it: a
        partially-erased answer still quotes the erased document.

        That matters more here than it reads. ``delete_document`` is the path DPDP erasure
        cascades through, and before Phase 12 nothing wrote the table, so nothing could leak. The
        moment a row carries an answer quoting a document, a document-scoped delete that left it
        behind would make the erasure certificate false."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            # The same lock `replace_document_structure` takes, and now load-bearing here too: that
            # path REPLACES a document's structure as delete-then-insert, so an erasure landing
            # between its two halves would be followed by the insert and leave `sections`/`blocks`
            # rows — holding verbatim document text — for a document this call has just erased.
            await conn.execute(_ADVISORY_LOCK, {"schema": schema})
            await conn.execute(
                text(
                    f'DELETE FROM "{schema}".map_cache WHERE section_id IN '
                    f'(SELECT section_id FROM "{schema}".sections WHERE doc_id = :d)'
                ),
                {"d": doc_id},
            )
            for table in ("blocks", "sections", "facets", "doc_relations"):
                await conn.execute(text(f'DELETE FROM "{schema}".{table} WHERE doc_id = :d'), {"d": doc_id})
            # ...and every query-log row that quoted this document (Phase 12 — see the docstring).
            # Inside the SAME transaction as the rest of the erasure, so there is no window in
            # which the document is gone and an answer quoting it is still readable.
            await conn.execute(
                text(f'DELETE FROM "{schema}".query_log WHERE cited_doc_ids && :docs'), {"docs": [doc_id]}
            )
            rows = (
                await conn.execute(
                    text(f'DELETE FROM "{schema}".chunks WHERE doc_id = :d RETURNING chunk_id'), {"d": doc_id}
                )
            ).all()
            await conn.execute(text(f'DELETE FROM "{schema}".documents WHERE doc_id = :d'), {"d": doc_id})
        return [str(r.chunk_id) for r in rows]

    async def delete_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        """Remove chunks without touching the document row (write-then-swap re-index)."""
        if not chunk_ids:
            return
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'DELETE FROM "{schema}".chunks WHERE chunk_id = ANY(:ids)'), {"ids": list(chunk_ids)}
            )

    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        schema = pg_schema(tenant_id)
        # A tenant that has never ingested anything has no schema yet — ensure it
        # exists (idempotent) so the query returns "no rows", not an undefined-table
        # error. Matches every other CatalogAdapter method, which all ensure first.
        await self.ensure_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(f'SELECT chunk_id, record FROM "{schema}".chunks WHERE chunk_id = ANY(:ids)'),
                    {"ids": chunk_ids},
                )
            ).all()
        return {row.chunk_id: Chunk.model_validate_json(_as_text(row.record)) for row in rows}

    async def document_chunks(self, tenant_id: str, doc_id: str) -> list[Chunk]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f'SELECT record FROM "{schema}".chunks WHERE doc_id = :d '
                        # `span.start` orders a v1 chunk. A v2 chunk has no span — markup-derived
                        # text has no offset into the extracted text — so every one of them tied on
                        # NULL and intra-page order became whatever Postgres returned, for the
                        # citation viewer, whole-document answering and neighbour expansion alike.
                        # `sequence` is the chunker's own document order and breaks that tie;
                        # `chunk_id` last makes the order TOTAL, so no shape can tie again. The
                        # in-memory twin mirrors this key exactly (`memory._document_order`) —
                        # it used to sort stably where this did not, which is precisely why
                        # `make test` could not see the defect.
                        "ORDER BY page NULLS FIRST, (record->'span'->>'start')::int NULLS FIRST, "
                        # The promoted column first, the record second: Phase 6 writes the column,
                        # but every row written before it has the value ONLY in `record`, and a
                        # corpus is not re-indexed just because a column was added. COALESCE keeps
                        # one ordering for both generations instead of a flag day.
                        "COALESCE(sequence, (record->>'sequence')::int) NULLS FIRST, chunk_id"
                    ),
                    {"d": doc_id},
                )
            ).all()
        return [Chunk.model_validate_json(_as_text(r.record)) for r in rows]

    async def document_relations(self, tenant_id: str, doc_id: str) -> list[DeclaredRelation]:
        """The relationships this document DECLARES about itself — Phase 3's `doc_relations` rows and
        Phase 4's `sections.amends` — as the graph-seeding input (RAG v2 Phase 6).

        Returns NAMES, not ids, because the graph is keyed on names and so are questions: the
        document's own `title` where Phase 3 has written one (its `doc_id` until then), the
        section's `breadcrumb`, and the target's title where the reference resolved to a document
        in this KB (`target_doc_id`) or the reference exactly as the document wrote it where it did
        not — an unresolved "Corrigendum-II" is still the right node name, and the day its document
        is ingested the two collapse onto one node by name.

        BOTH HALVES HAVE A PRODUCER AS OF PHASE 4. `doc_relations` is written by
        :meth:`replace_document_profile` (Phase 3) and `sections.amends` by
        :meth:`replace_section_enrichment` (Phase 4), so a document that declares an amendment at
        either grain answers with it here. An empty result now means the document declared
        nothing — it no longer also means "this phase has not been built".
        """
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            declared = (
                await conn.execute(
                    text(
                        "SELECT r.rel_type, r.evidence_quote, "
                        "COALESCE(d.title, r.doc_id) AS source_ref, "
                        "COALESCE(t.title, r.target_ref) AS target_ref "
                        f'FROM "{schema}".doc_relations r '
                        f'JOIN "{schema}".documents d ON d.doc_id = r.doc_id '
                        # LEFT join: a reference that names a document this KB does not hold is the
                        # NORMAL case for a corrigendum (the amended tender may never be ingested),
                        # and it must still produce an edge under the name the document used.
                        f'LEFT JOIN "{schema}".documents t ON t.doc_id = r.target_doc_id '
                        "WHERE r.doc_id = :d"
                    ),
                    {"d": doc_id},
                )
            ).all()
            amended = (
                await conn.execute(
                    text(
                        "SELECT breadcrumb, amends->>'target_ref' AS target_ref, amends->>'quote' AS quote "
                        f'FROM "{schema}".sections WHERE doc_id = :d AND amends IS NOT NULL'
                    ),
                    {"d": doc_id},
                )
            ).all()
        relations = [
            DeclaredRelation(
                rel_type=str(row.rel_type or ""),
                source_ref=str(row.source_ref or ""),
                target_ref=str(row.target_ref or ""),
                evidence_quote=str(row.evidence_quote or ""),
            )
            for row in declared
        ]
        # A section-level `amends` is the same shape with the relation type implied by the column.
        relations.extend(
            DeclaredRelation(
                rel_type="amends",
                source_ref=str(row.breadcrumb or ""),
                target_ref=str(row.target_ref or ""),
                evidence_quote=str(row.quote or ""),
            )
            for row in amended
        )
        return relations

    async def chunk_hashes(self, tenant_id: str, doc_id: str) -> dict[str, str]:
        """Two columns for one document — no `record` deserialised, and an index scan now that
        `chunks_by_doc` exists. Rows predating the `chunk_hash` write are excluded rather than
        returned as None: a caller must not be able to read "no hash" as "hash matched"."""
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f'SELECT chunk_id, chunk_hash FROM "{schema}".chunks '
                        "WHERE doc_id = :d AND chunk_hash IS NOT NULL"
                    ),
                    {"d": doc_id},
                )
            ).all()
        return {str(row.chunk_id): str(row.chunk_hash) for row in rows}

    async def count_by_level(self, tenant_id: str) -> dict[str | None, int]:
        """Aggregated in Postgres (``GROUP BY``), never by reading the rows back.

        Reads the ``level`` COLUMN first and falls back to the serialised record, the same
        ``COALESCE`` discipline the ordering key uses: the column is in every tenant's schema but
        is not written yet (Phase 6 promotes it, as it does ``sequence``), so a count that read
        only the column would report every v2 chunk as level-less, and one that read only the
        record would stop seeing them the day the column starts being written.

        Parent context records are excluded per the Protocol — they are deliberately never indexed,
        so counting them would make a correct parent/child KB fail the reconciliation.
        """
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT COALESCE(level, record->>'level') AS lvl, count(*) AS n "
                        f'FROM "{schema}".chunks '
                        "WHERE record->'metadata'->>'chunk_role' IS DISTINCT FROM 'parent' "
                        "GROUP BY 1"
                    )
                )
            ).all()
        return {row.lvl: int(row.n) for row in rows}

    async def get_tenant_config(self, tenant_id: str) -> dict[str, str]:
        schema = pg_schema(tenant_id)
        await self.ensure_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(f'SELECT key, value FROM "{schema}".tenant_config'))).all()
        return {row.key: row.value for row in rows}

    async def set_tenant_config(self, tenant_id: str, key: str, value: str) -> None:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'INSERT INTO "{schema}".tenant_config (key, value) VALUES (:k, :v) '
                    "ON CONFLICT (key) DO UPDATE SET value = :v"
                ),
                {"k": key, "v": value},
            )

    async def aclose(self) -> None:
        await self._engine.dispose()


def _row_to_doc(m: dict[str, Any], tenant_id: str) -> DocumentEnvelope:
    acl_raw = m.get("source_acl") or []
    meta_raw = m.get("metadata") or {}
    acl = json.loads(acl_raw) if isinstance(acl_raw, str) else acl_raw
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    return DocumentEnvelope(
        tenant_id=tenant_id,
        doc_id=str(m["doc_id"]),
        source_id=str(m["source_id"]),
        source_kind=SourceKind(m["source_kind"]),
        mime=str(m["mime"]),
        uri=m.get("uri") or None,
        lineage_id=m.get("lineage_id") or None,
        source_acl=list(acl),
        metadata=dict(meta),
        content_hash=m.get("content_hash") or None,
        ingested_at=m["ingested_at"],
    )


def _json(value: object) -> str:
    return json.dumps(value)


def _as_text(value: object) -> str:
    """psycopg may hand back JSONB as dict or str depending on codec."""
    if isinstance(value, str):
        return value
    return json.dumps(value)
