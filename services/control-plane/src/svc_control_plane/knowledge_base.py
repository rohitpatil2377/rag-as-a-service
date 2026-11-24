"""Knowledge bases within a tenant (org) — the real KB model (FR-H1).

An organisation (the authenticated ``tenant_id``) owns MANY knowledge bases: HR, Finance,
Sales, Health, etc. Each KB is its OWN namespace across every store (vector collection,
BM25 index, object prefix, Postgres schema), so documents, retrieval, model connection and
audit are hard-isolated PER KB — exactly as tenants are isolated from each other.

A KB's id is ``{tenant}-{slug}`` so ownership is provable by prefix (the tenant guard needs
no lookup), and the "Default" KB (id == tenant) always exists for the org's shared space.
This registry only holds the human names/list; isolation is by the id, per store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_settings, pg_schema


class KbRecord(BaseModel):
    kb_id: str
    name: str
    is_default: bool = False
    trashed_at: datetime | None = None  # soft-delete: set when moved to Trash, cleared on restore
    # Publish gate (banking pack, maker-checker): an unpublished KB does not answer end-user
    # queries. Default True so existing/normal KBs are published; only the dual-control publish
    # flow creates a KB as draft (False) until a second admin approves.
    published: bool = True
    # RBAC governance: unit_id None = tenant ROOT unit (tenant-wide); classification_floor is
    # the MINIMUM clearance to see the KB (0=Public..3=Restricted), enforced via rbac.can_read_kb.
    unit_id: str | None = None
    classification_floor: int = 0


KbRecords = list[KbRecord]  # alias: the method named ``list`` shadows the builtin in annotations

# The one platform-wide shared KB (Federated RAG cross-department Tier C — see the design
# discussion this implements). Not owned by any tenant: any authenticated principal may
# ingest into it (choosing this id instead of their own KB's, at ingest time — a one-time,
# per-document decision, no later toggle) and every principal's default /v1/ask searches it
# alongside their own KB, unconditionally, no grant/federation involved. Kept deliberately
# separate from Tier A (per-tenant classification/ACL) and Tier B (signed cross-dept Grants,
# see federation.py) — this is the "everyone" tier, all-or-nothing by design.
SHARED_KB_ID = "public__company-wide"
SHARED_KB_NAME = "Company-Wide (Shared)"


def slug_kb_id(tenant_id: str, name: str) -> str:
    """Derive a KB id ``{tenant}__{slug(name)}``. The ``__`` separator can't appear in an
    org id, so ownership-by-prefix is unambiguous even when one org id prefixes another."""
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    slug = "-".join(p for p in slug.split("-") if p) or "kb"
    return f"{tenant_id}__{slug}"


@runtime_checkable
class KnowledgeBaseStore(Protocol):
    async def ensure(self, tenant_id: str) -> None: ...
    async def list(self, tenant_id: str) -> KbRecords: ...  # active only (excludes trashed)
    async def list_trashed(self, tenant_id: str) -> KbRecords: ...
    async def get(self, tenant_id: str, kb_id: str) -> KbRecord | None: ...
    async def add(
        self,
        tenant_id: str,
        kb_id: str,
        name: str,
        published: bool = True,
        *,
        unit_id: str | None = None,
        classification_floor: int = 0,
    ) -> None: ...
    async def set_governance(
        self, tenant_id: str, kb_id: str, *, unit_id: str | None, classification_floor: int
    ) -> None: ...
    async def set_published(self, tenant_id: str, kb_id: str, published: bool) -> None: ...
    async def trash(self, tenant_id: str, kb_id: str) -> None: ...
    async def restore(self, tenant_id: str, kb_id: str) -> None: ...
    async def remove(self, tenant_id: str, kb_id: str) -> None: ...  # hard remove from the registry (purge)


class InMemoryKnowledgeBaseStore:
    def __init__(self) -> None:
        self._kbs: dict[str, dict[str, KbRecord]] = {}

    async def ensure(self, tenant_id: str) -> None:
        return None

    async def list(self, tenant_id: str) -> KbRecords:
        return [r for r in self._kbs.get(tenant_id, {}).values() if r.trashed_at is None]

    async def list_trashed(self, tenant_id: str) -> KbRecords:
        return [r for r in self._kbs.get(tenant_id, {}).values() if r.trashed_at is not None]

    async def get(self, tenant_id: str, kb_id: str) -> KbRecord | None:
        return self._kbs.get(tenant_id, {}).get(kb_id)

    async def add(
        self,
        tenant_id: str,
        kb_id: str,
        name: str,
        published: bool = True,
        *,
        unit_id: str | None = None,
        classification_floor: int = 0,
    ) -> None:
        self._kbs.setdefault(tenant_id, {})[kb_id] = KbRecord(
            kb_id=kb_id,
            name=name,
            published=published,
            unit_id=unit_id,
            classification_floor=classification_floor,
        )

    async def set_governance(
        self, tenant_id: str, kb_id: str, *, unit_id: str | None, classification_floor: int
    ) -> None:
        r = self._kbs.get(tenant_id, {}).get(kb_id)
        if r is not None:
            self._kbs[tenant_id][kb_id] = r.model_copy(
                update={"unit_id": unit_id, "classification_floor": classification_floor}
            )

    async def set_published(self, tenant_id: str, kb_id: str, published: bool) -> None:
        r = self._kbs.get(tenant_id, {}).get(kb_id)
        if r:
            self._kbs[tenant_id][kb_id] = r.model_copy(update={"published": published})

    async def trash(self, tenant_id: str, kb_id: str) -> None:
        r = self._kbs.get(tenant_id, {}).get(kb_id)
        if r:
            self._kbs[tenant_id][kb_id] = r.model_copy(update={"trashed_at": datetime.now(UTC)})

    async def restore(self, tenant_id: str, kb_id: str) -> None:
        r = self._kbs.get(tenant_id, {}).get(kb_id)
        if r:
            self._kbs[tenant_id][kb_id] = r.model_copy(update={"trashed_at": None})

    async def remove(self, tenant_id: str, kb_id: str) -> None:
        self._kbs.get(tenant_id, {}).pop(kb_id, None)


_DDL = """CREATE TABLE IF NOT EXISTS "{schema}".knowledge_bases (
    kb_id TEXT PRIMARY KEY,
    name TEXT NOT NULL
)"""
# Migration-safe soft-delete column (older deployments created the table without it).
_DDL_TRASH = 'ALTER TABLE "{schema}".knowledge_bases ADD COLUMN IF NOT EXISTS trashed_at TIMESTAMPTZ'
# Migration-safe publish column — default TRUE so existing KBs stay published (banking pack).
_DDL_PUBLISHED = (
    'ALTER TABLE "{schema}".knowledge_bases ADD COLUMN IF NOT EXISTS published BOOLEAN NOT NULL DEFAULT TRUE'
)
# Migration-safe RBAC governance columns (default: root unit / Public floor = current behaviour).
_DDL_UNIT = 'ALTER TABLE "{schema}".knowledge_bases ADD COLUMN IF NOT EXISTS unit_id TEXT'
_DDL_FLOOR = (
    'ALTER TABLE "{schema}".knowledge_bases ADD COLUMN IF NOT EXISTS classification_floor INTEGER NOT NULL DEFAULT 0'
)


class PostgresKnowledgeBaseStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            await conn.execute(text(_DDL.format(schema=schema)))
            await conn.execute(text(_DDL_TRASH.format(schema=schema)))
            await conn.execute(text(_DDL_PUBLISHED.format(schema=schema)))
            await conn.execute(text(_DDL_UNIT.format(schema=schema)))
            await conn.execute(text(_DDL_FLOOR.format(schema=schema)))

    async def list(self, tenant_id: str) -> KbRecords:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"SELECT kb_id, name, trashed_at, published, unit_id, classification_floor "
                        f'FROM "{schema}".knowledge_bases WHERE trashed_at IS NULL ORDER BY name'
                    )
                )
            ).all()
        return [
            KbRecord(
                kb_id=r.kb_id,
                name=r.name,
                published=r.published,
                unit_id=r.unit_id,
                classification_floor=r.classification_floor,
            )
            for r in rows
        ]

    async def list_trashed(self, tenant_id: str) -> KbRecords:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f'SELECT kb_id, name, trashed_at FROM "{schema}".knowledge_bases '
                        "WHERE trashed_at IS NOT NULL ORDER BY trashed_at DESC"
                    )
                )
            ).all()
        return [KbRecord(kb_id=r.kb_id, name=r.name, trashed_at=r.trashed_at) for r in rows]

    async def get(self, tenant_id: str, kb_id: str) -> KbRecord | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        f"SELECT kb_id, name, trashed_at, published, unit_id, classification_floor "
                        f'FROM "{schema}".knowledge_bases WHERE kb_id = :k'
                    ),
                    {"k": kb_id},
                )
            ).first()
        return (
            KbRecord(
                kb_id=row.kb_id,
                name=row.name,
                trashed_at=row.trashed_at,
                published=row.published,
                unit_id=row.unit_id,
                classification_floor=row.classification_floor,
            )
            if row
            else None
        )

    async def add(
        self,
        tenant_id: str,
        kb_id: str,
        name: str,
        published: bool = True,
        *,
        unit_id: str | None = None,
        classification_floor: int = 0,
    ) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'INSERT INTO "{schema}".knowledge_bases (kb_id, name, published, unit_id, classification_floor) '
                    "VALUES (:kb_id, :name, :published, :unit_id, :classification_floor) "
                    # Re-creating a KB whose slug matches a TRASHED one must revive it as a fresh,
                    # ACTIVE KB — clear trashed_at (else it 'creates' but stays hidden in Trash) and
                    # reset governance to the new create's values.
                    "ON CONFLICT (kb_id) DO UPDATE SET name = :name, published = :published, "
                    "trashed_at = NULL, unit_id = :unit_id, classification_floor = :classification_floor"
                ),
                {
                    "kb_id": kb_id,
                    "name": name,
                    "published": published,
                    "unit_id": unit_id,
                    "classification_floor": classification_floor,
                },
            )

    async def set_governance(
        self, tenant_id: str, kb_id: str, *, unit_id: str | None, classification_floor: int
    ) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'UPDATE "{schema}".knowledge_bases SET unit_id = :u, classification_floor = :c WHERE kb_id = :k'),
                {"u": unit_id, "c": classification_floor, "k": kb_id},
            )

    async def set_published(self, tenant_id: str, kb_id: str, published: bool) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'UPDATE "{schema}".knowledge_bases SET published = :p WHERE kb_id = :k'),
                {"p": published, "k": kb_id},
            )

    async def trash(self, tenant_id: str, kb_id: str) -> None:
        await self._set_trashed(tenant_id, kb_id, datetime.now(UTC))

    async def restore(self, tenant_id: str, kb_id: str) -> None:
        await self._set_trashed(tenant_id, kb_id, None)

    async def _set_trashed(self, tenant_id: str, kb_id: str, when: datetime | None) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'UPDATE "{schema}".knowledge_bases SET trashed_at = :t WHERE kb_id = :k'),
                {"t": when, "k": kb_id},
            )

    async def remove(self, tenant_id: str, kb_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(text(f'DELETE FROM "{schema}".knowledge_bases WHERE kb_id = :kb_id'), {"kb_id": kb_id})
