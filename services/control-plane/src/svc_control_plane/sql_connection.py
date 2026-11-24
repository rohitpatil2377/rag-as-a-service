"""Per-KB SQL connection registry (FR-D10 extended — multiple named DB connections).

A knowledge base can register MANY read-only database connections (e.g. a "Sales DB"
and an "Inventory DB"), one marked as its **default**. Mirrors model_connection.py's
shape exactly, for the same reason: a single replaceable slot doesn't fit a KB that
legitimately needs more than one governed database. The DSN is encrypted at rest via
``tenant_crypto`` (same as the legacy single-connection slot did).

Query-time routing is EXPLICIT, not auto-detected: ``sql_query.ask``/``execute_sql``
take a ``connection_id`` (falling back to the KB's default when omitted) — chosen over
schema-sniffing across every connection on every question, which would add real latency
and ambiguity when two DBs have similarly-named tables.

Isolation is per-KB, like every other store. The legacy single-connection tenant_config
keys (``sql_dsn``/``sql_tables``/``sql_dialect``) are migrated into this registry on
first access, so an existing connection keeps working with no operator action.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings, pg_schema

_log = get_logger(__name__)

# Legacy single-slot tenant_config keys (see the old sql_query.py) — read once at
# migration time only, never written again.
_LEGACY_DSN_KEY = "sql_dsn"
_LEGACY_TABLES_KEY = "sql_tables"
_LEGACY_DIALECT_KEY = "sql_dialect"


class StoredSqlConnection(BaseModel):
    """One registered read-only DB connection. ``dsn_ciphertext`` is encrypted at rest."""

    tenant_id: str
    conn_id: str = Field(..., description="Stable id within the KB.")
    name: str = Field(..., description="Human label, e.g. 'Sales DB'.")
    dsn_ciphertext: str = Field(..., description="Envelope-encrypted SQLAlchemy DSN.")
    tables: list[str] = Field(..., min_length=1, description="Allowlist of readable tables.")
    dialect: str = Field(default="postgres")
    is_default: bool = Field(default=False, description="Used when a query doesn't name a connection.")
    created_at: datetime | None = None


@runtime_checkable
class SqlConnectionStore(Protocol):
    async def ensure(self, tenant_id: str) -> None: ...
    async def list(self, tenant_id: str) -> list[StoredSqlConnection]: ...
    async def get(self, tenant_id: str, conn_id: str) -> StoredSqlConnection | None: ...
    async def get_default(self, tenant_id: str) -> StoredSqlConnection | None: ...
    async def put(self, conn: StoredSqlConnection) -> None: ...
    async def delete(self, tenant_id: str, conn_id: str) -> None: ...
    async def set_default(self, tenant_id: str, conn_id: str) -> None: ...


class InMemorySqlConnectionStore:
    def __init__(self) -> None:
        self._c: dict[str, dict[str, StoredSqlConnection]] = {}

    async def ensure(self, tenant_id: str) -> None:
        return None

    async def list(self, tenant_id: str) -> list[StoredSqlConnection]:
        conns = list(self._c.get(tenant_id, {}).values())
        return sorted(conns, key=lambda c: (not c.is_default, c.name))

    async def get(self, tenant_id: str, conn_id: str) -> StoredSqlConnection | None:
        return self._c.get(tenant_id, {}).get(conn_id)

    async def get_default(self, tenant_id: str) -> StoredSqlConnection | None:
        for c in self._c.get(tenant_id, {}).values():
            if c.is_default:
                return c
        return None

    async def put(self, conn: StoredSqlConnection) -> None:
        self._c.setdefault(conn.tenant_id, {})[conn.conn_id] = conn
        if conn.is_default:
            await self.set_default(conn.tenant_id, conn.conn_id)

    async def delete(self, tenant_id: str, conn_id: str) -> None:
        self._c.get(tenant_id, {}).pop(conn_id, None)

    async def set_default(self, tenant_id: str, conn_id: str) -> None:
        conns = self._c.get(tenant_id, {})
        for cid, c in conns.items():
            conns[cid] = c.model_copy(update={"is_default": cid == conn_id})


_DDL = """CREATE TABLE IF NOT EXISTS "{schema}".sql_connections (
    conn_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dsn_ciphertext TEXT NOT NULL,
    tables TEXT NOT NULL,
    dialect TEXT NOT NULL DEFAULT 'postgres',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL
)"""


class PostgresSqlConnectionStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            await conn.execute(text(_DDL.format(schema=schema)))
            await self._migrate_legacy(conn, tenant_id, schema)

    async def _migrate_legacy(self, conn: AsyncConnection, tenant_id: str, schema: str) -> None:
        """One-time: fold the old single-slot tenant_config DSN into the registry, so an
        already-configured KB keeps answering after the upgrade with no operator action."""
        count = (await conn.execute(text(f'SELECT COUNT(*) FROM "{schema}".sql_connections'))).scalar()
        if count:
            return
        # tenant_config is provisioned by the catalog adapter, not by us — it may not exist
        # yet for a brand-new KB that has never set any config (mirrors model_connection.py's
        # own to_regclass guard before touching a table another module owns).
        exists = (await conn.execute(text(f"SELECT to_regclass('\"{schema}\".tenant_config')"))).scalar()
        if exists is None:
            return
        row = (
            (
                await conn.execute(
                    text(f'SELECT key, value FROM "{schema}".tenant_config WHERE key = ANY(:keys)'),
                    {"keys": [_LEGACY_DSN_KEY, _LEGACY_TABLES_KEY, _LEGACY_DIALECT_KEY]},
                )
            )
            .mappings()
            .all()
        )
        cfg = {r["key"]: r["value"] for r in row}
        dsn_ct = cfg.get(_LEGACY_DSN_KEY)
        if not dsn_ct:
            return
        tables = [t for t in cfg.get(_LEGACY_TABLES_KEY, "").split(",") if t]
        await conn.execute(
            text(
                f'INSERT INTO "{schema}".sql_connections '
                "(conn_id, name, dsn_ciphertext, tables, dialect, is_default, created_at) "
                "VALUES (:cid, :name, :dsn, :tables, :dialect, TRUE, :at)"
            ),
            {
                "cid": "legacy",
                "name": "Database",
                "dsn": dsn_ct,
                "tables": ",".join(tables),
                "dialect": cfg.get(_LEGACY_DIALECT_KEY) or "postgres",
                "at": datetime.now(UTC),
            },
        )
        _log.info("sql_connection_migrated", tenant_id=tenant_id)

    async def list(self, tenant_id: str) -> list[StoredSqlConnection]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            rows = (
                (await c.execute(text(f'SELECT * FROM "{schema}".sql_connections ORDER BY is_default DESC, name')))
                .mappings()
                .all()
            )
        return [_row(dict(r), tenant_id) for r in rows]

    async def get(self, tenant_id: str, conn_id: str) -> StoredSqlConnection | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT * FROM "{schema}".sql_connections WHERE conn_id = :cid'), {"cid": conn_id}
                    )
                )
                .mappings()
                .first()
            )
        return _row(dict(row), tenant_id) if row else None

    async def get_default(self, tenant_id: str) -> StoredSqlConnection | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            row = (
                (await c.execute(text(f'SELECT * FROM "{schema}".sql_connections WHERE is_default LIMIT 1')))
                .mappings()
                .first()
            )
        return _row(dict(row), tenant_id) if row else None

    async def put(self, conn: StoredSqlConnection) -> None:
        await self.ensure(conn.tenant_id)
        schema = pg_schema(conn.tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{schema}".sql_connections '
                    "(conn_id, name, dsn_ciphertext, tables, dialect, is_default, created_at) "
                    "VALUES (:cid, :name, :dsn, :tables, :dialect, :dflt, :at) "
                    "ON CONFLICT (conn_id) DO UPDATE SET name = :name, dsn_ciphertext = :dsn, "
                    "tables = :tables, dialect = :dialect, is_default = :dflt"
                ),
                {
                    "cid": conn.conn_id,
                    "name": conn.name,
                    "dsn": conn.dsn_ciphertext,
                    "tables": ",".join(conn.tables),
                    "dialect": conn.dialect,
                    "dflt": conn.is_default,
                    "at": datetime.now(UTC),
                },
            )
            if conn.is_default:
                await c.execute(
                    text(f'UPDATE "{schema}".sql_connections SET is_default = (conn_id = :cid)'),
                    {"cid": conn.conn_id},
                )

    async def delete(self, tenant_id: str, conn_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{schema}".sql_connections WHERE conn_id = :cid'), {"cid": conn_id})

    async def set_default(self, tenant_id: str, conn_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{schema}".sql_connections SET is_default = (conn_id = :cid)'),
                {"cid": conn_id},
            )


def _row(m: dict[str, object], tenant_id: str) -> StoredSqlConnection:
    return StoredSqlConnection(
        tenant_id=tenant_id,
        conn_id=str(m["conn_id"]),
        name=str(m["name"]),
        dsn_ciphertext=str(m["dsn_ciphertext"]),
        tables=[t for t in str(m["tables"]).split(",") if t],
        dialect=str(m.get("dialect") or "postgres"),
        is_default=bool(m["is_default"]),
        created_at=m.get("created_at") if isinstance(m.get("created_at"), datetime) else None,
    )
