"""Per-KB model registry (FR-E1 — bring-your-own sovereign models).

A knowledge base can register MANY OpenAI-compatible model endpoints (e.g. two AIOps
deployments, a self-hosted vLLM), one marked as its **default**. Each connection's URL,
model/deployment id, auth header and API key are stored in the KB's own Postgres schema
(in-memory twin for CI); the key is encrypted at rest (never plaintext) and the endpoint
is validated against the sovereignty allowlist before use.

Answers use the KB's default connection unless a specific connection is chosen for the
request (per-question model pick); with no connections, the platform default gateway.

Isolation is per-KB, like every other store. The legacy single-connection table is
migrated into this registry on first access, so existing connections keep working.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings, pg_schema

_log = get_logger(__name__)


MODEL_ROLES = ("generation", "embedding", "reranking")


class StoredModelConnection(BaseModel):
    """One registered model endpoint. ``key_ciphertext`` is encrypted at rest."""

    tenant_id: str
    conn_id: str = Field(..., description="Stable id within the KB.")
    name: str = Field(..., description="Human label, e.g. 'AIOps Qwen3-32B'.")
    endpoint: str = Field(..., description="OpenAI-compatible base URL (host must be egress-allowlisted).")
    model: str = Field(..., description="Model/deployment id to request (from the endpoint's /models).")
    header: str = Field(default="apikey", description="Header carrying the API key (the AIOps platform uses 'apikey').")
    key_ciphertext: str = Field(default="", description="Encrypted API key; empty for unauthenticated endpoints.")
    is_default: bool = Field(default=False, description="The connection its role uses unless chosen per request.")
    role: str = Field(
        default="generation",
        pattern="^(generation|embedding|reranking|ocr|vision)$",
        description=(
            "What the model does: answers (generation), vectors (embedding), scoring (reranking), "
            "reading scans (ocr) or describing images (vision)."
        ),
    )
    embed_dim: int | None = Field(
        default=None,
        description="Embedding dimension the endpoint returns (probed at attach time; embedding role only).",
    )
    api_shape: str = Field(
        default="",
        description="Reranker wire shape probed at attach time, e.g. 'cohere+sigmoid' (reranking role only).",
    )
    backend: str = Field(
        default="openai_compat",
        pattern="^(openai_compat|bge_m3_multi)$",
        description=(
            "Embedding ADAPTER type (embedding role only; other roles ignore it). 'openai_compat' = "
            "the /embeddings endpoint with a raw apikey header (default, unchanged). 'bge_m3_multi' = "
            "the bge-m3 /embed endpoint (Bearer auth) that returns dense+learned-sparse+ColBERT — the "
            "adapter the multivector retrieval path (Slice 2b) duck-types."
        ),
    )


@runtime_checkable
class ModelConnectionStore(Protocol):
    async def ensure(self, tenant_id: str) -> None: ...
    async def list(self, tenant_id: str, role: str | None = None) -> list[StoredModelConnection]: ...
    async def get(self, tenant_id: str, conn_id: str) -> StoredModelConnection | None: ...
    async def get_default(self, tenant_id: str, role: str = "generation") -> StoredModelConnection | None: ...
    async def put(self, conn: StoredModelConnection) -> None: ...
    async def delete(self, tenant_id: str, conn_id: str) -> None: ...
    async def set_default(self, tenant_id: str, conn_id: str) -> None: ...


class InMemoryModelConnectionStore:
    def __init__(self) -> None:
        self._c: dict[str, dict[str, StoredModelConnection]] = {}

    async def ensure(self, tenant_id: str) -> None:
        return None

    async def list(self, tenant_id: str, role: str | None = None) -> list[StoredModelConnection]:
        conns = [c for c in self._c.get(tenant_id, {}).values() if role is None or c.role == role]
        return sorted(conns, key=lambda c: (not c.is_default, c.name))

    async def get(self, tenant_id: str, conn_id: str) -> StoredModelConnection | None:
        return self._c.get(tenant_id, {}).get(conn_id)

    async def get_default(self, tenant_id: str, role: str = "generation") -> StoredModelConnection | None:
        conns = [c for c in self._c.get(tenant_id, {}).values() if c.role == role]
        for c in conns:
            if c.is_default:
                return c
        # Only generation keeps the any-row fallback: falling back to a non-default
        # embedder/reranker would silently activate something nobody chose.
        return next(iter(conns), None) if role == "generation" else None

    async def put(self, conn: StoredModelConnection) -> None:
        self._c.setdefault(conn.tenant_id, {})[conn.conn_id] = conn
        if conn.is_default:
            await self.set_default(conn.tenant_id, conn.conn_id)

    async def delete(self, tenant_id: str, conn_id: str) -> None:
        self._c.get(tenant_id, {}).pop(conn_id, None)

    async def set_default(self, tenant_id: str, conn_id: str) -> None:
        conns = self._c.get(tenant_id, {})
        role = conns[conn_id].role if conn_id in conns else "generation"
        for cid, c in conns.items():
            if c.role == role:
                conns[cid] = c.model_copy(update={"is_default": cid == conn_id})


_DDL = """CREATE TABLE IF NOT EXISTS "{schema}".model_connections (
    conn_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    header TEXT NOT NULL DEFAULT 'apikey',
    key_ciphertext TEXT NOT NULL DEFAULT '',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    role TEXT NOT NULL DEFAULT 'generation',
    embed_dim INTEGER,
    api_shape TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT 'openai_compat',
    created_at TIMESTAMPTZ NOT NULL
)"""

# Pre-role tables gain the new columns in place; existing rows are generation models.
_MIGRATE = (
    "ALTER TABLE \"{schema}\".model_connections ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'generation'",
    'ALTER TABLE "{schema}".model_connections ADD COLUMN IF NOT EXISTS embed_dim INTEGER',
    "ALTER TABLE \"{schema}\".model_connections ADD COLUMN IF NOT EXISTS api_shape TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE \"{schema}\".model_connections ADD COLUMN IF NOT EXISTS backend TEXT NOT NULL DEFAULT 'openai_compat'",
)


class PostgresModelConnection:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            await conn.execute(text(_DDL.format(schema=schema)))
            for stmt in _MIGRATE:
                await conn.execute(text(stmt.format(schema=schema)))
            # Migrate the legacy singleton table (one connection) into the registry once.
            legacy = (await conn.execute(text(f"SELECT to_regclass('\"{schema}\".model_connection')"))).scalar()
            if legacy is not None:
                empty = (await conn.execute(text(f'SELECT COUNT(*) FROM "{schema}".model_connections'))).scalar()
                if not empty:
                    old = (
                        (await conn.execute(text(f'SELECT * FROM "{schema}".model_connection WHERE id = 1')))
                        .mappings()
                        .first()
                    )
                    if old is not None:
                        await conn.execute(
                            text(
                                f'INSERT INTO "{schema}".model_connections '
                                "(conn_id, name, endpoint, model, header, key_ciphertext, is_default, created_at) "
                                "VALUES (:cid, :name, :endpoint, :model, :header, :key, :dflt, :at)"
                            ),
                            {
                                "cid": "legacy",
                                "name": str(old["model"]),
                                "endpoint": str(old["endpoint"]),
                                "model": str(old["model"]),
                                "header": str(old["header"]),
                                "key": str(old["key_ciphertext"]),
                                "dflt": bool(old["active"]),
                                "at": datetime.now(UTC),
                            },
                        )
                        _log.info("model_connection_migrated", schema=schema)

    async def list(self, tenant_id: str, role: str | None = None) -> list[StoredModelConnection]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        where = "WHERE role = :role" if role is not None else ""
        params = {"role": role} if role is not None else {}
        async with self._engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        text(f'SELECT * FROM "{schema}".model_connections {where} ORDER BY is_default DESC, name'),
                        params,
                    )
                )
                .mappings()
                .all()
            )
        return [_row(dict(r), tenant_id) for r in rows]

    async def get(self, tenant_id: str, conn_id: str) -> StoredModelConnection | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT * FROM "{schema}".model_connections WHERE conn_id = :cid'), {"cid": conn_id}
                    )
                )
                .mappings()
                .first()
            )
        return _row(dict(row), tenant_id) if row else None

    async def get_default(self, tenant_id: str, role: str = "generation") -> StoredModelConnection | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        # Only generation keeps the any-row fallback: a non-default embedder/reranker
        # must never activate implicitly.
        clause = "" if role == "generation" else "AND is_default"
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(
                            f'SELECT * FROM "{schema}".model_connections WHERE role = :role {clause} '
                            "ORDER BY is_default DESC, created_at LIMIT 1"
                        ),
                        {"role": role},
                    )
                )
                .mappings()
                .first()
            )
        return _row(dict(row), tenant_id) if row else None

    async def put(self, conn: StoredModelConnection) -> None:
        await self.ensure(conn.tenant_id)
        schema = pg_schema(conn.tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{schema}".model_connections '
                    "(conn_id, name, endpoint, model, header, key_ciphertext, is_default, "
                    "role, embed_dim, api_shape, backend, created_at) "
                    "VALUES (:cid, :name, :endpoint, :model, :header, :key, :dflt, :role, :dim, :shape, :backend, :at) "
                    "ON CONFLICT (conn_id) DO UPDATE SET name = :name, endpoint = :endpoint, model = :model, "
                    "header = :header, key_ciphertext = :key, is_default = :dflt, role = :role, "
                    "embed_dim = :dim, api_shape = :shape, backend = :backend"
                ),
                {
                    "cid": conn.conn_id,
                    "name": conn.name,
                    "endpoint": conn.endpoint,
                    "model": conn.model,
                    "header": conn.header,
                    "key": conn.key_ciphertext,
                    "dflt": conn.is_default,
                    "role": conn.role,
                    "dim": conn.embed_dim,
                    "shape": conn.api_shape,
                    "backend": conn.backend,
                    "at": datetime.now(UTC),
                },
            )
            if conn.is_default:
                await c.execute(
                    text(f'UPDATE "{schema}".model_connections SET is_default = (conn_id = :cid) WHERE role = :role'),
                    {"cid": conn.conn_id, "role": conn.role},
                )

    async def delete(self, tenant_id: str, conn_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{schema}".model_connections WHERE conn_id = :cid'), {"cid": conn_id})

    async def set_default(self, tenant_id: str, conn_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'UPDATE "{schema}".model_connections SET is_default = (conn_id = :cid) '
                    f'WHERE role = (SELECT role FROM "{schema}".model_connections WHERE conn_id = :cid)'
                ),
                {"cid": conn_id},
            )


def _row(m: dict[str, object], tenant_id: str) -> StoredModelConnection:
    dim = m.get("embed_dim")
    return StoredModelConnection(
        tenant_id=tenant_id,
        conn_id=str(m["conn_id"]),
        name=str(m["name"]),
        endpoint=str(m["endpoint"]),
        model=str(m["model"]),
        header=str(m["header"]),
        key_ciphertext=str(m["key_ciphertext"]),
        is_default=bool(m["is_default"]),
        role=str(m.get("role") or "generation"),
        embed_dim=dim if isinstance(dim, int) else None,
        api_shape=str(m.get("api_shape") or ""),
        backend=str(m.get("backend") or "openai_compat"),
    )
