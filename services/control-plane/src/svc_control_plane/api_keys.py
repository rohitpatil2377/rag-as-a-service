"""Scoped service-account API keys (FR-I4) — programmatic access to the /v1 surface.

A key belongs to an organisation and carries a scope (read / ingest / admin). It is shown
in full exactly ONCE at creation; only its SHA-256 hash is stored, plus a short prefix for
display. On a request the key is resolved back to a :class:`Principal` (org + roles), so
every downstream tenancy/RBAC/ACL control applies unchanged.

Keys live in the GLOBAL ``rag_identity`` schema (like accounts) — the lookup is by prefix,
so it must not require knowing the org up front. This is the seam SDKs, the embeddable
widget and bots authenticate through.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings

_log = get_logger(__name__)

_IDENTITY_SCHEMA = "rag_identity"
_PREFIX = "ek_"
_KEY_PARTS = 3  # ek_<prefix>_<secret>

# A scope maps to the roles the resulting Principal carries (RBAC downstream is unchanged).
SCOPE_ROLES: dict[str, list[str]] = {
    "read": ["reader"],
    "ingest": ["reader", "ingestor"],
    "admin": ["reader", "ingestor", "admin"],
}


class ApiKeyRecord(BaseModel):
    """A key as stored/listed — never carries the secret."""

    key_id: str
    org_id: str
    name: str
    prefix: str = Field(description="Short public identifier, shown as ek_<prefix>_…")
    scope: str
    created_at: datetime


def mint_key() -> tuple[str, str, str]:
    """Return ``(full_key, prefix, sha256_hash)``. The full key is shown once, never stored."""
    prefix = secrets.token_hex(4)
    full = f"{_PREFIX}{prefix}_{secrets.token_urlsafe(24)}"
    return full, prefix, hashlib.sha256(full.encode()).hexdigest()


def looks_like_api_key(token: str) -> bool:
    return token.startswith(_PREFIX)


def key_prefix(full_key: str) -> str | None:
    """The public prefix of a full key, or None if malformed. The secret (token_urlsafe) may
    contain '_', so we bound the split to the first two separators (ek_<prefix>_<secret>)."""
    parts = full_key.split("_", 2)
    if len(parts) != _KEY_PARTS or parts[0] != _PREFIX.rstrip("_"):
        return None
    return parts[1]


@runtime_checkable
class ApiKeyStore(Protocol):
    async def ensure(self) -> None: ...
    async def create(self, rec: ApiKeyRecord, key_hash: str) -> None: ...
    async def list(self, org_id: str) -> list[ApiKeyRecord]: ...
    async def resolve(self, full_key: str) -> ApiKeyRecord | None: ...
    async def revoke(self, org_id: str, key_id: str) -> None: ...


class InMemoryApiKeyStore:
    def __init__(self) -> None:
        self._by_id: dict[str, ApiKeyRecord] = {}
        self._hash: dict[str, str] = {}  # key_id -> hash

    async def ensure(self) -> None:
        return None

    async def create(self, rec: ApiKeyRecord, key_hash: str) -> None:
        self._by_id[rec.key_id] = rec
        self._hash[rec.key_id] = key_hash

    async def list(self, org_id: str) -> list[ApiKeyRecord]:
        return sorted((r for r in self._by_id.values() if r.org_id == org_id), key=lambda r: r.created_at, reverse=True)

    async def resolve(self, full_key: str) -> ApiKeyRecord | None:
        h = hashlib.sha256(full_key.encode()).hexdigest()
        for kid, stored in self._hash.items():
            if secrets.compare_digest(stored, h):
                return self._by_id.get(kid)
        return None

    async def revoke(self, org_id: str, key_id: str) -> None:
        rec = self._by_id.get(key_id)
        if rec and rec.org_id == org_id:
            self._by_id.pop(key_id, None)
            self._hash.pop(key_id, None)


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".api_keys (
        key_id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        name TEXT NOT NULL,
        prefix TEXT NOT NULL,
        scope TEXT NOT NULL,
        key_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )''',
    f'CREATE INDEX IF NOT EXISTS api_keys_by_prefix ON "{_IDENTITY_SCHEMA}".api_keys (prefix)',
)


class PostgresApiKeyStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def create(self, rec: ApiKeyRecord, key_hash: str) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".api_keys '
                    "(key_id, org_id, name, prefix, scope, key_hash, created_at) "
                    "VALUES (:id, :org, :name, :prefix, :scope, :hash, :at)"
                ),
                {
                    "id": rec.key_id,
                    "org": rec.org_id,
                    "name": rec.name,
                    "prefix": rec.prefix,
                    "scope": rec.scope,
                    "hash": key_hash,
                    "at": rec.created_at,
                },
            )

    async def list(self, org_id: str) -> list[ApiKeyRecord]:
        await self.ensure()
        async with self._engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        text(
                            f"SELECT key_id, org_id, name, prefix, scope, created_at "
                            f'FROM "{_IDENTITY_SCHEMA}".api_keys WHERE org_id = :org ORDER BY created_at DESC'
                        ),
                        {"org": org_id},
                    )
                )
                .mappings()
                .all()
            )
        return [ApiKeyRecord(**dict(r)) for r in rows]

    async def resolve(self, full_key: str) -> ApiKeyRecord | None:
        prefix = key_prefix(full_key)  # ek_<prefix>_<secret>; secret may itself contain "_"
        if prefix is None:
            return None
        h = hashlib.sha256(full_key.encode()).hexdigest()
        await self.ensure()
        async with self._engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        text(
                            f"SELECT key_id, org_id, name, prefix, scope, key_hash, created_at "
                            f'FROM "{_IDENTITY_SCHEMA}".api_keys WHERE prefix = :p'
                        ),
                        {"p": prefix},
                    )
                )
                .mappings()
                .all()
            )
        for r in rows:
            if secrets.compare_digest(str(r["key_hash"]), h):
                data = {k: r[k] for k in ("key_id", "org_id", "name", "prefix", "scope", "created_at")}
                return ApiKeyRecord(**data)
        return None

    async def revoke(self, org_id: str, key_id: str) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(f'DELETE FROM "{_IDENTITY_SCHEMA}".api_keys WHERE key_id = :id AND org_id = :org'),
                {"id": key_id, "org": org_id},
            )
