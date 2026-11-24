"""One-time auth tokens for account activation and password reset (FR-H3 member lifecycle).

A token is a random secret handed to a user out-of-band (email link, or a link the admin
copies); only its SHA-256 hash is stored. Tokens are single-use (deleted on consume) and
expire. Global table (users can't be enumerated cross-tenant), with an in-memory twin for CI.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_settings

_IDENTITY_SCHEMA = "rag_identity"

# purpose values
INVITE = "invite"
RESET = "reset"


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@runtime_checkable
class AuthTokenStore(Protocol):
    async def issue(self, email: str, purpose: str, ttl_seconds: int) -> str: ...
    async def consume(self, raw_token: str) -> tuple[str, str] | None: ...


class InMemoryAuthTokenStore:
    def __init__(self) -> None:
        self._toks: dict[str, tuple[str, str, datetime]] = {}  # hash -> (email, purpose, expires)

    async def issue(self, email: str, purpose: str, ttl_seconds: int) -> str:
        raw = secrets.token_urlsafe(32)
        self._toks[_hash(raw)] = (email.lower(), purpose, datetime.now(UTC) + timedelta(seconds=ttl_seconds))
        return raw

    async def consume(self, raw_token: str) -> tuple[str, str] | None:
        rec = self._toks.pop(_hash(raw_token), None)
        if rec is None or rec[2] < datetime.now(UTC):
            return None
        return rec[0], rec[1]


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".auth_tokens (
        token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, purpose TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL)''',
)


class PostgresAuthTokenStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
        self._ready = False

    async def ensure(self) -> None:
        if self._ready:
            return
        async with self._engine.begin() as conn:
            for stmt in _DDL:
                await conn.execute(text(stmt))
        self._ready = True

    async def issue(self, email: str, purpose: str, ttl_seconds: int) -> str:
        await self.ensure()
        raw = secrets.token_urlsafe(32)
        expires = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".auth_tokens (token_hash, email, purpose, expires_at) '
                    "VALUES (:h, :e, :p, :x)"
                ),
                {"h": _hash(raw), "e": email.lower(), "p": purpose, "x": expires},
            )
        return raw

    async def consume(self, raw_token: str) -> tuple[str, str] | None:
        await self.ensure()
        async with self._engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            f'DELETE FROM "{_IDENTITY_SCHEMA}".auth_tokens WHERE token_hash = :h '
                            "RETURNING email, purpose, expires_at"
                        ),
                        {"h": _hash(raw_token)},
                    )
                )
                .mappings()
                .first()
            )
        if row is None or row["expires_at"] < datetime.now(UTC):
            return None
        return row["email"], row["purpose"]
