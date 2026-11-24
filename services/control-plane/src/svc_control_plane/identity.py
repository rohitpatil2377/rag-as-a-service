"""Accounts & organisations — the multi-tenant SaaS identity layer (FR-H1 / auth).

Anyone can sign up: signup creates an ORGANISATION (which IS the isolation tenant) and its
owner user. Additional users are added to the org with a role. Identity lives in a single
global ``rag_identity`` schema (orgs + users) — distinct from the per-tenant data schemas,
which stay hard-isolated. Passwords are scrypt-hashed; OIDC/SSO layers on later and yields
the same Principal.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_settings

# Role -> capability roles carried in the JWT so existing RBAC/tenant checks keep working.
ROLE_CAPS: dict[str, list[str]] = {
    "owner": ["owner", "admin", "reader", "ingestor", "reviewer"],
    "admin": ["admin", "reader", "ingestor", "reviewer"],
    "member": ["reader", "ingestor", "reviewer"],
    "viewer": ["reader"],
}
# The fine-grained capability roles an admin can select/deselect per user (owner is reserved
# for the account creator and always implies admin).
CAP_ROLES: tuple[str, ...] = ("reader", "ingestor", "reviewer", "admin")
_IDENTITY_SCHEMA = "rag_identity"


def effective_roles(user: User) -> list[str]:
    """The capability roles carried in a user's JWT. Prefer the explicit ``roles`` list
    (fine-grained, set per user); fall back to the coarse ``role`` bundle for older users."""
    roles = list(user.roles) if user.roles else list(ROLE_CAPS.get(user.role, ["reader"]))
    if user.role == "owner" and "owner" not in roles:
        roles = ["owner", *roles]
    return roles


class Org(BaseModel):
    org_id: str
    name: str


class User(BaseModel):
    email: str
    org_id: str
    name: str
    role: str = "owner"
    password_hash: str = ""
    # Account lifecycle: invited (created by an admin, not yet activated) -> active (set a
    # password and can sign in) -> suspended (access revoked). Owners/self-signup are active.
    status: str = "active"
    # Fine-grained capability roles (reader/ingestor/reviewer/admin). Empty -> derived from `role`.
    roles: list[str] = Field(default_factory=list)
    # Knowledge bases this user may access (kb_ids). Empty -> ALL KBs in the org.
    kb_access: list[str] = Field(default_factory=list)
    # Hierarchical RBAC: the org unit the user belongs to (None = tenant ROOT) and the data
    # clearance ceiling (0=Public..3=Restricted). Set by an admin, escalation-gated.
    unit_id: str | None = None
    clearance: int = 0


@runtime_checkable
class IdentityStore(Protocol):
    async def ensure(self) -> None: ...
    async def get_org(self, org_id: str) -> Org | None: ...
    async def create_org(self, org: Org) -> None: ...
    async def list_orgs(self) -> list[Org]: ...
    async def get_user(self, email: str) -> User | None: ...
    async def create_user(self, user: User) -> None: ...
    async def list_users(self, org_id: str) -> list[User]: ...
    async def set_user_access(self, email: str, *, roles: list[str], kb_access: list[str]) -> None: ...
    async def set_governance(self, email: str, *, clearance: int, unit_id: str | None) -> None: ...
    async def set_password(self, email: str, password_hash: str) -> None: ...
    async def set_status(self, email: str, status: str) -> None: ...
    async def delete_user(self, email: str) -> None: ...


class InMemoryIdentityStore:
    def __init__(self) -> None:
        self._orgs: dict[str, Org] = {}
        self._users: dict[str, User] = {}

    async def ensure(self) -> None:
        return None

    async def get_org(self, org_id: str) -> Org | None:
        return self._orgs.get(org_id)

    async def create_org(self, org: Org) -> None:
        self._orgs[org.org_id] = org

    async def list_orgs(self) -> list[Org]:
        return list(self._orgs.values())

    async def get_user(self, email: str) -> User | None:
        return self._users.get(email.lower())

    async def create_user(self, user: User) -> None:
        self._users[user.email.lower()] = user

    async def list_users(self, org_id: str) -> list[User]:
        return [u for u in self._users.values() if u.org_id == org_id]

    async def set_user_access(self, email: str, *, roles: list[str], kb_access: list[str]) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            self._users[email.lower()] = u.model_copy(update={"roles": roles, "kb_access": kb_access})

    async def set_governance(self, email: str, *, clearance: int, unit_id: str | None) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            self._users[email.lower()] = u.model_copy(update={"clearance": clearance, "unit_id": unit_id})

    async def set_password(self, email: str, password_hash: str) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            self._users[email.lower()] = u.model_copy(update={"password_hash": password_hash, "status": "active"})

    async def set_status(self, email: str, status: str) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            self._users[email.lower()] = u.model_copy(update={"status": status})

    async def delete_user(self, email: str) -> None:
        self._users.pop(email.lower(), None)


_DDL = [
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".orgs (org_id TEXT PRIMARY KEY, name TEXT NOT NULL)',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".users (
        email TEXT PRIMARY KEY, org_id TEXT NOT NULL, name TEXT NOT NULL,
        role TEXT NOT NULL, password_hash TEXT NOT NULL)''',
    # Migration-safe: fine-grained roles + per-KB access for RBAC (added after initial release).
    f'''ALTER TABLE "{_IDENTITY_SCHEMA}".users ADD COLUMN IF NOT EXISTS roles JSONB DEFAULT '[]'::jsonb''',
    f'''ALTER TABLE "{_IDENTITY_SCHEMA}".users ADD COLUMN IF NOT EXISTS kb_access JSONB DEFAULT '[]'::jsonb''',
    # Account lifecycle (added after RBAC): existing users default to active (backward-compatible).
    f'''ALTER TABLE "{_IDENTITY_SCHEMA}".users ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active' NOT NULL''',
    # Hierarchical RBAC (added later): unit + clearance default to root/Public = prior behaviour.
    f'''ALTER TABLE "{_IDENTITY_SCHEMA}".users ADD COLUMN IF NOT EXISTS unit_id TEXT''',
    f'''ALTER TABLE "{_IDENTITY_SCHEMA}".users ADD COLUMN IF NOT EXISTS clearance INTEGER DEFAULT 0 NOT NULL''',
]


def _as_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw:
        try:
            return [str(x) for x in json.loads(raw)]
        except (ValueError, TypeError):
            return []
    return []


class PostgresIdentityStore:
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

    async def get_org(self, org_id: str) -> Org | None:
        await self.ensure()
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(f'SELECT org_id, name FROM "{_IDENTITY_SCHEMA}".orgs WHERE org_id = :i'), {"i": org_id}
                    )
                )
                .mappings()
                .first()
            )
        return Org(org_id=row["org_id"], name=row["name"]) if row else None

    async def create_org(self, org: Org) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'INSERT INTO "{_IDENTITY_SCHEMA}".orgs (org_id, name) VALUES (:i, :n) ON CONFLICT DO NOTHING'),
                {"i": org.org_id, "n": org.name},
            )

    async def list_orgs(self) -> list[Org]:
        await self.ensure()
        async with self._engine.connect() as conn:
            rows = (
                (await conn.execute(text(f'SELECT org_id, name FROM "{_IDENTITY_SCHEMA}".orgs ORDER BY org_id')))
                .mappings()
                .all()
            )
        return [Org(org_id=r["org_id"], name=r["name"]) for r in rows]

    async def get_user(self, email: str) -> User | None:
        await self.ensure()
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(f'SELECT * FROM "{_IDENTITY_SCHEMA}".users WHERE email = :e'), {"e": email.lower()}
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        return User(
            email=row["email"],
            org_id=row["org_id"],
            name=row["name"],
            role=row["role"],
            password_hash=row["password_hash"],
            status=row.get("status") or "active",
            roles=_as_list(row.get("roles")),
            kb_access=_as_list(row.get("kb_access")),
            unit_id=row.get("unit_id"),
            clearance=int(row.get("clearance") or 0),
        )

    async def create_user(self, user: User) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".users '
                    "(email, org_id, name, role, password_hash, status, roles, kb_access, unit_id, clearance) "
                    "VALUES (:e, :o, :n, :r, :p, :s, CAST(:rl AS JSONB), CAST(:ka AS JSONB), :u, :c)"
                ),
                {
                    "e": user.email.lower(),
                    "o": user.org_id,
                    "n": user.name,
                    "r": user.role,
                    "p": user.password_hash,
                    "s": user.status,
                    "rl": json.dumps(user.roles),
                    "ka": json.dumps(user.kb_access),
                    "u": user.unit_id,
                    "c": user.clearance,
                },
            )

    async def list_users(self, org_id: str) -> list[User]:
        await self.ensure()
        query = (
            f"SELECT email, org_id, name, role, status, roles, kb_access, unit_id, clearance "
            f'FROM "{_IDENTITY_SCHEMA}".users WHERE org_id = :o ORDER BY email'
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(query), {"o": org_id})).mappings().all()
        return [
            User(
                email=r["email"],
                org_id=r["org_id"],
                name=r["name"],
                role=r["role"],
                status=r.get("status") or "active",
                roles=_as_list(r.get("roles")),
                kb_access=_as_list(r.get("kb_access")),
                unit_id=r.get("unit_id"),
                clearance=int(r.get("clearance") or 0),
            )
            for r in rows
        ]

    async def set_user_access(self, email: str, *, roles: list[str], kb_access: list[str]) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'UPDATE "{_IDENTITY_SCHEMA}".users SET roles = CAST(:rl AS JSONB), '
                    "kb_access = CAST(:ka AS JSONB) WHERE email = :e"
                ),
                {"e": email.lower(), "rl": json.dumps(roles), "ka": json.dumps(kb_access)},
            )

    async def set_governance(self, email: str, *, clearance: int, unit_id: str | None) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".users SET clearance = :c, unit_id = :u WHERE email = :e'),
                {"e": email.lower(), "c": clearance, "u": unit_id},
            )

    async def set_password(self, email: str, password_hash: str) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"UPDATE \"{_IDENTITY_SCHEMA}\".users SET password_hash = :p, status = 'active' WHERE email = :e"),
                {"e": email.lower(), "p": password_hash},
            )

    async def set_status(self, email: str, status: str) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".users SET status = :s WHERE email = :e'),
                {"e": email.lower(), "s": status},
            )

    async def delete_user(self, email: str) -> None:
        await self.ensure()
        async with self._engine.begin() as conn:
            await conn.execute(text(f'DELETE FROM "{_IDENTITY_SCHEMA}".users WHERE email = :e'), {"e": email.lower()})
