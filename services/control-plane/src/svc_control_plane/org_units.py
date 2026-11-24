"""Org-unit tree — the nested departmental scope each tenant's RBAC hangs on (Sub-step A).

A tenant owns a TREE of units (division -> department -> team). A user belongs to one unit; a KB
belongs to one unit. The tree is what :mod:`rag_common.rbac` walks to decide need-to-know scope:
data at your unit or an ANCESTOR flows down to you; a UNIT_ADMIN governs its own SUBTREE. Every
tenant has a ROOT unit whose id EQUALS the tenant id, so a user/KB with no explicit unit sits at
the root (tenant-wide) — least surprise, and back-compatible with everything that predates units.

Lives in one small GLOBAL table in ``rag_identity`` alongside the other cross-tenant registries
(grants, tenants, eval schedules). Provider-operated, per-tenant scoped by ``tenant_id``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings
from rag_common.errors import ValidationError
from rag_common.rbac import Tier

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"


class OrgUnit(BaseModel):
    unit_id: str
    tenant_id: str
    name: str
    parent_id: str | None = Field(default=None, description="None only for the tenant ROOT unit.")


# --- tenant role -> management tier (the JWT coarse role maps to an RBAC Tier) --------
# owner/admin administer the whole tenant; UNIT_ADMIN is assigned per-unit in Sub-step D (an
# admin scoped to a non-root unit). SUPER_ADMIN never comes from a tenant role — only the
# platform allowlist (tenant_control.is_platform_admin).
_ROLE_TIER: dict[str, Tier] = {
    "owner": Tier.TENANT_ADMIN,
    "admin": Tier.TENANT_ADMIN,
    "member": Tier.MEMBER,
    "viewer": Tier.VIEWER,
}


def tier_for_role(role: str, *, is_platform_admin: bool = False, unit_admin: bool = False) -> Tier:
    """Map a tenant role to its management :class:`Tier`.

    ``is_platform_admin`` (the Acme operator allowlist) wins as SUPER_ADMIN. ``unit_admin`` demotes
    a tenant admin to UNIT_ADMIN when the account is scoped to a specific (non-root) unit."""
    if is_platform_admin:
        return Tier.SUPER_ADMIN
    tier = _ROLE_TIER.get(role, Tier.VIEWER)
    if unit_admin and tier is Tier.TENANT_ADMIN:
        return Tier.UNIT_ADMIN
    return tier


# --- store ---------------------------------------------------------------------------


@runtime_checkable
class OrgUnitStore(Protocol):
    async def ensure(self) -> None: ...
    async def ensure_root(self, tenant_id: str) -> OrgUnit: ...
    async def create(self, unit: OrgUnit) -> None: ...
    async def get(self, tenant_id: str, unit_id: str) -> OrgUnit | None: ...
    async def list_for_tenant(self, tenant_id: str) -> list[OrgUnit]: ...
    async def delete(self, tenant_id: str, unit_id: str) -> None: ...


async def parents_map(store: OrgUnitStore, tenant_id: str) -> dict[str, str | None]:
    """The ``{unit_id: parent_id}`` mapping the rbac tree math walks. Includes the root
    (parent None). A tenant with no explicit units yields ``{tenant_id: None}`` (root only)."""
    units = await store.list_for_tenant(tenant_id)
    if not any(u.unit_id == tenant_id for u in units):
        units = [OrgUnit(unit_id=tenant_id, tenant_id=tenant_id, name="root", parent_id=None), *units]
    return {u.unit_id: u.parent_id for u in units}


class InMemoryOrgUnitStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], OrgUnit] = {}

    async def ensure(self) -> None:
        return None

    async def ensure_root(self, tenant_id: str) -> OrgUnit:
        key = (tenant_id, tenant_id)
        if key not in self._rows:
            self._rows[key] = OrgUnit(unit_id=tenant_id, tenant_id=tenant_id, name="root", parent_id=None)
        return self._rows[key]

    async def create(self, unit: OrgUnit) -> None:
        if unit.parent_id is not None and (unit.tenant_id, unit.parent_id) not in self._rows:
            raise ValidationError(f"parent unit '{unit.parent_id}' does not exist")
        key = (unit.tenant_id, unit.unit_id)
        if key in self._rows:
            raise ValidationError(f"unit '{unit.unit_id}' already exists")
        self._rows[key] = unit

    async def get(self, tenant_id: str, unit_id: str) -> OrgUnit | None:
        return self._rows.get((tenant_id, unit_id))

    async def list_for_tenant(self, tenant_id: str) -> list[OrgUnit]:
        return [u for (t, _), u in self._rows.items() if t == tenant_id]

    async def delete(self, tenant_id: str, unit_id: str) -> None:
        if unit_id == tenant_id:
            raise ValidationError("the tenant root unit cannot be deleted")
        if any(u.parent_id == unit_id for u in await self.list_for_tenant(tenant_id)):
            raise ValidationError("cannot delete a unit that still has child units")
        self._rows.pop((tenant_id, unit_id), None)


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".org_units (
        tenant_id TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        name TEXT NOT NULL,
        parent_id TEXT,
        PRIMARY KEY (tenant_id, unit_id)
    )''',
    f'CREATE INDEX IF NOT EXISTS ix_org_units_parent ON "{_IDENTITY_SCHEMA}".org_units (tenant_id, parent_id)',
)
_COLS = "tenant_id, unit_id, name, parent_id"


class PostgresOrgUnitStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def ensure_root(self, tenant_id: str) -> OrgUnit:
        await self.ensure()
        root = OrgUnit(unit_id=tenant_id, tenant_id=tenant_id, name="root", parent_id=None)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".org_units ({_COLS}) '
                    "VALUES (:tenant_id, :unit_id, :name, :parent_id) ON CONFLICT DO NOTHING"
                ),
                root.model_dump(),
            )
        return root

    async def create(self, unit: OrgUnit) -> None:
        await self.ensure()
        if unit.parent_id is not None and await self.get(unit.tenant_id, unit.parent_id) is None:
            raise ValidationError(f"parent unit '{unit.parent_id}' does not exist")
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".org_units ({_COLS}) '
                    "VALUES (:tenant_id, :unit_id, :name, :parent_id)"
                ),
                unit.model_dump(),
            )

    async def get(self, tenant_id: str, unit_id: str) -> OrgUnit | None:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".org_units WHERE tenant_id = :t AND unit_id = :u'
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"t": tenant_id, "u": unit_id})).mappings().first()
        return OrgUnit(**dict(row)) if row else None

    async def list_for_tenant(self, tenant_id: str) -> list[OrgUnit]:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".org_units WHERE tenant_id = :t ORDER BY unit_id'
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"t": tenant_id})).mappings().all()
        return [OrgUnit(**dict(r)) for r in rows]

    async def delete(self, tenant_id: str, unit_id: str) -> None:
        await self.ensure()
        if unit_id == tenant_id:
            raise ValidationError("the tenant root unit cannot be deleted")
        kids_sql = f'SELECT 1 FROM "{_IDENTITY_SCHEMA}".org_units WHERE tenant_id = :t AND parent_id = :u LIMIT 1'
        async with self._engine.begin() as c:
            has_kids = (await c.execute(text(kids_sql), {"t": tenant_id, "u": unit_id})).first()
            if has_kids:
                raise ValidationError("cannot delete a unit that still has child units")
            await c.execute(
                text(f'DELETE FROM "{_IDENTITY_SCHEMA}".org_units WHERE tenant_id = :t AND unit_id = :u'),
                {"t": tenant_id, "u": unit_id},
            )
