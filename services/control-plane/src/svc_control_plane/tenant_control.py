"""Tenant-provisioning control plane — the PROVIDER's lifecycle for managed-cloud tenants (T3).

Self-serve signup (``/auth/signup``) creates one org from the outside. This is the INSIDE view: an
Acme platform operator (SUPER_ADMIN, gated by ``platform_admin_emails``) provisions, plans,
suspends and deprovisions tenant orgs across the whole managed-cloud fleet — the keystone of the
RAGaaS SKU.

Separation of duties: a SUPER_ADMIN manages tenant LIFECYCLE but never reads tenant DATA
(:func:`rag_common.rbac.can_read_kb` refuses SUPER_ADMIN). Provisioning REUSES the exact
primitives signup uses (create org -> create owner -> provision the default-KB namespace + a root
org-unit), so a provider-created tenant and a self-serve one are byte-identical; the control plane
only adds the fleet record + plan + status on top.

Lives in one small GLOBAL table (``rag_identity.tenants``) next to the other cross-tenant
registries. Self-serve orgs predating the control plane are simply ABSENT from it and treated as
active, so this is purely additive.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import assert_registrable, get_logger, get_settings, hash_password
from rag_common.errors import ValidationError
from rag_common.rbac import Clearance
from rag_schemas.common import AuditAction
from svc_control_plane import pipeline, wiring
from svc_control_plane.identity import Org, User

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"
_ORG_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,62}$")


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"  # provisioned but service withheld (non-payment / hold)
    DEPROVISIONED = "deprovisioned"  # stores purged; the record is a tombstone


# Named plans -> resolved caps, stamped onto the tenant AT provision time so re-pricing the plan
# catalogue later never silently changes an existing tenant's limits.
PLANS: dict[str, dict[str, float]] = {
    "trial": {"max_documents_per_kb": 100, "monthly_budget": 0.0},
    "standard": {"max_documents_per_kb": 1000, "monthly_budget": 50000.0},
    "enterprise": {"max_documents_per_kb": 100000, "monthly_budget": 1000000.0},
}
DEFAULT_PLAN = "standard"


class TenantRecord(BaseModel):
    org_id: str
    name: str
    admin_email: str
    status: TenantStatus = TenantStatus.ACTIVE
    plan: str = DEFAULT_PLAN
    max_documents_per_kb: int = 1000
    monthly_budget: float = 0.0
    provisioned_at: datetime | None = None
    updated_at: datetime | None = None
    # False = the org exists in the identity store but was never fleet-provisioned (self-serve
    # signup / legacy direct creation). Surfaced read-only so the super-admin view is never blind
    # to an org the system knows about; lifecycle actions (plan/suspend/delete) don't apply until
    # it's adopted into the fleet registry.
    managed: bool = True


class TenantSummary(BaseModel):
    tenant: TenantRecord
    knowledge_bases: int = 0
    users: int = 0


def _plan_caps(plan: str) -> dict[str, float]:
    if plan not in PLANS:
        raise ValidationError(f"unknown plan '{plan}'", details={"plans": sorted(PLANS)})
    return PLANS[plan]


# --- Registry (global rag_identity.tenants table + in-memory twin) -------------------


@runtime_checkable
class TenantRegistry(Protocol):
    async def ensure(self) -> None: ...
    async def create(self, rec: TenantRecord) -> None: ...
    async def get(self, org_id: str) -> TenantRecord | None: ...
    async def list_all(self) -> list[TenantRecord]: ...
    async def set_status(self, org_id: str, status: TenantStatus) -> None: ...
    async def set_plan(self, org_id: str, plan: str, max_documents_per_kb: int, monthly_budget: float) -> None: ...


class InMemoryTenantRegistry:
    def __init__(self) -> None:
        self._rows: dict[str, TenantRecord] = {}

    async def ensure(self) -> None:
        return None

    async def create(self, rec: TenantRecord) -> None:
        if rec.org_id in self._rows:
            raise ValidationError(f"tenant {rec.org_id} already exists")
        assert_registrable(rec.org_id, self._rows.keys())  # no slug-colliding sibling org
        self._rows[rec.org_id] = rec

    async def get(self, org_id: str) -> TenantRecord | None:
        return self._rows.get(org_id)

    async def list_all(self) -> list[TenantRecord]:
        return sorted(self._rows.values(), key=lambda r: r.org_id)

    async def set_status(self, org_id: str, status: TenantStatus) -> None:
        rec = self._rows.get(org_id)
        if rec is not None:
            self._rows[org_id] = rec.model_copy(update={"status": status, "updated_at": datetime.now(UTC)})

    async def set_plan(self, org_id: str, plan: str, max_documents_per_kb: int, monthly_budget: float) -> None:
        rec = self._rows.get(org_id)
        if rec is not None:
            self._rows[org_id] = rec.model_copy(
                update={
                    "plan": plan,
                    "max_documents_per_kb": max_documents_per_kb,
                    "monthly_budget": monthly_budget,
                    "updated_at": datetime.now(UTC),
                }
            )


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".tenants (
        org_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        admin_email TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        plan TEXT NOT NULL DEFAULT 'standard',
        max_documents_per_kb INTEGER NOT NULL DEFAULT 1000,
        monthly_budget DOUBLE PRECISION NOT NULL DEFAULT 0,
        provisioned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ
    )''',
)
_COLS = "org_id, name, admin_email, status, plan, max_documents_per_kb, monthly_budget, provisioned_at, updated_at"


class PostgresTenantRegistry:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def create(self, rec: TenantRecord) -> None:
        await self.ensure()
        assert_registrable(rec.org_id, [t.org_id for t in await self.list_all()])  # no slug-collision
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".tenants ({_COLS}) VALUES '
                    "(:org_id, :name, :admin_email, :status, :plan, :max_documents_per_kb, "
                    ":monthly_budget, :provisioned_at, :updated_at)"
                ),
                {
                    **rec.model_dump(),
                    "status": rec.status.value,
                    "provisioned_at": rec.provisioned_at or datetime.now(UTC),
                },
            )

    async def get(self, org_id: str) -> TenantRecord | None:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".tenants WHERE org_id = :o'
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"o": org_id})).mappings().first()
        return TenantRecord(**dict(row)) if row else None

    async def list_all(self) -> list[TenantRecord]:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".tenants ORDER BY org_id'
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql))).mappings().all()
        return [TenantRecord(**dict(r)) for r in rows]

    async def set_status(self, org_id: str, status: TenantStatus) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".tenants SET status = :s, updated_at = :u WHERE org_id = :o'),
                {"s": status.value, "u": datetime.now(UTC), "o": org_id},
            )

    async def set_plan(self, org_id: str, plan: str, max_documents_per_kb: int, monthly_budget: float) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'UPDATE "{_IDENTITY_SCHEMA}".tenants SET plan = :p, max_documents_per_kb = :m, '
                    "monthly_budget = :b, updated_at = :u WHERE org_id = :o"
                ),
                {"p": plan, "m": max_documents_per_kb, "b": monthly_budget, "u": datetime.now(UTC), "o": org_id},
            )


# --- Lifecycle -----------------------------------------------------------------------


async def assert_active(org_id: str) -> None:
    """Reject a request for a SUSPENDED/DEPROVISIONED tenant. A tenant ABSENT from the registry
    (self-serve signup, or any org predating the control plane) is treated as active — purely
    additive, never blocks the existing self-serve flow."""
    rec = await wiring.tenant_registry().get(org_id)
    if rec is not None and rec.status is not TenantStatus.ACTIVE:
        raise ValidationError(
            f"tenant '{org_id}' is {rec.status.value}", details={"org_id": org_id, "status": rec.status.value}
        )


async def provision(
    *, org_id: str, name: str, admin_email: str, password: str, plan: str = DEFAULT_PLAN, actor: str
) -> TenantRecord:
    """Provision a managed-cloud tenant end to end: org + owner (top clearance) + default-KB
    namespace + root org-unit + fleet record + audit. Reuses the SAME primitives as signup."""
    org_id = org_id.strip().lower()
    if not _ORG_ID_RE.match(org_id):
        raise ValidationError(
            "org_id must be lowercase alnum/-/_ (2-63 chars, letter first)", details={"org_id": org_id}
        )
    email = admin_email.strip().lower()
    ident = wiring.identity_store()
    registry = wiring.tenant_registry()
    # A present FLEET record means fully provisioned — a real duplicate. But an identity org with
    # NO fleet record is a PARTIAL provision: a prior attempt created the org/owner then died
    # before the registry write (e.g. a transient store timeout in provision_kb). Re-provisioning
    # must COMPLETE it (idempotent resume), not 422 into an unrecoverable stuck state. Every step
    # below is made safe to repeat: create_* are guarded here; provision_kb (collection_exists
    # guard) and ensure_root are already idempotent.
    if await registry.get(org_id) is not None:
        raise ValidationError(f"tenant '{org_id}' already exists")
    existing_org = await ident.get_org(org_id)
    existing_user = await ident.get_user(email)
    # A fresh provision must not collide with an unrelated account; on resume the owner is THIS org's.
    if existing_org is None and existing_user is not None:
        raise ValidationError("an account with this email already exists")
    if existing_user is not None and existing_user.org_id != org_id:
        raise ValidationError("an account with this email already exists")
    caps = _plan_caps(plan)

    if existing_org is None:
        await ident.create_org(Org(org_id=org_id, name=name))
    if existing_user is None:
        # The owner sits at the top of the tenant hierarchy -> Restricted clearance, so it can grant
        # any clearance within its own tenant (no chicken-and-egg for the first admin).
        await ident.create_user(
            User(
                email=email,
                org_id=org_id,
                name=email,
                role="owner",
                password_hash=hash_password(password),
                clearance=int(Clearance.RESTRICTED),
            )
        )
    await wiring.provision_kb(org_id)  # the org's default-KB namespace (same as signup); idempotent
    await wiring.org_unit_store().ensure_root(org_id)  # the RBAC scope root; idempotent

    now = datetime.now(UTC)
    rec = TenantRecord(
        org_id=org_id,
        name=name,
        admin_email=email,
        status=TenantStatus.ACTIVE,
        plan=plan,
        max_documents_per_kb=int(caps["max_documents_per_kb"]),
        monthly_budget=caps["monthly_budget"],
        provisioned_at=now,
        updated_at=now,
    )
    await registry.create(rec)
    await _audit(
        org_id,
        AuditAction.APPROVAL,
        actor,
        {"event": "tenant_provisioned", "plan": plan, "admin_email": email},
    )
    _log.info("tenant_provisioned", org_id=org_id, plan=plan, actor=actor)
    return rec


async def summary(org_id: str) -> TenantSummary | None:
    rec = await wiring.tenant_registry().get(org_id)
    if rec is None:
        return None
    kbs = len(await wiring.kb_store().list(org_id))
    users = len(await wiring.identity_store().list_users(org_id))
    return TenantSummary(tenant=rec, knowledge_bases=kbs, users=users)


async def set_status(org_id: str, status: TenantStatus, *, actor: str) -> TenantRecord:
    registry = wiring.tenant_registry()
    if await registry.get(org_id) is None:
        raise ValidationError(f"tenant '{org_id}' is not provisioned")
    await registry.set_status(org_id, status)
    await _audit(org_id, AuditAction.APPROVAL, actor, {"event": "tenant_status", "status": status.value})
    _log.info("tenant_status_changed", org_id=org_id, status=status.value, actor=actor)
    rec = await registry.get(org_id)
    assert rec is not None
    return rec


async def set_plan(org_id: str, plan: str, *, actor: str) -> TenantRecord:
    registry = wiring.tenant_registry()
    if await registry.get(org_id) is None:
        raise ValidationError(f"tenant '{org_id}' is not provisioned")
    caps = _plan_caps(plan)
    await registry.set_plan(org_id, plan, int(caps["max_documents_per_kb"]), caps["monthly_budget"])
    await _audit(org_id, AuditAction.APPROVAL, actor, {"event": "tenant_plan", "plan": plan})
    rec = await registry.get(org_id)
    assert rec is not None
    return rec


async def deprovision(org_id: str, *, actor: str) -> dict[str, int]:
    """Purge every non-default KB of a tenant from all stores, then tombstone the fleet record.
    The identity rows stay as an inert tombstone (deprovisioned, no KBs) — auditable, and not
    resurrectable without a fresh provision. Idempotent for an already-deprovisioned tenant."""
    registry = wiring.tenant_registry()
    if await registry.get(org_id) is None:
        raise ValidationError(f"tenant '{org_id}' is not provisioned")
    kbs = await wiring.kb_store().list(org_id)
    purged = 0
    for kb in kbs:
        if kb.kb_id == org_id:
            continue  # the org-root default KB is left; its docs are handled by store teardown
        result = await pipeline.delete_knowledge_base(org_id, kb.kb_id, actor=actor)
        purged += int(result.get("chunks", 0))
    await registry.set_status(org_id, TenantStatus.DEPROVISIONED)
    await _audit(
        org_id,
        AuditAction.DELETE,
        actor,
        {"event": "tenant_deprovisioned", "knowledge_bases": len(kbs), "chunks_purged": purged},
    )
    _log.info("tenant_deprovisioned", org_id=org_id, kbs=len(kbs), actor=actor)
    return {"knowledge_bases": len(kbs), "chunks_purged": purged}


async def _audit(org_id: str, action: AuditAction, actor: str, payload: dict[str, object]) -> None:
    await wiring.audit_service().record(
        tenant_id=org_id,
        trace_id=datetime.now(UTC).isoformat(),
        action=action,
        actor=actor,
        payload={"scope": "tenant_control", "org_id": org_id, **payload},
    )
