"""Egress-host approval workflow — self-service, without weakening the sovereignty guard.

A blocked external host (web/S3 connector) becomes a REQUEST an owner/admin approves, instead
of a dead end. The guard is never disabled: approval only ADDS one host to the allowlist, which
the guard already enforces. Every request and decision is audited, so widening the allowlist is
always attributable.

Reuses the runtime-override pattern (see feature_flags): an approved host is appended to the
live ``RAG_ALLOWED_EGRESS_HOSTS`` and the settings cache is rebuilt, so ``assert_sovereign``
picks it up with no restart. Approved hosts persist in one small global table and are re-applied
at startup. Platform-wide in v1.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings
from rag_common.errors import NotFoundError, ValidationError
from rag_schemas.common import AuditAction

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"
_EGRESS_ENV = "RAG_ALLOWED_EGRESS_HOSTS"

Status = Literal["pending", "approved", "rejected", "revoked"]


class EgressRequest(BaseModel):
    id: str
    host: str
    tenant_id: str
    kb_id: str | None = None
    requested_by: str
    reason: str = ""
    status: Status = "pending"
    created_at: datetime
    decided_by: str | None = None
    decided_at: datetime | None = None


def normalize_host(host_or_url: str) -> str:
    """Extract a bare hostname from a URL or ``host[:port][/path]`` string."""
    s = (host_or_url or "").strip()
    if not s:
        raise ValidationError("a host or URL is required")
    parsed = urlsplit(s if "://" in s else f"//{s}")
    host = parsed.hostname
    if not host:
        raise ValidationError("could not parse a host", details={"input": host_or_url})
    return host.lower()


# The allowlist the DEPLOYMENT was configured with, captured before any runtime approval mutates
# it. The approval workflow exists to WIDEN the allowlist; letting it narrow the list below the
# operator's own configuration is a different power, and one click of it took generation down --
# an admin revoked the generation model's host and every answer failed with a sovereignty error
# that pointed at the guard rather than at the revoke. A restart "fixed" it, which is worse: the
# cause disappears with the symptom.
_BASE_HOSTS: frozenset[str] | None = None


def _base_hosts() -> frozenset[str]:
    global _BASE_HOSTS  # noqa: PLW0603 -- process-wide snapshot, taken once
    if _BASE_HOSTS is None:
        current = os.environ.get(_EGRESS_ENV, get_settings().allowed_egress_hosts)
        _BASE_HOSTS = frozenset(h.strip() for h in current.split(",") if h.strip())
    return _BASE_HOSTS


def reset_base_hosts() -> None:
    """Re-take the snapshot. For tests; production takes it once at first use."""
    global _BASE_HOSTS  # noqa: PLW0603
    _BASE_HOSTS = None


def _apply_host_to_allowlist(host: str) -> None:
    """Append an approved host to the live egress allowlist (env + settings rebuild)."""
    _base_hosts()  # pin the deployment's own list BEFORE the first widening moves it
    current = os.environ.get(_EGRESS_ENV, get_settings().allowed_egress_hosts)
    hosts = [h.strip() for h in current.split(",") if h.strip()]
    if host not in hosts:
        hosts.append(host)
        os.environ[_EGRESS_ENV] = ",".join(hosts)
        get_settings.cache_clear()
        _log.info("egress_host_allowlisted", host=host)


def _remove_host_from_allowlist(host: str) -> None:
    """Remove a RUNTIME-APPROVED host from the live egress allowlist (revoke).

    A host the deployment itself configured is never removed -- see :data:`_BASE_HOSTS`. The
    request is still recorded as revoked, because the tenant's permission to use it is a separate
    question from whether the platform can reach its own model endpoints."""
    if host in _base_hosts():
        _log.warning("egress_revoke_ignored_base_host", host=host, detail="configured by the deployment")
        return
    current = os.environ.get(_EGRESS_ENV, get_settings().allowed_egress_hosts)
    hosts = [h.strip() for h in current.split(",") if h.strip() and h.strip() != host]
    os.environ[_EGRESS_ENV] = ",".join(hosts)
    get_settings.cache_clear()
    _log.info("egress_host_revoked", host=host)


# --- store (global table; in-memory twin for CI) -----------------------------
@runtime_checkable
class EgressRequestStore(Protocol):
    async def ensure(self) -> None: ...
    async def get(self, req_id: str) -> EgressRequest | None: ...
    async def put(self, req: EgressRequest) -> None: ...
    async def list(self, status: Status | None = None) -> list[EgressRequest]: ...
    async def find_active(self, host: str) -> EgressRequest | None: ...


class InMemoryEgressRequestStore:
    def __init__(self) -> None:
        self._rows: dict[str, EgressRequest] = {}

    async def ensure(self) -> None:
        return None

    async def get(self, req_id: str) -> EgressRequest | None:
        return self._rows.get(req_id)

    async def put(self, req: EgressRequest) -> None:
        self._rows[req.id] = req

    async def list(self, status: Status | None = None) -> list[EgressRequest]:
        rows = sorted(self._rows.values(), key=lambda r: r.created_at, reverse=True)
        return [r for r in rows if status is None or r.status == status]

    async def find_active(self, host: str) -> EgressRequest | None:
        for r in self._rows.values():
            if r.host == host and r.status in ("pending", "approved"):
                return r
        return None


_COLS = "id, host, tenant_id, kb_id, requested_by, reason, status, created_at, decided_by, decided_at"
_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".egress_requests ('
    "id TEXT PRIMARY KEY, host TEXT NOT NULL, tenant_id TEXT NOT NULL, kb_id TEXT, "
    "requested_by TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL, decided_by TEXT, decided_at TIMESTAMPTZ)",
)


class PostgresEgressRequestStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def get(self, req_id: str) -> EgressRequest | None:
        await self.ensure()
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".egress_requests WHERE id = :i'), {"i": req_id}
                    )
                )
                .mappings()
                .first()
            )
        return EgressRequest(**dict(row)) if row else None

    async def put(self, req: EgressRequest) -> None:
        await self.ensure()
        sql = (
            f'INSERT INTO "{_IDENTITY_SCHEMA}".egress_requests ({_COLS}) '
            "VALUES (:id, :host, :tenant_id, :kb_id, :requested_by, :reason, :status, "
            ":created_at, :decided_by, :decided_at) "
            "ON CONFLICT (id) DO UPDATE SET status = :status, decided_by = :decided_by, decided_at = :decided_at"
        )
        async with self._engine.begin() as c:
            await c.execute(text(sql), req.model_dump())

    async def list(self, status: Status | None = None) -> list[EgressRequest]:
        await self.ensure()
        where = "WHERE status = :s " if status else ""
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".egress_requests {where}ORDER BY created_at DESC'
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"s": status} if status else {})).mappings().all()
        return [EgressRequest(**dict(r)) for r in rows]

    async def find_active(self, host: str) -> EgressRequest | None:
        await self.ensure()
        sql = (
            f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".egress_requests '
            "WHERE host = :h AND status IN ('pending','approved') ORDER BY created_at DESC LIMIT 1"
        )
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"h": host})).mappings().first()
        return EgressRequest(**dict(row)) if row else None


# --- operations --------------------------------------------------------------
async def submit(
    store: EgressRequestStore, *, host_or_url: str, tenant_id: str, requested_by: str, kb_id: str | None, reason: str
) -> EgressRequest:
    """Create (or return the existing) request for a host. Already-approved → returned as-is."""
    host = normalize_host(host_or_url)
    existing = await store.find_active(host)
    if existing is not None:
        return existing
    req = EgressRequest(
        id=str(uuid.uuid4()),
        host=host,
        tenant_id=tenant_id,
        kb_id=kb_id,
        requested_by=requested_by,
        reason=reason.strip(),
        status="pending",
        created_at=datetime.now(UTC),
    )
    await store.put(req)
    await _audit(tenant_id, "requested", host, requested_by)
    _log.info("egress_requested", host=host, by=requested_by)
    return req


async def decide(store: EgressRequestStore, *, req_id: str, approve: bool, decided_by: str) -> EgressRequest:
    """Approve (allowlist the host) or reject a pending request. Owner/admin only (enforced at API)."""
    req = await store.get(req_id)
    if req is None:
        raise NotFoundError("egress request not found", details={"id": req_id})
    if req.status != "pending":
        return req  # idempotent
    if approve:
        _apply_host_to_allowlist(req.host)
    updated = req.model_copy(
        update={
            "status": "approved" if approve else "rejected",
            "decided_by": decided_by,
            "decided_at": datetime.now(UTC),
        }
    )
    await store.put(updated)
    await _audit(req.tenant_id, "approved" if approve else "rejected", req.host, decided_by)
    _log.info("egress_decided", host=req.host, approved=approve, by=decided_by)
    return updated


async def revoke(store: EgressRequestStore, *, req_id: str, decided_by: str) -> EgressRequest:
    """Revoke a previously-approved host — remove it from the allowlist (the guard blocks it
    again) and record the revocation. Owner/admin only (enforced at the API)."""
    req = await store.get(req_id)
    if req is None:
        raise NotFoundError("egress request not found", details={"id": req_id})
    if req.status != "approved":
        return req  # only an approved host can be revoked
    _remove_host_from_allowlist(req.host)
    updated = req.model_copy(update={"status": "revoked", "decided_by": decided_by, "decided_at": datetime.now(UTC)})
    await store.put(updated)
    await _audit(req.tenant_id, "revoked", req.host, decided_by)
    _log.info("egress_revoked", host=req.host, by=decided_by)
    return updated


async def load_approved(store: EgressRequestStore) -> None:
    """Re-apply every approved host to the live allowlist at startup."""
    approved = await store.list(status="approved")
    for req in approved:
        _apply_host_to_allowlist(req.host)
    if approved:
        _log.info("egress_approved_loaded", count=len(approved))


async def _audit(tenant_id: str, event: str, host: str, actor: str) -> None:
    from svc_control_plane import wiring  # noqa: PLC0415 — avoid import cycle

    try:
        await wiring.audit_service().record(
            tenant_id=tenant_id,
            trace_id=str(uuid.uuid4()),
            action=AuditAction.EGRESS,
            actor=actor,
            payload={"event": event, "host": host},
        )
    except Exception as exc:  # audit must never block the workflow
        _log.warning("egress_audit_failed", error=str(exc)[:160])
