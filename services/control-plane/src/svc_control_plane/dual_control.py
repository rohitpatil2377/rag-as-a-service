"""Maker-checker (dual-control / four-eyes) approval workflow — banking pack.

Sensitive actions do not execute immediately: the *maker* submits a request, and a DIFFERENT
admin (the *checker*) must approve before it runs. This is standard segregation-of-duties in a
bank — no single person can publish knowledge, ingest a classified document, destroy records, or
change a security setting alone. The maker can NEVER approve their own request (enforced here);
both parties must be owner/admin (enforced at the API).

Gated actions (operator-selectable via config):
* ``publish_kb``        — make a knowledge base answerable to end-users
* ``ingest_classified`` — add a Confidential/Restricted document
* ``purge`` / ``erasure`` — destroy documents/records (irreversible)
* ``security_change``   — change a feature flag / auth / egress / KMS setting

Every submit and decision is audited (``AuditAction.APPROVAL``). Default OFF: with
``dual_control_enabled`` false, nothing is gated and behaviour is unchanged. Mirrors the
egress-approval store (global table + in-memory twin). Platform-wide in v1.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings
from rag_common.errors import ForbiddenError, NotFoundError
from rag_schemas.common import AuditAction

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"

Action = Literal["publish_kb", "ingest_classified", "purge", "erasure", "security_change"]
Status = Literal["pending", "approved", "rejected", "executed", "failed"]
ALL_ACTIONS: tuple[Action, ...] = ("publish_kb", "ingest_classified", "purge", "erasure", "security_change")


class ApprovalRequest(BaseModel):
    id: str
    tenant_id: str
    kb_id: str | None = None
    action: Action
    params: dict[str, Any] = Field(default_factory=dict)  # enough to execute the action later
    summary: str = ""  # human-readable description for the reviewer
    requested_by: str
    reason: str = ""
    status: Status = "pending"
    created_at: datetime
    decided_by: str | None = None
    decided_at: datetime | None = None
    error: str | None = None


# --- executor registry -------------------------------------------------------
# An executor performs the action from its stored params and returns a short result string.
# Registered by the binding module (dual_control_actions) so this engine imports no services.
Executor = Callable[[ApprovalRequest], Awaitable[str]]
_EXECUTORS: dict[str, Executor] = {}


def register(action: Action, fn: Executor) -> None:
    _EXECUTORS[action] = fn


def is_gated(action: Action) -> bool:
    """True if this action currently requires dual-control approval (config-driven)."""
    s = get_settings()
    if not s.dual_control_enabled:
        return False
    chosen = {a.strip() for a in s.dual_control_actions.split(",") if a.strip()}
    return not chosen or action in chosen  # empty list = all actions gated


# --- store (global table; in-memory twin for CI) -----------------------------
@runtime_checkable
class ApprovalStore(Protocol):
    async def ensure(self) -> None: ...
    async def get(self, req_id: str) -> ApprovalRequest | None: ...
    async def put(self, req: ApprovalRequest) -> None: ...
    async def list(self, tenant_id: str, status: Status | None = None) -> list[ApprovalRequest]: ...


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._rows: dict[str, ApprovalRequest] = {}

    async def ensure(self) -> None:
        return None

    async def get(self, req_id: str) -> ApprovalRequest | None:
        return self._rows.get(req_id)

    async def put(self, req: ApprovalRequest) -> None:
        self._rows[req.id] = req

    async def list(self, tenant_id: str, status: Status | None = None) -> list[ApprovalRequest]:
        rows = sorted(self._rows.values(), key=lambda r: r.created_at, reverse=True)
        return [r for r in rows if r.tenant_id == tenant_id and (status is None or r.status == status)]


_COLS = "id, tenant_id, kb_id, action, params, summary, requested_by, reason, status, created_at, decided_by, decided_at, error"  # noqa: E501
_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".approval_requests ('
    "id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kb_id TEXT, action TEXT NOT NULL, "
    "params JSONB NOT NULL DEFAULT '{}', summary TEXT NOT NULL DEFAULT '', requested_by TEXT NOT NULL, "
    "reason TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, "
    "decided_by TEXT, decided_at TIMESTAMPTZ, error TEXT)",
)


def _row_to_req(row: dict[str, Any]) -> ApprovalRequest:
    data = dict(row)
    params = data.get("params")
    if isinstance(params, str):  # some drivers return JSONB as text
        data["params"] = json.loads(params or "{}")
    return ApprovalRequest(**data)


class PostgresApprovalStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def get(self, req_id: str) -> ApprovalRequest | None:
        await self.ensure()
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".approval_requests WHERE id = :i'), {"i": req_id}
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_req(dict(row)) if row else None

    async def put(self, req: ApprovalRequest) -> None:
        await self.ensure()
        sql = (
            f'INSERT INTO "{_IDENTITY_SCHEMA}".approval_requests ({_COLS}) '
            "VALUES (:id, :tenant_id, :kb_id, :action, CAST(:params AS JSONB), :summary, :requested_by, "
            ":reason, :status, :created_at, :decided_by, :decided_at, :error) "
            "ON CONFLICT (id) DO UPDATE SET status = :status, decided_by = :decided_by, "
            "decided_at = :decided_at, error = :error"
        )
        payload = req.model_dump()
        payload["params"] = json.dumps(payload["params"])
        async with self._engine.begin() as c:
            await c.execute(text(sql), payload)

    async def list(self, tenant_id: str, status: Status | None = None) -> list[ApprovalRequest]:
        await self.ensure()
        where = "WHERE tenant_id = :t" + (" AND status = :s" if status else "")
        params: dict[str, Any] = {"t": tenant_id} | ({"s": status} if status else {})
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".approval_requests {where} ORDER BY created_at DESC'
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), params)).mappings().all()
        return [_row_to_req(dict(r)) for r in rows]


# --- operations --------------------------------------------------------------
async def submit(
    store: ApprovalStore,
    *,
    action: Action,
    tenant_id: str,
    kb_id: str | None,
    params: dict[str, Any],
    summary: str,
    requested_by: str,
    reason: str = "",
) -> ApprovalRequest:
    """Stage a pending approval request for a gated action (the maker's half)."""
    req = ApprovalRequest(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        kb_id=kb_id,
        action=action,
        params=params,
        summary=summary,
        requested_by=requested_by,
        reason=reason.strip(),
        status="pending",
        created_at=datetime.now(UTC),
    )
    await store.put(req)
    await _audit(tenant_id, "requested", req, requested_by)
    _log.info("approval_requested", action=action, kb_id=kb_id, by=requested_by, req=req.id)
    return req


async def decide(store: ApprovalStore, *, req_id: str, approve: bool, decided_by: str) -> ApprovalRequest:
    """Approve (⇒ execute the action) or reject a pending request — the checker's half.

    Separation of duties: the maker can never approve their own request. Both parties must be
    owner/admin (enforced at the API)."""
    req = await store.get(req_id)
    if req is None:
        raise NotFoundError("approval request not found", details={"id": req_id})
    if req.status != "pending":
        return req  # idempotent
    if approve and decided_by == req.requested_by:
        raise ForbiddenError(
            "separation of duties: a request cannot be approved by the same person who made it",
            details={"id": req_id, "requested_by": req.requested_by},
        )
    if not approve:
        updated = req.model_copy(
            update={"status": "rejected", "decided_by": decided_by, "decided_at": datetime.now(UTC)}
        )
        await store.put(updated)
        await _audit(req.tenant_id, "rejected", req, decided_by)
        return updated

    # Approved → run the executor now; record executed or failed (never silently drop).
    executor = _EXECUTORS.get(req.action)
    now = datetime.now(UTC)
    if executor is None:
        updated = req.model_copy(
            update={
                "status": "failed",
                "decided_by": decided_by,
                "decided_at": now,
                "error": f"no executor registered for action '{req.action}'",
            }
        )
        await store.put(updated)
        await _audit(req.tenant_id, "failed", req, decided_by)
        return updated
    try:
        result = await executor(req)
        updated = req.model_copy(
            update={
                "status": "executed",
                "decided_by": decided_by,
                "decided_at": now,
                "error": None,
            }
        )
        _log.info("approval_executed", action=req.action, req=req.id, result=result[:120], by=decided_by)
    except Exception as exc:
        updated = req.model_copy(
            update={
                "status": "failed",
                "decided_by": decided_by,
                "decided_at": now,
                "error": str(exc)[:300],
            }
        )
        _log.warning("approval_execution_failed", action=req.action, req=req.id, error=str(exc)[:200])
    await store.put(updated)
    await _audit(req.tenant_id, updated.status, req, decided_by)
    return updated


async def _audit(tenant_id: str, event: str, req: ApprovalRequest, actor: str) -> None:
    from svc_control_plane import wiring  # noqa: PLC0415 — avoid import cycle

    try:
        await wiring.audit_service().record(
            tenant_id=tenant_id,
            trace_id=str(uuid.uuid4()),
            action=AuditAction.APPROVAL,
            actor=actor,
            payload={"event": event, "action": req.action, "kb_id": req.kb_id, "request_id": req.id},
        )
    except Exception as exc:
        _log.warning("approval_audit_failed", error=str(exc)[:160])
