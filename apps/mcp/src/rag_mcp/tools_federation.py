"""Built-in federation tool pack (registered through the plugin framework).

Exposes Federated RAG federation capabilities to agents over MCP that the console does not
yet surface as buttons — but which the REST API already implements. Each tool calls the
SAME control-plane functions the ``/v1`` routes do (``svc_control_plane.grants`` /
``svc_control_plane.provenance``), so an agent gets identical fail-closed semantics:

* ``rag.delegate_grant`` — transitive A->B->C re-delegation of a RECEIVED grant, bounded
  (the child may only NARROW the parent) and chain-depth capped; owner/admin only.
* ``rag.verify_receipt`` — verify a supplied trust receipt's signature (stateless).
* ``rag.get_receipt``   — pull the signed trust receipt for an answer from the audit chain.

Every tool runs under the same per-request tenancy as the core tools (``rag_mcp.context``),
so a credential can only delegate grants held by, and read receipts of, its OWN department.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from svc_control_plane import provenance, wiring
from svc_control_plane.grants import DelegationError, attenuate, sign_grant

from rag_common import ForbiddenError, NotFoundError, UnauthorizedError, ValidationError, get_logger, get_settings
from rag_common.trust_receipt import TrustReceipt
from rag_mcp.context import ctx_roles, ctx_tenant, current_tenant
from rag_mcp.registry import ToolRegistry
from rag_schemas import Grant
from rag_schemas.common import AuditAction

_log = get_logger(__name__)


def _require_federation() -> None:
    """Federation must be enabled, mirroring the REST ``_require_federation`` gate."""
    if not get_settings().federation_enabled:
        raise UnauthorizedError("federation is not enabled")


def _require_admin_or_owner() -> None:
    """Owner/admin gate for delegation. Mirrors ``rag_mcp.context.require_role``'s transport
    rule: an HTTP request is always bound to a context (so the role check is strict); the
    stdio transport is the operator-launched, full-trust binding and is exempt."""
    if ctx_tenant.get() is None:
        return  # stdio: operator-launched, full-trust binding — no per-request context
    roles = ctx_roles.get()
    if "owner" not in roles and "admin" not in roles:
        raise ForbiddenError("only an owner or admin can delegate federation grants")


async def delegate_grant_tool(
    parent_grant_id: str,
    subject_dept_id: str,
    topics: list[str],
    kb_ids: list[str],
    max_results_per_query: int = 5,
    ttl_days: int = 180,
) -> dict[str, Any]:
    """Re-delegate a grant this department RECEIVED (transitive A->B->C). The caller must be
    the parent grant's subject; the new grant may only NARROW it (``topics``/``kb_ids`` a
    subset, cap and validity no larger), and the chain may not exceed the configured max
    depth. Revoking any ancestor later kills this grant too. Owner/admin, federation on."""
    _require_federation()
    _require_admin_or_owner()
    me = current_tenant()
    store = wiring.grant_store()
    parent = await store.get(parent_grant_id)
    if parent is None:
        raise NotFoundError("grant not found", details={"grant_id": parent_grant_id})
    if parent.subject_dept_id != me:
        raise ForbiddenError("only the grant's subject can re-delegate it")
    if not parent.is_active:
        raise ValidationError("cannot delegate a revoked or expired grant")
    if subject_dept_id == me:
        raise ValidationError("subject_dept_id cannot be the delegator itself")

    max_depth = get_settings().federation_max_grant_depth
    parent_depth = len(await store.resolve_chain(parent_grant_id))
    if parent_depth + 1 > max_depth:
        raise ValidationError(
            f"delegation would exceed the max chain depth ({max_depth})",
            details={"parent_depth": parent_depth, "max_depth": max_depth},
        )

    # Whole-second precision: the Go responder's canonical-claims formatter drops fractional
    # seconds, so a microsecond-precision "now" would produce a signature it cannot reproduce.
    now = datetime.now(UTC).replace(microsecond=0)
    # Clamp the child window INSIDE the parent (inherits the parent ceiling).
    not_before = max(now, parent.not_before)
    expires_at = min(now + timedelta(days=ttl_days), parent.expires_at)
    child = Grant(
        grant_id=f"grant_{uuid.uuid4().hex[:16]}",
        issuer_dept_id=me,
        subject_dept_id=subject_dept_id,
        topics=topics,
        kb_ids=kb_ids,
        max_results_per_query=max_results_per_query,
        not_before=not_before,
        expires_at=expires_at,
        signed_token="",
        created_at=now,
        parent_grant_id=parent_grant_id,
    )
    try:
        attenuate(parent, child)  # bounded trust: the child may only narrow the parent
    except DelegationError as exc:
        raise ValidationError(str(exc)) from exc
    child = child.model_copy(update={"signed_token": sign_grant(child)})
    await store.create(child)
    await wiring.audit_service().record(
        tenant_id=me,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.GRANT_ISSUE,
        actor="mcp-agent",
        payload={
            "grant_id": child.grant_id,
            "subject_dept_id": child.subject_dept_id,
            "topics": child.topics,
            "kb_ids": child.kb_ids,
            "expires_at": child.expires_at.isoformat(),
            "parent_grant_id": parent_grant_id,
        },
    )
    return child.model_dump(mode="json")


async def verify_receipt_tool(receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify a supplied trust receipt against the provider public key — stateless and
    fail-closed. A malformed receipt is reported invalid, never raised."""
    try:
        parsed = TrustReceipt(**receipt)
    except Exception:
        return {"valid": False, "reason": "malformed_receipt"}
    return {"valid": provenance.verify_supplied_receipt(parsed)}


async def get_receipt_tool(answer_id: str) -> dict[str, Any]:
    """Pull the signed trust receipt for one answer (by its trace id) from the caller's audit
    chain. Errors if the answer produced no receipt (it abstained, or receipts were off)."""
    org = current_tenant()
    namespaces = {org}
    try:
        namespaces |= {k.kb_id for k in await wiring.kb_store().list(org)}
    except Exception as exc:
        # Widening to per-KB namespaces is an optimisation; the org namespace alone still finds
        # the receipt. Log rather than swallow, so a persistently failing store is visible.
        _log.debug("kb_list_failed_for_receipt_lookup", org=org, error=str(exc)[:120])
    for ns in namespaces:
        for ev in await wiring.audit_service().list(ns, trace_id=answer_id):
            receipt = ev.payload.get("receipt") if ev.action == AuditAction.RECEIPT else None
            if receipt:
                return TrustReceipt(**receipt).model_dump(mode="json")
    raise NotFoundError("no trust receipt for this answer", details={"answer_id": answer_id})


# (name, description, fn) — the federation pack surface.
_FEDERATION_TOOLS = (
    (
        "rag.delegate_grant",
        "Transitively re-delegate a received Federated RAG grant (bounded, owner/admin, federation on).",
        delegate_grant_tool,
    ),
    (
        "rag.verify_receipt",
        "Verify a supplied trust receipt's signature (stateless, fail-closed).",
        verify_receipt_tool,
    ),
    (
        "rag.get_receipt",
        "Pull the signed trust receipt for an answer (by trace id) from the audit chain.",
        get_receipt_tool,
    ),
)


def register(registry: ToolRegistry) -> None:
    """Framework entrypoint: contribute the federation tools to ``registry``."""
    for name, description, fn in _FEDERATION_TOOLS:
        registry.register(name, description, fn)
