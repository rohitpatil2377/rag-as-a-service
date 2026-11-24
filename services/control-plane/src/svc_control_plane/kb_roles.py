"""Per-KB custom named roles — a KB admin's own labels ('associate', 'senior associate')
layered ON TOP of the existing RBAC spine (``rag_common.rbac``), never a parallel engine.

A KB role is a thin, per-KB convenience: it carries a data-access **clearance** (the same
0..3 ladder :mod:`rag_common.rbac` enforces) plus two coarse permission flags. Assigning a
member such a role can only **ELEVATE** their effective clearance within THIS KB — it can
never restrict it, and it can never mint management tier. So a KB with no roles, or a caller
with no assignment, behaves EXACTLY as today (zero regression): the additive-only invariant.

Everything lives in ``tenant_config`` as JSON strings (keys ``kb_roles`` /
``kb_role_assignments``) — no migration, mirroring the taxonomy/persona storage pattern.

This module is pure model + tenant_config load/save + the two decision helpers
(:func:`effective_kb_clearance`, the escalation checks). It reuses the spine's
:func:`rag_common.rbac.can_grant_clearance` for the define/assign escalation gate — it does
NOT re-implement any access logic.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from rag_common.auth import Principal
from rag_common.rbac import Clearance, Subject, Tier, can_grant_clearance
from rag_core.interfaces import CatalogAdapter
from svc_control_plane.rbac_context import subject_of

# tenant_config keys (JSON string values, no migration — same pattern as taxonomy/persona).
ROLES_KEY = "kb_roles"
ASSIGNMENTS_KEY = "kb_role_assignments"
# Cross-KB access grants (approved access-requests). Stored in THIS KB's tenant_config as
# {email: clearance} — the LIVE source of truth for both "can this caller reach the KB" and their
# effective clearance in it, so grant/revoke take effect with no re-login. See api.py access-requests.
ACCESS_GRANTS_KEY = "access_grants"

_CLEARANCE_MIN = int(Clearance.PUBLIC)
_CLEARANCE_MAX = int(Clearance.RESTRICTED)


class KbRole(BaseModel):
    """A KB-admin-defined role. A label like 'associate' maps to a data-access ceiling plus a
    couple of coarse permission flags. It NEVER grants management tier — it only raises a
    member's clearance ceiling within this KB (and gates ingest via ``can_ingest``)."""

    clearance: int = Field(default=_CLEARANCE_MIN, ge=_CLEARANCE_MIN, le=_CLEARANCE_MAX)
    can_ingest: bool = False
    can_review: bool = False
    description: str = Field(default="", max_length=280)


class KbRoles(BaseModel):
    """The KB's defined roles (``{role_name: KbRole}``) — empty when none are configured."""

    roles: dict[str, KbRole] = Field(default_factory=dict)


class KbRoleAssignments(BaseModel):
    """Who holds which role (``{subject_email: role_name}``) — empty when none assigned."""

    assignments: dict[str, str] = Field(default_factory=dict)


# --- tenant_config load / save (pure I/O over the CatalogAdapter seam) ----------------


async def load_roles(catalog: CatalogAdapter, kb_id: str) -> KbRoles:
    """The KB's defined roles (empty if unset or malformed)."""
    raw = (await catalog.get_tenant_config(kb_id)).get(ROLES_KEY)
    if not raw:
        return KbRoles()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return KbRoles()
    if not isinstance(data, dict):
        return KbRoles()
    roles: dict[str, KbRole] = {}
    for name, spec in data.items():
        if isinstance(spec, dict):
            roles[str(name)] = KbRole.model_validate(spec)
    return KbRoles(roles=roles)


async def save_roles(catalog: CatalogAdapter, kb_id: str, roles: KbRoles) -> None:
    """Persist the defined roles as a JSON string in tenant_config."""
    payload = {name: role.model_dump() for name, role in roles.roles.items()}
    await catalog.set_tenant_config(kb_id, ROLES_KEY, json.dumps(payload))


async def load_assignments(catalog: CatalogAdapter, kb_id: str) -> KbRoleAssignments:
    """The KB's current role assignments (empty if unset or malformed)."""
    raw = (await catalog.get_tenant_config(kb_id)).get(ASSIGNMENTS_KEY)
    if not raw:
        return KbRoleAssignments()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return KbRoleAssignments()
    if not isinstance(data, dict):
        return KbRoleAssignments()
    return KbRoleAssignments(assignments={str(k): str(v) for k, v in data.items()})


async def save_assignments(catalog: CatalogAdapter, kb_id: str, assignments: KbRoleAssignments) -> None:
    """Persist the assignments as a JSON string in tenant_config."""
    await catalog.set_tenant_config(kb_id, ASSIGNMENTS_KEY, json.dumps(assignments.assignments))


# --- the additive-only clearance elevation (the safety invariant) ---------------------


def _assigned_role(roles: KbRoles, assignments: KbRoleAssignments, subject_email: str) -> KbRole | None:
    """The caller's assigned KB role, if any AND still defined. A dangling assignment (role
    since deleted) resolves to None — fail-closed, no elevation."""
    name = assignments.assignments.get(subject_email)
    if name is None:
        return None
    return roles.roles.get(name)


def effective_clearance(base_clearance: int, roles: KbRoles, assignments: KbRoleAssignments, subject_email: str) -> int:
    """The caller's effective clearance for this KB = ``max(base, assigned role clearance)``.

    ELEVATE-only: it can never fall below ``base_clearance``, so a KB with no roles or a caller
    with no assignment returns ``base_clearance`` unchanged (zero regression). Pure — takes the
    already-loaded roles/assignments so it stays trivially testable."""
    role = _assigned_role(roles, assignments, subject_email)
    if role is None:
        return base_clearance
    return max(base_clearance, role.clearance)


async def effective_kb_clearance(catalog: CatalogAdapter, kb_id: str, principal: Principal) -> int:
    """The caller's effective data-clearance for ``kb_id``: the max of their base JWT/attrs
    clearance and the clearance of their assigned KB role (if any). ADDITIVE-ONLY — it can only
    raise the ceiling, never lower it, so with no roles/assignment it equals the base clearance
    and every existing read/ingest decision is unchanged.

    Also honours ``can_ingest``: :func:`role_allows_ingest` exposes the ingest flag separately so
    the ingest path can additionally gate on it."""
    base = int(subject_of(principal).clearance)
    roles = await load_roles(catalog, kb_id)
    assignments = await load_assignments(catalog, kb_id)
    clr = effective_clearance(base, roles, assignments, principal.subject)
    # A cross-KB access grant also raises the ceiling (elevate-only, same as roles).
    grant = (await load_access_grants(catalog, kb_id)).get(principal.subject.strip().lower())
    return max(clr, grant) if grant is not None else clr


async def load_access_grants(catalog: CatalogAdapter, kb_id: str) -> dict[str, int]:
    """Approved cross-KB access grants for this KB: ``{email: clearance}``. Empty when none."""
    cfg = await catalog.get_tenant_config(kb_id)
    raw = cfg.get(ACCESS_GRANTS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    out: dict[str, int] = {}
    for k, v in data.items() if isinstance(data, dict) else []:
        try:
            out[str(k).strip().lower()] = max(_CLEARANCE_MIN, min(_CLEARANCE_MAX, int(v)))
        except (ValueError, TypeError):
            continue
    return out


async def elevated_subject(catalog: CatalogAdapter, kb_id: str, principal: Principal) -> Subject:
    """The caller's access :class:`Subject` with its clearance raised to the effective KB clearance
    (base or assigned-role, whichever is higher). Every other axis (tier, tenant, unit) is
    UNCHANGED — a KB role never touches management tier or scope. Use this at the ``can_read_kb`` /
    ``can_ingest_at`` call sites so the additive elevation flows through the spine unchanged. With
    no roles/assignment the clearance equals the base, so the Subject is identical to today."""
    base = subject_of(principal)
    roles = await load_roles(catalog, kb_id)
    assignments = await load_assignments(catalog, kb_id)
    eff = effective_clearance(int(base.clearance), roles, assignments, principal.subject)
    if eff == int(base.clearance):
        return base
    return Subject(
        tier=base.tier,
        clearance=Clearance(eff),
        tenant_id=base.tenant_id,
        unit_id=base.unit_id,
        attrs=base.attrs,
    )


async def role_allows_ingest(catalog: CatalogAdapter, kb_id: str, principal: Principal) -> bool:
    """Does the caller's assigned KB role (if any) permit ingest? A caller with NO assignment
    returns True (no additional constraint — the base RBAC ingest gate stands alone). A caller
    WITH a role must have ``can_ingest`` on that role. Additive: absence of roles never blocks."""
    roles = await load_roles(catalog, kb_id)
    assignments = await load_assignments(catalog, kb_id)
    role = _assigned_role(roles, assignments, principal.subject)
    return True if role is None else role.can_ingest


# --- escalation gate: define + assign (reuses the spine's can_grant_clearance) ---------


class RoleEscalationError(Exception):
    """A KB admin tried to define/assign a role above their OWN effective clearance, or to act
    without admin authority over this KB. Mapped to 403 at the API edge."""


class UnknownRoleError(Exception):
    """An assignment referenced a role name that is not defined on this KB. → 422 at the edge."""


def _is_kb_admin(subject: Subject) -> bool:
    """Only an admin/owner of THIS KB (tenant_guard already proved ownership) may define roles
    or assign members — tier at least UNIT_ADMIN. A member/viewer manages nothing."""
    return subject.tier >= Tier.UNIT_ADMIN


def assert_can_define(caller: Principal, roles: KbRoles) -> None:
    """Escalation gate at DEFINE time. The caller must administer this KB, and may not define a
    role whose clearance exceeds their OWN clearance — reusing the spine's
    :func:`can_grant_clearance` so 'you cannot hand out access you don't hold' is enforced in
    exactly one place. Raises :class:`RoleEscalationError` (→ 403) otherwise."""
    subject = subject_of(caller)
    if not _is_kb_admin(subject):
        raise RoleEscalationError("only a KB admin/owner may define roles")
    for name, role in roles.roles.items():
        if not can_grant_clearance(subject.tier, subject.clearance, Clearance(role.clearance)):
            raise RoleEscalationError(
                f"role '{name}' requests clearance {role.clearance} above your own {int(subject.clearance)}"
            )


def assert_can_assign(caller: Principal, roles: KbRoles, assignments: KbRoleAssignments) -> None:
    """Escalation gate at ASSIGN time. The caller must administer this KB; every referenced role
    must be defined (:class:`UnknownRoleError` → 422); and the caller may not assign a role above
    their own clearance (:class:`RoleEscalationError` → 403) — same anti-escalation rule as
    define, so a low-clearance admin cannot elevate someone past themselves."""
    subject = subject_of(caller)
    if not _is_kb_admin(subject):
        raise RoleEscalationError("only a KB admin/owner may assign roles")
    for member, role_name in assignments.assignments.items():
        role = roles.roles.get(role_name)
        if role is None:
            raise UnknownRoleError(f"role '{role_name}' assigned to '{member}' is not defined on this KB")
        if not can_grant_clearance(subject.tier, subject.clearance, Clearance(role.clearance)):
            raise RoleEscalationError(
                f"cannot assign '{role_name}' (clearance {role.clearance}) above your own {int(subject.clearance)}"
            )
