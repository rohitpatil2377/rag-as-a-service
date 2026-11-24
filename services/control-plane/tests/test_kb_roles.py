"""Per-KB custom named roles (extends the RBAC spine, additive-only).

Covers the safety invariants: a KB-admin defines roles only up to their OWN clearance
(escalation-gated); an assigned member's effective clearance ELEVATES; an UNassigned member
is UNCHANGED (regression-safe); the can_ingest flag is respected; an unknown role name on
assign is refused.
"""

from __future__ import annotations

import pytest
from svc_control_plane import kb_roles
from svc_control_plane.kb_roles import (
    KbRole,
    KbRoleAssignments,
    KbRoles,
    RoleEscalationError,
    UnknownRoleError,
)

from rag_clients.memory import InMemoryCatalog
from rag_common.auth import Principal


def _admin(clearance: int = 3, *, subject: str = "admin@x", tenant: str = "acme") -> Principal:
    """A tenant admin (TENANT_ADMIN tier) with the given data-clearance."""
    return Principal(
        subject=subject,
        tenant_id=tenant,
        roles=frozenset({"owner", "admin", "reader", "ingestor"}),
        attrs={"clearance": clearance},
    )


def _member(clearance: int = 0, *, subject: str = "u@x", tenant: str = "acme") -> Principal:
    return Principal(
        subject=subject,
        tenant_id=tenant,
        roles=frozenset({"reader", "ingestor"}),
        attrs={"clearance": clearance},
    )


@pytest.fixture
def catalog() -> InMemoryCatalog:
    return InMemoryCatalog()


# --- define: escalation-gated -----------------------------------------------


async def test_admin_defines_roles_within_own_clearance(catalog: InMemoryCatalog) -> None:
    admin = _admin(clearance=2)  # CONFIDENTIAL
    roles = KbRoles(roles={"associate": KbRole(clearance=1, can_ingest=True)})
    kb_roles.assert_can_define(admin, roles)  # no raise
    await kb_roles.save_roles(catalog, "acme", roles)
    assert (await kb_roles.load_roles(catalog, "acme")).roles["associate"].clearance == 1


async def test_low_clearance_admin_cannot_define_restricted_role(catalog: InMemoryCatalog) -> None:
    # A MEMBER-clearance (PUBLIC/0) admin may NOT define a RESTRICTED(3) role -> escalation.
    admin = _admin(clearance=0)
    roles = KbRoles(roles={"senior associate": KbRole(clearance=3)})
    with pytest.raises(RoleEscalationError):
        kb_roles.assert_can_define(admin, roles)


async def test_non_admin_cannot_define(catalog: InMemoryCatalog) -> None:
    member = _member(clearance=3)  # high clearance but no admin tier
    with pytest.raises(RoleEscalationError):
        kb_roles.assert_can_define(member, KbRoles(roles={"associate": KbRole(clearance=1)}))


# --- assign + effective clearance -------------------------------------------


async def test_effective_clearance_elevates_assigned_member(catalog: InMemoryCatalog) -> None:
    kb = "acme"
    await kb_roles.save_roles(catalog, kb, KbRoles(roles={"associate": KbRole(clearance=2)}))
    await kb_roles.save_assignments(catalog, kb, KbRoleAssignments(assignments={"u@x": "associate"}))
    member = _member(clearance=0)  # base PUBLIC
    eff = await kb_roles.effective_kb_clearance(catalog, kb, member)
    assert eff == 2  # elevated to the role's CONFIDENTIAL ceiling


async def test_effective_clearance_unchanged_for_unassigned_member(catalog: InMemoryCatalog) -> None:
    kb = "acme"
    await kb_roles.save_roles(catalog, kb, KbRoles(roles={"associate": KbRole(clearance=3)}))
    await kb_roles.save_assignments(catalog, kb, KbRoleAssignments(assignments={"other@x": "associate"}))
    member = _member(clearance=1, subject="u@x")  # not assigned
    eff = await kb_roles.effective_kb_clearance(catalog, kb, member)
    assert eff == 1  # UNCHANGED — regression-safe


async def test_no_roles_configured_is_identity(catalog: InMemoryCatalog) -> None:
    # A KB with no roles at all behaves exactly as today.
    member = _member(clearance=2)
    assert await kb_roles.effective_kb_clearance(catalog, "acme", member) == 2


async def test_role_never_lowers_clearance(catalog: InMemoryCatalog) -> None:
    kb = "acme"
    await kb_roles.save_roles(catalog, kb, KbRoles(roles={"associate": KbRole(clearance=0)}))
    await kb_roles.save_assignments(catalog, kb, KbRoleAssignments(assignments={"u@x": "associate"}))
    member = _member(clearance=3)  # base higher than the role
    assert await kb_roles.effective_kb_clearance(catalog, kb, member) == 3  # max, never lowered


async def test_dangling_assignment_does_not_elevate(catalog: InMemoryCatalog) -> None:
    # Assignment references a role that is no longer defined -> no elevation (fail-closed).
    kb = "acme"
    await kb_roles.save_roles(catalog, kb, KbRoles(roles={}))
    await kb_roles.save_assignments(catalog, kb, KbRoleAssignments(assignments={"u@x": "ghost"}))
    assert await kb_roles.effective_kb_clearance(catalog, kb, _member(clearance=0)) == 0


# --- assign: escalation-gated + unknown role --------------------------------


async def test_admin_cannot_assign_role_above_own_clearance(catalog: InMemoryCatalog) -> None:
    roles = KbRoles(roles={"senior": KbRole(clearance=3)})
    admin = _admin(clearance=1)  # only INTERNAL
    with pytest.raises(RoleEscalationError):
        kb_roles.assert_can_assign(admin, roles, KbRoleAssignments(assignments={"u@x": "senior"}))


async def test_assign_unknown_role_is_rejected(catalog: InMemoryCatalog) -> None:
    roles = KbRoles(roles={"associate": KbRole(clearance=1)})
    admin = _admin(clearance=3)
    with pytest.raises(UnknownRoleError):
        kb_roles.assert_can_assign(admin, roles, KbRoleAssignments(assignments={"u@x": "nope"}))


async def test_admin_assigns_within_clearance(catalog: InMemoryCatalog) -> None:
    roles = KbRoles(roles={"associate": KbRole(clearance=2)})
    admin = _admin(clearance=2)
    kb_roles.assert_can_assign(admin, roles, KbRoleAssignments(assignments={"u@x": "associate"}))  # no raise


# --- can_ingest flag respected ----------------------------------------------


async def test_can_ingest_flag_respected(catalog: InMemoryCatalog) -> None:
    kb = "acme"
    await kb_roles.save_roles(
        catalog,
        kb,
        KbRoles(
            roles={
                "reader_role": KbRole(clearance=2, can_ingest=False),
                "writer_role": KbRole(clearance=2, can_ingest=True),
            }
        ),
    )
    await kb_roles.save_assignments(
        catalog, kb, KbRoleAssignments(assignments={"reader@x": "reader_role", "writer@x": "writer_role"})
    )
    reader = _member(clearance=0, subject="reader@x")
    writer = _member(clearance=0, subject="writer@x")
    unassigned = _member(clearance=0, subject="none@x")
    assert await kb_roles.role_allows_ingest(catalog, kb, reader) is False
    assert await kb_roles.role_allows_ingest(catalog, kb, writer) is True
    assert await kb_roles.role_allows_ingest(catalog, kb, unassigned) is True  # additive: no role = no extra gate


async def test_elevated_subject_only_raises_clearance(catalog: InMemoryCatalog) -> None:
    kb = "acme"
    await kb_roles.save_roles(catalog, kb, KbRoles(roles={"associate": KbRole(clearance=3)}))
    await kb_roles.save_assignments(catalog, kb, KbRoleAssignments(assignments={"u@x": "associate"}))
    member = _member(clearance=0, subject="u@x")
    subj = await kb_roles.elevated_subject(catalog, kb, member)
    assert int(subj.clearance) == 3
    # tier + scope untouched — a KB role never grants management authority.
    from svc_control_plane.rbac_context import subject_of

    from rag_common.rbac import Tier

    base = subject_of(member)
    assert subj.tier is base.tier is Tier.MEMBER
    assert subj.unit_id == base.unit_id
