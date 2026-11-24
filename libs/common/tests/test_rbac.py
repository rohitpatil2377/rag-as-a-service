"""Hierarchical RBAC spine — exhaustive tests for the one rule that must never break:
a lower level must not read higher-level data. Covers the tier/escalation matrix, the nested
org-unit scope, the clearance ceiling, provider SoD, and least-privilege defaults.
"""

from __future__ import annotations

from rag_common.rbac import (
    Clearance,
    Subject,
    Tier,
    ancestors,
    can_admin_unit,
    can_grant_clearance,
    can_grant_role,
    can_ingest_kb,
    can_read_kb,
    in_subtree,
    is_ancestor_or_self,
)

# A bank org tree:  company (root) → {branch, exec};  branch → teller_desk
_PARENTS = {"company": None, "branch": "company", "exec": "company", "teller_desk": "branch"}


def _subj(tier: Tier, clr: Clearance, unit: str) -> Subject:
    return Subject(tier=tier, clearance=clr, tenant_id="bank", unit_id=unit)


# --- tree math -----------------------------------------------------------------------
def test_ancestors_and_subtree() -> None:
    assert ancestors(_PARENTS, "teller_desk") == ["branch", "company"]
    assert ancestors(_PARENTS, "company") == []
    assert is_ancestor_or_self(_PARENTS, "company", "teller_desk")
    assert is_ancestor_or_self(_PARENTS, "teller_desk", "teller_desk")
    assert not is_ancestor_or_self(_PARENTS, "branch", "exec")  # siblings' subtrees
    assert in_subtree(_PARENTS, "branch", "teller_desk")
    assert not in_subtree(_PARENTS, "exec", "branch")


def test_ancestors_is_cycle_guarded() -> None:
    cyclic = {"a": "b", "b": "a"}
    assert ancestors(cyclic, "a") == ["b"]  # finite, not an infinite loop


# --- no escalation -------------------------------------------------------------------
def test_a_grantor_may_only_grant_a_tier_below_its_own() -> None:
    assert can_grant_role(Tier.TENANT_ADMIN, Tier.UNIT_ADMIN)
    assert can_grant_role(Tier.UNIT_ADMIN, Tier.MEMBER)
    # never a peer or a superior
    assert not can_grant_role(Tier.TENANT_ADMIN, Tier.TENANT_ADMIN)
    assert not can_grant_role(Tier.UNIT_ADMIN, Tier.TENANT_ADMIN)
    assert not can_grant_role(Tier.UNIT_ADMIN, Tier.UNIT_ADMIN)
    # non-admins grant nothing
    assert not can_grant_role(Tier.MEMBER, Tier.VIEWER)


def test_clearance_cannot_be_granted_above_the_grantor() -> None:
    assert can_grant_clearance(Tier.TENANT_ADMIN, Clearance.RESTRICTED, Clearance.CONFIDENTIAL)
    assert can_grant_clearance(Tier.UNIT_ADMIN, Clearance.CONFIDENTIAL, Clearance.CONFIDENTIAL)  # at own level ok
    assert not can_grant_clearance(Tier.TENANT_ADMIN, Clearance.CONFIDENTIAL, Clearance.RESTRICTED)  # above own
    assert not can_grant_clearance(Tier.MEMBER, Clearance.RESTRICTED, Clearance.PUBLIC)  # not an admin


# --- the core read rule: lower level never reads higher data -------------------------
def test_a_teller_cannot_read_executive_data() -> None:
    teller = _subj(Tier.MEMBER, Clearance.INTERNAL, "teller_desk")
    # exec KB: Restricted, in the exec unit — blocked on BOTH clearance and scope.
    assert not can_read_kb(teller, kb_classification=Clearance.RESTRICTED, kb_unit_id="exec", parents=_PARENTS)


def test_a_teller_reads_inherited_company_policy_but_not_a_sibling_branch() -> None:
    teller = _subj(Tier.MEMBER, Clearance.INTERNAL, "teller_desk")
    # company-wide Public policy at the ROOT flows down.
    assert can_read_kb(teller, kb_classification=Clearance.PUBLIC, kb_unit_id="company", parents=_PARENTS)
    # its own desk's Internal KB.
    assert can_read_kb(teller, kb_classification=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS)
    # a SIBLING branch's KB — never, even at Internal (no scope path).
    assert not can_read_kb(teller, kb_classification=Clearance.INTERNAL, kb_unit_id="exec", parents=_PARENTS)


def test_the_clearance_ceiling_is_checked_first_and_cannot_be_scoped_around() -> None:
    # Even for its OWN unit's KB, a below-clearance subject is refused (ceiling before scope).
    under = _subj(Tier.MEMBER, Clearance.INTERNAL, "teller_desk")
    assert not can_read_kb(under, kb_classification=Clearance.CONFIDENTIAL, kb_unit_id="teller_desk", parents=_PARENTS)


def test_a_unit_admin_reads_its_subtree_downward() -> None:
    branch_admin = _subj(Tier.UNIT_ADMIN, Clearance.CONFIDENTIAL, "branch")
    # its own subtree (teller_desk) — allowed (downward management scope).
    assert can_read_kb(
        branch_admin, kb_classification=Clearance.CONFIDENTIAL, kb_unit_id="teller_desk", parents=_PARENTS
    )
    # the sibling exec subtree — still no.
    assert not can_read_kb(branch_admin, kb_classification=Clearance.CONFIDENTIAL, kb_unit_id="exec", parents=_PARENTS)
    # and still bounded by clearance: a Restricted exec-subtree KB is refused regardless.
    assert not can_read_kb(branch_admin, kb_classification=Clearance.RESTRICTED, kb_unit_id="branch", parents=_PARENTS)


def test_tenant_admin_reads_the_whole_tenant_within_clearance() -> None:
    admin = _subj(Tier.TENANT_ADMIN, Clearance.RESTRICTED, "company")
    assert can_read_kb(admin, kb_classification=Clearance.RESTRICTED, kb_unit_id="exec", parents=_PARENTS)
    # but a Confidential-cleared tenant admin still cannot read Restricted (ceiling holds for all).
    mid = _subj(Tier.TENANT_ADMIN, Clearance.CONFIDENTIAL, "company")
    assert not can_read_kb(mid, kb_classification=Clearance.RESTRICTED, kb_unit_id="exec", parents=_PARENTS)


def test_super_admin_never_reads_tenant_data_separation_of_duties() -> None:
    god = _subj(Tier.SUPER_ADMIN, Clearance.RESTRICTED, "company")
    # highest tier + highest clearance, yet blocked from ALL tenant content.
    assert not can_read_kb(god, kb_classification=Clearance.PUBLIC, kb_unit_id="company", parents=_PARENTS)


def test_default_subject_sees_only_root_public_data() -> None:
    fresh = _subj(Tier.VIEWER, Clearance.PUBLIC, "company")
    assert can_read_kb(fresh, kb_classification=Clearance.PUBLIC, kb_unit_id="company", parents=_PARENTS)
    assert not can_read_kb(fresh, kb_classification=Clearance.INTERNAL, kb_unit_id="company", parents=_PARENTS)
    # a non-root unit is not in a root viewer scope, even at Public.
    assert not can_read_kb(fresh, kb_classification=Clearance.PUBLIC, kb_unit_id="branch", parents=_PARENTS)


# --- admin + ingest ------------------------------------------------------------------
def test_unit_admin_administers_only_its_subtree() -> None:
    branch_admin = _subj(Tier.UNIT_ADMIN, Clearance.CONFIDENTIAL, "branch")
    assert can_admin_unit(branch_admin, unit_id="teller_desk", parents=_PARENTS)
    assert not can_admin_unit(branch_admin, unit_id="exec", parents=_PARENTS)
    assert not can_admin_unit(branch_admin, unit_id="company", parents=_PARENTS)  # cannot admin upward
    tadmin = _subj(Tier.TENANT_ADMIN, Clearance.RESTRICTED, "company")
    assert can_admin_unit(tadmin, unit_id="exec", parents=_PARENTS)


def test_a_viewer_cannot_ingest_even_where_it_can_read() -> None:
    viewer = _subj(Tier.VIEWER, Clearance.INTERNAL, "teller_desk")
    assert can_read_kb(viewer, kb_classification=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS)
    assert not can_ingest_kb(viewer, kb_classification=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS)
    member = _subj(Tier.MEMBER, Clearance.INTERNAL, "teller_desk")
    assert can_ingest_kb(member, kb_classification=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS)


# --- the ingestion-level rule --------------------------------------------------------
def test_ingestion_level_is_bounded_by_kb_floor_and_ingestor_clearance() -> None:
    from rag_common.rbac import can_ingest_at

    # A Confidential-cleared member ingesting into an Internal-floor KB in scope.
    m = _subj(Tier.MEMBER, Clearance.CONFIDENTIAL, "teller_desk")

    def ok(req: Clearance) -> bool:
        return can_ingest_at(m, kb_floor=Clearance.INTERNAL, requested=req, kb_unit_id="teller_desk", parents=_PARENTS)

    assert ok(Clearance.INTERNAL)  # at the floor
    assert ok(Clearance.CONFIDENTIAL)  # up to own clearance
    assert not ok(Clearance.PUBLIC)  # below the KB floor
    assert not ok(Clearance.RESTRICTED)  # above own clearance (exfil guard)


def test_a_viewer_or_out_of_scope_user_cannot_ingest_at_any_level() -> None:
    from rag_common.rbac import can_ingest_at

    viewer = _subj(Tier.VIEWER, Clearance.RESTRICTED, "teller_desk")
    assert not can_ingest_at(  # viewers never write, even where cleared/in-scope
        viewer, kb_floor=Clearance.INTERNAL, requested=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS
    )
    outsider = _subj(Tier.MEMBER, Clearance.RESTRICTED, "exec")
    assert not can_ingest_at(  # not in the KB scope
        outsider, kb_floor=Clearance.INTERNAL, requested=Clearance.INTERNAL, kb_unit_id="teller_desk", parents=_PARENTS
    )
