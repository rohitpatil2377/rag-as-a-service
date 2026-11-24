"""Org-unit tree store + role->tier mapping (Sub-step A foundation)."""

from __future__ import annotations

import pytest
from svc_control_plane.org_units import (
    InMemoryOrgUnitStore,
    OrgUnit,
    parents_map,
    tier_for_role,
)

from rag_common.errors import ValidationError
from rag_common.rbac import Tier


def test_role_maps_to_management_tier() -> None:
    assert tier_for_role("owner") is Tier.TENANT_ADMIN
    assert tier_for_role("admin") is Tier.TENANT_ADMIN
    assert tier_for_role("member") is Tier.MEMBER
    assert tier_for_role("viewer") is Tier.VIEWER
    assert tier_for_role("unknown") is Tier.VIEWER  # unknown -> least privilege
    assert tier_for_role("owner", is_platform_admin=True) is Tier.SUPER_ADMIN
    assert tier_for_role("admin", unit_admin=True) is Tier.UNIT_ADMIN  # scoped admin -> unit tier
    assert tier_for_role("member", unit_admin=True) is Tier.MEMBER  # unit_admin only demotes an admin


async def test_ensure_root_is_idempotent_and_parentless() -> None:
    store = InMemoryOrgUnitStore()
    root = await store.ensure_root("bank")
    assert root.unit_id == "bank" and root.parent_id is None
    again = await store.ensure_root("bank")
    assert again == root


async def test_build_a_tree_and_derive_parents_map() -> None:
    store = InMemoryOrgUnitStore()
    await store.ensure_root("bank")
    await store.create(OrgUnit(unit_id="branch", tenant_id="bank", name="Branch", parent_id="bank"))
    await store.create(OrgUnit(unit_id="teller", tenant_id="bank", name="Teller", parent_id="branch"))

    pm = await parents_map(store, "bank")
    assert pm == {"bank": None, "branch": "bank", "teller": "branch"}


async def test_parents_map_synthesises_root_for_a_bare_tenant() -> None:
    # A tenant that never created units still yields a valid single-root map.
    store = InMemoryOrgUnitStore()
    assert await parents_map(store, "solo") == {"solo": None}


async def test_a_child_needs_an_existing_parent() -> None:
    store = InMemoryOrgUnitStore()
    await store.ensure_root("bank")
    with pytest.raises(ValidationError):
        await store.create(OrgUnit(unit_id="x", tenant_id="bank", name="X", parent_id="ghost"))


async def test_root_and_non_empty_units_cannot_be_deleted() -> None:
    store = InMemoryOrgUnitStore()
    await store.ensure_root("bank")
    await store.create(OrgUnit(unit_id="branch", tenant_id="bank", name="Branch", parent_id="bank"))
    await store.create(OrgUnit(unit_id="teller", tenant_id="bank", name="Teller", parent_id="branch"))
    with pytest.raises(ValidationError):
        await store.delete("bank", "bank")  # root
    with pytest.raises(ValidationError):
        await store.delete("bank", "branch")  # has a child
    await store.delete("bank", "teller")  # leaf -> ok
    assert await store.get("bank", "teller") is None
