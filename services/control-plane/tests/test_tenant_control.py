"""Tenant-provisioning control plane — lifecycle + registry (T3a)."""

from __future__ import annotations

import pytest

from rag_common.errors import ValidationError
from rag_common.rbac import Clearance


async def test_provision_creates_org_owner_kb_and_record(inmemory: None) -> None:
    from svc_control_plane import tenant_control, wiring

    rec = await tenant_control.provision(
        org_id="acme", name="Acme", admin_email="admin@acme.io", password="pw", plan="enterprise", actor="ops"
    )
    assert rec.org_id == "acme" and rec.status is tenant_control.TenantStatus.ACTIVE
    assert rec.plan == "enterprise" and rec.max_documents_per_kb == 100000

    # owner user exists at the top clearance (can grant anything within the tenant).
    owner = await wiring.identity_store().get_user("admin@acme.io")
    assert owner is not None and owner.role == "owner" and owner.clearance == int(Clearance.RESTRICTED)
    # the RBAC root unit was ensured.
    assert await wiring.org_unit_store().get("acme", "acme") is not None
    # listed in the fleet.
    assert any(t.org_id == "acme" for t in await wiring.tenant_registry().list_all())


async def test_provision_rejects_bad_id_and_duplicates(inmemory: None) -> None:
    from svc_control_plane import tenant_control

    with pytest.raises(ValidationError):
        await tenant_control.provision(org_id="Bad Id!", name="x", admin_email="a@b", password="p", actor="ops")
    await tenant_control.provision(org_id="dup", name="Dup", admin_email="a@dup", password="p", actor="ops")
    with pytest.raises(ValidationError):
        await tenant_control.provision(org_id="dup", name="Dup2", admin_email="b@dup", password="p", actor="ops")


async def test_suspend_blocks_then_resume_restores(inmemory: None) -> None:
    from svc_control_plane import tenant_control

    await tenant_control.provision(org_id="hold", name="Hold", admin_email="a@hold", password="p", actor="ops")
    await tenant_control.assert_active("hold")  # active -> ok

    await tenant_control.set_status("hold", tenant_control.TenantStatus.SUSPENDED, actor="ops")
    with pytest.raises(ValidationError):
        await tenant_control.assert_active("hold")

    await tenant_control.set_status("hold", tenant_control.TenantStatus.ACTIVE, actor="ops")
    await tenant_control.assert_active("hold")  # resumed


async def test_absent_tenant_is_treated_active(inmemory: None) -> None:
    # A self-serve org not in the registry must not be blocked (additive).
    from svc_control_plane import tenant_control

    await tenant_control.assert_active("self-serve-org")  # no raise


async def test_set_plan_and_deprovision(inmemory: None) -> None:
    from svc_control_plane import tenant_control, wiring

    await tenant_control.provision(org_id="corp", name="Corp", admin_email="a@corp", password="p", actor="ops")
    rec = await tenant_control.set_plan("corp", "trial", actor="ops")
    assert rec.plan == "trial" and rec.max_documents_per_kb == 100

    out = await tenant_control.deprovision("corp", actor="ops")
    assert "knowledge_bases" in out
    got = await wiring.tenant_registry().get("corp")
    assert got is not None and got.status is tenant_control.TenantStatus.DEPROVISIONED
