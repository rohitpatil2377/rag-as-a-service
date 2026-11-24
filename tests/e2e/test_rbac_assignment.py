"""E2E: user assignment API — set a member's clearance + unit, escalation-gated (wiring D2)."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _client(tenant: str, roles: list[str], *, clearance: str | int = 0) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("officer", tenant, roles=roles, attrs={"clearance": clearance})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _seed(tenant: str) -> None:
    from svc_control_plane import wiring
    from svc_control_plane.identity import Org, User
    from svc_control_plane.org_units import OrgUnit

    await wiring.identity_store().create_org(Org(org_id=tenant, name="Bank"))
    await wiring.identity_store().create_user(
        User(email="m@x", org_id=tenant, name="M", role="member", roles=["reader"])
    )
    await wiring.org_unit_store().ensure_root(tenant)
    await wiring.org_unit_store().create(OrgUnit(unit_id="branch", tenant_id=tenant, name="Branch", parent_id=tenant))


async def test_admin_assigns_clearance_and_unit(inmemory: None) -> None:
    from svc_control_plane import wiring

    tenant = "bank"
    await _seed(tenant)
    hi = _client(tenant, ["admin"], clearance="restricted")  # cleared to grant up to Restricted
    async with hi:
        r = await hi.patch(
            "/v1/org/members/m@x",
            json={"roles": ["reader"], "kb_access": [], "clearance": "confidential", "unit_id": "branch"},
        )
        assert r.status_code == 200, r.text
    u = await wiring.identity_store().get_user("m@x")
    assert u is not None and u.clearance == 2 and u.unit_id == "branch"


async def test_cannot_grant_clearance_above_your_own(inmemory: None) -> None:
    tenant = "bank"
    await _seed(tenant)
    mid = _client(tenant, ["admin"], clearance="confidential")  # only Confidential-cleared
    async with mid:
        r = await mid.patch(
            "/v1/org/members/m@x",
            json={"roles": ["reader"], "kb_access": [], "clearance": "restricted"},  # above the grantor
        )
        assert r.status_code == 403, r.text


async def test_assigning_into_an_unknown_unit_is_404(inmemory: None) -> None:
    tenant = "bank"
    await _seed(tenant)
    hi = _client(tenant, ["admin"], clearance="restricted")
    async with hi:
        r = await hi.patch(
            "/v1/org/members/m@x",
            json={"roles": ["reader"], "kb_access": [], "clearance": "internal", "unit_id": "ghost"},
        )
        assert r.status_code == 404, r.text
