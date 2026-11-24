"""E2E: org-unit management API — build the tree, escalation-gated (wiring D1)."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _client(tenant: str, roles: list[str], *, unit: str | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    attrs: dict[str, object] = {}
    if unit is not None:
        attrs["unit"] = unit
    token = issue_dev_jwt("officer", tenant, roles=roles, attrs=attrs)
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_tenant_admin_builds_the_tree(inmemory: None) -> None:
    admin = _client("bank", ["owner", "admin"])
    async with admin:
        r = await admin.post("/v1/units", json={"name": "Branch"})
        assert r.status_code == 201, r.text
        assert r.json()["unit_id"] == "branch" and r.json()["parent_id"] == "bank"
        assert (await admin.post("/v1/units", json={"name": "Teller", "parent_id": "branch"})).status_code == 201
        units = {u["unit_id"] for u in (await admin.get("/v1/units")).json()}
        assert {"bank", "branch", "teller"} <= units


async def test_a_non_admin_cannot_create_a_unit(inmemory: None) -> None:
    reader = _client("bank", ["reader"])
    async with reader:
        assert (await reader.post("/v1/units", json={"name": "Sneaky"})).status_code == 403


async def test_a_unit_admin_grows_only_its_own_subtree(inmemory: None) -> None:
    # Seed the tree as tenant admin.
    admin = _client("bank", ["owner", "admin"])
    async with admin:
        await admin.post("/v1/units", json={"name": "Branch"})

    # A UNIT_ADMIN scoped to "branch" (role admin + unit=branch).
    branch_admin = _client("bank", ["admin"], unit="branch")
    async with branch_admin:
        # ...may create under its own subtree.
        assert (await branch_admin.post("/v1/units", json={"name": "Teller", "parent_id": "branch"})).status_code == 201
        # ...but NOT under the root (it does not govern upward).
        assert (await branch_admin.post("/v1/units", json={"name": "Exec", "parent_id": "bank"})).status_code == 403


async def test_delete_guards_root_and_non_empty(inmemory: None) -> None:
    admin = _client("bank", ["owner", "admin"])
    async with admin:
        await admin.post("/v1/units", json={"name": "Branch"})
        await admin.post("/v1/units", json={"name": "Teller", "parent_id": "branch"})
        assert (await admin.request("DELETE", "/v1/units/bank")).status_code == 422  # root
        assert (await admin.request("DELETE", "/v1/units/branch")).status_code == 422  # has a child
        assert (await admin.request("DELETE", "/v1/units/teller")).status_code == 204  # leaf
        assert "teller" not in {u["unit_id"] for u in (await admin.get("/v1/units")).json()}
