"""E2E: platform control plane — SUPER_ADMIN tenant lifecycle + suspend enforcement (T3b)."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "ops@acme.example")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(subject: str, tenant: str, roles: list[str], *, clearance: str | int = 0) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt(subject, tenant, roles=roles, attrs={"clearance": clearance})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_only_a_platform_admin_may_provision(inmemory: None, platform: None) -> None:
    stranger = _client("nobody@x.com", "x", ["owner", "admin"])
    async with stranger:
        r = await stranger.post(
            "/v1/platform/tenants",
            json={"org_id": "acme", "name": "Acme", "admin_email": "a@acme.io", "password": "supersecret"},
        )
        assert r.status_code == 403, r.text


async def test_provision_list_and_summary(inmemory: None, platform: None) -> None:
    ops = _client("ops@acme.example", "platform", ["owner"])
    async with ops:
        r = await ops.post(
            "/v1/platform/tenants",
            json={
                "org_id": "acme",
                "name": "Acme",
                "admin_email": "a@acme.io",
                "password": "supersecret",
                "plan": "enterprise",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["plan"] == "enterprise"
        assert any(t["org_id"] == "acme" for t in (await ops.get("/v1/platform/tenants")).json())
        summ = (await ops.get("/v1/platform/tenants/acme")).json()
        assert summ["tenant"]["org_id"] == "acme" and summ["users"] >= 1


def _ops() -> httpx.AsyncClient:
    return _client("ops@acme.example", "platform", ["owner"])


def _acme() -> httpx.AsyncClient:
    return _client("a@acme.io", "acme", ["owner"], clearance="restricted")


async def test_suspend_blocks_the_tenant_then_resume_restores(inmemory: None, platform: None) -> None:
    async with _ops() as ops:
        await ops.post(
            "/v1/platform/tenants",
            json={"org_id": "acme", "name": "Acme", "admin_email": "a@acme.io", "password": "supersecret"},
        )
    async with _acme() as user:  # active -> reachable
        assert (await user.get("/v1/kb/acme/model-status")).status_code == 200

    async with _ops() as ops:
        await ops.post("/v1/platform/tenants/acme/suspend")
    async with _acme() as user:  # suspended -> withheld
        assert (await user.get("/v1/kb/acme/model-status")).status_code == 422

    async with _ops() as ops:
        await ops.post("/v1/platform/tenants/acme/resume")
    async with _acme() as user:  # resumed
        assert (await user.get("/v1/kb/acme/model-status")).status_code == 200


async def test_deprovision_tombstones(inmemory: None, platform: None) -> None:
    ops = _client("ops@acme.example", "platform", ["owner"])
    async with ops:
        await ops.post(
            "/v1/platform/tenants",
            json={"org_id": "gone", "name": "Gone", "admin_email": "a@gone.io", "password": "supersecret"},
        )
        r = await ops.request("DELETE", "/v1/platform/tenants/gone")
        assert r.status_code == 200, r.text
        summ = (await ops.get("/v1/platform/tenants/gone")).json()
        assert summ["tenant"]["status"] == "deprovisioned"
