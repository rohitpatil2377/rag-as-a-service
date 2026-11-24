"""E2E: Federated RAG federation-grant lifecycle over the /v1 API.

Proves: an owner/admin can publish a grant, list it (issued + received from the
subject's side), and revoke it; a non-admin cannot; grants over foreign KBs are
refused; and the whole surface is gated behind RAG_FEDERATION_ENABLED.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def federation_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable the federation feature flag. Cache-clear so get_settings re-reads it."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(tenant: str, roles: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=roles)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_issue_list_and_revoke_grant(inmemory: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner", "reader"])
    async with hr:
        # HR owns a KB it can grant against.
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})

        r = await hr.post(
            "/v1/grants",
            json={
                "subject_dept_id": "finance",
                "topics": ["leave-policy", "headcount"],
                "kb_ids": ["hr__policies"],
                "ttl_days": 90,
            },
        )
        assert r.status_code == 201, r.text
        grant = r.json()
        assert grant["issuer_dept_id"] == "hr"
        assert grant["subject_dept_id"] == "finance"
        assert grant["active"] is True
        grant_id = grant["grant_id"]

        # HR sees it in its issued list.
        issued = (await hr.get("/v1/grants")).json()
        assert any(g["grant_id"] == grant_id for g in issued)

        # HR revokes it.
        r = await hr.delete(f"/v1/grants/{grant_id}")
        assert r.status_code == 204, r.text

        issued_after = (await hr.get("/v1/grants")).json()
        row = next(g for g in issued_after if g["grant_id"] == grant_id)
        assert row["active"] is False
        assert row["revoked_at"] is not None


async def test_subject_sees_received_grant(inmemory: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 201, r.text

    finance = _client("finance", roles=["reader"])
    async with finance:
        received = (await finance.get("/v1/grants/received")).json()
        assert len(received) == 1
        assert received[0]["issuer_dept_id"] == "hr"
        assert received[0]["topics"] == ["leave-policy"]

        # Finance (not the issuer) cannot revoke HR's grant.
        gid = received[0]["grant_id"]
        r = await finance.delete(f"/v1/grants/{gid}")
        assert r.status_code == 403, r.text


async def test_non_admin_cannot_issue(inmemory: None, federation_on: None) -> None:
    hr = _client("hr", roles=["reader", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["x"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 403, r.text


async def test_cannot_grant_foreign_kb(inmemory: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner"])
    async with hr:
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["x"], "kb_ids": ["acme__secret"]},
        )
        assert r.status_code == 403, r.text
        assert r.json()["code"] == "tenant_isolation_violation"


async def test_cannot_self_grant(inmemory: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "hr", "topics": ["x"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 422, r.text


async def test_federation_disabled_blocks_all(inmemory: None) -> None:
    """Without the feature flag, the grant surface is a hard 403 (default isolation stands)."""
    hr = _client("hr", roles=["owner"])
    async with hr:
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["x"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 403, r.text
        assert r.json()["details"]["flag"] == "federation_enabled"
