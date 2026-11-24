"""E2E: scoped service-account API keys (FR-I4) — programmatic access.

A key authenticates like a bearer token, resolves to the org with scope-derived roles, is
shown once, is revocable, and never crosses org boundaries.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(headers: dict[str, str]) -> httpx.AsyncClient:
    from rag_api.app import create_app

    app = create_app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=headers)


def _jwt(org: str, role: str = "owner") -> dict[str, str]:
    from svc_control_plane.identity import ROLE_CAPS

    from rag_common import issue_dev_jwt

    return {"Authorization": f"Bearer {issue_dev_jwt('u@x', org, roles=ROLE_CAPS[role])}"}


async def test_key_authenticates_and_is_scoped(inmemory: None) -> None:
    org = "acme"
    owner = await _client(_jwt(org))
    async with owner:
        await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await owner.put(f"/v1/kb/{org}/settings", json={"faithful_threshold": 0.1})
        doc = base64.b64encode(b"The uptime target is 99.5 percent.").decode()
        await owner.post(f"/v1/kb/{org}/ingest", json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc})

        # Mint a read-scoped key; the full secret is returned exactly once.
        created = (await owner.post("/v1/api-keys", json={"name": "ci-bot", "scope": "read"})).json()
        full = created["key"]
        assert full.startswith("ek_")
        # Listing never returns the secret.
        listed = (await owner.get("/v1/api-keys")).json()
        assert listed[0]["prefix"] == created["prefix"] and "key" not in listed[0]

    # The key authenticates a query as the org…
    bot = await _client({"Authorization": f"Bearer {full}"})
    async with bot:
        ans = (await bot.post(f"/v1/kb/{org}/query", json={"query": "What is the uptime target?"})).json()
        assert ans["status"] == "grounded"
        # …but a read scope cannot ingest (no ingestor role).
        blocked = await bot.post(
            f"/v1/kb/{org}/ingest", json={"doc_id": "x", "mime": "text/plain", "content_b64": "eA=="}
        )
        assert blocked.status_code == 403


async def test_revoked_key_is_rejected(inmemory: None) -> None:
    org = "acme"
    owner = await _client(_jwt(org))
    async with owner:
        await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})
        created = (await owner.post("/v1/api-keys", json={"name": "temp", "scope": "read"})).json()
        full, kid = created["key"], created["key_id"]

    bot = await _client({"Authorization": f"Bearer {full}"})
    async with bot:
        assert (await bot.get(f"/v1/kb/{org}/documents")).status_code == 200

    async with await _client(_jwt(org)) as owner2:
        assert (await owner2.delete(f"/v1/api-keys/{kid}")).status_code == 204

    async with await _client({"Authorization": f"Bearer {full}"}) as revoked:
        assert (await revoked.get(f"/v1/kb/{org}/documents")).status_code == 401


async def test_key_cannot_reach_another_org(inmemory: None) -> None:
    async with await _client(_jwt("orga")) as a:
        await a.post("/v1/knowledge-bases", json={"display_name": "KB"})
        full = (await a.post("/v1/api-keys", json={"name": "k", "scope": "admin"})).json()["key"]

    # orga's key resolves to orga → cannot address orgb's KB (tenant isolation).
    async with await _client({"Authorization": f"Bearer {full}"}) as bot:
        assert (await bot.get("/v1/kb/orgb/documents")).status_code == 403


async def test_reader_jwt_cannot_mint_keys(inmemory: None) -> None:
    async with await _client(_jwt("acme", role="viewer")) as viewer:
        await viewer.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await viewer.post("/v1/api-keys", json={"name": "x", "scope": "read"})
        assert r.status_code == 403


def test_key_prefix_handles_underscore_in_secret() -> None:
    """Regression: token_urlsafe secrets can contain '_'; prefix parsing must still work
    (a Postgres-only resolve bug surfaced by the FR-I6 widget session exchange)."""
    from svc_control_plane.api_keys import key_prefix

    assert key_prefix("ek_ab12cd34_sec_ret_with_unders") == "ab12cd34"
    assert key_prefix("ek_ab12cd34_plainsecret") == "ab12cd34"
    assert key_prefix("not-a-key") is None
    assert key_prefix("ek_only") is None
