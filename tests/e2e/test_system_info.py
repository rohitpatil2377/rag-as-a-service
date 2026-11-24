"""E2E: /v1/system/info exposes non-secret platform config for the console.

Guards the invariant that this endpoint surfaces feature flags/thresholds/backend names
for the Settings screen but NEVER a secret (JWT signing key, store credentials, DSNs).
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_system_info_is_non_secret_config(inmemory: None) -> None:
    client = await _client("acme")
    async with client:
        r = await client.get("/v1/system/info")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["sovereign_enforce"] is True
        assert set(body["features"]) >= {"guardrails", "faithfulness", "adaptive_router"}
        assert set(body["thresholds"]) == {"abstain", "faithful", "flag"}

        # No secret value may appear anywhere in the payload.
        blob = r.text
        for secret in ("dev-only-insecure", "minioadmin", "postgresql+psycopg", "secret_key"):
            assert secret not in blob


async def test_system_info_requires_auth(inmemory: None) -> None:
    from rag_api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        r = await anon.get("/v1/system/info")
        assert r.status_code == 401
