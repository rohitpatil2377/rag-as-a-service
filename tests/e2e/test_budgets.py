"""E2E: per-KB token budget & hard cap (FR-H7).

A KB's monthly token spend is metered from the audit trail (FR-H4). Over the hard cap,
generation is refused with a 429; under it, answers flow. 0 = unlimited.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from svc_control_plane.identity import ROLE_CAPS

    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("owner", tenant, roles=ROLE_CAPS["owner"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _seed(c: httpx.AsyncClient, tenant: str) -> None:
    await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
    await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
    doc = base64.b64encode(b"The uptime target is 99.5 percent.").decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc})


async def test_budget_hard_cap_refuses_over_limit(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)

        # Unlimited by default → answers flow and accrue tokens.
        assert (await c.get(f"/v1/kb/{tenant}/budget")).json()["budget_tokens"] == 0
        first = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        assert first["status"] == "grounded"

        used = (await c.get(f"/v1/kb/{tenant}/budget")).json()["used_tokens"]
        assert used > 0  # metered from the audit trail

        # Set a hard cap BELOW what's already spent → next generation is refused (429).
        r = await c.put(f"/v1/kb/{tenant}/budget", json={"budget_tokens": 1})
        assert r.status_code == 200 and r.json()["over_budget"] is True
        blocked = await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})
        assert blocked.status_code == 429
        assert "budget" in blocked.json()["message"].lower()

        # Raise the cap well above usage → answers resume.
        await c.put(f"/v1/kb/{tenant}/budget", json={"budget_tokens": 10_000_000})
        resumed = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        assert resumed["status"] == "grounded"


async def test_budget_requires_admin(inmemory: None) -> None:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    tenant = "acme"
    app = create_app()
    viewer = issue_dev_jwt("v", tenant, roles=["reader"])  # read scope, no admin
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {viewer}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        # The admin gate fires before any KB work, so a reader is refused outright.
        assert (await c.put(f"/v1/kb/{tenant}/budget", json={"budget_tokens": 100})).status_code == 403
