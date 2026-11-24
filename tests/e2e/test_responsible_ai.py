"""E2E: Responsible-AI framework attestation endpoint (FR-H8+)."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_frameworks_report(inmemory: None) -> None:
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.get(f"/v1/kb/{tenant}/compliance/frameworks")
        assert r.status_code == 200
        body = r.json()
        ids = {f["id"] for f in body["frameworks"]}
        # IRDAI joined the catalogue (insurance regulator — localization, access control, audit
        # trail). Asserted as an exact set on purpose: a framework silently disappearing from a
        # compliance report is exactly the regression worth failing on.
        assert ids == {"niti", "dpdp", "rbi", "sebi", "irdai", "iso42001", "owasp"}
        assert body["summary"]["controls"] > 20
        # Every control carries evidence and a valid status.
        for fw in body["frameworks"]:
            assert fw["satisfied"] <= fw["total"]
            for ctrl in fw["controls"]:
                assert ctrl["evidence"]
                assert ctrl["status"] in {"satisfied", "partial", "manual", "not_applicable"}


async def test_single_framework_filter(inmemory: None) -> None:
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.get(f"/v1/kb/{tenant}/compliance/frameworks", params={"framework": "dpdp"})
        body = r.json()
        assert [f["id"] for f in body["frameworks"]] == ["dpdp"]
        assert any(c["id"] == "dpdp.erasure" and c["status"] == "satisfied" for c in body["frameworks"][0]["controls"])
