"""E2E: permission-aware retrieval (FR-A6/D8).

Two docs in one tenant — a finance-restricted brief and a public one. A caller only ever
sees spans whose ACL is empty (tenant-public) or intersects the caller's group memberships,
so a restricted document never reaches a caller outside its group, at retrieval OR in an
answer.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_FINANCE = "The classified quarterly budget allocation for Project Trishul is 4200 crore rupees."
_PUBLIC = "The platform uptime target is 99.5 percent baseline and 99.9 percent at peak."


async def _client(tenant: str, groups: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"], attrs={"groups": groups})
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


async def _ask(client: httpx.AsyncClient, tenant: str, q: str) -> dict:
    return (await client.post(f"/v1/kb/{tenant}/query", json={"query": q, "top_k": 5})).json()


async def _ingest(client: httpx.AsyncClient, tenant: str, doc_id: str, text: str, acl: list[str] | None = None) -> int:
    body: dict = {"doc_id": doc_id, "mime": "text/plain", "content_b64": _b64(text)}
    if acl:
        body["acl"] = acl
    return (await client.post(f"/v1/kb/{tenant}/ingest", json=body)).status_code


async def test_restricted_doc_is_invisible_outside_its_group(inmemory: None) -> None:
    tenant = "cag"
    # Ingest both docs as a finance-entitled officer.
    admin = await _client(tenant, ["finance"])
    async with admin:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert await _ingest(admin, tenant, "budget", _FINANCE, acl=["finance"]) == 200
        assert await _ingest(admin, tenant, "sla", _PUBLIC) == 200

    # A finance officer CAN see the restricted budget.
    fin = await _client(tenant, ["finance"])
    async with fin:
        a = await _ask(fin, tenant, "What is the budget for Project Trishul?")
        assert a["status"] == "grounded"
        assert "4200" in a["answer"]

    # An HR officer (different group) CANNOT — the restricted fact never appears...
    hr = await _client(tenant, ["hr"])
    async with hr:
        a = await _ask(hr, tenant, "What is the budget for Project Trishul?")
        assert "4200" not in a["answer"]
        assert all(c["doc_id"] != "budget" for c in a["citations"])

        # ...but the public document is still visible to them.
        pub = await _ask(hr, tenant, "What is the uptime target?")
        assert pub["status"] == "grounded"
        assert "99.5" in pub["answer"]


async def test_retrieve_endpoint_filters_by_entitlement(inmemory: None) -> None:
    tenant = "cag"
    admin = await _client(tenant, ["finance"])
    async with admin:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(admin, tenant, "budget", _FINANCE, acl=["finance"])

    hr = await _client(tenant, ["hr"])
    async with hr:
        r = (await hr.post(f"/v1/kb/{tenant}/retrieve", json={"query": "Project Trishul budget", "top_k": 5})).json()
        assert all(s["doc_id"] != "budget" for s in r["spans"])
