"""E2E: an org owns MANY knowledge bases, each hard-isolated (FR-H1).

HR / Finance as separate KBs under one tenant: creating a KB makes its own namespace,
documents in one KB never leak into another, and a different tenant can't touch them.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(client: httpx.AsyncClient, kb: str, doc_id: str, text: str) -> int:
    r = await client.post(
        f"/v1/kb/{kb}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": base64.b64encode(text.encode()).decode()},
    )
    return r.status_code


async def test_org_owns_many_isolated_kbs(inmemory: None) -> None:
    org = await _client("cag")
    async with org:
        # Create two knowledge bases under the same org.
        hr = (await org.post("/v1/knowledge-bases", json={"display_name": "HR Policies"})).json()
        fin = (await org.post("/v1/knowledge-bases", json={"display_name": "Finance"})).json()
        assert hr["kb_id"] == "cag__hr-policies" and fin["kb_id"] == "cag__finance"

        # Only the explicitly-created KBs appear — there is no auto-provisioned 'Default' KB.
        # The shared company-wide KB is listed for every org on purpose: it is a platform-wide
        # resource, deliberately outside this org's own kb_access/unit governance, so it is
        # excluded here rather than treated as one of the org's KBs.
        from svc_control_plane.api import SHARED_KB_ID

        kbs = (await org.get("/v1/knowledge-bases")).json()
        ids = {k["kb_id"] for k in kbs} - {SHARED_KB_ID}
        assert ids == {"cag__hr-policies", "cag__finance"}

        # Ingest a different doc into each KB.
        assert await _ingest(org, "cag__hr-policies", "leave", "Employees get 24 days of annual leave.") == 200
        assert await _ingest(org, "cag__finance", "budget", "The Q1 budget is 4200 crore rupees.") == 200

        # Each KB sees ONLY its own document — hard isolation.
        hr_docs = {d["doc_id"] for d in (await org.get("/v1/kb/cag__hr-policies/documents")).json()}
        fin_docs = {d["doc_id"] for d in (await org.get("/v1/kb/cag__finance/documents")).json()}
        assert hr_docs == {"leave"} and fin_docs == {"budget"}

        # Asking the HR KB about the budget → grounded on HR only (no finance leak).
        ans = (await org.post("/v1/kb/cag__hr-policies/query", json={"query": "What is the budget?"})).json()
        assert "4200" not in ans["answer"]


async def test_other_tenant_cannot_touch_a_kb(inmemory: None) -> None:
    other = await _client("acme")
    async with other:
        r = await other.get("/v1/kb/cag__hr-policies/documents")
        assert r.status_code == 403
        assert r.json()["code"] == "tenant_isolation_violation"
