"""E2E: the platform-wide shared KB (Federated RAG Tier C — company-wide opt-in).

Distinct from Tier B (signed cross-department Grants, see test_federated_ask.py /
test_federation_*): the shared KB needs no grant and no ``federate_from`` — any tenant's
ingestor may choose it as an ingest destination instead of their own KB, and every
tenant's plain ``/v1/ask`` searches it by default alongside their own KBs.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from svc_control_plane.knowledge_base import SHARED_KB_ID

pytestmark = pytest.mark.e2e


async def _client(org: str, roles: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", org, roles=roles or ["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, kb: str, doc_id: str, text: str, classification: int = 0) -> httpx.Response:
    return await c.post(
        f"/v1/kb/{kb}/ingest",
        json={
            "doc_id": doc_id,
            "mime": "text/plain",
            "content_b64": base64.b64encode(text.encode()).decode(),
            "classification": classification,
        },
    )


async def test_shared_kb_listed_for_every_tenant_with_no_setup(inmemory: None) -> None:
    hr = await _client("hr")
    async with hr:
        kbs = (await hr.get("/v1/knowledge-bases")).json()
        assert any(k["kb_id"] == SHARED_KB_ID for k in kbs)


async def test_other_department_reads_shared_content_with_no_grant(inmemory: None, lexical_gate: None) -> None:
    hr = await _client("hr")
    async with hr:
        r = await _ingest(hr, SHARED_KB_ID, "leave", "Employees are entitled to 24 days of paid annual leave.")
        assert r.status_code == 200, r.text

    tech = await _client("tech")
    async with tech:
        r = await tech.post("/v1/ask", json={"query": "How many days of annual leave do employees get?"})
        assert r.status_code == 200, r.text
        ans = r.json()
        assert ans["status"] == "grounded", ans
        assert "24" in ans["answer"]
        assert any(c["doc_id"].startswith("company-wide/") for c in ans["citations"])


async def test_own_private_kb_still_isolated_from_other_tenants(inmemory: None, lexical_gate: None) -> None:
    hr = await _client("hr")
    async with hr:
        kb = (await hr.post("/v1/knowledge-bases", json={"display_name": "Comp Bands"})).json()
        kb_id = kb["kb_id"]
        r = await _ingest(hr, kb_id, "bands", "Senior engineers are paid 45 lakh per year.")
        assert r.status_code == 200, r.text

    tech = await _client("tech")
    async with tech:
        # Direct cross-tenant access to HR's own (non-shared) KB must still be refused.
        r = await tech.post(f"/v1/kb/{kb_id}/retrieve", json={"query": "salary"})
        assert r.status_code in (403, 404), r.text
        # And it must not leak into Tech's own default ask either.
        r = await tech.post("/v1/ask", json={"query": "How much are senior engineers paid?"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] != "grounded"


async def test_shared_kb_rejects_above_public_classification(inmemory: None) -> None:
    hr = await _client("hr")
    async with hr:
        r = await _ingest(hr, SHARED_KB_ID, "restricted-thing", "This should never be company-wide.", classification=2)
        assert r.status_code == 403, r.text
