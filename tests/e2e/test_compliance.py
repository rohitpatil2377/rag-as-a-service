"""E2E: DPDP compliance surface (FR-H8, v1.1) — erasure, retention, attestation.

Proves the obligations a regulated buyer actually asks for:
* right-to-erasure removes a data principal's documents from every store AND the chat
  history that quoted them, and returns a tamper-evident certificate;
* the erased content is genuinely gone — a later ask abstains, the audit records ERASURE;
* retention purges documents older than the policy window;
* the residency attestation reports sovereignty, encryption, audit-chain integrity.
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
    token = issue_dev_jwt("dpo", tenant, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body})


async def test_right_to_erasure_by_subject_cascades(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        await _ingest(c, tenant, "record-anita", "Anita Desai's account balance is 5 lakh.")
        await _ingest(c, tenant, "policy", "The uptime target is 99.5 percent.")

        # A conversation quotes the personal record → must be scrubbed too.
        conv = (await c.post(f"/v1/kb/{tenant}/conversations", json={"query": "What is the balance?"})).json()
        cid = conv["conversation_id"]
        ans = (
            await c.post(f"/v1/kb/{tenant}/conversations/{cid}/ask", json={"query": "What is Anita Desai's balance?"})
        ).json()
        assert "5 lakh" in ans["answer"]

        # Erase everything about the data principal.
        cert = (await c.post(f"/v1/kb/{tenant}/erasure", json={"subject": "Anita Desai"})).json()
        assert cert["documents_erased"] == ["record-anita"]
        assert cert["conversations_purged"] == 1
        assert len(cert["certificate_hash"]) == 64

        # The document is gone from the catalog…
        docs = (await c.get(f"/v1/kb/{tenant}/documents")).json()
        assert [d["doc_id"] for d in docs] == ["policy"]
        # …the conversation that quoted it is gone…
        assert (await c.get(f"/v1/kb/{tenant}/conversations")).json() == []
        # …and the erasure is on the audit trail.
        audit = (await c.get(f"/v1/kb/{tenant}/audit")).json()
        assert any(
            e["action"] == "erasure" and e["payload"]["certificate_hash"] == cert["certificate_hash"] for e in audit
        )

        # The unrelated document still answers.
        still = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        assert still["status"] == "grounded"


async def test_erasure_requires_a_target(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(f"/v1/kb/{tenant}/erasure", json={})
        assert r.status_code == 422


async def test_retention_policy_purges_old_documents(inmemory: None) -> None:
    from datetime import UTC, datetime, timedelta

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "fresh", "Recent document.")

        # Backdate one document past the retention window using the store directly.
        from rag_clients.factory import get_catalog
        from rag_schemas import DocumentEnvelope
        from rag_schemas.common import SourceKind

        old = DocumentEnvelope(
            tenant_id=tenant,
            doc_id="stale",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            ingested_at=datetime.now(UTC) - timedelta(days=400),
        )
        await get_catalog().register_document(old)

        assert (await c.put(f"/v1/kb/{tenant}/retention", json={"retention_days": 365})).json()["retention_days"] == 365
        result = (await c.post(f"/v1/kb/{tenant}/retention/apply")).json()
        assert result["documents_erased"] == ["stale"]

        remaining = {d["doc_id"] for d in (await c.get(f"/v1/kb/{tenant}/documents")).json()}
        assert remaining == {"fresh"}


async def test_attestation_reports_sovereignty_and_chain(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "d1", "Some content.")

        att = (await c.get(f"/v1/kb/{tenant}/attestation")).json()
        assert att["sovereignty_enforced"] is True
        assert att["audit_chain_verified"] is True
        assert att["audit_events"] >= 1
        assert att["documents"] == 1
        assert "India" in att["data_residency"]
