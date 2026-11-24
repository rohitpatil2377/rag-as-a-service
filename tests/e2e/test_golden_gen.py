"""E2E: FR-G8 synthetic golden-set generation — draft from the KB's own documents."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str, roles: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=roles)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_generate_golden_from_documents(inmemory: None) -> None:
    tenant = "g8demo"
    c = await _client(tenant, ["reader", "ingestor"])
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        text = "The Acme Cloud platform guarantees 99.9% uptime. Support responds to Critical tickets within 4 hours."
        body = base64.b64encode(text.encode()).decode()
        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": body, "strategy": "clause"},
        )
        gen = (await c.post(f"/v1/kb/{tenant}/golden-set/generate?count=3")).json()
        assert len(gen["questions"]) >= 1
        # Every drafted item is a well-formed question (heuristic fallback under CI's dev gateway).
        assert all(q["question"].endswith("?") for q in gen["questions"])
        # Generation does NOT auto-save — the stored golden set is still empty until PUT.
        assert (await c.get(f"/v1/kb/{tenant}/golden-set")).json()["questions"] == []


async def test_generate_requires_ingestor(inmemory: None) -> None:
    tenant = "g8ro"
    c = await _client(tenant, ["reader"])
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        resp = await c.post(f"/v1/kb/{tenant}/golden-set/generate")
        assert resp.status_code == 403


async def test_generate_empty_kb_returns_no_drafts(inmemory: None) -> None:
    tenant = "g8empty"
    c = await _client(tenant, ["reader", "ingestor"])
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        gen = (await c.post(f"/v1/kb/{tenant}/golden-set/generate")).json()
        assert gen["questions"] == []
