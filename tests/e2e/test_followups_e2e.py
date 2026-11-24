"""E2E: FR-I8 suggested follow-ups — offered on grounded answers, from context, flag-gated."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FOLLOWUPS_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body, "strategy": "clause"},
    )


async def test_followups_offered_on_grounded_answer(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    tenant = "i8demo"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        await _ingest(c, tenant, "d", "The Acme Cloud Platform guarantees 99.9% uptime and encrypts data with AES-256.")
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime?"})).json()
        assert ans["status"] == "grounded"
        assert len(ans["suggested_followups"]) >= 1
        assert all(q.endswith("?") for q in ans["suggested_followups"])


async def test_followups_absent_when_disabled(inmemory: None) -> None:
    tenant = "i8off"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        await _ingest(c, tenant, "d", "The Acme Cloud Platform guarantees 99.9% uptime.")
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime?"})).json()
        assert ans["suggested_followups"] == []
