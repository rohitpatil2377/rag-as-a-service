"""E2E: per-tenant quotas & rate limits (FR-H1 — noisy-neighbour isolation).

Rate limits are per org over a sliding minute; the document cap is per KB. A breach is a
clear 429 with a stable code and retry hint — and one org's limit never throttles another.
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
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _set_quota(monkeypatch: pytest.MonkeyPatch, **kv: str) -> None:
    from rag_common.config import get_settings

    for k, v in kv.items():
        monkeypatch.setenv(f"RAG_QUOTA_{k.upper()}", v)
    get_settings.cache_clear()


async def test_query_rate_limit_is_per_org(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_quota(monkeypatch, queries_per_minute="3")

    a = await _client("acme")
    async with a:
        for _ in range(3):
            assert (await a.post("/v1/kb/acme/query", json={"query": "hello?"})).status_code == 200
        r = await a.post("/v1/kb/acme/query", json={"query": "hello?"})
        assert r.status_code == 429
        body = r.json()
        assert body["code"] == "rate_limited"
        assert body["details"]["retry_after_seconds"] >= 1

    # A different org is NOT throttled by acme's burst.
    b = await _client("globex")
    async with b:
        assert (await b.post("/v1/kb/globex/query", json={"query": "hi?"})).status_code == 200


async def test_document_cap_per_kb(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_quota(monkeypatch, max_documents_per_kb="2", ingests_per_minute="0")  # 0 = rate off

    c = await _client("acme")
    doc = base64.b64encode(b"content").decode()
    async with c:
        for i in range(2):
            r = await c.post("/v1/kb/acme/ingest", json={"doc_id": f"d{i}", "mime": "text/plain", "content_b64": doc})
            assert r.status_code == 200
        r = await c.post("/v1/kb/acme/ingest", json={"doc_id": "d2", "mime": "text/plain", "content_b64": doc})
        assert r.status_code == 429
        assert r.json()["code"] == "quota_exceeded"

        # Deleting a document frees capacity.
        assert (await c.delete("/v1/kb/acme/documents/d0")).status_code == 204
        r = await c.post("/v1/kb/acme/ingest", json={"doc_id": "d2", "mime": "text/plain", "content_b64": doc})
        assert r.status_code == 200


async def test_quotas_can_be_disabled(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_quota(monkeypatch, enabled="false", queries_per_minute="1")

    c = await _client("acme")
    async with c:
        for _ in range(3):
            assert (await c.post("/v1/kb/acme/query", json={"query": "q?"})).status_code == 200
