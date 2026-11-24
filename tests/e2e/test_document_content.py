"""E2E: in-document citation viewer source endpoint (FR-I3, v1.1).

The viewer reconstructs a document's readable text from its ordered chunks so a citation
can be shown in context with its span highlighted. The endpoint is ACL-aware: it never
returns a chunk the caller could not have retrieved (FR-A6).
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str, groups: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    attrs = {"groups": groups} if groups is not None else None
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"], attrs=attrs)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_document_content_returns_ordered_chunks(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        text = "The uptime target is 99.5 percent. The latency target is 3 seconds. Support is 24 by 7."
        doc = base64.b64encode(text.encode()).decode()
        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc, "strategy": "clause"},
        )

        # Ask so we have a citation with a chunk_id to open.
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        cited_chunk = ans["citations"][0]["chunk_id"]

        content = (await c.get(f"/v1/kb/{tenant}/documents/sla/content")).json()
        assert content["doc_id"] == "sla"
        assert content["mime"] == "text/plain"
        assert len(content["chunks"]) >= 1
        # The cited chunk is present in the reconstructed document, so the viewer can anchor.
        assert any(ch["chunk_id"] == cited_chunk for ch in content["chunks"])
        # Full source text is recoverable from the chunks in order.
        joined = " ".join(ch["text"] for ch in content["chunks"])
        assert "99.5 percent" in joined
        assert "3 seconds" in joined


async def test_content_is_acl_filtered(inmemory: None) -> None:
    """A caller only sees chunks they could have retrieved — restricted spans never leak
    through the viewer."""
    tenant = "acme"
    ingestor = await _client(tenant, groups=["finance"])
    async with ingestor:
        await ingestor.post("/v1/knowledge-bases", json={"display_name": "KB"})
        secret = base64.b64encode(b"The finance-only budget ceiling is 5 crore.").decode()
        await ingestor.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "budget", "mime": "text/plain", "content_b64": secret, "acl": ["finance"]},
        )

    # A public caller (no groups) must not see the restricted document's content.
    public = await _client(tenant, groups=[])
    async with public:
        r = await public.get(f"/v1/kb/{tenant}/documents/budget/content")
        assert r.status_code == 404  # nothing readable for this caller

    # The entitled caller sees it.
    entitled = await _client(tenant, groups=["finance"])
    async with entitled:
        r = await entitled.get(f"/v1/kb/{tenant}/documents/budget/content")
        assert r.status_code == 200
        assert "5 crore" in " ".join(ch["text"] for ch in r.json()["chunks"])
