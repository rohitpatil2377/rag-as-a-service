"""E2E: job-centre activity feed (recent ingest/index operations)."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_activity_lists_ingest(inmemory: None) -> None:
    kb = "actco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        b = base64.b64encode(b"The night shift allowance policy covers all staff.").decode()
        await c.post(
            f"/v1/kb/{kb}/ingest", json={"doc_id": "d", "mime": "text/plain", "content_b64": b, "strategy": "clause"}
        )
        acts = (await c.get(f"/v1/kb/{kb}/activity")).json()
        assert any(a["action"] == "index" and a["label"] == "d" and "chunk" in a["detail"] for a in acts)
        assert any(a["action"] == "ingest" for a in acts)
