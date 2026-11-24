"""E2E: hierarchical data-classification clearance (banking pack).

A Confidential document is invisible to a caller cleared only to Internal, but visible to a
caller cleared to Confidential (who, being higher, also sees Public/Internal). Enforced at
retrieval AND in the answer — the same guarantee A6 gives for need-to-know groups, but for
sensitivity LEVELS. The two are independent: a span must clear the level AND the ACL.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_CONFIDENTIAL = "The confidential merger consideration for Project Kohinoor is 8600 crore rupees."
_PUBLIC = "The bank's branch network uptime target is 99.5 percent."


async def _client(tenant: str, clearance: int | str = 0) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"], attrs={"clearance": clearance})
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


async def _ask(client: httpx.AsyncClient, tenant: str, q: str) -> dict:
    return (await client.post(f"/v1/kb/{tenant}/query", json={"query": q, "top_k": 5})).json()


async def test_confidential_doc_hidden_below_clearance(inmemory: None) -> None:
    tenant = "bank"
    # A Restricted-cleared officer ingests a Confidential doc + a Public doc.
    admin = await _client(tenant, clearance="restricted")
    async with admin:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r1 = await admin.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "merger",
                "mime": "text/plain",
                "content_b64": _b64(_CONFIDENTIAL),
                "classification": "confidential",
            },
        )
        assert r1.status_code == 200
        r2 = await admin.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": _b64(_PUBLIC)},  # default public
        )
        assert r2.status_code == 200

    # Internal clearance (1) < Confidential (2): the merger figure NEVER appears.
    low = await _client(tenant, clearance="internal")
    async with low:
        a = await _ask(low, tenant, "What is the consideration for Project Kohinoor?")
        assert "8600" not in a["answer"]
        assert all(c["doc_id"] != "merger" for c in a["citations"])
        # ...but a Public doc is still visible to them (assert the fact is retrieved; the
        # faithfulness verdict is orthogonal).
        pub = await _ask(low, tenant, "What is the uptime target?")
        assert pub["status"] in {"grounded", "flagged_for_review"}
        assert "99.5" in pub["answer"]

    # Confidential clearance (2) >= the doc's level: the officer CAN see it. (We assert
    # VISIBILITY — the fact + citation are retrieved — not the faithfulness verdict, which is
    # an orthogonal control; the small dev judge may flag it for review.)
    high = await _client(tenant, clearance="confidential")
    async with high:
        a = await _ask(high, tenant, "What is the consideration for Project Kohinoor?")
        assert a["status"] in {"grounded", "flagged_for_review"}
        assert "8600" in a["answer"]
        assert any(c["doc_id"] == "merger" for c in a["citations"])


async def test_retrieve_endpoint_filters_by_clearance(inmemory: None) -> None:
    tenant = "bank"
    admin = await _client(tenant, clearance="restricted")
    async with admin:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await admin.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "merger", "mime": "text/plain", "content_b64": _b64(_CONFIDENTIAL), "classification": 2},
        )

    low = await _client(tenant, clearance=0)
    async with low:
        r = (await low.post(f"/v1/kb/{tenant}/retrieve", json={"query": "Kohinoor consideration", "top_k": 5})).json()
        assert all(s["doc_id"] != "merger" for s in r["spans"])
