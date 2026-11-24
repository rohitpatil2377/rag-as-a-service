"""E2E: end-user answer feedback (FR-I2, v1.1) — deterministic, in-memory.

A 👍/👎 on an answer is trace-correlated, audited as a distinct FEEDBACK event, and rolls
up into the Quality dashboard's satisfaction rate. Feedback never changes an answer — it's
signal for evaluation.
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


async def test_feedback_recorded_and_aggregated(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        doc = base64.b64encode(b"The uptime target is 99.5 percent.").decode()
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc})

        a1 = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        a2 = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target again?"})).json()

        # Two thumbs-up, one thumbs-down.
        assert (
            await c.post(
                f"/v1/kb/{tenant}/feedback", json={"trace_id": a1["trace_id"], "rating": "up", "comment": "spot on"}
            )
        ).status_code == 204
        assert (
            await c.post(f"/v1/kb/{tenant}/feedback", json={"trace_id": a1["trace_id"], "rating": "up"})
        ).status_code == 204
        assert (
            await c.post(f"/v1/kb/{tenant}/feedback", json={"trace_id": a2["trace_id"], "rating": "down"})
        ).status_code == 204

        report = (await c.get(f"/v1/kb/{tenant}/quality?window=all")).json()
        assert report["feedback_up"] == 2
        assert report["feedback_down"] == 1
        assert report["satisfaction_rate"] == pytest.approx(2 / 3, abs=1e-3)


async def test_feedback_rating_validated(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        bad = await c.post(f"/v1/kb/{tenant}/feedback", json={"trace_id": "t", "rating": "meh"})
        assert bad.status_code == 422  # only up/down allowed


async def test_no_feedback_leaves_satisfaction_null(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        report = (await c.get(f"/v1/kb/{tenant}/quality?window=all")).json()
        assert report["feedback_up"] == 0
        assert report["satisfaction_rate"] is None
