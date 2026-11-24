"""E2E: usage metering + FinOps showback (FR-H4).

Drives real activity through the API, then asserts /v1/kb/{id}/usage aggregates the audit
trail into per-action counts + an indicative showback that matches the rate card, and that
the time window bounds the counts.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_DOC = "The platform uptime target is 99.5 percent baseline and 99.9 percent at peak."


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_usage_meters_audit_trail_and_prices_showback(inmemory: None, lexical_gate: None) -> None:
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "brief", "mime": "text/plain", "content_b64": base64.b64encode(_DOC.encode()).decode()},
        )
        await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?", "top_k": 3})
        await client.post(f"/v1/kb/{tenant}/query", json={"query": "Who won the 1998 world cup?", "top_k": 3})

        r = await client.get(f"/v1/kb/{tenant}/usage?window=all")
        assert r.status_code == 200, r.text
        u = r.json()

        assert u["documents"] == 1  # one ingest
        assert u["queries"] == 2  # two queries
        assert u["counts"]["query"] == 2
        assert u["abstains"] >= 1  # the world-cup query has no grounding

        # Token metering: the two answered/abstained queries produced >=1 generation with
        # (estimated) token usage; totals aggregate over GENERATE events.
        assert u["total_tokens"] > 0
        assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0

        # Showback = every line priced from the (indicative) rate card, summed.
        expected = round(sum(line["cost"] for line in u["lines"]), 2)
        assert u["total_cost"] == expected
        rates = {line["unit"]: line for line in u["lines"]}
        assert rates["1k tokens"]["count"] == u["total_tokens"]
        assert rates["1k tokens"]["cost"] == round(u["total_tokens"] / 1000 * rates["1k tokens"]["rate"], 2)
        assert u["currency"] == "INR"
        assert u["rate_card_indicative"] is True


async def test_usage_window_bounds_counts(inmemory: None) -> None:
    """A 24h window still counts just-created events; an unknown window is a 422."""
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await client.post(f"/v1/kb/{tenant}/query", json={"query": "anything?"})

        r = await client.get(f"/v1/kb/{tenant}/usage?window=24h")
        assert r.status_code == 200
        assert r.json()["queries"] == 1

        bad = await client.get(f"/v1/kb/{tenant}/usage?window=nonsense")
        assert bad.status_code == 422
