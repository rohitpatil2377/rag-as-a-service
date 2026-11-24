"""E2E: FR-G9 continuous evaluation — enrol, run-now (golden + live sampling), breach alert.

Exercises the composition end-to-end through the real API: a run scores the golden set
plus a sample of recent live QUERY traffic, records an EVAL_RUN, and raises an EVAL_ALERT
when a metric floor is breached. Deterministic; no scheduler needed (run-now triggers it).
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
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body, "strategy": "clause"},
    )


async def test_run_now_scores_golden_and_live_sample(inmemory: None) -> None:
    tenant = "evalco"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        await _ingest(c, tenant, "d", "The office WiFi password is rotated every 90 days by IT.")

        # Generate live QUERY traffic (feeds the sampler).
        for q in ("How often is the WiFi password rotated?", "Who rotates the WiFi password?"):
            await c.post(f"/v1/kb/{tenant}/query", json={"query": q})

        # Enrol with permissive floors (healthy).
        await c.put(
            f"/v1/kb/{tenant}/eval/schedule",
            json={"enabled": True, "interval_hours": 24, "min_grounded_rate": 0.1, "min_faithfulness": 0.0},
        )
        run = (await c.post(f"/v1/kb/{tenant}/eval/run-now")).json()
        assert run["sampled_n"] >= 1  # re-scored real queries
        assert run["total"] >= 1
        assert run["breached"] is False

        sched = (await c.get(f"/v1/kb/{tenant}/eval/schedule")).json()
        assert sched["schedule"]["enabled"] is True
        assert any(r["action"] == "eval_run" for r in sched["runs"])


async def test_breach_raises_alert(inmemory: None) -> None:
    tenant = "evalco2"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "d", "Annual leave is 24 days per year.")
        # A retrieval gate nothing can clear, so the sampled question ABSTAINS when the run
        # re-asks it -> grounded_rate 0.0, under any floor above it. The faithfulness floor is
        # not used to force the breach here: it cannot be, now that the extractive twin quotes
        # its source cleanly and every CI answer scores 1.00. Which floors fire on which metrics
        # is unit-tested (test_continuous_eval_unit.test_breach_reasons_flags_each_floor); what
        # this test owns is the WIRING -- a breached run raises the alert on the audit chain.
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.99})
        await c.post(f"/v1/kb/{tenant}/query", json={"query": "How many annual leave days?"})

        await c.put(
            f"/v1/kb/{tenant}/eval/schedule",
            json={"enabled": True, "interval_hours": 24, "min_grounded_rate": 0.9, "min_faithfulness": 0.0},
        )
        run = (await c.post(f"/v1/kb/{tenant}/eval/run-now")).json()
        assert run["sampled_n"] >= 1
        assert run["breached"] is True, run
        assert any("grounded_rate" in r for r in run["breach_reasons"]), run

        runs = (await c.get(f"/v1/kb/{tenant}/eval/schedule")).json()["runs"]
        assert any(r["action"] == "eval_alert" for r in runs)


async def test_run_now_requires_ingestor(inmemory: None) -> None:
    from rag_common import issue_dev_jwt

    tenant = "evalco3"
    from rag_api.app import create_app

    app = create_app()
    reader = issue_dev_jwt("ro", tenant, roles=["reader"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {reader}"}
    ) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        resp = await c.post(f"/v1/kb/{tenant}/eval/run-now")
        assert resp.status_code == 403
