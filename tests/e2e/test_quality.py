"""E2E: quality dashboards + on-demand golden-set evaluation (FR-G5)."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_DOC = "The platform uptime target is 99.5 percent baseline. The P95 query latency target is 3 seconds or less."


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor", "reviewer"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_quality_report_reflects_live_traffic(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": base64.b64encode(_DOC.encode()).decode()},
        )
        # One answerable question + one that abstains on retrieval confidence.
        await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})
        await c.post(f"/v1/kb/{tenant}/query", json={"query": "Who won the 1998 football world cup final?"})

        r = await c.get(f"/v1/kb/{tenant}/quality?window=all")
        assert r.status_code == 200, r.text
        q = r.json()
        assert q["answers"] >= 1  # the grounded/flagged one went through faithfulness
        buckets = q["faithfulness_buckets"]
        assert sum(buckets.values()) == q["answers"]
        assert q["faithfulness_avg"] is not None
        assert q["abstain_reasons"].get("retrieval_confidence_below_threshold", 0) >= 1
        assert (await c.get(f"/v1/kb/{tenant}/quality?window=bogus")).status_code == 422


async def test_eval_runs_golden_questions_and_scores(inmemory: None, lexical_gate: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": base64.b64encode(_DOC.encode()).decode()},
        )

        r = await c.post(
            f"/v1/kb/{tenant}/eval",
            json={
                "questions": [
                    {"question": "What is the uptime target?", "expected": "99.5"},
                    {"question": "What is the P95 latency target?", "expected": "3 seconds"},
                    {"question": "What is the capital of France?", "expected": "Paris"},
                ]
            },
        )
        assert r.status_code == 200, r.text
        e = r.json()
        assert e["total"] == 3
        assert e["with_expected"] == 3
        assert e["hits"] >= 2  # the two SLA facts are answerable; Paris is not in the KB
        assert e["abstained"] >= 1  # the France question abstains (grounded-or-abstain)
        assert len(e["rows"]) == 3
        assert e["rows"][0]["hit"] is True


async def test_a_golden_set_over_the_cap_says_so(inmemory: None, lexical_gate: None) -> None:
    """`questions[:MAX_EVAL_QUESTIONS]` dropped everything past the cap in silence, so a tenant
    who submitted 40 questions read a report over 20 and had no way to know. A truncated run
    that looks complete is worse than a refused one — the score is real, the coverage is not."""
    from svc_control_plane.quality import MAX_EVAL_QUESTIONS, EvalQuestion, run_eval

    questions = [EvalQuestion(question=f"question number {i}?") for i in range(MAX_EVAL_QUESTIONS + 5)]
    report = await run_eval("qtrunc", questions, acl=[], actor="t")
    assert report.total == MAX_EVAL_QUESTIONS
    assert report.submitted == MAX_EVAL_QUESTIONS + 5
    assert report.truncated is True


async def test_a_golden_set_within_the_cap_is_not_flagged(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane.quality import EvalQuestion, run_eval

    report = await run_eval("qsmall", [EvalQuestion(question="anything?")], acl=[], actor="t")
    assert report.truncated is False
    assert report.submitted == 1
