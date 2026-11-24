"""Hybrid audit fix + abstain/feedback analytics (Layer-1 item 3).

Two things are proven here:

* the bug's regression test — a multi-KB org's KB-scoped ``count_actions_kb`` now sees a query-time
  event that landed in the ORG schema (the old ``count_actions`` misses it), while a single-KB org
  (``kb_id == org``) is unaffected;
* the analytics surface — abstain rate, clustered abstained queries, thumbs-down (correlated back
  to the question), and low-faithfulness answers, all over the KB-scoped union read.
"""

from __future__ import annotations

import pytest
from svc_control_plane import wiring
from svc_control_plane.analytics import analytics_gaps
from svc_guardrails_eval import AuditService

from rag_clients.memory import InMemoryAuditSink
from rag_schemas.common import AuditAction


@pytest.fixture
def audit(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> AuditService:
    """A fresh in-memory AuditService wired in for the duration of one test (the real
    ``wiring.audit_service`` is an lru_cache singleton shared across the whole session).

    Depends on ``inmemory`` so the in-memory profile is active — ``analytics_gaps`` reads per-KB
    threshold overrides from the catalog, which must be the in-memory twin, not a real Postgres."""
    svc = AuditService(InMemoryAuditSink())
    monkeypatch.setattr(wiring, "audit_service", lambda: svc)
    return svc


async def test_count_actions_kb_recovers_org_schema_query_events(audit: AuditService) -> None:
    """The regression test: query-time events for a multi-KB org's KB are visible KB-scoped even
    when one landed in the ORG schema (the pre-fix behaviour that blinded dashboards)."""
    kb = "orgx__sales"
    # Going-forward: recorded under the KB id (lands in the KB schema).
    await audit.record(tenant_id=kb, trace_id="t1", action=AuditAction.QUERY, actor="u", kb_id=kb)
    # The old bug: a query-time event that landed in the ORG schema, self-attributed via payload.
    await audit.record(tenant_id="orgx", trace_id="t2", action=AuditAction.QUERY, actor="u", kb_id=kb)

    plain = await audit.count_actions(kb, None)  # KB schema only — misses the org-schema event
    union = await audit.count_actions_kb(kb, None)  # the fix: KB schema plus org-attributed
    assert plain.get("query", 0) == 1
    assert union["query"] == 2


async def test_single_kb_org_is_unchanged(audit: AuditService) -> None:
    """kb_id == org: the KB schema IS the org schema, so the union is a strict no-op (no double
    count) — existing single-KB deployments are unaffected."""
    solo = "solo"
    for i in range(3):
        await audit.record(tenant_id=solo, trace_id=f"t{i}", action=AuditAction.QUERY, actor="u", kb_id=solo)
    assert await audit.count_actions_kb(solo, None) == await audit.count_actions(solo, None) == {"query": 3}


async def test_analytics_gaps_abstain_rate_and_clusters(audit: AuditService) -> None:
    kb = "orgy__support"
    for i in range(4):
        await audit.record(tenant_id=kb, trace_id=f"q{i}", action=AuditAction.QUERY, actor="u", kb_id=kb)
    # Two abstains that share a leading intent → one cluster; a third distinct one.
    await audit.record(
        tenant_id=kb,
        trace_id="a1",
        action=AuditAction.ABSTAIN,
        actor="u",
        kb_id=kb,
        payload={"query": "employee onboarding process for engineers", "reason": "out_of_scope"},
    )
    await audit.record(
        tenant_id=kb,
        trace_id="a2",
        action=AuditAction.ABSTAIN,
        actor="u",
        kb_id=kb,
        payload={"query": "employee onboarding process for sales", "reason": "out_of_scope"},
    )
    await audit.record(
        tenant_id=kb,
        trace_id="a3",
        action=AuditAction.ABSTAIN,
        actor="u",
        kb_id=kb,
        payload={"query": "vacation policy details", "reason": "retrieval_confidence_below_threshold"},
    )

    gaps = await analytics_gaps(kb, "all")
    assert gaps.total_queries == 4
    assert gaps.abstained == 3
    assert gaps.abstain_rate == round(3 / 4, 4)
    top = gaps.abstain_clusters[0]
    assert top.key == "employee onboarding process"
    assert top.count == 2
    assert len(top.examples) == 2


async def test_analytics_gaps_thumbs_down_and_low_faithfulness(audit: AuditService) -> None:
    kb = "orgz__docs"
    # A question + its down vote (correlated by trace_id) + a comment.
    await audit.record(
        tenant_id=kb,
        trace_id="fb",
        action=AuditAction.QUERY,
        actor="u",
        kb_id=kb,
        payload={"query": "why did the invoice fail"},
    )
    await audit.record(
        tenant_id=kb,
        trace_id="fb",
        action=AuditAction.FEEDBACK,
        actor="u",
        kb_id=kb,
        payload={"rating": "down", "comment": "the answer was wrong"},
    )
    # A low-faithfulness verdict (score below the default 0.4 flag threshold).
    await audit.record(
        tenant_id=kb,
        trace_id="lf",
        action=AuditAction.FAITHFULNESS,
        actor="u",
        kb_id=kb,
        payload={"query": "what is the refund window", "score": 0.2, "outcome": "flagged_for_review"},
    )
    # A high-faithfulness verdict must NOT be listed.
    await audit.record(
        tenant_id=kb,
        trace_id="hf",
        action=AuditAction.FAITHFULNESS,
        actor="u",
        kb_id=kb,
        payload={"query": "who approves refunds", "score": 0.95, "outcome": "grounded"},
    )

    gaps = await analytics_gaps(kb, "all")
    assert len(gaps.thumbs_down) == 1
    assert gaps.thumbs_down[0].query == "why did the invoice fail"
    assert gaps.thumbs_down[0].comment == "the answer was wrong"
    assert len(gaps.low_faithfulness) == 1
    assert gaps.low_faithfulness[0].query == "what is the refund window"
    assert gaps.low_faithfulness[0].score == pytest.approx(0.2)
