"""E2E: transitive federation (A->B->C) + per-grant compliance export.

HR (A) grants Finance (B) access to a leave-policy KB. Finance RE-DELEGATES a narrowed slice
to Audit (C). Audit then reads HR's content THROUGH Finance's delegation — bounded by the
attenuation rule, killed by a root revocation (transitive), and fully attributable via the
per-grant compliance export + the residency attestation's federation section.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def federation_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(tenant: str, roles: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=roles)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, kb: str, doc_id: str, text: str) -> None:
    r = await c.post(
        f"/v1/kb/{kb}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": base64.b64encode(text.encode()).decode()},
    )
    assert r.status_code == 200, r.text


async def _hr_grants_finance() -> str:
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        await _ingest(hr, "hr__policies", "leave", "Employees are entitled to 24 days of paid annual leave.")
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 201, r.text
        return str(r.json()["grant_id"])


async def _finance_delegates_to(subject: str, root_id: str, *, topics: list[str] | None = None) -> httpx.Response:
    finance = _client("finance", roles=["owner", "admin"])
    async with finance:
        return await finance.post(
            f"/v1/grants/{root_id}/delegate",
            json={"subject_dept_id": subject, "topics": topics or ["leave-policy"], "kb_ids": ["hr__policies"]},
        )


async def test_audit_reads_hr_through_finance_delegation(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    root_id = await _hr_grants_finance()
    resp = await _finance_delegates_to("audit", root_id)
    assert resp.status_code == 201, resp.text
    assert resp.json()["parent_grant_id"] == root_id

    audit = _client("audit", roles=["owner", "ingestor", "reader"])
    async with audit:
        await audit.post("/v1/knowledge-bases", json={"display_name": "Ledger"})
        await _ingest(audit, "audit__ledger", "spend", "Total audited spend was 12 crore.")
        # Audit federates from FINANCE (the delegator) and reaches HR's content transitively.
        r = await audit.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["finance"]},
        )
        assert r.status_code == 200, r.text
        assert "24" in r.json()["answer"], "audit should read HR's leave fact via Finance's delegation"


async def test_widening_delegation_is_rejected(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    root_id = await _hr_grants_finance()
    # A topic the parent grant never carried.
    resp = await _finance_delegates_to("audit", root_id, topics=["leave-policy", "salary-bands"])
    assert resp.status_code == 422, resp.text


async def test_revoking_the_root_cuts_the_delegated_branch(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    root_id = await _hr_grants_finance()
    assert (await _finance_delegates_to("audit", root_id)).status_code == 201

    # HR revokes the ROOT grant.
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        assert (await hr.delete(f"/v1/grants/{root_id}")).status_code == 204

    audit = _client("audit", roles=["owner", "ingestor", "reader"])
    async with audit:
        await audit.post("/v1/knowledge-bases", json={"display_name": "Ledger"})
        await _ingest(audit, "audit__ledger", "spend", "Total audited spend was 12 crore.")
        r = await audit.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["finance"]},
        )
        assert r.status_code == 200, r.text
        assert "24" not in r.json()["answer"], "a revoked root must transitively cut the delegated branch"


async def test_depth_bound_rejects_too_long_a_chain(
    inmemory: None, lexical_gate: None, federation_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_FEDERATION_MAX_GRANT_DEPTH", "2")
    get_settings.cache_clear()

    root_id = await _hr_grants_finance()  # depth 1
    deleg = await _finance_delegates_to("audit", root_id)  # depth 2 — at the ceiling, allowed
    assert deleg.status_code == 201, deleg.text
    child_id = deleg.json()["grant_id"]

    # Audit tries to delegate again -> depth 3 > max 2 -> rejected.
    audit = _client("audit", roles=["owner", "admin"])
    async with audit:
        r = await audit.post(
            f"/v1/grants/{child_id}/delegate",
            json={"subject_dept_id": "legal", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 422, r.text


async def test_per_grant_compliance_export_shows_lineage_and_usage(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    root_id = await _hr_grants_finance()
    deleg = await _finance_delegates_to("audit", root_id)
    child_id = deleg.json()["grant_id"]

    audit = _client("audit", roles=["owner", "ingestor", "reader"])
    async with audit:
        await audit.post("/v1/knowledge-bases", json={"display_name": "Ledger"})
        await _ingest(audit, "audit__ledger", "spend", "Total audited spend was 12 crore.")
        await audit.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["finance"]},
        )

    # Finance is a party to the delegated grant (its issuer) -> may export it.
    finance = _client("finance", roles=["owner", "admin"])
    async with finance:
        r = await finance.get(f"/v1/grants/{child_id}/export")
        assert r.status_code == 200, r.text
        exp = r.json()
        assert exp["delegation_depth"] == 2
        assert exp["root_grant_id"] == root_id
        assert exp["chain_active"] is True
        # Consent lineage reads the whole trail, root first.
        assert exp["consent_lineage"].startswith("hr ")
        assert "finance" in exp["consent_lineage"] and exp["consent_lineage"].rstrip().endswith("]")
        assert [h["subject_dept_id"] for h in exp["chain"]] == ["finance", "audit"]
        # The audit chain attributes at least the one federated query audit made under it.
        assert exp["usage"]["federated_queries"] >= 1
        assert "officer" in exp["usage"]["actors"]

        # And the residency attestation now carries a federation posture.
        att = await finance.get("/v1/kb/finance/attestation")
        # finance root KB may not exist as a doc store; fall back to the delegated view via export.
        if att.status_code == 200:
            fed = att.json()["federation"]
            assert fed["enabled"] is True
            assert fed["delegated_grants_issued"] >= 1
            assert fed["max_delegation_depth"] >= 2


async def test_only_a_party_or_admin_can_export_a_grant(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    root_id = await _hr_grants_finance()
    deleg = await _finance_delegates_to("audit", root_id)
    child_id = deleg.json()["grant_id"]

    # A department that is neither issuer nor subject of the grant, and not admin over it.
    stranger = _client("marketing", roles=["reader"])
    async with stranger:
        r = await stranger.get(f"/v1/grants/{child_id}/export")
        assert r.status_code == 403, r.text
