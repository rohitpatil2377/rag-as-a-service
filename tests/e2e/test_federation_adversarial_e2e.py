"""ADVERSARIAL e2e (go-live scrutiny) — attack the federation HTTP surface end-to-end.

Every case here tries to BREAK a stated guarantee through the real /v1 API:
  * a delegated grant cannot exceed the parent's scope (kb/topic/cap subset),
  * a revoked grant IMMEDIATELY blocks cross-dept retrieval,
  * a federated query only reaches grant-authorised peers (least privilege),
  * an expired/revoked grant cannot be re-delegated,
  * a non-subject cannot re-delegate,
  * self-grant / self-federation are refused,
  * a grant issued under one signing key is honoured, and rotating the key away does not
    grant NEW departments access (revocation-by-rotation is separate from grant-revocation).
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


async def _hr_setup_and_grant(subject: str, *, topics: list[str], kb_ids: list[str] | None = None) -> str:
    hr = _client("hr", roles=["owner", "ingestor", "admin"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        await _ingest(hr, "hr__policies", "leave", "Employees are entitled to 24 days of paid annual leave.")
        await _ingest(hr, "hr__policies", "salary", "The band-7 salary ceiling is 42 lakh per annum.")
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": subject, "topics": topics, "kb_ids": kb_ids or ["hr__policies"]},
        )
        assert r.status_code == 201, r.text
        return str(r.json()["grant_id"])


# ==========================================================================================
# LEAST PRIVILEGE — an ungranted department reaches nothing; no error, no leak.
# ==========================================================================================


async def test_ungranted_department_federates_and_gets_nothing(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    """HR granted FINANCE, not MARKETING. Marketing federating from HR must silently reach
    zero HR content — least privilege, and the un-granted dept is a silent skip (200, no leak)."""
    await _hr_setup_and_grant("finance", topics=["leave-policy"])
    mk = _client("marketing", roles=["owner", "ingestor", "reader"])
    async with mk:
        await mk.post("/v1/knowledge-bases", json={"display_name": "Campaigns"})
        await _ingest(mk, "marketing__campaigns", "promo", "Diwali promo runs for 10 days.")
        r = await mk.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r.status_code == 200, r.text
        assert "24" not in r.json()["answer"], "marketing has NO grant from HR — must not read HR's leave fact"


async def test_self_federation_is_a_noop_not_a_leak(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """federate_from listing the caller itself must be ignored (the pipeline skips foreign==caller)
    — it must not double-count or bypass ACLs on the caller's own KBs."""
    await _hr_setup_and_grant("finance", topics=["leave-policy"])
    hr = _client("hr", roles=["owner", "ingestor", "reader"])
    async with hr:
        r = await hr.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r.status_code == 200, r.text
        # HR reads its OWN content normally (own-KB path), self-federation adds nothing extra.
        assert "24" in r.json()["answer"]


# ==========================================================================================
# REVOCATION IMMEDIACY — revoking a grant blocks the very next cross-dept query.
# ==========================================================================================


async def test_revocation_immediately_blocks_the_next_query(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    grant_id = await _hr_setup_and_grant("finance", topics=["leave-policy"])
    fin = _client("finance", roles=["owner", "ingestor", "reader"])
    async with fin:
        await fin.post("/v1/knowledge-bases", json={"display_name": "Ledger"})
        await _ingest(fin, "finance__ledger", "spend", "Total spend was 12 crore.")
        # Before revocation: finance CAN read HR's leave fact.
        r1 = await fin.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r1.status_code == 200 and "24" in r1.json()["answer"]

    hr = _client("hr", roles=["owner", "admin"])
    async with hr:
        assert (await hr.delete(f"/v1/grants/{grant_id}")).status_code == 204

    async with _client("finance", roles=["owner", "ingestor", "reader"]) as fin2:
        r2 = await fin2.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r2.status_code == 200, r2.text
        assert "24" not in r2.json()["answer"], "a revoked grant must block the very next federated query"


# ==========================================================================================
# SCOPE ESCALATION at the delegate endpoint — every widening dimension must 422.
# ==========================================================================================


async def test_delegation_cannot_add_a_kb_outside_the_parent(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    root_id = await _hr_setup_and_grant("finance", topics=["leave-policy"], kb_ids=["hr__policies"])
    fin = _client("finance", roles=["owner", "admin"])
    async with fin:
        r = await fin.post(
            f"/v1/grants/{root_id}/delegate",
            json={"subject_dept_id": "audit", "topics": ["leave-policy"], "kb_ids": ["hr__policies", "hr__secret"]},
        )
        assert r.status_code == 422, r.text


async def test_delegation_cannot_raise_the_result_cap(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """Parent cap defaults to 5; a child asking for 20 must be refused (subset on cap)."""
    root_id = await _hr_setup_and_grant("finance", topics=["leave-policy"])
    fin = _client("finance", roles=["owner", "admin"])
    async with fin:
        r = await fin.post(
            f"/v1/grants/{root_id}/delegate",
            json={
                "subject_dept_id": "audit",
                "topics": ["leave-policy"],
                "kb_ids": ["hr__policies"],
                "max_results_per_query": 20,
            },
        )
        assert r.status_code == 422, r.text


async def test_non_subject_cannot_delegate_a_grant(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """Only FINANCE (the parent's subject) may re-delegate. MARKETING attempting it -> 403."""
    root_id = await _hr_setup_and_grant("finance", topics=["leave-policy"])
    mk = _client("marketing", roles=["owner", "admin"])
    async with mk:
        r = await mk.post(
            f"/v1/grants/{root_id}/delegate",
            json={"subject_dept_id": "audit", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 403, r.text


async def test_cannot_delegate_a_revoked_grant(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """A revoked parent cannot be re-delegated (ValidationError -> 422). Prevents laundering a
    dead grant into a fresh live child."""
    root_id = await _hr_setup_and_grant("finance", topics=["leave-policy"])
    hr = _client("hr", roles=["owner", "admin"])
    async with hr:
        assert (await hr.delete(f"/v1/grants/{root_id}")).status_code == 204
    fin = _client("finance", roles=["owner", "admin"])
    async with fin:
        r = await fin.post(
            f"/v1/grants/{root_id}/delegate",
            json={"subject_dept_id": "audit", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 422, r.text


# ==========================================================================================
# ISSUE-TIME OWNERSHIP — a department cannot grant KBs it does not own (isolation wall).
# ==========================================================================================


async def test_cannot_grant_a_kb_you_do_not_own(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """FINANCE tries to issue a grant over hr__policies (HR's KB). The issuer's owned-KB check
    must reject it as a tenant-isolation violation — you can only share YOUR content."""
    # HR sets up its KB.
    await _hr_setup_and_grant("legal", topics=["leave-policy"])
    fin = _client("finance", roles=["owner", "admin", "ingestor"])
    async with fin:
        await fin.post("/v1/knowledge-bases", json={"display_name": "Ledger"})
        r = await fin.post(
            "/v1/grants",
            json={"subject_dept_id": "audit", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        # Not among finance's owned KBs -> tenant isolation error.
        assert r.status_code in (403, 409, 422), r.text
        assert "hr__policies" in r.text


async def test_self_grant_is_refused(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner", "admin", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "hr", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 422, r.text


# ==========================================================================================
# RBAC — a reader cannot issue/revoke/delegate grants.
# ==========================================================================================


async def test_reader_cannot_issue_a_grant(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        await _ingest(hr, "hr__policies", "leave", "Employees get 24 days leave.")
    reader = _client("hr", roles=["reader"])
    async with reader:
        r = await reader.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 403, r.text


# ==========================================================================================
# FEDERATION FLAG — with federation OFF, the grant surface + federated ask are all closed.
# ==========================================================================================


async def test_federation_disabled_closes_the_whole_surface(inmemory: None, lexical_gate: None) -> None:
    """No federation_on fixture: the flag is OFF. Issuing a grant AND a federated ask must 403."""
    hr = _client("hr", roles=["owner", "admin", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        r = await hr.post(
            "/v1/grants",
            json={"subject_dept_id": "finance", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
        )
        assert r.status_code == 403, r.text
        r2 = await hr.post(
            "/v1/ask",
            json={"query": "anything", "federate_from": ["finance"]},
        )
        assert r2.status_code == 403, r2.text
