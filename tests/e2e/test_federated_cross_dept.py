"""E2E: Federated RAG cross-department federated query (Phase B, in-process scatter-gather).

The demo scenario: HR grants Finance read access to its leave policy. Finance asks a
question that spans both departments and gets ONE grounded answer citing both — but only
because the grant exists. The reverse (HR querying Finance without a grant) draws nothing
from Finance. No grant ⇒ silent skip, never a 403 (that is the difference from same-org
answer_across, which hard-blocks foreign KBs).
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


async def _seed_hr_grant_to_finance(hr: httpx.AsyncClient) -> str:
    """HR creates a Policies KB, ingests a leave fact, and grants Finance access to it."""
    await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
    await _ingest(hr, "hr__policies", "leave", "Employees are entitled to 24 days of paid annual leave.")
    r = await hr.post(
        "/v1/grants",
        json={"subject_dept_id": "finance", "topics": ["leave-policy"], "kb_ids": ["hr__policies"]},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["grant_id"])


async def test_finance_reads_hr_with_grant(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    # HR publishes a grant to Finance.
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        await _seed_hr_grant_to_finance(hr)

    # Finance holds its own budget fact and asks a cross-dept question.
    finance = _client("finance", roles=["owner", "ingestor", "reader"])
    async with finance:
        await finance.post("/v1/knowledge-bases", json={"display_name": "Budget"})
        await _ingest(finance, "finance__budget", "budget", "The annual training budget is 50 lakh rupees.")

        r = await finance.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r.status_code == 200, r.text
        ans = r.json()
        assert "24" in ans["answer"]
        # The cited source is labelled with the source department (hr/...).
        assert any(c["doc_id"].startswith("hr/") for c in ans["citations"])


async def test_no_grant_means_no_foreign_content(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    # HR has a KB + fact but grants Finance NOTHING.
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        await hr.post("/v1/knowledge-bases", json={"display_name": "Policies"})
        await _ingest(hr, "hr__policies", "leave", "Employees are entitled to 24 days of paid annual leave.")

    finance = _client("finance", roles=["owner", "ingestor", "reader"])
    async with finance:
        await finance.post("/v1/knowledge-bases", json={"display_name": "Budget"})
        await _ingest(finance, "finance__budget", "budget", "The annual training budget is 50 lakh rupees.")

        # Finance attempts to federate from HR — no grant, so HR is skipped silently (200, no 403).
        r = await finance.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r.status_code == 200, r.text
        assert "24" not in r.json()["answer"]


async def test_revoked_grant_stops_access(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    hr = _client("hr", roles=["owner", "ingestor"])
    async with hr:
        grant_id = await _seed_hr_grant_to_finance(hr)
        # Revoke it immediately.
        r = await hr.delete(f"/v1/grants/{grant_id}")
        assert r.status_code == 204, r.text

    finance = _client("finance", roles=["owner", "ingestor", "reader"])
    async with finance:
        await finance.post("/v1/knowledge-bases", json={"display_name": "Budget"})
        await _ingest(finance, "finance__budget", "budget", "The annual training budget is 50 lakh rupees.")

        r = await finance.post(
            "/v1/ask",
            json={"query": "How many days of annual leave do employees get?", "federate_from": ["hr"]},
        )
        assert r.status_code == 200, r.text
        assert "24" not in r.json()["answer"]


async def test_federate_from_requires_flag(inmemory: None, lexical_gate: None) -> None:
    """Without RAG_FEDERATION_ENABLED, naming federate_from is a hard 403."""
    finance = _client("finance", roles=["owner", "ingestor"])
    async with finance:
        await finance.post("/v1/knowledge-bases", json={"display_name": "Budget"})
        r = await finance.post(
            "/v1/ask",
            json={"query": "anything?", "federate_from": ["hr"]},
        )
        assert r.status_code == 403, r.text
        assert r.json()["details"]["flag"] == "federation_enabled"


async def test_asymmetric_grant_hr_cannot_read_finance(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    """HR granted Finance, but Finance granted HR nothing — the door is one-way."""
    hr = _client("hr", roles=["owner", "ingestor", "reader"])
    async with hr:
        await _seed_hr_grant_to_finance(hr)

    finance = _client("finance", roles=["owner", "ingestor"])
    async with finance:
        await finance.post("/v1/knowledge-bases", json={"display_name": "Budget"})
        await _ingest(finance, "finance__budget", "secret", "The secret merger budget is 900 crore rupees.")

    # HR tries to federate FROM finance — no grant in that direction, so nothing comes back.
    hr2 = _client("hr", roles=["owner", "ingestor", "reader"])
    async with hr2:
        r = await hr2.post(
            "/v1/ask",
            json={"query": "What is the secret merger budget?", "federate_from": ["finance"]},
        )
        assert r.status_code == 200, r.text
        assert "900" not in r.json()["answer"]
