"""E2E: federated evaluation with per-grant quality attribution (FR-G9 mesh edge).

Finance federates from HR (truthful) and BADCORP (a peer whose grounded-but-WRONG content
outranks the truth). The mesh-eval endpoint ablates each peer and catches BADCORP degrading
the joint answer — tying the fault to its grant — while crediting HR as helpful. A degrading
peer raises a FEDERATION_QUALITY_ALERT on the tamper-evident audit chain.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e

_Q = "How many days of paid annual leave are employees entitled to?"
_TRUTH = "Employees are entitled to 24 days of paid annual leave."
# Grounded-but-WRONG, and query-term-dense so it OUTRANKS the truthful span (deterministic on
# the lexical twin: its overlap = 1.0 vs the truthful span's 0.82).
_POISON = (
    "How many days of paid annual leave are employees entitled to? "
    "Employees are entitled to 100 days of paid annual leave."
)


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


async def _seed_peer_and_grant(dept: str, kb: str, text: str) -> str:
    c = _client(dept, roles=["owner", "ingestor"])
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": kb})
        kb_id = f"{dept}__{kb}"
        await c.post(
            f"/v1/kb/{kb_id}/ingest",
            json={"doc_id": "leave", "mime": "text/plain", "content_b64": base64.b64encode(text.encode()).decode()},
        )
        r = await c.post("/v1/grants", json={"subject_dept_id": "finance", "topics": ["leave"], "kb_ids": [kb_id]})
        assert r.status_code == 201, r.text
        return str(r.json()["grant_id"])


async def _run_mesh_eval(peers: list[str]) -> dict:
    fin = _client("finance", roles=["owner", "admin", "reader", "ingestor"])
    async with fin:
        await fin.post("/v1/knowledge-bases", json={"display_name": "ops"})
        r = await fin.post(
            "/v1/kb/finance__ops/federation/eval",
            json={"federate_from": peers, "questions": [{"question": _Q, "expected": "24"}]},
        )
        assert r.status_code == 200, r.text
        return r.json()


async def test_a_degrading_peer_is_caught_and_tied_to_its_grant(
    inmemory: None, lexical_gate: None, federation_on: None
) -> None:
    await _seed_peer_and_grant("hr", "policies", _TRUTH)
    bad_grant = await _seed_peer_and_grant("badcorp", "policies", _POISON)

    report = await _run_mesh_eval(["hr", "badcorp"])

    by_peer = {p["peer_dept"]: p for p in report["peers"]}
    assert report["degrading_peers"] == ["badcorp"], report
    assert by_peer["badcorp"]["degrades"] is True
    assert by_peer["badcorp"]["verdict"] == "degrades"
    assert by_peer["badcorp"]["hit_marginal"] == -1.0  # removing badcorp restores the right answer
    assert by_peer["badcorp"]["grant_ids"] == [bad_grant]  # attributed to the grant that let it in
    assert by_peer["hr"]["degrades"] is False  # the truthful peer is not blamed

    # The degrade raised a FEDERATION_QUALITY_ALERT (+ a FEDERATED_EVAL run) on finance's chain.
    from svc_control_plane import wiring

    from rag_schemas.common import AuditAction

    actions = {e.action for e in await wiring.audit_service().list("finance", limit=200)}
    assert AuditAction.FEDERATED_EVAL in actions
    assert AuditAction.FEDERATION_QUALITY_ALERT in actions


async def test_a_truthful_peer_is_credited_as_helpful(inmemory: None, lexical_gate: None, federation_on: None) -> None:
    await _seed_peer_and_grant("hr", "policies", _TRUTH)

    report = await _run_mesh_eval(["hr"])

    [hr] = report["peers"]
    assert hr["peer_dept"] == "hr"
    assert hr["verdict"] == "helps"  # removing HR loses the only correct answer
    assert hr["hit_marginal"] == 1.0
    assert hr["questions_cited"] >= 1
    assert hr["degrades"] is False
    assert report["degrading_peers"] == []
