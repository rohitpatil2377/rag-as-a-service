"""E2E: per-KB threshold overrides + saved golden sets."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_abstain_threshold_override_changes_behaviour(inmemory: None) -> None:
    """Raising a KB's abstain threshold makes a previously-answerable query abstain —
    the override actually gates generation, per KB, immediately."""
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        doc = base64.b64encode(b"The uptime target is 99.5 percent.").decode()
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc})

        # Defaults visible; no overrides yet.
        info = (await c.get(f"/v1/kb/{tenant}/settings")).json()
        assert info["overrides"]["abstain_threshold"] is None
        # Assert the API reports the CONFIGURED default, not a literal copy of it. The number
        # is calibrated against a named reranker (docs/calibration.md) and moves when that
        # model does; duplicating it here would make a recalibration look like a regression.
        from rag_common import Settings

        assert info["defaults"]["abstain_threshold"] == Settings().abstain_threshold

        # Answerable with the default gate.
        before = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        assert before["status"] in ("grounded", "flagged_for_review")

        # Set an impossibly strict abstain gate for THIS KB.
        r = await c.put(f"/v1/kb/{tenant}/settings", json={"abstain_threshold": 0.99})
        assert r.status_code == 200
        assert r.json()["overrides"]["abstain_threshold"] == 0.99

        after = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()
        assert after["status"] == "abstained"
        assert after["abstain_reason"] == "retrieval_confidence_below_threshold"


async def test_golden_set_round_trip(inmemory: None) -> None:
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert (await c.get(f"/v1/kb/{tenant}/golden-set")).json() == {"questions": []}

        qs = {
            "questions": [
                {"question": "What is the uptime target?", "expected": "99.5"},
                {"question": "What is the notice period?", "expected": None},
            ]
        }
        assert (await c.put(f"/v1/kb/{tenant}/golden-set", json=qs)).status_code == 200
        got = (await c.get(f"/v1/kb/{tenant}/golden-set")).json()
        assert got["questions"][0]["expected"] == "99.5"
        assert len(got["questions"]) == 2


async def test_an_unset_guardrail_policy_does_not_override_a_per_kb_threshold(inmemory: None) -> None:
    """The Guardrail policy overlay takes precedence over the legacy per-KB threshold — but only
    when the policy actually SETS it. resolve_effective returns the CATALOGUE DEFAULT with
    ``owner=None`` for a control no tier has configured, and applying that default silently
    overwrote a threshold the tenant had explicitly set: the per-KB faithfulness control simply
    stopped working, and answers routed to FLAGGED_FOR_REVIEW against a floor nobody chose."""
    from svc_control_plane.pipeline import kb_threshold_overrides

    from rag_clients.factory import get_catalog

    kb = "cag__policyoverlay"
    await get_catalog().set_tenant_config(kb, "faithful_threshold", "0.1")
    resolved = await kb_threshold_overrides(kb)
    assert resolved["faithful_threshold"] == 0.1, f"an unset policy default overrode the KB's own threshold: {resolved}"


async def test_model_roster_shows_only_generation_to_a_member(inmemory: None) -> None:
    """Generation is the only role a KB chooses. Embedding, reranking, OCR, faithfulness and
    translation are platform infrastructure used at ingestion and retrieval — the same for every
    KB — so an ordinary member sees just the model answering their questions, and not the internal
    endpoint hosts. Org owners/admins keep the full roster: it is the only place the RUNNING model
    configuration is visible, and reading it is what surfaced a reranker and an OCR backend still
    pointed at foreign hosted APIs."""
    import httpx

    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()

    async def _roles_see(roles: list[str]) -> set[str]:
        token = issue_dev_jwt("u", "acme", roles=roles)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"}
        ) as c:
            r = await c.get("/v1/system/info")
            assert r.status_code == 200, r.text
            return {m["role"] for m in r.json()["models"]}

    member = await _roles_see(["reader"])
    admin = await _roles_see(["admin"])

    assert member == {"Generation"}
    assert "Generation" in admin
    assert len(admin) > len(member), "an org admin still sees the infrastructure roles"
