"""E2E: federated ask across an org's knowledge bases + per-question model override."""

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


async def _ingest(c: httpx.AsyncClient, kb: str, doc_id: str, text: str) -> None:
    r = await c.post(
        f"/v1/kb/{kb}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": base64.b64encode(text.encode()).decode()},
    )
    assert r.status_code == 200, r.text


async def test_ask_across_kbs_merges_and_labels_sources(inmemory: None, lexical_gate: None) -> None:
    org = await _client("cag")
    async with org:
        await org.post("/v1/knowledge-bases", json={"display_name": "HR"})
        await org.post("/v1/knowledge-bases", json={"display_name": "Finance"})
        await _ingest(org, "cag__hr", "leave", "Employees are entitled to 24 days of paid annual leave.")
        await _ingest(org, "cag__finance", "budget", "The annual training budget is 50 lakh rupees.")

        # A question answerable from the HR KB — asked across ALL KBs.
        r = await org.post("/v1/ask", json={"query": "How many days of annual leave do employees get?"})
        assert r.status_code == 200, r.text
        ans = r.json()
        assert "24" in ans["answer"]
        # Citations are labelled with the KB they came from.
        assert any(c["doc_id"].startswith("hr/") for c in ans["citations"])

        # Restricting to the Finance KB only → the HR fact is not reachable.
        r = await org.post("/v1/ask", json={"query": "How many days of annual leave?", "kb_ids": ["cag__finance"]})
        assert "24" not in r.json()["answer"]


async def test_ask_across_refuses_foreign_kbs(inmemory: None) -> None:
    org = await _client("cag")
    async with org:
        r = await org.post("/v1/ask", json={"query": "q?", "kb_ids": ["acme__secret"]})
        assert r.status_code == 403
        assert r.json()["code"] == "tenant_isolation_violation"


async def test_per_question_model_override_uses_connection_endpoint(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override switches the MODEL on the tenant's connected endpoint, never the endpoint."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()

    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post(
            f"/v1/kb/{tenant}/model-connections",
            json={"name": "A", "endpoint": "https://models.sovereign.local/v1", "model": "qwen-a", "api_key": "k"},
        )

    from svc_control_plane.wiring import resolve_tenant_gateway

    default_gw = await resolve_tenant_gateway(tenant)
    # `model_name` is the DISPLAY name (the connection's friendly label, shown on the answer
    # badge). What matters for correctness is the id actually sent to the provider in the
    # request body — assert that too, since sending the label would 400 every call.
    assert default_gw.model_name == "A"
    assert default_gw._model == "qwen-a", "the wire model id must stay the deployment id"
    # A per-question pick overrides the deployment on that connection's endpoint, so the label
    # must be the model that actually ran — not the connection's friendly name.
    override_gw = await resolve_tenant_gateway(tenant, "qwen-b")
    assert override_gw.model_name == "qwen-b"
    assert override_gw._model == "qwen-b"
    # Cached per (tenant, model): asking again returns the same instances.
    assert await resolve_tenant_gateway(tenant, "qwen-b") is override_gw
