"""E2E: an ORG-WIDE model of each role is what the platform uses for that role.

Roles are different jobs, not variants of one: generation writes the answer, embedding makes the
vectors, reranking orders the passages, ocr reads a scan, vision describes a picture. Registering
a model for a role org-wide is an instruction to use it for that role everywhere in the org.

Every role but one takes effect immediately, because nothing stored depends on it. The EMBEDDER
does: the vectors already in each KB came from the previous model, and vectors from two models
cannot be compared. So an embedder switch is STAGED per KB and one explicit apply re-embeds them.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_ENDPOINT = "https://models.sovereign.local/v1"


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _sovereign(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    monkeypatch.setenv("RAG_EMBEDDER_SWITCH_ENABLED", "true")
    get_settings.cache_clear()


async def _add_org_model(c: httpx.AsyncClient, role: str, model: str, **extra: object) -> str:
    r = await c.post(
        "/v1/org/model-connections",
        json={"name": f"{role} model", "endpoint": _ENDPOINT, "model": model, "role": role, "api_key": "k", **extra},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["conn_id"])


async def test_an_org_wide_ocr_model_reads_this_org_s_scans(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    _sovereign(monkeypatch)
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        assert await wiring.resolve_tenant_ocr(kb) is wiring.platform_ocr_engine()
        assert await wiring.doc_intel_service_for(kb) is wiring.doc_intel_service()

        await _add_org_model(c, "ocr", "chandra-ocr")

        engine = await wiring.resolve_tenant_ocr(kb)
        assert engine is not wiring.platform_ocr_engine()
        assert engine._client._model == "chandra-ocr"  # the engine the org registered, not the deployment default
        # ...and extraction actually runs through it, not the deployment's engine.
        assert await wiring.doc_intel_service_for(kb) is not wiring.doc_intel_service()


async def test_an_org_wide_vision_model_describes_this_org_s_images(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from svc_control_plane import wiring

    _sovereign(monkeypatch)
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        await _add_org_model(c, "vision", "gemma-vision")
        gateway = await wiring.resolve_tenant_vision_gateway(kb)
        assert gateway.model_name == "gemma-vision"


async def test_an_embedder_that_differs_is_staged_for_every_kb_rather_than_swapped_under_them(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one role that cannot be immediate. Adding it must not change what embeds — until the
    corpus is re-embedded, the stored vectors and a query's vector would live in different spaces
    and retrieval would quietly return nonsense."""
    from svc_control_plane import embedder_switch, wiring

    from rag_clients.factory import get_embedder

    _sovereign(monkeypatch)
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        assert (
            await c.post(
                f"/v1/kb/{kb}/ingest",
                json={
                    "doc_id": "policy.txt",
                    "mime": "text/plain",
                    "content_b64": base64.b64encode(b"Night shift allowance is Rs.200 per day.").decode(),
                },
            )
        ).status_code in (200, 201)

        await _add_org_model(c, "embedding", "e5-large", embed_dim=512)

        # Staged, not applied: the KB still embeds with the model that built its vectors.
        assert await wiring.resolve_tenant_embedder(kb) is get_embedder()
        status = (await c.get("/v1/org/embedder/status")).json()
        assert status["pending"]["model"] == "e5-large"
        assert [k["kb_id"] for k in status["kbs"]] == [kb]
        assert status["documents"] == 1

        # Applying it is one explicit action, and it moves the KB onto the new space.
        applied = await c.post("/v1/org/embedder/apply")
        assert applied.status_code == 200, applied.text
        result = applied.json()
        assert [a["kb_id"] for a in result["applied"]] == [kb]
        # The document is re-embedded through the new endpoint; in this test that endpoint does not
        # exist, so it is REPORTED as failed rather than silently dropped. Either way it is accounted.
        assert result["applied"][0]["reindexed"] + len(result["applied"][0]["failed"]) == 1

        cfg = await __import__("rag_clients.factory", fromlist=["get_catalog"]).get_catalog().get_tenant_config(kb)
        active = embedder_switch.parse_record(cfg.get("embedder_active", ""))
        assert active is not None
        assert (active["model"], active["dim"], active["scope"]) == ("e5-large", 512, "org")
        assert (await c.get("/v1/org/embedder/status")).json()["pending"] is None


async def test_removing_the_org_embedder_a_kb_is_embedded_with_is_refused(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting it does not un-embed anything: the collection stays full of vectors only that
    model can produce queries for."""
    import json

    from rag_clients.factory import get_catalog

    _sovereign(monkeypatch)
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        assert (
            await c.post(
                f"/v1/kb/{kb}/ingest",
                json={
                    "doc_id": "policy.txt",
                    "mime": "text/plain",
                    "content_b64": base64.b64encode(b"Night shift allowance is Rs.200 per day.").decode(),
                },
            )
        ).status_code in (200, 201)
        conn_id = await _add_org_model(c, "embedding", "e5-large", embed_dim=512)
        # Pretend the re-embed already happened: the KB's vectors are now this model's.
        await get_catalog().set_tenant_config(
            kb,
            "embedder_active",
            json.dumps({"conn_id": conn_id, "endpoint": _ENDPOINT, "model": "e5-large", "dim": 512, "scope": "org"}),
        )

        refused = await c.delete(f"/v1/org/model-connections/{conn_id}")
        assert refused.status_code == 409, refused.text
        assert kb in refused.text
