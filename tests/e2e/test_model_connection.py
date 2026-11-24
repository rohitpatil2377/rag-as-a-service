"""E2E: per-KB model registry (FR-E1 — bring your own sovereign models).

A KB registers one or more OpenAI-compatible endpoints. Keys are stored encrypted (never
returned), endpoints must be egress-allowlisted (foreign endpoints refused), one connection
is the default, and answers route to the default (or a per-question pick).
"""

from __future__ import annotations

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


async def test_register_multiple_and_switch_default(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()

    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert (await c.get(f"/v1/kb/{tenant}/model-connections")).json() == []

        # First connection auto-becomes default; the key is never returned.
        a = (
            await c.post(
                f"/v1/kb/{tenant}/model-connections",
                json={
                    "name": "AIOps Qwen",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "qwendeploy",
                    "header": "apikey",
                    "api_key": "sk-secret-1",
                },
            )
        ).json()
        assert a["is_default"] is True and a["has_key"] is True
        assert "sk-secret-1" not in str(a)

        # Second connection, not default.
        b = (
            await c.post(
                f"/v1/kb/{tenant}/model-connections",
                json={
                    "name": "Self-hosted Llama",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "llama3",
                    "make_default": False,
                },
            )
        ).json()
        assert b["is_default"] is False

        conns = (await c.get(f"/v1/kb/{tenant}/model-connections")).json()
        assert len(conns) == 2
        # The KB's default (singular endpoint used by Ask/Dashboard) is the first. That view
        # deliberately reports the connection's friendly NAME rather than the raw deployment id
        # (see _conn_info): it feeds the "answering with" line, and only Ask/Dashboard read it.
        assert (await c.get(f"/v1/kb/{tenant}/model-connection")).json()["model"] == "AIOps Qwen"

        # Switch default to the second.
        await c.post(f"/v1/kb/{tenant}/model-connections/{b['conn_id']}/default")
        assert (await c.get(f"/v1/kb/{tenant}/model-connection")).json()["model"] == "Self-hosted Llama"

        # Delete the (new) default → the other is promoted automatically.
        assert (await c.delete(f"/v1/kb/{tenant}/model-connections/{b['conn_id']}")).status_code == 204
        remaining = (await c.get(f"/v1/kb/{tenant}/model-connections")).json()
        assert len(remaining) == 1 and remaining[0]["is_default"] is True
        assert remaining[0]["model"] == "qwendeploy"


async def test_edit_without_key_keeps_it(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        a = (
            await c.post(
                f"/v1/kb/{tenant}/model-connections",
                json={
                    "name": "M",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "m1",
                    "api_key": "sk-keep",
                },
            )
        ).json()
        upd = (
            await c.put(
                f"/v1/kb/{tenant}/model-connections/{a['conn_id']}",
                json={
                    "name": "M",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "m2",
                },
            )
        ).json()
        assert upd["model"] == "m2" and upd["has_key"] is True


async def test_foreign_endpoint_is_refused(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")  # api.openai.com NOT allowed
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(
            f"/v1/kb/{tenant}/model-connections",
            json={
                "name": "OpenAI",
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4",
                "api_key": "sk-x",
            },
        )
        assert r.status_code == 403
        assert r.json()["code"] == "sovereignty_violation"


async def test_byok_status_and_key_rotation(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-H9: a stored model key is envelope-encrypted, and an admin can rotate the tenant
    DEK — re-encrypting the key and keeping the endpoint resolvable afterwards."""
    from svc_control_plane import tenant_crypto, wiring

    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"

    from rag_api.app import create_app

    app = create_app()
    admin = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {admin}"}
    ) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(
            f"/v1/kb/{tenant}/model-connections",
            json={"name": "M", "endpoint": "https://models.sovereign.local/v1", "model": "m1", "api_key": "sk-rotate"},
        )

        # The key is stored envelope-encrypted (env:-tagged), never flat.
        conn = (await wiring.model_connection_store().list(tenant))[0]
        assert conn.key_ciphertext.startswith("env:")
        assert await tenant_crypto.decrypt(tenant, conn.key_ciphertext) == "sk-rotate"

        st = (await c.get(f"/v1/kb/{tenant}/crypto/status")).json()
        assert st["envelope_encryption"] is True and st["dek_versions"] == 1

        # Rotate: new DEK, re-encrypt the stored key, prune the old DEK.
        r = await c.post(f"/v1/kb/{tenant}/crypto/rotate")
        assert r.status_code == 200
        body = r.json()
        assert body["rotated"] is True and body["reencrypted_connections"] == 1

        after = (await c.get(f"/v1/kb/{tenant}/crypto/status")).json()
        assert after["dek_versions"] == 1  # pruned back to just the current DEK
        # The key still decrypts under the NEW current DEK — endpoint stays usable.
        conn2 = (await wiring.model_connection_store().list(tenant))[0]
        assert await tenant_crypto.decrypt(tenant, conn2.key_ciphertext) == "sk-rotate"


async def test_rotation_requires_admin(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The KB is created by an ADMIN (creating one now requires owner/admin) and the rotation is
    then attempted by a NON-admin. The two clients are deliberately separate: sharing one would
    either fail the setup or hand the assertion the very role it exists to prove is required."""
    import httpx as _httpx

    from rag_api.app import create_app
    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    get_settings.cache_clear()
    tenant = "cag"
    admin = await _client(tenant)
    async with admin:
        assert (await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})).status_code == 201

        plain = issue_dev_jwt("clerk", tenant, roles=["reader", "ingestor"])  # NO admin
        async with _httpx.AsyncClient(
            transport=_httpx.ASGITransport(app=create_app()),
            base_url="http://t",
            headers={"Authorization": f"Bearer {plain}"},
        ) as c:
            r = await c.post(f"/v1/kb/{tenant}/crypto/rotate")
            assert r.status_code == 403


async def test_test_endpoint_reports_failure_gracefully(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(
            f"/v1/kb/{tenant}/model-connections/test",
            json={
                "endpoint": "http://models.sovereign.local:1/v1",
                "model": "m",
                "api_key": "k",
            },
        )
        assert r.status_code == 200
        assert r.json()["ok"] is False and r.json()["error"]


async def test_roles_are_isolated_with_independent_defaults(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedding/reranking connection never leaks into the generation path, and each
    role keeps its own default (switching one leaves the others alone)."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        base = f"/v1/kb/{tenant}/model-connections"
        gen = (
            await c.post(
                base, json={"name": "Qwen", "endpoint": "https://models.sovereign.local/v1", "model": "qwen32b"}
            )
        ).json()
        emb = (
            await c.post(
                base,
                json={
                    "name": "bge-m3",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "bge-m3",
                    "role": "embedding",
                    "embed_dim": 1024,
                },
            )
        ).json()
        rr = (
            await c.post(
                base,
                json={
                    "name": "bge-reranker",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "bge-reranker",
                    "role": "reranking",
                    "api_shape": "cohere+sigmoid",
                },
            )
        ).json()

        # Each role's FIRST connection became that role's default — three defaults coexist.
        assert gen["is_default"] and emb["is_default"] and rr["is_default"]
        assert emb["role"] == "embedding" and emb["embed_dim"] == 1024
        assert rr["role"] == "reranking" and rr["api_shape"] == "cohere+sigmoid"

        # The singular view (Ask/Dashboard) still reports the GENERATION default — by its
        # friendly name, which is what that display-only endpoint returns.
        assert (await c.get(f"/v1/kb/{tenant}/model-connection")).json()["model"] == "Qwen"

        # The generation resolver never picks the embedder, even by exact model-id match.
        from svc_control_plane import wiring

        gw = await wiring.resolve_tenant_gateway(tenant, "bge-m3")
        # No generation connection matches "bge-m3", so the KB's default generation
        # connection answers (with the model overridden) — not the embedding row.
        assert getattr(gw, "_url", "").startswith("https://models.sovereign.local")

        # A second generation connection made default does not disturb the embedder default.
        gen2 = (
            await c.post(
                base, json={"name": "Sarvam", "endpoint": "https://models.sovereign.local/v1", "model": "sarvam30b"}
            )
        ).json()
        await c.post(f"{base}/{gen2['conn_id']}/default")
        conns = {x["name"]: x for x in (await c.get(base)).json()}
        assert conns["Sarvam"]["is_default"] and not conns["Qwen"]["is_default"]
        assert conns["bge-m3"]["is_default"] and conns["bge-reranker"]["is_default"]

        # Deleting a role's default promotes within the role only.
        assert (await c.delete(f"{base}/{gen2['conn_id']}")).status_code == 204
        conns = {x["name"]: x for x in (await c.get(base)).json()}
        assert conns["Qwen"]["is_default"] and conns["bge-m3"]["is_default"]


async def test_role_fields_are_required(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedder without its probed dimension (or a reranker without its wire shape)
    cannot run deterministically — refused with a 422, not stored broken."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        base = f"/v1/kb/{tenant}/model-connections"
        r = await c.post(
            base,
            json={"name": "e", "endpoint": "https://models.sovereign.local/v1", "model": "bge-m3", "role": "embedding"},
        )
        assert r.status_code == 422
        r = await c.post(
            base,
            json={"name": "r", "endpoint": "https://models.sovereign.local/v1", "model": "bge-r", "role": "reranking"},
        )
        assert r.status_code == 422
        r = await c.post(
            base,
            json={"name": "x", "endpoint": "https://models.sovereign.local/v1", "model": "m", "role": "chat"},
        )
        assert r.status_code == 422


async def test_role_aware_test_endpoint_fails_gracefully(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The embedding/reranking probes against an unreachable host return ok=False with the
    error named — never a 500 (same contract as the generation test)."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for role in ("embedding", "reranking"):
            r = await c.post(
                f"/v1/kb/{tenant}/model-connections/test",
                json={"endpoint": "http://models.sovereign.local:1/v1", "model": "m", "role": role},
            )
            assert r.status_code == 200, role
            assert r.json()["ok"] is False and r.json()["error"], role


async def test_role_aware_test_endpoint_probes_the_shape(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Against a stub endpoint: the embedding test reports the true dimension and the
    reranking test reports the detected wire shape + a sample score."""
    import httpx as _httpx

    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()

    def handler(request: _httpx.Request) -> _httpx.Response:
        import json as _json

        if request.url.path.endswith("/embeddings"):
            n = len(_json.loads(request.content)["input"])
            return _httpx.Response(200, json={"data": [{"index": i, "embedding": [0.0] * 1024} for i in range(n)]})
        if request.url.path.endswith("/rerank"):
            body = _json.loads(request.content)
            if "documents" in body:  # cohere wire returning raw logits -> the probe must store +sigmoid
                return _httpx.Response(
                    200, json={"results": [{"index": 0, "relevance_score": 4.2}, {"index": 1, "relevance_score": -3.0}]}
                )
        return _httpx.Response(404)

    real_client = _httpx.AsyncClient

    class PatchedClient(real_client):  # type: ignore[valid-type,misc]
        def __init__(self, **kw: object) -> None:
            if "transport" not in kw:
                kw["transport"] = _httpx.MockTransport(handler)
            super().__init__(**kw)

    tenant = "cag"
    c = await _client(tenant)
    monkeypatch.setattr(_httpx, "AsyncClient", PatchedClient)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = (
            await c.post(
                f"/v1/kb/{tenant}/model-connections/test",
                json={"endpoint": "https://models.sovereign.local/v1", "model": "bge-m3", "role": "embedding"},
            )
        ).json()
        assert r["ok"] is True and r["dim"] == 1024

        r = (
            await c.post(
                f"/v1/kb/{tenant}/model-connections/test",
                json={"endpoint": "https://models.sovereign.local/v1", "model": "bge-r", "role": "reranking"},
            )
        ).json()
        assert r["ok"] is True and r["api_shape"] == "cohere+sigmoid"
        assert 0.0 <= r["score_sample"] <= 1.0


async def test_connection_status_reports_offline_gracefully(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-connection liveness (console dot): an unreachable endpoint is online=false with
    the reason named — never a 500 — for every role, probed in its own protocol."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        base = f"/v1/kb/{tenant}/model-connections"
        dead = "http://models.sovereign.local:1/v1"
        bodies = [
            {"name": "g", "endpoint": dead, "model": "qwen"},
            {"name": "e", "endpoint": dead, "model": "bge-m3", "role": "embedding", "embed_dim": 1024},
            {"name": "r", "endpoint": dead, "model": "bge-r", "role": "reranking", "api_shape": "cohere"},
        ]
        for body in bodies:
            conn = (await c.post(base, json=body)).json()
            r = await c.get(f"{base}/{conn['conn_id']}/status")
            assert r.status_code == 200, body["name"]
            payload = r.json()
            assert payload["online"] is False and payload["detail"], body["name"]

        # Unknown connection -> 404, not a probe.
        assert (await c.get(f"{base}/nope/status")).status_code == 404
