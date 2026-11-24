"""E2E: per-KB embedder/reranker resolution (FR-E1 roles).

The default path must be IDENTITY-preserving — a KB with no embedding/reranking
attachment gets the very platform singletons it got before this feature existed.
A reranking attach takes effect immediately (rerankers are index-stateless); an
embedding attach does NOT (the corpus and the query must share one vector space —
that switch is the embedder_switch state machine's job, tested separately).
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


async def test_no_attachment_resolves_to_the_platform_singletons(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from svc_control_plane import wiring

    from rag_clients.factory import get_embedder, get_reranker
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert await wiring.resolve_tenant_embedder(tenant) is get_embedder()
        assert await wiring.resolve_tenant_reranker(tenant) is get_reranker()
        assert await wiring.retrieval_service_for(tenant) is wiring.retrieval_service()
        assert await wiring.indexing_service_for(tenant) is wiring.indexing_service()


async def test_a_reranking_attach_takes_effect_immediately_and_reverts_on_delete(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from svc_control_plane import wiring

    from rag_clients.factory import get_reranker
    from rag_clients.reranker import OpenAICompatReranker
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(
            f"/v1/kb/{tenant}/model-connections",
            json={
                "name": "bge-reranker",
                "endpoint": "https://models.sovereign.local/v1",
                "model": "bge-reranker",
                "role": "reranking",
                "api_shape": "cohere+sigmoid",
                "api_key": "k",
            },
        )
        assert r.status_code == 201
        reranker = await wiring.resolve_tenant_reranker(tenant)
        assert isinstance(reranker, OpenAICompatReranker)

        # ...and the retrieval service is now per-tenant, not the platform singleton.
        assert await wiring.retrieval_service_for(tenant) is not wiring.retrieval_service()

        # Remove it: the platform reranker is back and the singleton service with it.
        conn_id = r.json()["conn_id"]
        assert (await c.delete(f"/v1/kb/{tenant}/model-connections/{conn_id}")).status_code == 204
        assert await wiring.resolve_tenant_reranker(tenant) is get_reranker()
        assert await wiring.retrieval_service_for(tenant) is wiring.retrieval_service()


async def test_an_embedding_attach_alone_does_not_change_what_embeds(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving an embedding connection must NOT repoint queries/ingest — only the explicit
    embedder switch (apply) does. Until then the platform embedder still owns the space."""
    from svc_control_plane import wiring

    from rag_clients.factory import get_embedder
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # Ingest one document so the KB is non-empty (the immediate-activation shortcut
        # for empty KBs is exercised in the embedder-switch tests).
        import base64

        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "doc.txt",
                "mime": "text/plain",
                "content_b64": base64.b64encode(b"Leave policy: 20 days annually.").decode(),
            },
        )
        r = await c.post(
            f"/v1/kb/{tenant}/model-connections",
            json={
                "name": "bge-m3",
                "endpoint": "https://models.sovereign.local/v1",
                "model": "bge-m3",
                "role": "embedding",
                "embed_dim": 1024,
            },
        )
        assert r.status_code == 201
        assert await wiring.resolve_tenant_embedder(tenant) is get_embedder()
        assert await wiring.indexing_service_for(tenant) is wiring.indexing_service()


async def test_an_org_wide_reranker_serves_a_kb_that_has_none_of_its_own(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model added ORG-WIDE in the console is what the platform uses — the registry's whole
    point, already true of generation. A reranker can be adopted the moment it is added: no
    stored vector depends on it, so there is nothing to invalidate."""
    from svc_control_plane import wiring

    from rag_clients.reranker import OpenAICompatReranker
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        r = await c.post(
            "/v1/org/model-connections",
            json={
                "name": "BGE reranker (shared)",
                "endpoint": "https://models.sovereign.local/v1",
                "model": "bge-reranker-v2-m3",
                "role": "reranking",
                "api_shape": "cohere+sigmoid",
                "api_key": "k",
            },
        )
        assert r.status_code == 201, r.text
        reranker = await wiring.resolve_tenant_reranker(kb)
        assert isinstance(reranker, OpenAICompatReranker)
        assert reranker._model == "bge-reranker-v2-m3"

        # The KB's OWN attach still wins over the org default.
        assert (
            await c.post(
                f"/v1/kb/{kb}/model-connections",
                json={
                    "name": "kb reranker",
                    "endpoint": "https://models.sovereign.local/v1",
                    "model": "kb-local-reranker",
                    "role": "reranking",
                    "api_shape": "tei",
                    "api_key": "k",
                },
            )
        ).status_code == 201
        assert (await wiring.resolve_tenant_reranker(kb))._model == "kb-local-reranker"


async def test_an_org_wide_embedder_that_matches_the_corpus_is_adopted_with_no_re_embed(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one case where an embedder CAN change hands for free: the org connection names the same
    model at the same width as the embedder that built the corpus, so it is the same vector space —
    a different endpoint and credential for the identical vectors. Adopting it is safe, and staging
    a re-embed would be an expensive no-op. A DIFFERENT org embedder is staged instead; that is
    tests/e2e/test_org_model_roles.py."""
    from svc_control_plane import wiring

    from rag_clients.openai_embedder import OpenAICompatEmbedder
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    settings = get_settings()
    org = "cag"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR"})).json()["kb_id"]
        import base64

        await c.post(
            f"/v1/kb/{kb}/ingest",
            json={
                "doc_id": "doc.txt",
                "mime": "text/plain",
                "content_b64": base64.b64encode(b"Leave policy: 20 days annually.").decode(),
            },
        )
        matching = await c.post(
            "/v1/org/model-connections",
            json={
                "name": "Embedding",
                "endpoint": "https://models.sovereign.local/v1",
                "model": settings.embed_model,
                "role": "embedding",
                "embed_dim": settings.embed_dim,
                "api_key": "k",
            },
        )
        assert matching.status_code == 201, matching.text
        embedder = await wiring.resolve_tenant_embedder(kb)
        assert isinstance(embedder, OpenAICompatEmbedder)
        assert embedder.dim == settings.embed_dim
        assert (await c.get("/v1/org/embedder/status")).json()["pending"] is None, (
            "the same vector space is not a switch — re-embedding it would be an expensive no-op"
        )
