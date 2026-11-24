"""E2E: the embedder-switch state machine (FR-E1 embedding role).

The invariant under test: saving an embedding attach NEVER silently re-points a KB
that has documents — it stages a PENDING switch; only the explicit apply rebuilds the
vector space (and an empty KB activates immediately because nothing indexed can
disagree). Deleting the embedder a live index depends on is refused.

The OpenAI-compat embedder is faked at the wiring seam so no network is involved;
what matters here is the state machine, not the arithmetic.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_ENDPOINT = "https://models.sovereign.local/v1"


class FakeEmbedder:
    """Stands in for OpenAICompatEmbedder: right shape, no network."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        dim: int,
        api_key: str = "",
        api_key_header: str = "apikey",
        timeout: float | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Deterministic, text-sensitive, dim-correct.
        return [[(len(t) % 7) / 7.0] * self._dim for t in texts]


async def _client(tenant: str, *, admin: bool = True) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    roles = ["reader", "ingestor", "admin"] if admin else ["reader", "ingestor"]
    token = issue_dev_jwt("officer", tenant, roles=roles)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _setup(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    monkeypatch.setattr(wiring, "OpenAICompatEmbedder", FakeEmbedder)


async def _ingest(c: httpx.AsyncClient, tenant: str, name: str, text: str) -> None:
    r = await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={
            "doc_id": name,
            "mime": "text/plain",
            "content_b64": base64.b64encode(text.encode()).decode(),
        },
    )
    assert r.status_code in (200, 201), r.text


def _attach_body(model: str = "bge-m3", dim: int = 1024) -> dict[str, object]:
    return {
        "name": model,
        "endpoint": _ENDPOINT,
        "model": model,
        "role": "embedding",
        "embed_dim": dim,
    }


async def test_attach_with_documents_stages_pending_and_apply_rebuilds(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from svc_control_plane import wiring

    _setup(monkeypatch)
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "leave.txt", "Leave policy: 20 days annually, prorated by joining month.")
        await _ingest(c, tenant, "expenses.txt", "Expense claims need a receipt within 30 days.")

        r = await c.post(f"/v1/kb/{tenant}/model-connections", json=_attach_body())
        assert r.status_code == 201

        # Staged, not live: status shows pending, the resolver still returns the platform embedder.
        status = (await c.get(f"/v1/kb/{tenant}/embedder/status")).json()
        assert status["active"] is None and status["pending"]["model"] == "bge-m3"
        assert status["documents"] == 2
        from rag_clients.factory import get_embedder

        assert await wiring.resolve_tenant_embedder(tenant) is get_embedder()

        # Queries keep answering from the old space while pending.
        ask = await c.post(f"/v1/kb/{tenant}/query", json={"query": "How many leave days?", "top_k": 3})
        assert ask.status_code == 200

        # Apply: flips active, re-embeds every document, and audits the switch.
        applied = (await c.post(f"/v1/kb/{tenant}/embedder/apply")).json()
        assert applied["reindexed"] == 2 and applied["failed"] == []
        status = (await c.get(f"/v1/kb/{tenant}/embedder/status")).json()
        assert status["pending"] is None and status["active"]["model"] == "bge-m3"

        resolved = await wiring.resolve_tenant_embedder(tenant)
        assert isinstance(resolved, FakeEmbedder) and resolved.dim == 1024

        # The KB still answers — the whole corpus now lives in the new space.
        ask = await c.post(f"/v1/kb/{tenant}/query", json={"query": "How many leave days?", "top_k": 3})
        assert ask.status_code == 200

        # The switch is on the audit chain.
        audit = (await c.get(f"/v1/kb/{tenant}/audit?limit=50")).json()
        switch_events = [e for e in audit if e["action"] == "index" and e.get("payload", {}).get("embedder_switch")]
        assert switch_events and switch_events[0]["payload"]["to"].startswith(_ENDPOINT)

        # Nothing pending any more: a second apply is a 409, not a silent rebuild.
        assert (await c.post(f"/v1/kb/{tenant}/embedder/apply")).status_code == 409


async def test_attach_to_an_empty_kb_activates_immediately(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    _setup(monkeypatch)
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(f"/v1/kb/{tenant}/model-connections", json=_attach_body())
        assert r.status_code == 201
        status = (await c.get(f"/v1/kb/{tenant}/embedder/status")).json()
        assert status["pending"] is None and status["active"]["model"] == "bge-m3"
        resolved = await wiring.resolve_tenant_embedder(tenant)
        assert isinstance(resolved, FakeEmbedder)

        # Ingest lands directly in the new space; no re-embed ever needed.
        await _ingest(c, tenant, "leave.txt", "Leave policy: 20 days annually.")
        ask = await c.post(f"/v1/kb/{tenant}/query", json={"query": "leave days", "top_k": 3})
        assert ask.status_code == 200


async def test_deleting_the_active_embedder_with_documents_is_refused(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        conn = (await c.post(f"/v1/kb/{tenant}/model-connections", json=_attach_body())).json()
        await _ingest(c, tenant, "leave.txt", "Leave policy: 20 days annually.")

        r = await c.delete(f"/v1/kb/{tenant}/model-connections/{conn['conn_id']}")
        assert r.status_code == 409
        # Still attached, still active.
        status = (await c.get(f"/v1/kb/{tenant}/embedder/status")).json()
        assert status["active"]["model"] == "bge-m3"


async def test_apply_requires_admin(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(monkeypatch)
    tenant = "cag"
    admin = await _client(tenant)
    reader = await _client(tenant, admin=False)
    async with admin, reader:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(admin, tenant, "leave.txt", "Leave policy: 20 days annually.")
        await admin.post(f"/v1/kb/{tenant}/model-connections", json=_attach_body())
        assert (await reader.post(f"/v1/kb/{tenant}/embedder/apply")).status_code == 403
