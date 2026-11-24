"""E2E: web-page ingestion + external-bucket folder sync (P0-e, FR-A3/A2b batch).

The web connector is sovereignty-gated exactly like an external bucket: a URL on a
non-allowlisted host is refused before any network call. Folder sync batch-ingests every
key under a prefix, reporting per-file outcomes.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e

_HTML = b"""<html><head><title>SLA</title><style>body{color:red}</style>
<script>alert('x')</script></head>
<body><nav>menu</nav><h1>Service Levels</h1>
<p>The platform uptime target is 99.7 percent for the web tier.</p></body></html>"""


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _allow(monkeypatch: pytest.MonkeyPatch, *hosts: str) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", ",".join(("localhost", *hosts)))
    get_settings.cache_clear()


async def test_web_ingest_extracts_visible_text(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch, "docs.sovereign.local")

    # Stub the page fetch (no real network in tests) — sovereignty still enforced upstream.
    from svc_ingestion import connectors

    class _StubWeb(connectors.WebConnector):
        async def fetch(self, doc):  # type: ignore[no-untyped-def]
            from rag_common import assert_sovereign

            assert_sovereign(doc.uri or "")
            return _HTML

    monkeypatch.setitem(connectors._REGISTRY, "web", lambda ctx: _StubWeb())

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "sla-page", "source_kind": "web", "url": "https://docs.sovereign.local/sla"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        # The page is answerable — and scripts/styles never entered the index.
        ans = (await client.post(f"/v1/kb/{tenant}/query", json={"query": "uptime target for the web tier?"})).json()
        assert "99.7" in ans["answer"]
        spans = (await client.post(f"/v1/kb/{tenant}/retrieve", json={"query": "uptime"})).json()["spans"]
        assert all("alert(" not in s["text"] and "color:red" not in s["text"] for s in spans)


async def test_web_ingest_refuses_non_allowlisted_host(inmemory: None) -> None:
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest", json={"doc_id": "x", "source_kind": "web", "url": "https://example.com/page"}
        )
        assert r.status_code == 403
        assert r.json()["code"] == "sovereignty_violation"


async def test_folder_sync_ingests_prefix(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch, "obj.sovereign.local")
    monkeypatch.setenv("RAG_SRC1_ACCESS_KEY", "ak")
    monkeypatch.setenv("RAG_SRC1_SECRET_KEY", "sk")

    from rag_clients.factory import get_external_object_fetcher

    fetcher = get_external_object_fetcher()
    fetcher.seed("obj.sovereign.local", "exports", "reports/a.txt", b"Report A: revenue rose 12 percent.")  # type: ignore[attr-defined]
    fetcher.seed("obj.sovereign.local", "exports", "reports/b.txt", b"Report B: costs fell 3 percent.")  # type: ignore[attr-defined]
    fetcher.seed("obj.sovereign.local", "exports", "other/c.txt", b"unrelated")  # type: ignore[attr-defined]

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest/sync",
            json={
                "external_source": {
                    "endpoint": "obj.sovereign.local",
                    "bucket": "exports",
                    "secure": True,
                    "credentials_ref": "src1",
                },
                "prefix": "reports/",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["found"] == 2 and body["ingested"] == 2 and body["failed"] == 0

        docs = {d["doc_id"] for d in (await client.get(f"/v1/kb/{tenant}/documents")).json()}
        assert docs == {"reports/a.txt", "reports/b.txt"}  # the other/ prefix was not touched
