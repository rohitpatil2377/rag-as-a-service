"""E2E: FR-A9 web crawler — bounded BFS over a stubbed site, sovereign + scoped."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e

_SITE = {
    "https://site.local/": "<html><body><h1>Home</h1><p>Welcome to the docs portal.</p>"
    '<a href="/a">A</a> <a href="/b">B</a> <a href="https://evil.local/x">off</a></body></html>',
    "https://site.local/a": "<html><body><h1>Page A</h1><p>Alpha content about uptime targets.</p>"
    '<a href="/c">C</a></body></html>',
    "https://site.local/b": "<html><body><h1>Page B</h1><p>Beta content about security.</p></body></html>",
    "https://site.local/c": "<html><body><h1>Page C</h1><p>Deeper page, depth two.</p></body></html>",
}


def _allow(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,site.local")
    get_settings.cache_clear()


def _stub_web(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_ingestion import connectors

    from rag_common import assert_sovereign

    class _Stub(connectors.WebConnector):
        async def fetch(self, doc):  # type: ignore[no-untyped-def]
            assert_sovereign(doc.uri or "")  # keep sovereignty enforcement
            html = _SITE.get(doc.uri or "")
            if html is None:
                from rag_common import NotFoundError

                raise NotFoundError("not found", details={"url": doc.uri})
            return html.encode()

    monkeypatch.setitem(connectors._REGISTRY, "web", lambda ctx: _Stub())


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_crawl_depth_and_scope(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch)
    _stub_web(monkeypatch)
    kb = "crawlco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # depth 1: seed + its direct links (a, b) — NOT c (depth 2), NOT the offsite host.
        r = (
            await c.post(
                f"/v1/kb/{kb}/ingest/crawl",
                json={
                    "seed_url": "https://site.local/",
                    "max_depth": 1,
                    "max_pages": 10,
                    "same_host_only": True,
                    "respect_robots": False,
                },
            )
        ).json()
        assert r["pages"] == 3
        docs = {d["doc_id"] for d in (await c.get(f"/v1/kb/{kb}/documents")).json()}
        assert docs == {"https://site.local/", "https://site.local/a", "https://site.local/b"}
        assert "https://site.local/c" not in docs  # depth-limited


async def test_crawl_depth_two_and_page_cap(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch)
    _stub_web(monkeypatch)
    kb = "crawlcap"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # depth 2 reaches c via a; but cap at 2 pages.
        r = (
            await c.post(
                f"/v1/kb/{kb}/ingest/crawl",
                json={
                    "seed_url": "https://site.local/",
                    "max_depth": 2,
                    "max_pages": 2,
                    "same_host_only": True,
                    "respect_robots": False,
                },
            )
        ).json()
        assert r["pages"] == 2  # page cap honoured


async def test_crawl_requires_ingestor(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    _allow(monkeypatch)
    app = create_app()
    reader = issue_dev_jwt("ro", "crawlro", roles=["reader"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {reader}"}
    ) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        resp = await c.post("/v1/kb/crawlro/ingest/crawl", json={"seed_url": "https://site.local/"})
        assert resp.status_code == 403
