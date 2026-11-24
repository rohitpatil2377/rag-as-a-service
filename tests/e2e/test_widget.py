"""E2E: FR-I6 embeddable widget — config + the key-never-in-browser session exchange."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _app():
    from rag_api.app import create_app

    return create_app()


async def _admin(app, tenant: str) -> httpx.AsyncClient:
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("admin", tenant, roles=["reader", "ingestor", "admin"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {token}"}
    )


async def _mint_api_key(app, tenant: str, scope: str) -> str:
    """Create a real service-account API key for `tenant` (org) with `scope`."""
    async with await _admin(app, tenant) as c:
        r = await c.post("/v1/api-keys", json={"name": "widget", "scope": scope})
        return r.json()["key"]


async def test_widget_session_exchange(inmemory: None) -> None:
    app = _app()
    tenant = "widgetco"
    async with await _admin(app, tenant) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        body = base64.b64encode(b"Acme Cloud guarantees 99.9% uptime.").decode()
        await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "d", "mime": "text/plain", "content_b64": body, "strategy": "clause"},
        )
        # Session refused until the widget is enabled.
        await c.put(f"/v1/kb/{tenant}/widget/config", json={"enabled": True, "title": "Ask Acme", "accent": "#1a3d7c"})

    key = await _mint_api_key(app, tenant, "read")

    # Host SERVER exchanges its read key for a short-lived browser token (key stays server-side).
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {key}"}
    ) as srv:
        sess = await srv.post("/v1/widget/session", json={"kb_id": tenant})
        assert sess.status_code == 200
        data = sess.json()
        assert data["expires_in"] > 0 and data["config"]["title"] == "Ask Acme"
        widget_token = data["token"]

    # The BROWSER token (no API key) can query — and only this KB.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {widget_token}"}
    ) as browser:
        ans = await browser.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime?"})
        assert ans.status_code == 200
        assert ans.json()["status"] == "grounded"
        # Bound to one KB: a different KB is rejected.
        other = await browser.post("/v1/kb/otherorg/query", json={"query": "x"})
        assert other.status_code == 403
        # Reader-only: it cannot ingest.
        blocked = await browser.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "x", "mime": "text/plain", "content_b64": "eA==", "strategy": "clause"},
        )
        assert blocked.status_code == 403


async def test_widget_session_refused_when_disabled(inmemory: None) -> None:
    app = _app()
    tenant = "widgetoff"
    async with await _admin(app, tenant) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
    key = await _mint_api_key(app, tenant, "read")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {key}"}
    ) as srv:
        resp = await srv.post("/v1/widget/session", json={"kb_id": tenant})
        assert resp.status_code == 404


async def test_widget_session_rejects_foreign_kb(inmemory: None) -> None:
    app = _app()
    key = await _mint_api_key(app, "orgA", "read")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", headers={"Authorization": f"Bearer {key}"}
    ) as srv:
        resp = await srv.post("/v1/widget/session", json={"kb_id": "orgB"})
        assert resp.status_code in (403, 404)
