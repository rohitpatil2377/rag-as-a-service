"""E2E: FR-I7 chat bots — Mattermost outgoing-webhook adapter (token-auth, grounded reply)."""

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


async def test_mattermost_bot_answers_grounded(inmemory: None) -> None:
    app = _app()
    kb = "botco"
    async with await _admin(app, kb) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        body = base64.b64encode(b"Acme Cloud guarantees 99.9% uptime.").decode()
        await c.post(
            f"/v1/kb/{kb}/ingest",
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": body, "strategy": "clause"},
        )
        tok = (await c.post(f"/v1/kb/{kb}/bot/token")).json()
        assert tok["token"] and tok["webhook_path"] == f"/v1/bots/mattermost/{kb}"
        assert (await c.get(f"/v1/kb/{kb}/bot/config")).json()["configured"] is True

    # Mattermost posts a form-encoded outgoing-webhook payload; we answer synchronously.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as mm:
        resp = await mm.post(
            f"/v1/bots/mattermost/{kb}",
            data={
                "token": tok["token"],
                "text": "@rag What is the uptime?",
                "trigger_word": "@rag",
                "user_name": "alice",
            },
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["response_type"] == "comment"
        assert "99.9%" in out["text"]
        assert "Sources:" in out["text"]  # citations rendered

    # Wrong token is rejected.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as mm:
        bad = await mm.post(f"/v1/bots/mattermost/{kb}", data={"token": "nope", "text": "hi"})
        assert bad.status_code == 401


async def test_bot_disabled_by_default(inmemory: None) -> None:
    app = _app()
    kb = "botoff"
    async with await _admin(app, kb) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        cfg = (await c.get(f"/v1/kb/{kb}/bot/config")).json()
        assert cfg["enabled"] is False and cfg["configured"] is False
    # No token set → any inbound is rejected.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as mm:
        resp = await mm.post(f"/v1/bots/mattermost/{kb}", data={"token": "x", "text": "hi"})
        assert resp.status_code == 401
