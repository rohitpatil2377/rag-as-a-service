"""E2E: KB delete safeguards — soft-delete to Trash, restore, and purge with typed confirmation."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["owner", "admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_delete_is_soft_then_restore_then_purge(inmemory: None) -> None:
    org = await _client("cag")
    async with org:
        kb = (await org.post("/v1/knowledge-bases", json={"display_name": "HR Policies"})).json()
        kb_id = kb["kb_id"]
        await org.post(
            f"/v1/kb/{kb_id}/ingest",
            json={"doc_id": "d1", "mime": "text/plain", "content_b64": base64.b64encode(b"leave policy").decode()},
        )

        # DELETE is a soft-delete → moves to Trash, not purged.
        r = await org.delete(f"/v1/knowledge-bases/{kb_id}")
        assert r.status_code == 200 and r.json()["status"] == "trashed"
        assert kb_id not in {k["kb_id"] for k in (await org.get("/v1/knowledge-bases")).json()}  # hidden from active
        assert kb_id in {k["kb_id"] for k in (await org.get("/v1/knowledge-bases/trash")).json()}  # in Trash

        # Restore → back in the active list; its document is still there (nothing was purged).
        assert (await org.post(f"/v1/knowledge-bases/{kb_id}/restore")).status_code == 200
        assert kb_id in {k["kb_id"] for k in (await org.get("/v1/knowledge-bases")).json()}
        assert len((await org.get(f"/v1/kb/{kb_id}/documents")).json()) == 1

        # Purge requires the exact typed name — a wrong name is refused.
        await org.delete(f"/v1/knowledge-bases/{kb_id}")  # back to trash
        assert (await org.post(f"/v1/knowledge-bases/{kb_id}/purge", json={"confirm_name": "wrong"})).status_code == 422
        # Correct name → permanent delete, gone from Trash and the document purged.
        ok = await org.post(f"/v1/knowledge-bases/{kb_id}/purge", json={"confirm_name": "HR Policies"})
        assert ok.status_code == 200 and ok.json()["documents"] == 1
        assert kb_id not in {k["kb_id"] for k in (await org.get("/v1/knowledge-bases/trash")).json()}


async def test_purge_requires_admin(inmemory: None) -> None:
    from rag_common import issue_dev_jwt

    org = await _client("cag")
    async with org:
        kb_id = (await org.post("/v1/knowledge-bases", json={"display_name": "Finance"})).json()["kb_id"]
        await org.delete(f"/v1/knowledge-bases/{kb_id}")
    # A reader-only principal may not purge.
    from rag_api.app import create_app

    app = create_app()
    reader = issue_dev_jwt("clerk", "cag", roles=["reader"])
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {reader}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        r = await c.post(f"/v1/knowledge-bases/{kb_id}/purge", json={"confirm_name": "Finance"})
        assert r.status_code == 403
